# VAD Integration Spec: TEN VAD Gate for Audio Input

## 1. Current Audio Flow

### Path: WebRTC inbound → OpenAI Realtime API

```
[Browser mic]
  getUserMedia({ sampleRate:24000, channelCount:1, echoCancellation, noiseSuppression, AGC })
  frontend/static/app.js:139-148

  │ WebRTC PeerConnection.addTrack
  frontend/static/app.js:158-161

  ▼
[aiortc RTCPeerConnection]
  Receives track, stores as session.input_track
  backend/transport/webrtc_server.py:244-253
  Format: 48kHz, mono, s16, 960 samples/frame (20ms)

  │ frame = await input_track.recv()
  backend/transport/realtime_bridge.py:525

  ▼
[AudioConverter.aiortc_frame_to_realtime_b64(frame)]
  backend/audio_convert.py:51-72
  - av.AudioResampler: 48kHz → 24kHz (anti-aliased, stateful)
  - Extract int16 array, np.concatenate
  - base64.b64encode → str
  Returns: base64-encoded PCM16 at 24kHz, or "" if resampler buffering

  │ audio_b64 = ...
  backend/transport/realtime_bridge.py:528

  ▼ ← VAD GATE INSERTION POINT (line 528–534)

[OpenAI Realtime API WebSocket send]
  await self._realtime_ws.send(json.dumps({
      "type": "input_audio_buffer.append",
      "audio": audio_b64,
  }))
  backend/transport/realtime_bridge.py:534-541
```

### Concurrent loops (realtime_bridge.py:250-256)

All four run under `asyncio.gather`:
- `_audio_input_loop` — hot path described above
- `_event_handler_loop` — reads OpenAI events, handles tool calls, barge-in
- `_idle_monitor_loop` — 15s nudge, 60s disconnect
- `_audio_drain_monitor_loop` — resets idle timer on audio playback end

### Current turn detection

Session configured at `realtime_bridge.py:285-303` with:
```python
"turn_detection": {"type": "semantic_vad"}
```
OpenAI's server-side VAD handles all turn boundaries today.

---

## 2. Proposed Audio Flow with VAD Gate

```
[aiortc input_track.recv()]
  backend/transport/realtime_bridge.py:525

  ▼
[AudioConverter.aiortc_frame_to_pcm16(frame)]    ← NEW METHOD
  backend/audio_convert.py
  - Same av.AudioResampler downsampling path (48kHz → 24kHz)
  - Returns raw bytes instead of base64 string
  Rationale: VAD and WebSocket send both need the PCM; avoid double resampler call

  │ pcm16_bytes: bytes (480 samples @ 24kHz = 20ms)

  ▼
[VADGate.process(pcm16_bytes)]                    ← NEW MODULE
  backend/vad.py
  - Downsample 24kHz → 16kHz for TEN VAD
  - Accumulate samples into VAD frame size (512 samples @ 16kHz = 32ms)
  - Run ten_vad.process() → speech probability
  - Apply threshold, pre-roll buffer, post-roll hangover
  Returns: VADResult(is_speech: bool, frames_to_flush: list[bytes])

  │ result.frames_to_flush — frames to send (includes pre-roll if onset detected)

  ▼ (only if result.frames_to_flush is non-empty)
[pcm16_bytes_to_b64(frame)]
  backend/audio_convert.py:105-107 (existing helper)
  base64.b64encode(pcm_bytes)

  ▼
[OpenAI Realtime API WebSocket send]
  backend/transport/realtime_bridge.py
  Same input_audio_buffer.append event
```

### Turn detection: two options

**Option A — Gate + keep OpenAI VAD (recommended for initial integration)**
Local VAD gates what audio reaches OpenAI. OpenAI's `semantic_vad` still handles
turn boundaries. Minimal disruption; turn detection quality stays the same.

**Option B — Gate + disable OpenAI VAD (full local control)**
Set `"turn_detection": {"type": "none"}` in session config.
On speech-end detection: send `input_audio_buffer.commit` + `response.create`.
More control over latency but requires managing turn state in `_event_handler_loop`.

Implement Option A first; Option B is a follow-on.

---

## 3. Interface Contracts

### 3.1 `AudioConverter.aiortc_frame_to_pcm16` (new method)

**File:** `backend/audio_convert.py`

```python
def aiortc_frame_to_pcm16(self, frame: av.AudioFrame) -> bytes:
    """
    Convert an aiortc AudioFrame to raw PCM16 bytes at 24kHz mono.

    Uses the same stateful downsampler as aiortc_frame_to_realtime_b64.
    Returns empty bytes b"" if the resampler is still buffering (first call).

    Caller is responsible for base64 encoding if the Realtime API is the
    destination. Use audio_convert.pcm16_bytes_to_b64(pcm_bytes) for that.
    """
```

- Input: `av.AudioFrame` (48kHz, mono, s16, 960 samples)
- Output: `bytes` — raw int16 little-endian PCM at 24kHz; `b""` on resampler buffering
- Side effects: advances `self._downsampler` internal filter state (stateful)
- Thread safety: not thread-safe; per-session instance only

### 3.2 `VADConfig`

**File:** `backend/config.py` — add as a nested model on `Settings`, or as a standalone dataclass.

```python
class VADConfig(BaseModel):
    enabled: bool = True

    # Detection threshold
    threshold: float = 0.5          # TEN VAD probability [0.0, 1.0]; raise to be less sensitive

    # TEN VAD internal parameters
    vad_sample_rate: int = 16000    # TEN VAD supports 8000 or 16000
    vad_frame_ms: int = 32          # ms per VAD inference frame (16 or 32 at 16kHz)

    # Buffering
    pre_roll_ms: int = 100          # audio to prepend before detected speech onset
    post_roll_ms: int = 300         # silence to append after detected speech offset
    hangover_frames: int = 8        # consecutive sub-threshold frames before speech-end
```

Add to `Settings`:
```python
vad: VADConfig = VADConfig()
```

Env-var surface (pydantic-settings flattening):
```
VAD__ENABLED=true
VAD__THRESHOLD=0.5
VAD__PRE_ROLL_MS=100
VAD__POST_ROLL_MS=300
VAD__HANGOVER_FRAMES=8
```

### 3.3 `VADResult`

```python
@dataclass
class VADResult:
    is_speech: bool                 # current frame classified as speech
    speech_probability: float       # raw TEN VAD probability [0.0, 1.0]
    frames_to_flush: list[bytes]    # PCM16 byte chunks to send this cycle
                                    # empty when silent and no buffered flush
```

### 3.4 `VADGate`

**File:** `backend/vad.py`

```python
class VADGate:
    def __init__(self, config: VADConfig) -> None:
        """
        Create a VAD gate for a single audio session.

        Owns:
        - TEN VAD model instance
        - 16kHz resampler (24kHz → 16kHz, for VAD inference only)
        - Pre-roll ring buffer (deque of raw PCM16 byte chunks)
        - Hangover counter

        One VADGate per RealtimeBridge (per session).
        """

    def process(self, pcm16_bytes: bytes) -> VADResult:
        """
        Process one PCM16 chunk (24kHz mono) through the VAD gate.

        Steps:
          1. Downsample pcm16_bytes to 16kHz (VAD inference sample rate)
          2. Buffer into vad_frame_ms-sized chunks
          3. For each complete VAD frame: run ten_vad, get probability
          4. Apply threshold and hangover logic to determine state transition
          5. Build frames_to_flush:
             - On onset  (silence→speech): flush pre-roll buffer + current chunk
             - In speech : yield current chunk immediately
             - On offset (speech→silence): yield post-roll via hangover countdown
             - In silence: hold in pre-roll ring buffer only
          6. Maintain pre-roll ring buffer (discard oldest when full)
          7. Return VADResult

        Not thread-safe. Call from a single asyncio task (_audio_input_loop).
        """

    def reset(self) -> None:
        """
        Clear all internal state (buffers, counters).
        Call on session start or after a hard interrupt.
        """
```

### 3.5 `_audio_input_loop` modifications

**File:** `backend/transport/realtime_bridge.py:503-554`

Replace the existing inner loop body:

```python
# Before (lines 525-541):
frame = await input_track.recv()
audio_b64 = self._audio_converter.aiortc_frame_to_realtime_b64(frame)
if not audio_b64:
    continue
await self._realtime_ws.send(json.dumps({
    "type": "input_audio_buffer.append",
    "audio": audio_b64,
}))

# After:
frame = await input_track.recv()
pcm16_bytes = self._audio_converter.aiortc_frame_to_pcm16(frame)
if not pcm16_bytes:
    continue

if self._vad_gate is not None:
    result = self._vad_gate.process(pcm16_bytes)
    chunks_to_send = result.frames_to_flush
else:
    chunks_to_send = [pcm16_bytes]

for chunk in chunks_to_send:
    await self._realtime_ws.send(json.dumps({
        "type": "input_audio_buffer.append",
        "audio": pcm16_bytes_to_b64(chunk),
    }))
```

`self._vad_gate` is `VADGate | None` — `None` when `settings.vad.enabled` is False.
Constructed in `RealtimeBridge.__init__` alongside `self._audio_converter`.

---

## 4. Config Surface Summary

| Parameter | Default | Effect |
|---|---|---|
| `VAD__ENABLED` | `true` | Master switch; `false` restores current passthrough behavior |
| `VAD__THRESHOLD` | `0.5` | Raise (→1.0) to require louder speech; lower (→0.0) to be more permissive |
| `VAD__PRE_ROLL_MS` | `100` | ms of audio prepended on speech onset; prevents clipping word starts |
| `VAD__POST_ROLL_MS` | `300` | ms of audio appended on speech offset; prevents clipping word ends |
| `VAD__HANGOVER_FRAMES` | `8` | Consecutive sub-threshold frames before speech end declared (~250ms at 32ms/frame) |
| `VAD__VAD_FRAME_MS` | `32` | TEN VAD inference frame size; 16 or 32 supported |
| `VAD__VAD_SAMPLE_RATE` | `16000` | TEN VAD input rate; 8000 or 16000 |

### Latency impact

- VAD inference per 32ms frame: ~1–3ms CPU (Silero/ONNX)
- Pre-roll adds 0ms latency to forwarded audio (buffered, flushed on onset)
- No extra network round-trips; gate only affects whether `input_audio_buffer.append` is sent

### Interaction with OpenAI's `semantic_vad`

With Option A (recommended): OpenAI's VAD remains active. Local VAD reduces total
audio bytes sent to OpenAI and eliminates background noise frames from transcription.
OpenAI's VAD still controls when it commits the buffer and generates a response.

---

## 5. New Files

| File | Role |
|---|---|
| `backend/vad.py` | `VADGate`, `VADResult`, `VADConfig` |

## 6. Modified Files

| File | Change |
|---|---|
| `backend/audio_convert.py` | Add `aiortc_frame_to_pcm16` method |
| `backend/config.py` | Add `vad: VADConfig` field to `Settings` |
| `backend/transport/realtime_bridge.py` | Construct `VADGate`, wire into `_audio_input_loop` |
| `pyproject.toml` | Add `ten-vad` dependency |
