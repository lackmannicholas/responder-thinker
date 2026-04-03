# VAD Integration Task Breakdown

See `docs/vad-integration-spec.md` for full context, interface contracts, and config.

---

## Task List

### T1 — Add `ten-vad` dependency
**File:** `pyproject.toml`
**Scope:** Add `"ten-vad"` (or `"ten_vad"`) to the `dependencies` list.
**Note:** Confirm package name on PyPI before adding. TEN VAD ships an ONNX runtime
dependency; verify it doesn't conflict with existing `av`/`numpy` pins.
**Independent of all other tasks.**

---

### T2 — Add `VADConfig` and wire into `Settings`
**File:** `backend/config.py`

1. Define `VADConfig` as a `pydantic.BaseModel` with fields:
   `enabled`, `threshold`, `vad_sample_rate`, `vad_frame_ms`,
   `pre_roll_ms`, `post_roll_ms`, `hangover_frames`
   (defaults in spec §3.2).
2. Add `vad: VADConfig = VADConfig()` to `Settings`.

Pydantic-settings will pick up `VAD__*` env vars automatically.
**Independent of all other tasks.**

---

### T3 — Add `AudioConverter.aiortc_frame_to_pcm16`
**File:** `backend/audio_convert.py`

Add a new public method that runs the existing downsampler and returns raw `bytes`
instead of base64. The existing `aiortc_frame_to_realtime_b64` can be reimplemented
to call this method + `pcm16_bytes_to_b64` (already exists at line 105), or left as-is
since both share the same `_downsampler` instance and the loop will call one or the
other — not both — per frame.

Signature:
```python
def aiortc_frame_to_pcm16(self, frame: av.AudioFrame) -> bytes:
    """Returns raw PCM16 bytes at 24kHz mono. Returns b"" while resampler buffers."""
```

**Independent of all other tasks.**

---

### T4 — Implement `VADGate`
**File:** `backend/vad.py` (new file)

Implement `VADResult` dataclass and `VADGate` class per spec §3.3–§3.4.

Internal structure:
- `_vad`: TEN VAD model instance (loaded once in `__init__`)
- `_resampler`: `av.AudioResampler(format="s16", layout="mono", rate=16000)` for
  downsampling the 24kHz input to 16kHz for TEN VAD
- `_vad_buffer`: `bytearray` accumulating samples until a full VAD frame is ready
- `_pre_roll`: `collections.deque` of recent PCM16 byte chunks (24kHz),
  max length = `ceil(pre_roll_ms / 20)` frames
- `_hangover_remaining`: `int` countdown for post-roll

State machine for `process()`:
```
SILENCE ──(prob >= threshold)──► SPEECH
SPEECH  ──(prob < threshold)───► HANGOVER  (start countdown)
HANGOVER──(countdown > 0)──────► HANGOVER  (keep sending, decrement)
HANGOVER──(countdown == 0)─────► SILENCE
HANGOVER──(prob >= threshold)──► SPEECH    (re-triggered)
```

`frames_to_flush` contents per state:
- **SILENCE**: `[]` (chunk goes into pre-roll ring buffer)
- **SILENCE → SPEECH onset**: `list(pre_roll_deque) + [current_chunk]`
- **SPEECH**: `[current_chunk]`
- **HANGOVER**: `[current_chunk]`
- **HANGOVER → SILENCE**: `[]`

**Depends on T1** (needs `ten_vad` installed to import).
**Does not depend on T2 or T3** — accepts `VADConfig` as constructor arg; can be
developed and unit-tested with a manually constructed `VADConfig`.

---

### T5 — Wire `VADGate` into `_audio_input_loop`
**File:** `backend/transport/realtime_bridge.py`

1. In `RealtimeBridge.__init__`: construct `self._vad_gate: VADGate | None`
   based on `settings.vad.enabled`.
2. Replace the inner loop body at lines 525–541 per spec §3.5:
   - Call `self._audio_converter.aiortc_frame_to_pcm16(frame)` → `pcm16_bytes`
   - If VAD enabled: route through `self._vad_gate.process(pcm16_bytes)`
   - Send each chunk in `frames_to_flush` as a separate `input_audio_buffer.append`
3. Import `pcm16_bytes_to_b64` from `audio_convert` (already defined at line 105).

**Depends on T3 and T4.**

---

### T6 — (Optional) Disable OpenAI VAD, manage turn commits locally
**File:** `backend/transport/realtime_bridge.py`

If you want full local VAD control (Option B from spec §2):

1. In `_configure_session`: change `"turn_detection": {"type": "none"}` when
   `settings.vad.enabled` is True.
2. On `VADGate` state transition SPEECH → HANGOVER (or HANGOVER → SILENCE):
   send `input_audio_buffer.commit` + `response.create` to the Realtime WS.
3. Propagate speech-end notification from `VADGate` back to `_audio_input_loop`
   (e.g., add a `speech_ended: bool` field to `VADResult`).

**Depends on T5. Implement after T5 is stable.**

---

## Dependency Graph

```
T1 ──────────────────────────────┐
                                  ▼
T2 (independent)            T4 (VADGate impl)
                                  │
T3 (independent)                  │
   │                              │
   └──────────────────────────────┤
                                  ▼
                            T5 (wire into loop)
                                  │
                                  ▼
                            T6 (optional, full local VAD)
```

**Parallel work possible:**
- T1, T2, T3 have no inter-dependencies — all can be done simultaneously.
- T4 can start in parallel with T2 and T3 once T1 is installed.

**Sequential gates:**
- T5 must wait for both T3 and T4.
- T6 must wait for T5.

---

## File Scope by Task

| Task | Files Touched |
|---|---|
| T1 | `pyproject.toml` |
| T2 | `backend/config.py` |
| T3 | `backend/audio_convert.py` |
| T4 | `backend/vad.py` (new) |
| T5 | `backend/transport/realtime_bridge.py` |
| T6 | `backend/transport/realtime_bridge.py` |

---

## Verification Checklist (post-T5)

- [ ] `VAD__ENABLED=false` → behavior identical to today (no frames dropped)
- [ ] `VAD__ENABLED=true` → silence frames are not forwarded to OpenAI WS
- [ ] Speech onset: pre-roll audio included (no clipped word starts)
- [ ] Speech offset: post-roll audio included (no clipped word ends)
- [ ] Barge-in still works: `input_audio_buffer.speech_started` event path unchanged
- [ ] `audio_convert.aiortc_frame_to_realtime_b64` still works (existing callers,
  tests in `test_full_pipeline.py` and `test_webrtc_echo.py`)
