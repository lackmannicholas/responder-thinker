"""
VAD gate: wraps TEN VAD to suppress silence before forwarding audio to OpenAI.

One VADGate per RealtimeBridge (per session). Not thread-safe; call from a
single asyncio task (_audio_input_loop).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction

import av
import numpy as np
import ten_vad


@dataclass
class VADConfig:
    enabled: bool = True
    threshold: float = 0.5
    vad_sample_rate: int = 16000
    vad_frame_ms: int = 32
    pre_roll_ms: int = 100
    post_roll_ms: int = 300
    hangover_frames: int = 8


@dataclass
class VADResult:
    is_speech: bool
    speech_probability: float
    frames_to_flush: list[bytes]
    speech_started: bool = False  # True on SILENCE → SPEECH transition (onset)
    speech_ended: bool = False    # True on HANGOVER → SILENCE transition (offset)


class _State(Enum):
    SILENCE = auto()
    SPEECH = auto()
    HANGOVER = auto()


_INPUT_SAMPLE_RATE = 24000  # PCM16 input rate (from AudioConverter)


class VADGate:
    def __init__(self, config: VADConfig) -> None:
        self._config = config
        self._hop_size = int(config.vad_sample_rate * config.vad_frame_ms / 1000)
        self._vad_model = ten_vad.TenVad(hop_size=self._hop_size)

        # 24kHz → 16kHz resampler, for VAD inference only
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=config.vad_sample_rate
        )

        # Pre-roll ring buffer: holds 24kHz PCM16 byte chunks
        max_pre_roll = math.ceil(config.pre_roll_ms / 20)
        self._pre_roll: deque[bytes] = deque(maxlen=max_pre_roll)

        # Accumulated 16kHz int16 samples waiting for a full VAD frame
        self._vad_samples: list[int] = []

        self._state = _State.SILENCE
        self._hangover_count = 0
        self._last_prob = 0.0

    def _downsample(self, pcm16_bytes: bytes) -> np.ndarray:
        """Return 16kHz int16 samples for the given 24kHz PCM16 bytes."""
        audio_np = np.frombuffer(pcm16_bytes, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(audio_np.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = _INPUT_SAMPLE_RATE
        frame.time_base = Fraction(1, _INPUT_SAMPLE_RATE)
        resampled = self._resampler.resample(frame)
        if not resampled:
            return np.array([], dtype=np.int16)
        chunks = [rf.to_ndarray().flatten().astype(np.int16) for rf in resampled]
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    def process(self, pcm16_bytes: bytes) -> VADResult:
        """
        Process one 24kHz mono PCM16 chunk through the VAD gate.

        Returns a VADResult whose frames_to_flush contains the 24kHz chunks
        that should be forwarded to the Realtime API this cycle.
        """
        # 1. Downsample for VAD inference and accumulate
        samples_16k = self._downsample(pcm16_bytes)
        if samples_16k.size:
            self._vad_samples.extend(samples_16k.tolist())

        # 2. Run VAD inference on every complete hop
        prob = self._last_prob
        while len(self._vad_samples) >= self._hop_size:
            frame = np.array(self._vad_samples[: self._hop_size], dtype=np.int16)
            self._vad_samples = self._vad_samples[self._hop_size :]
            prob, _ = self._vad_model.process(frame)
        self._last_prob = prob

        # 3. Apply state machine using the latest probability
        is_speech_frame = prob >= self._config.threshold
        frames_to_flush: list[bytes] = []
        speech_started = False
        speech_ended = False

        if self._state == _State.SILENCE:
            if is_speech_frame:
                # Onset: flush pre-roll + current chunk
                frames_to_flush = list(self._pre_roll) + [pcm16_bytes]
                self._pre_roll.clear()
                self._state = _State.SPEECH
                speech_started = True
            else:
                self._pre_roll.append(pcm16_bytes)

        elif self._state == _State.SPEECH:
            frames_to_flush = [pcm16_bytes]
            if not is_speech_frame:
                self._state = _State.HANGOVER
                self._hangover_count = self._config.hangover_frames

        elif self._state == _State.HANGOVER:
            if is_speech_frame:
                self._state = _State.SPEECH
                frames_to_flush = [pcm16_bytes]
            elif self._hangover_count > 0:
                frames_to_flush = [pcm16_bytes]
                self._hangover_count -= 1
                # Remain in HANGOVER; transition to SILENCE happens next frame
                # when countdown == 0 is detected (HANGOVER → SILENCE yields []).
            else:
                # countdown == 0: speech end declared
                self._state = _State.SILENCE
                self._pre_roll.append(pcm16_bytes)
                speech_ended = True

        return VADResult(
            is_speech=self._state in (_State.SPEECH, _State.HANGOVER),
            speech_probability=prob,
            frames_to_flush=frames_to_flush,
            speech_started=speech_started,
            speech_ended=speech_ended,
        )

    def reset(self) -> None:
        """Clear all buffers and counters. Call on session start or hard interrupt."""
        self._pre_roll.clear()
        self._vad_samples.clear()
        self._state = _State.SILENCE
        self._hangover_count = 0
        self._last_prob = 0.0
