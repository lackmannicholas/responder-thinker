"""
Audio format conversion between aiortc and OpenAI Realtime API.

aiortc delivers audio as av.AudioFrame objects:
  - Default: 48000 Hz, stereo (2 channels), s16 format
  - Frame size: 960 samples per channel per frame (20ms at 48kHz)

OpenAI Realtime API expects/produces:
  - 24000 Hz, mono (1 channel), PCM16 (signed 16-bit little-endian)

This module handles the conversion in both directions using av.AudioResampler
(backed by libswresample) for proper anti-aliased resampling.

IMPORTANT: Resamplers are stateful — they maintain internal filter state
for gapless audio across consecutive frames. Each session MUST use its own
AudioConverter instance. Sharing resamplers across sessions corrupts the
filter state and produces audio glitches.
"""

import base64
from fractions import Fraction

import av
import numpy as np


# OpenAI Realtime API audio parameters
OPENAI_SAMPLE_RATE = 24000
OPENAI_CHANNELS = 1
OPENAI_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

# aiortc default audio parameters
AIORTC_SAMPLE_RATE = 48000
AIORTC_CHANNELS = 1  # We request mono from the browser
AIORTC_FRAME_SAMPLES = 960  # 20ms at 48kHz


class AudioConverter:
    """
    Per-session audio format converter.

    Each instance owns its own av.AudioResampler pair so that the internal
    filter state (used for gapless, anti-aliased resampling) is never shared
    across concurrent sessions. Create one per RealtimeBridge.
    """

    def __init__(self):
        self._downsampler = av.AudioResampler(format="s16", layout="mono", rate=OPENAI_SAMPLE_RATE)
        self._upsampler = av.AudioResampler(format="s16", layout="mono", rate=AIORTC_SAMPLE_RATE)

    def aiortc_frame_to_realtime_b64(self, frame: av.AudioFrame) -> str:
        """
        Convert an aiortc AudioFrame to base64-encoded PCM16 for the Realtime API.

        Uses av.AudioResampler (libswresample) for proper anti-aliased downsampling
        from 48kHz to 24kHz, avoiding the aliasing artifacts that naive decimation
        produces (which degrade transcription accuracy).

        Returns empty string if the resampler is buffering (no output yet).
        """
        resampled_frames = self._downsampler.resample(frame)

        if not resampled_frames:
            return ""

        chunks = []
        for rf in resampled_frames:
            audio = rf.to_ndarray().flatten().astype(np.int16)
            chunks.append(audio)

        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return base64.b64encode(audio.tobytes()).decode("utf-8")

    def realtime_b64_to_aiortc_frame(self, audio_b64: str) -> av.AudioFrame:
        """
        Convert base64-encoded PCM16 from the Realtime API to an aiortc AudioFrame.

        Uses av.AudioResampler for proper upsampling from 24kHz to 48kHz.
        """
        raw_bytes = base64.b64decode(audio_b64)
        audio = np.frombuffer(raw_bytes, dtype=np.int16)

        frame_24k = av.AudioFrame.from_ndarray(
            audio.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame_24k.sample_rate = OPENAI_SAMPLE_RATE
        frame_24k.time_base = Fraction(1, OPENAI_SAMPLE_RATE)

        resampled_frames = self._upsampler.resample(frame_24k)

        if not resampled_frames:
            silence = np.zeros(AIORTC_FRAME_SAMPLES, dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(silence.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = AIORTC_SAMPLE_RATE
            frame.time_base = Fraction(1, AIORTC_SAMPLE_RATE)
            return frame

        result = resampled_frames[0]
        result.time_base = Fraction(1, AIORTC_SAMPLE_RATE)
        return result


def pcm16_bytes_to_b64(pcm_bytes: bytes) -> str:
    """Simple helper: raw PCM16 bytes → base64 string."""
    return base64.b64encode(pcm_bytes).decode("utf-8")


def b64_to_pcm16_bytes(audio_b64: str) -> bytes:
    """Simple helper: base64 string → raw PCM16 bytes."""
    return base64.b64decode(audio_b64)
