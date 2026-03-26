"""
Audio format conversion between aiortc and OpenAI Realtime API.

aiortc delivers audio as av.AudioFrame objects:
  - Default: 48000 Hz, stereo (2 channels), s16 format
  - Frame size: 960 samples per channel per frame (20ms at 48kHz)

OpenAI Realtime API expects/produces:
  - 24000 Hz, mono (1 channel), PCM16 (signed 16-bit little-endian)

This module handles the conversion in both directions.
Getting this right is critical — wrong sample rates = chipmunk audio,
wrong channel count = silence or noise, wrong frame sizes = choppy playback.
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


def aiortc_frame_to_realtime_b64(frame: av.AudioFrame) -> str:
    """
    Convert an aiortc AudioFrame to base64-encoded PCM16 for the Realtime API.

    Steps:
      1. Extract raw samples from the frame as numpy array
      2. Convert to mono if stereo
      3. Resample from 48kHz to 24kHz (simple 2:1 decimation)
      4. Ensure 16-bit signed integer format
      5. Base64 encode
    """
    # Get raw audio data as numpy array
    # frame.to_ndarray() returns shape (channels, samples) for planar formats
    # or (samples,) for packed formats
    audio = frame.to_ndarray()

    # Handle different frame layouts
    if audio.ndim == 2:
        # Planar: (channels, samples) — average to mono
        audio = audio.mean(axis=0)

    # Ensure float for processing
    audio = audio.astype(np.float64)

    # Resample 48kHz → 24kHz (simple decimation — take every other sample)
    # This works cleanly because 48000/24000 = 2
    if frame.sample_rate == AIORTC_SAMPLE_RATE:
        audio = audio[::2]
    elif frame.sample_rate != OPENAI_SAMPLE_RATE:
        # Generic resampling for non-standard rates
        ratio = OPENAI_SAMPLE_RATE / frame.sample_rate
        indices = np.arange(0, len(audio), 1 / ratio).astype(int)
        indices = indices[indices < len(audio)]
        audio = audio[indices]

    # Clip and convert to int16
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    # Base64 encode the raw bytes
    return base64.b64encode(audio.tobytes()).decode("utf-8")


def realtime_b64_to_aiortc_frame(audio_b64: str) -> av.AudioFrame:
    """
    Convert base64-encoded PCM16 from the Realtime API to an aiortc AudioFrame.

    Steps:
      1. Base64 decode to raw PCM16 bytes
      2. Parse as numpy int16 array
      3. Upsample from 24kHz to 48kHz (simple 2x interpolation)
      4. Build an av.AudioFrame
    """
    # Decode base64 to raw bytes
    raw_bytes = base64.b64decode(audio_b64)

    # Parse as int16 samples
    audio = np.frombuffer(raw_bytes, dtype=np.int16)

    # Upsample 24kHz → 48kHz (repeat each sample — simple but effective)
    audio_48k = np.repeat(audio, 2)

    # Build an av.AudioFrame
    frame = av.AudioFrame.from_ndarray(
        audio_48k.reshape(1, -1),  # Shape: (channels, samples)
        format="s16",
        layout="mono",
    )
    frame.sample_rate = AIORTC_SAMPLE_RATE
    frame.time_base = Fraction(1, AIORTC_SAMPLE_RATE)

    return frame


def pcm16_bytes_to_b64(pcm_bytes: bytes) -> str:
    """Simple helper: raw PCM16 bytes → base64 string."""
    return base64.b64encode(pcm_bytes).decode("utf-8")


def b64_to_pcm16_bytes(audio_b64: str) -> bytes:
    """Simple helper: base64 string → raw PCM16 bytes."""
    return base64.b64decode(audio_b64)
