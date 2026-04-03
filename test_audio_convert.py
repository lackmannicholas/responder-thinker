"""
Tests for AudioConverter — covers both aiortc_frame_to_realtime_b64 and
the new aiortc_frame_to_pcm16 method.
"""

import base64
from fractions import Fraction

import av
import numpy as np
import pytest

from backend.audio_convert import (
    AudioConverter,
    AIORTC_SAMPLE_RATE,
    OPENAI_SAMPLE_RATE,
    AIORTC_FRAME_SAMPLES,
    pcm16_bytes_to_b64,
)


def _make_aiortc_frame(samples: int = AIORTC_FRAME_SAMPLES, frequency: int = 440) -> av.AudioFrame:
    """Create a synthetic mono s16 aiortc frame at 48kHz."""
    t = np.arange(samples) / AIORTC_SAMPLE_RATE
    sine = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(sine.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = AIORTC_SAMPLE_RATE
    frame.time_base = Fraction(1, AIORTC_SAMPLE_RATE)
    return frame


class TestAiortcFrameToPcm16:
    def test_returns_bytes(self):
        converter = AudioConverter()
        frame = _make_aiortc_frame()
        result = converter.aiortc_frame_to_pcm16(frame)
        assert isinstance(result, bytes)

    def test_buffering_returns_empty_bytes(self):
        """First call may return b"" while the resampler warms up."""
        converter = AudioConverter()
        frame = _make_aiortc_frame()
        result = converter.aiortc_frame_to_pcm16(frame)
        # Either empty (buffering) or non-empty bytes — both are valid.
        assert result == b"" or (isinstance(result, bytes) and len(result) > 0)

    def test_non_empty_after_warmup(self):
        """After enough frames the resampler always produces output."""
        converter = AudioConverter()
        outputs = []
        for _ in range(5):
            frame = _make_aiortc_frame()
            out = converter.aiortc_frame_to_pcm16(frame)
            if out:
                outputs.append(out)
        assert len(outputs) > 0, "Expected at least one non-empty output after warmup"

    def test_output_is_int16_aligned(self):
        """Output byte length must be a multiple of 2 (int16 = 2 bytes per sample)."""
        converter = AudioConverter()
        for _ in range(5):
            frame = _make_aiortc_frame()
            result = converter.aiortc_frame_to_pcm16(frame)
            if result:
                assert len(result) % 2 == 0

    def test_pcm16_matches_b64_decoded(self):
        """aiortc_frame_to_pcm16 and aiortc_frame_to_realtime_b64 must agree on PCM content.

        Both methods share the same downsampler, so we use separate converters
        and compare the decoded bytes from the b64 path against the raw bytes path.
        """
        converter_pcm = AudioConverter()
        converter_b64 = AudioConverter()

        frames = [_make_aiortc_frame() for _ in range(5)]

        pcm_outputs = []
        b64_outputs = []

        for frame in frames:
            # Build a fresh frame each time since resamplers consume the frame
            f1 = _make_aiortc_frame()
            f2 = _make_aiortc_frame()
            pcm = converter_pcm.aiortc_frame_to_pcm16(f1)
            b64 = converter_b64.aiortc_frame_to_realtime_b64(f2)
            if pcm:
                pcm_outputs.append(pcm)
            if b64:
                b64_outputs.append(base64.b64decode(b64))

        assert len(pcm_outputs) == len(b64_outputs)
        for pcm, decoded in zip(pcm_outputs, b64_outputs):
            assert pcm == decoded

    def test_statefulness_across_calls(self):
        """Resampler state persists — multiple calls on same instance are coherent."""
        converter = AudioConverter()
        results = []
        for _ in range(10):
            frame = _make_aiortc_frame()
            out = converter.aiortc_frame_to_pcm16(frame)
            if out:
                results.append(out)
        # Just verify we got consistent non-empty output and no exceptions
        assert all(isinstance(r, bytes) and len(r) > 0 for r in results)


class TestAiortcFrameToRealtimeB64:
    """Regression: ensure existing method still works after adding aiortc_frame_to_pcm16."""

    def test_returns_str(self):
        converter = AudioConverter()
        frame = _make_aiortc_frame()
        result = converter.aiortc_frame_to_realtime_b64(frame)
        assert isinstance(result, str)

    def test_non_empty_after_warmup(self):
        converter = AudioConverter()
        outputs = []
        for _ in range(5):
            frame = _make_aiortc_frame()
            out = converter.aiortc_frame_to_realtime_b64(frame)
            if out:
                outputs.append(out)
        assert len(outputs) > 0

    def test_output_is_valid_base64(self):
        converter = AudioConverter()
        for _ in range(5):
            frame = _make_aiortc_frame()
            out = converter.aiortc_frame_to_realtime_b64(frame)
            if out:
                decoded = base64.b64decode(out)
                assert len(decoded) % 2 == 0
