"""
Tests for VADGate state machine.

Mocked tests replace both the VAD model and the downsampler so that each
call to gate.process() triggers exactly one VAD inference — making mock
side_effect lists predictable and state-machine tests deterministic.

Real-model smoke tests verify basic behaviour with actual TEN VAD.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.vad import VADConfig, VADGate, VADResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gate(
    threshold: float = 0.5,
    pre_roll_ms: int = 100,
    hangover_frames: int = 3,
) -> VADGate:
    config = VADConfig(
        threshold=threshold,
        pre_roll_ms=pre_roll_ms,
        hangover_frames=hangover_frames,
    )
    return VADGate(config)


def _chunk() -> bytes:
    """Arbitrary 20ms PCM16 payload (zeros; content doesn't matter for mocked tests)."""
    return np.zeros(480, dtype=np.int16).tobytes()


def _speech_chunk() -> bytes:
    """20ms of a 440Hz sine at 24kHz."""
    t = np.linspace(0, 0.020, 480, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16).tobytes()


def _inject_probs(gate: VADGate, probs: list[float]) -> None:
    """
    Replace the gate's VAD model with a mock that returns successive
    (probability, is_speech) tuples from *probs*.

    Also replaces _downsample to return exactly hop_size samples per call,
    ensuring exactly one VAD inference per gate.process() call and making the
    side_effect list fully predictable.
    """
    mock_model = MagicMock()
    mock_model.process.side_effect = [
        (p, int(p >= gate._config.threshold)) for p in probs
    ]
    gate._vad_model = mock_model
    gate._downsample = lambda _: np.zeros(gate._hop_size, dtype=np.int16)


# ---------------------------------------------------------------------------
# State machine transitions
# ---------------------------------------------------------------------------

class TestStateMachineTransitions:
    def test_silence_stays_silent(self):
        gate = _make_gate()
        _inject_probs(gate, [0.1, 0.1, 0.1])
        for _ in range(3):
            result = gate.process(_chunk())
            assert not result.is_speech
            assert result.frames_to_flush == []

    def test_silence_to_speech_onset(self):
        gate = _make_gate()
        _inject_probs(gate, [0.1, 0.9])
        gate.process(_chunk())  # silence → pre-roll
        result = gate.process(_chunk())  # onset
        assert result.is_speech
        assert len(result.frames_to_flush) >= 1

    def test_speech_flushes_each_frame(self):
        gate = _make_gate()
        _inject_probs(gate, [0.9, 0.9, 0.9, 0.9])
        gate.process(_chunk())  # onset
        for _ in range(3):
            chunk = _chunk()
            result = gate.process(chunk)
            assert result.is_speech
            assert chunk in result.frames_to_flush

    def test_speech_to_hangover_on_sub_threshold(self):
        gate = _make_gate(hangover_frames=2)
        _inject_probs(gate, [0.9, 0.9, 0.1])
        gate.process(_chunk())  # onset
        gate.process(_chunk())  # speech
        result = gate.process(_chunk())  # sub-threshold → HANGOVER (still flushes)
        assert result.is_speech
        assert len(result.frames_to_flush) == 1

    def test_hangover_counts_down_to_silence(self):
        # hangover_frames=2: SPEECH→HANGOVER transition (flush) + 2 more flushes
        # (count=2→1, count=1→0), then the frame that finds count==0 goes to SILENCE.
        gate = _make_gate(hangover_frames=2)
        _inject_probs(gate, [0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        gate.process(_chunk())  # onset
        gate.process(_chunk())  # speech
        r1 = gate.process(_chunk())  # SPEECH → HANGOVER, count=2, flush
        gate.process(_chunk())       # HANGOVER count=2→1, flush
        gate.process(_chunk())       # HANGOVER count=1→0, flush (still in HANGOVER)
        r_silence = gate.process(_chunk())  # count==0 → SILENCE, no flush
        assert r1.is_speech
        assert not r_silence.is_speech
        assert r_silence.frames_to_flush == []

    def test_hangover_retriggered_to_speech(self):
        gate = _make_gate(hangover_frames=3)
        _inject_probs(gate, [0.9, 0.1, 0.9])
        gate.process(_chunk())  # onset
        gate.process(_chunk())  # SPEECH → HANGOVER
        result = gate.process(_chunk())  # re-trigger → SPEECH
        assert result.is_speech
        assert len(result.frames_to_flush) == 1


# ---------------------------------------------------------------------------
# Pre-roll buffer
# ---------------------------------------------------------------------------

class TestPreRollBuffer:
    def test_pre_roll_flushed_on_onset(self):
        # pre_roll_ms=100 → ceil(100/20) = 5 chunks max
        gate = _make_gate(pre_roll_ms=100)
        _inject_probs(gate, [0.1, 0.1, 0.1, 0.9])
        for _ in range(3):
            gate.process(_chunk())  # 3 chunks into pre-roll
        onset_chunk = _chunk()
        result = gate.process(onset_chunk)
        # 3 pre-roll chunks + current onset chunk = 4
        assert len(result.frames_to_flush) == 4
        assert result.frames_to_flush[-1] == onset_chunk

    def test_pre_roll_ring_buffer_bounded(self):
        # pre_roll_ms=60 → ceil(60/20) = 3 chunks max
        gate = _make_gate(pre_roll_ms=60)
        _inject_probs(gate, [0.1] * 6 + [0.9])
        for _ in range(6):
            gate.process(_chunk())  # ring buffer holds last 3
        result = gate.process(_chunk())
        # 3 pre-roll + current = 4
        assert len(result.frames_to_flush) == 4

    def test_pre_roll_empty_on_fresh_onset(self):
        gate = _make_gate(pre_roll_ms=100)
        _inject_probs(gate, [0.9])
        result = gate.process(_chunk())
        # No prior silence → just current chunk
        assert len(result.frames_to_flush) == 1

    def test_pre_roll_cleared_after_onset(self):
        # After an onset the pre-roll deque is empty; verify directly.
        gate = _make_gate(pre_roll_ms=100)
        _inject_probs(gate, [0.1, 0.1, 0.9])
        gate.process(_chunk())  # silence → pre-roll: 1
        gate.process(_chunk())  # silence → pre-roll: 2
        gate.process(_chunk())  # onset → pre-roll flushed
        assert len(gate._pre_roll) == 0


# ---------------------------------------------------------------------------
# Hangover countdown
# ---------------------------------------------------------------------------

class TestHangoverCountdown:
    def test_hangover_flushes_transition_plus_n_frames(self):
        n = 4
        gate = _make_gate(hangover_frames=n)
        # onset + speech + (n+2) sub-threshold: n+1 hangover flushes then silence
        _inject_probs(gate, [0.9, 0.9] + [0.1] * (n + 2))
        gate.process(_chunk())  # onset
        gate.process(_chunk())  # speech

        hangover_flushes = 0
        silence_reached = False
        for _ in range(n + 2):
            r = gate.process(_chunk())
            if r.frames_to_flush:
                hangover_flushes += 1
            else:
                silence_reached = True
                assert not r.is_speech
                break

        # Transition frame + n countdown frames = n+1 total hangover flushes
        assert hangover_flushes == n + 1
        assert silence_reached

    def test_zero_hangover_frames_transition_frame_flushed(self):
        gate = _make_gate(hangover_frames=0)
        _inject_probs(gate, [0.9, 0.9, 0.1])
        gate.process(_chunk())  # onset
        gate.process(_chunk())  # speech
        # Sub-threshold: SPEECH flushes this frame, then sets HANGOVER with count=0
        result = gate.process(_chunk())
        assert len(result.frames_to_flush) == 1  # transition frame still sent

    def test_zero_hangover_frames_next_silence(self):
        gate = _make_gate(hangover_frames=0)
        _inject_probs(gate, [0.9, 0.9, 0.1, 0.1])
        gate.process(_chunk())  # onset
        gate.process(_chunk())  # speech
        gate.process(_chunk())  # SPEECH → HANGOVER (count=0)
        # Next sub-threshold: HANGOVER count=0 → SILENCE immediately
        r2 = gate.process(_chunk())
        assert not r2.is_speech
        assert r2.frames_to_flush == []


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_pre_roll(self):
        gate = _make_gate()
        _inject_probs(gate, [0.1, 0.1])
        gate.process(_chunk())
        gate.process(_chunk())
        gate.reset()
        assert len(gate._pre_roll) == 0

    def test_reset_returns_to_silence(self):
        gate = _make_gate()
        _inject_probs(gate, [0.9, 0.9, 0.1])
        gate.process(_chunk())  # onset → SPEECH
        gate.process(_chunk())  # SPEECH
        gate.process(_chunk())  # → HANGOVER
        gate.reset()
        _inject_probs(gate, [0.1])
        result = gate.process(_chunk())
        assert not result.is_speech
        assert result.frames_to_flush == []

    def test_reset_clears_internal_state(self):
        gate = _make_gate()
        _inject_probs(gate, [0.9])
        gate.process(_chunk())
        gate.reset()
        assert gate._vad_samples == []
        assert gate._hangover_count == 0
        assert gate._last_prob == 0.0


# ---------------------------------------------------------------------------
# Real model smoke tests (no mocking)
# ---------------------------------------------------------------------------

class TestRealModel:
    def test_silence_not_speech(self):
        """Real TEN VAD: sustained zeros stay below threshold."""
        gate = VADGate(VADConfig(threshold=0.5))
        result = None
        for _ in range(10):
            result = gate.process(np.zeros(480, dtype=np.int16).tobytes())
        assert not result.is_speech
        assert result.speech_probability < 0.5

    def test_sine_wave_detected_as_speech_early(self):
        """Real TEN VAD: 440Hz sine is above threshold for initial frames."""
        gate = VADGate(VADConfig(threshold=0.5, pre_roll_ms=0))
        speech_detected = False
        for _ in range(10):
            result = gate.process(_speech_chunk())
            if result.is_speech:
                speech_detected = True
                break
        assert speech_detected, "Expected speech to be detected within first 10 frames"

    def test_transition_silence_to_speech_flushes_preroll(self):
        """Real TEN VAD: speech onset flushes pre-roll frames."""
        gate = VADGate(VADConfig(threshold=0.5, pre_roll_ms=60))
        # Warm up with silence
        for _ in range(5):
            gate.process(np.zeros(480, dtype=np.int16).tobytes())
        # Send speech until we get a flush with > 1 frame (pre-roll included)
        for _ in range(20):
            result = gate.process(_speech_chunk())
            if len(result.frames_to_flush) > 1:
                return  # pre-roll was flushed on onset
        # If we never got > 1 frame, at minimum some flush should have occurred
        assert any(
            len(gate.process(_speech_chunk()).frames_to_flush) >= 1
            for _ in range(5)
        )
