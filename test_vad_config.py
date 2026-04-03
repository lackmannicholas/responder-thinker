"""
Tests for VADConfig defaults and env-var override.
"""

import pytest
from pydantic import ValidationError

from backend.config import VADConfig, Settings


def test_vadconfig_defaults():
    cfg = VADConfig()
    assert cfg.enabled is True
    assert cfg.threshold == 0.5
    assert cfg.vad_sample_rate == 16000
    assert cfg.vad_frame_ms == 32
    assert cfg.pre_roll_ms == 100
    assert cfg.post_roll_ms == 300
    assert cfg.hangover_frames == 8


def test_vadconfig_override():
    cfg = VADConfig(
        enabled=False,
        threshold=0.8,
        vad_sample_rate=8000,
        vad_frame_ms=16,
        pre_roll_ms=50,
        post_roll_ms=200,
        hangover_frames=4,
    )
    assert cfg.enabled is False
    assert cfg.threshold == 0.8
    assert cfg.vad_sample_rate == 8000
    assert cfg.vad_frame_ms == 16
    assert cfg.pre_roll_ms == 50
    assert cfg.post_roll_ms == 200
    assert cfg.hangover_frames == 4


def test_settings_has_vad_field():
    """Settings.vad should be a VADConfig with defaults."""
    # Use env vars to avoid needing a real .env for mandatory fields.
    settings = Settings(
        openai_api_key="sk-test",
        _env_file=None,
    )
    assert isinstance(settings.vad, VADConfig)
    assert settings.vad.enabled is True
    assert settings.vad.threshold == 0.5


def test_settings_vad_env_override(monkeypatch):
    """VAD__ENABLED and VAD__THRESHOLD env vars are picked up by pydantic-settings."""
    monkeypatch.setenv("VAD__ENABLED", "false")
    monkeypatch.setenv("VAD__THRESHOLD", "0.75")
    monkeypatch.setenv("VAD__HANGOVER_FRAMES", "12")

    settings = Settings(
        openai_api_key="sk-test",
        _env_file=None,
    )
    assert settings.vad.enabled is False
    assert settings.vad.threshold == 0.75
    assert settings.vad.hangover_frames == 12
