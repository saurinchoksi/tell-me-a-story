"""Tests for inspect_audio module."""

import sys
from pathlib import Path

# Add src to path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent / "src" ))

from pipeline import get_audio_info


def test_get_audio_info_returns_dict_for_valid_file():
    """Valid audio file should return a dict with expected keys."""
    info = get_audio_info("sessions/00000000-000000/audio.m4a")

    assert info is not None
    assert "filename" in info
    assert "format" in info
    assert "duration_seconds" in info
    assert "sample_rate" in info
    assert "channels" in info


def test_get_audio_info_returns_none_for_missing_file():
    """Missing file should return None."""
    info = get_audio_info("sessions/audio/does_not_exist.m4a")

    assert info is None

def test_audio_file_has_expected_properties():
    """Our test file should have the properties we observed."""
    info = get_audio_info("sessions/00000000-000000/audio.m4a")

    assert info["sample_rate"] == 48000
    assert info["channels"] == 2
    assert info["duration_seconds"] > 300 # roughly 5+ minutes
