"""Tests for diarize module."""

import sys
import os
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diarize import convert_to_wav_16k, diarize


# --- Unit tests (fast, no model needed) ---


def test_convert_to_wav_16k_creates_file():
    """convert_to_wav_16k should create a WAV file."""
    source = "sessions/audio/00000000-000000.m4a"
    
    wav_path = convert_to_wav_16k(source)
    
    try:
        assert os.path.exists(wav_path)
        assert wav_path.endswith(".wav")
    finally:
        os.unlink(wav_path)


def test_convert_to_wav_16k_correct_sample_rate():
    """Converted WAV should be 16kHz mono."""
    source = "sessions/audio/00000000-000000.m4a"
    
    wav_path = convert_to_wav_16k(source)
    
    try:
        # Use wave module to check format
        import wave
        with wave.open(wav_path, 'rb') as w:
            assert w.getframerate() == 16000  # 16kHz
            assert w.getnchannels() == 1      # mono
    finally:
        os.unlink(wav_path)


# --- Integration tests (slow, loads model) ---


@pytest.mark.slow
def test_diarize_returns_expected_structure():
    """diarize() should return list of dicts with start, end, speaker."""
    segments = diarize("sessions/audio/00000000-000000.m4a")
    
    assert isinstance(segments, list)
    assert len(segments) > 0
    
    # Check first segment has expected keys
    first = segments[0]
    assert "start" in first
    assert "end" in first
    assert "speaker" in first


@pytest.mark.slow
def test_diarize_segments_have_valid_timestamps():
    """Each segment should have valid timestamps (end > start)."""
    segments = diarize("sessions/audio/00000000-000000.m4a")
    
    for seg in segments:
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert seg["end"] > seg["start"]


@pytest.mark.slow
def test_diarize_segments_are_chronological():
    """Segments should be in chronological order."""
    segments = diarize("sessions/audio/00000000-000000.m4a")
    
    for i in range(1, len(segments)):
        # Each segment should start at or after the previous one started
        assert segments[i]["start"] >= segments[i-1]["start"]
