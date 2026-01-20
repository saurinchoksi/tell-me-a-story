"""Tests for transcribe module."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcribe import transcribe, save_transcript


# --- Unit tests (fast, no model needed) ---


def test_save_transcript_creates_file():
    """save_transcript should create a file with expected content."""
    #Mock result dict matching what mlx_whisper returns
    mock_result = {
            "text": "Once upon a time there were five brothers.",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": " Once upon a time"},
                {"start": 2.5, "end": 5.0, "text": " there were five brothers."},
            ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name

    try:
        save_transcript(mock_result, temp_path)

        with open(temp_path, 'r') as f:
            content = f.read()

        assert "--- TRANSCRIPT ---" in content
        assert "Once upon a time there were five brothers." in content
        assert "--- SEGMENTS ---" in content
        assert "[   0.0 -    2.5]" in content
        assert "[   2.5 -    5.0]" in content
    finally:
        os.unlink(temp_path)


def test_save_transcript_handles_empty_segments():
    """save_transcript should handle result with no segments."""
    mock_result = {
            "text": "",
            "segments": []
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name

    try: 
        save_transcript(mock_result, temp_path)

        with open(temp_path, 'r') as f:
            content = f.read()

        assert "--- TRANSCRIPT ---" in content
        assert "--- SEGMENTS ---" in content
    finally:
        os.unlink(temp_path)


# --- Inegration test (slow, runs actual model) ---
# Mark with pytest.mark.slow so we can skip in quick runs

import pytest

@pytest.mark.slow
def test_transcribe_returns_expected_structure():
    """transcribe() should return dict with text, language, segments."""
    result = transcribe("data/00000000-000000.m4a")

    assert "text" in result
    assert "language" in result
    assert "segments" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["segments"], list)
    assert len(result["text"]) > 0 # Should have some context

@pytest.mark.slow
def test_transcribe_segments_have_timestamps():
    """Each segment should have start, end, and text."""
    result = transcribe("data/00000000-000000.m4a")

    assert len(result["segments"]) > 0

    for seg in result["segments"]:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert seg["end"] > seg["start"]
