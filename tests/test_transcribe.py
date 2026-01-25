"""Tests for transcribe module."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcribe import transcribe, save_transcript, mark_hallucinated_segments


# --- Unit tests (fast, no model needed) ---


# --- mark_hallucinated_segments tests ---

def test_mark_keeps_normal_segments():
    """Normal segments pass through unchanged."""
    segments = [
        {
            "start": 0.0, "end": 2.0, "text": " Hello there",
            "temperature": 0.0, "compression_ratio": 1.2,
            "words": [{"word": " Hello", "start": 0.0, "end": 1.0, "probability": 0.95}]
        }
    ]

    result = mark_hallucinated_segments(segments)

    assert len(result) == 1
    assert result[0]["text"] == " Hello there"


def test_mark_flags_high_temperature():
    """Segments with temperature >= 1.0 get marked."""
    segments = [
        {
            "start": 0.0, "end": 2.0, "text": " kids kids kids",
            "temperature": 1.0, "compression_ratio": 1.5,
            "words": [{"word": " kids", "start": 0.0, "end": 0.5, "probability": 0.9}]
        }
    ]

    result = mark_hallucinated_segments(segments)

    assert result[0]["text"] == " [unintelligible]"
    assert len(result[0]["words"]) == 1
    assert result[0]["words"][0]["word"] == " [unintelligible]"


def test_mark_flags_high_compression():
    """Segments with compression_ratio > 2.5 get marked."""
    segments = [
        {
            "start": 5.0, "end": 8.0, "text": " silly silly silly",
            "temperature": 0.0, "compression_ratio": 20.0,
            "words": [{"word": " silly", "start": 5.0, "end": 5.5, "probability": 0.8}]
        }
    ]

    result = mark_hallucinated_segments(segments)

    assert result[0]["text"] == " [unintelligible]"
    assert result[0]["words"][0]["start"] == 5.0
    assert result[0]["words"][0]["end"] == 8.0


def test_mark_custom_thresholds():
    """Custom thresholds are respected."""
    segments = [
        {
            "start": 0.0, "end": 1.0, "text": " um",
            "temperature": 0.5, "compression_ratio": 2.0,
            "words": []
        }
    ]

    # Default thresholds — should pass
    result_default = mark_hallucinated_segments(segments)
    assert result_default[0]["text"] == " um"

    # Stricter thresholds — should mark
    result_strict = mark_hallucinated_segments(
        segments, temp_threshold=0.5, compression_threshold=1.5
    )
    assert result_strict[0]["text"] == " [unintelligible]"


def test_mark_empty_segments():
    """Empty input returns empty result."""
    result = mark_hallucinated_segments([])
    assert result == []


def test_mark_preserves_other_fields():
    """Other segment fields are preserved."""
    segments = [
        {
            "id": 5, "start": 0.0, "end": 1.0, "text": " test",
            "temperature": 1.0, "compression_ratio": 1.0,
            "words": [], "tokens": [1, 2, 3]
        }
    ]

    result = mark_hallucinated_segments(segments)

    assert result[0]["id"] == 5
    assert result[0]["tokens"] == [1, 2, 3]


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
    result = transcribe("sessions/audio/00000000-000000.m4a")

    assert "text" in result
    assert "language" in result
    assert "segments" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["segments"], list)
    assert len(result["text"]) > 0 # Should have some context

@pytest.mark.slow
def test_transcribe_segments_have_timestamps():
    """Each segment should have start, end, and text."""
    result = transcribe("sessions/audio/00000000-000000.m4a")

    assert len(result["segments"]) > 0

    for seg in result["segments"]:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert seg["end"] > seg["start"]


@pytest.mark.slow
def test_transcribe_with_word_timestamps():
    """word_timestamps=True should add words array to segments."""
    result = transcribe("sessions/audio/00000000-000000.m4a", word_timestamps=True)
    
    # Find a segment with content (some may be empty)
    seg_with_words = None
    for seg in result["segments"]:
        if seg.get("words"):
            seg_with_words = seg
            break
    
    assert seg_with_words is not None, "Expected at least one segment with words"
    
    # Check word structure
    word = seg_with_words["words"][0]
    assert "start" in word
    assert "end" in word
    assert "word" in word
