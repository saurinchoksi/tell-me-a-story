"""Tests for transcribe module."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcribe import transcribe, save_transcript, clean_transcript, compute_min_word_prob


# --- Unit tests (fast, no model needed) ---


# --- clean_transcript tests ---

def test_clean_removes_zero_duration_words():
    """Zero-duration words (fabrications) are removed."""
    transcript = {
        "text": "Hello world",
        "segments": [
            {
                "start": 0.0, "end": 2.0, "text": " Hello world",
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": " fake", "start": 1.0, "end": 1.0, "probability": 0.9},  # zero duration
                    {"word": " world", "start": 1.5, "end": 2.0, "probability": 0.92}
                ]
            }
        ]
    }

    result = clean_transcript(transcript)

    assert len(result["segments"]) == 1
    assert len(result["segments"][0]["words"]) == 2
    words = [w["word"] for w in result["segments"][0]["words"]]
    assert " Hello" in words
    assert " world" in words
    assert " fake" not in words


def test_clean_keeps_normal_words():
    """Words with positive duration pass through unchanged."""
    transcript = {
        "text": "Once upon a time",
        "segments": [
            {
                "start": 0.0, "end": 3.0, "text": " Once upon a time",
                "words": [
                    {"word": " Once", "start": 0.0, "end": 0.5, "probability": 0.98},
                    {"word": " upon", "start": 0.5, "end": 1.0, "probability": 0.97},
                    {"word": " a", "start": 1.0, "end": 1.2, "probability": 0.99},
                    {"word": " time", "start": 1.2, "end": 1.8, "probability": 0.96}
                ]
            }
        ]
    }

    result = clean_transcript(transcript)

    assert len(result["segments"][0]["words"]) == 4
    # Verify original transcript not mutated
    assert len(transcript["segments"][0]["words"]) == 4


def test_clean_adds_schema_version():
    """clean_transcript adds _schema_version to result."""
    transcript = {
        "text": "Hello",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " Hello", "words": []}
        ]
    }

    result = clean_transcript(transcript)

    assert "_schema_version" in result
    assert result["_schema_version"] == "1.1.0"
    assert "_generator_version" in result


def test_clean_adds_min_word_prob():
    """clean_transcript computes _min_word_prob for each segment."""
    transcript = {
        "text": "Hello world",
        "segments": [{
            "start": 0.0, "end": 2.0, "text": " Hello world",
            "words": [
                {"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                {"word": " world", "start": 1.5, "end": 2.0, "probability": 0.42}
            ]
        }]
    }

    result = clean_transcript(transcript)

    assert "_min_word_prob" in result["segments"][0]
    assert result["segments"][0]["_min_word_prob"] == 0.42


def test_clean_empty_segments():
    """Empty segments list returns empty result with schema version."""
    transcript = {
        "text": "",
        "segments": []
    }

    result = clean_transcript(transcript)

    assert result["segments"] == []
    assert "_schema_version" in result


def test_clean_removes_garbage_segments():
    """Segments with no text or zero duration are removed."""
    transcript = {
        "text": "Hello",
        "segments": [
            # Good segment
            {"start": 0.0, "end": 1.0, "text": " Hello", "words": []},
            # Garbage: empty text, zero duration
            {"start": 115.56, "end": 115.56, "text": "", "words": []},
            # Garbage: whitespace-only text
            {"start": 2.0, "end": 3.0, "text": "   ", "words": []},
            # Garbage: zero duration even with text
            {"start": 5.0, "end": 5.0, "text": " Something", "words": []},
        ]
    }

    result = clean_transcript(transcript)

    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == " Hello"


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


# --- compute_min_word_prob tests ---


class TestComputeMinWordProb:
    """Tests for compute_min_word_prob helper."""

    def test_normal_segment(self):
        """Returns minimum probability from words."""
        segment = {
            "words": [
                {"word": "Hello", "probability": 0.95},
                {"word": "world", "probability": 0.42},
                {"word": "!", "probability": 0.88},
            ]
        }
        assert compute_min_word_prob(segment) == 0.42

    def test_empty_words_array(self):
        """Returns 1.0 for empty words array."""
        segment = {"words": []}
        assert compute_min_word_prob(segment) == 1.0

    def test_missing_words_key(self):
        """Returns 1.0 when words key is absent."""
        segment = {"text": "Hello"}
        assert compute_min_word_prob(segment) == 1.0

    def test_word_missing_probability(self):
        """Treats missing probability as 1.0."""
        segment = {
            "words": [
                {"word": "Hello", "probability": 0.5},
                {"word": "world"},  # no probability key
            ]
        }
        assert compute_min_word_prob(segment) == 0.5

    def test_single_word(self):
        """Works with single word."""
        segment = {"words": [{"word": "Hi", "probability": 0.99}]}
        assert compute_min_word_prob(segment) == 0.99

    def test_all_low_probability(self):
        """Returns lowest when all are low."""
        segment = {
            "words": [
                {"word": "Um", "probability": 0.1},
                {"word": "uh", "probability": 0.05},
            ]
        }
        assert compute_min_word_prob(segment) == 0.05


# --- Inegration test (slow, runs actual model) ---
# Mark with pytest.mark.slow so we can skip in quick runs

import pytest

@pytest.mark.slow
def test_transcribe_returns_expected_structure():
    """transcribe() should return dict with text, language, segments."""
    result = transcribe("sessions/00000000-000000/audio.m4a")

    assert "text" in result
    assert "language" in result
    assert "segments" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["segments"], list)
    assert len(result["text"]) > 0 # Should have some context

@pytest.mark.slow
def test_transcribe_segments_have_timestamps():
    """Each segment should have start, end, and text."""
    result = transcribe("sessions/00000000-000000/audio.m4a")

    assert len(result["segments"]) > 0

    for seg in result["segments"]:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert seg["end"] > seg["start"]


@pytest.mark.slow
def test_transcribe_with_word_timestamps():
    """word_timestamps=True should add words array to segments."""
    result = transcribe("sessions/00000000-000000/audio.m4a", word_timestamps=True)

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


@pytest.mark.slow
def test_transcribe_adds_min_word_prob():
    """Transcription includes _min_word_prob on all segments."""
    raw = transcribe("sessions/00000000-000000/audio.m4a", word_timestamps=True)
    result = clean_transcript(raw)

    for segment in result["segments"]:
        assert "_min_word_prob" in segment
        assert isinstance(segment["_min_word_prob"], float)
        assert 0.0 <= segment["_min_word_prob"] <= 1.0


@pytest.mark.slow
def test_schema_version_updated():
    """Schema version reflects the _min_word_prob addition."""
    raw = transcribe("sessions/00000000-000000/audio.m4a")
    result = clean_transcript(raw)
    assert result.get("_schema_version") == "1.1.0"
