"""Tests for query module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query import (
    build_speaker_index,
    find_speaker,
    assign_speakers,
    to_utterances,
    format_transcript,
)


# --- build_speaker_index tests ---

def test_build_speaker_index_basic():
    """Creates sorted index from segments."""
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
    ]
    index = build_speaker_index(segments)
    assert len(index) == 2
    assert index[0] == (0.0, 1.0, "SPEAKER_00")
    assert index[1] == (2.0, 3.0, "SPEAKER_01")


def test_build_speaker_index_sorts():
    """Sorts segments by start time."""
    segments = [
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ]
    index = build_speaker_index(segments)
    assert index[0][0] == 0.0  # first should be 0.0
    assert index[1][0] == 2.0  # second should be 2.0


def test_build_speaker_index_empty():
    """Empty input returns empty index."""
    assert build_speaker_index([]) == []


# --- find_speaker tests ---

def test_find_speaker_in_segment():
    """Finds speaker when midpoint is in segment."""
    index = [(0.0, 1.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")]
    assert find_speaker(0.5, index) == "SPEAKER_00"
    assert find_speaker(2.5, index) == "SPEAKER_01"


def test_find_speaker_at_boundary():
    """Finds speaker at segment boundaries."""
    index = [(0.0, 1.0, "SPEAKER_00")]
    assert find_speaker(0.0, index) == "SPEAKER_00"
    assert find_speaker(1.0, index) == "SPEAKER_00"


def test_find_speaker_in_gap():
    """Returns None for gaps between segments."""
    index = [(0.0, 1.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")]
    assert find_speaker(1.5, index) is None


def test_find_speaker_before_first():
    """Returns None before first segment."""
    index = [(1.0, 2.0, "SPEAKER_00")]
    assert find_speaker(0.5, index) is None


def test_find_speaker_after_last():
    """Returns None after last segment."""
    index = [(0.0, 1.0, "SPEAKER_00")]
    assert find_speaker(1.5, index) is None


def test_find_speaker_empty_index():
    """Returns None for empty index."""
    assert find_speaker(0.5, []) is None


# --- assign_speakers tests ---

def test_assign_speakers_basic():
    """Assigns speaker to words by midpoint."""
    transcript = {
        "segments": [{
            "words": [
                {"start": 0.0, "end": 1.0, "word": " Hello"},
            ]
        }]
    }
    diarization = [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
    ]
    result = assign_speakers(transcript, diarization)
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_gap():
    """Words in gaps get None speaker."""
    transcript = {
        "segments": [{
            "words": [
                {"start": 1.0, "end": 2.0, "word": " gap"},  # midpoint 1.5
            ]
        }]
    }
    diarization = [
        {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    result = assign_speakers(transcript, diarization)
    assert result[0]["speaker"] is None


def test_assign_speakers_no_default_filter():
    """Without a filter, all words pass through."""
    transcript = {
        "segments": [{
            "words": [
                {"start": 0.0, "end": 0.5, "word": " Hello"},
                {"start": 0.5, "end": 0.5, "word": " fake"},  # zero duration
            ]
        }]
    }
    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ]
    result = assign_speakers(transcript, diarization)
    # No default filter - both words pass through
    assert len(result) == 2


def test_assign_speakers_custom_filter():
    """Custom filter is respected."""
    transcript = {
        "segments": [{
            "words": [
                {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.9},
                {"start": 0.5, "end": 1.0, "word": " bad", "probability": 0.3},
            ]
        }]
    }
    diarization = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
    ]

    from filters import min_probability
    result = assign_speakers(transcript, diarization, word_filter=min_probability(0.5))
    assert len(result) == 1
    assert result[0]["word"] == " Hello"


def test_assign_speakers_empty():
    """Empty transcript returns empty result."""
    transcript = {"segments": []}
    result = assign_speakers(transcript, [])
    assert result == []


# --- to_utterances tests ---

def test_to_utterances_consolidates_same_speaker():
    """Consecutive same-speaker words become one utterance."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": " there", "speaker": "SPEAKER_00"},
    ]
    result = to_utterances(words)
    assert len(result) == 1
    assert result[0]["text"] == "Hello there"
    assert result[0]["speaker"] == "SPEAKER_00"
    assert len(result[0]["words"]) == 2


def test_to_utterances_splits_different_speakers():
    """Different speakers produce separate utterances."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": " Hi", "speaker": "SPEAKER_01"},
    ]
    result = to_utterances(words)
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_to_utterances_does_not_consolidate_none():
    """None speaker utterances stay separate."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Uh", "speaker": None},
        {"start": 0.5, "end": 1.0, "word": " huh", "speaker": None},
    ]
    result = to_utterances(words)
    assert len(result) == 2
    assert result[0]["text"] == "Uh"
    assert result[1]["text"] == "huh"


def test_to_utterances_timestamps():
    """Utterances have correct start/end timestamps."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": " there", "speaker": "SPEAKER_00"},
    ]
    result = to_utterances(words)
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.0


def test_to_utterances_strips_whitespace():
    """Word text is stripped of whitespace."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello ", "speaker": "SPEAKER_00"},
    ]
    result = to_utterances(words)
    assert result[0]["text"] == "Hello"


def test_to_utterances_empty():
    """Empty input returns empty result."""
    assert to_utterances([]) == []


# --- format_transcript tests ---

def test_format_transcript_basic():
    """Formats utterances as SPEAKER: text lines."""
    utterances = [
        {"speaker": "SPEAKER_00", "text": "Hello"},
        {"speaker": "SPEAKER_01", "text": "Hi there"},
    ]
    result = format_transcript(utterances)
    assert "SPEAKER_00: Hello" in result
    assert "SPEAKER_01: Hi there" in result


def test_format_transcript_none_speaker():
    """None speaker shows as UNKNOWN."""
    utterances = [
        {"speaker": None, "text": "mystery"},
    ]
    result = format_transcript(utterances)
    assert "UNKNOWN: mystery" in result


def test_format_transcript_empty():
    """Empty utterances return empty string."""
    result = format_transcript([])
    assert result == ""
