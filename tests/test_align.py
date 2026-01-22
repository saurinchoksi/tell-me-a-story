"""Tests for align module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from align import (
    align_words_to_speakers,
    group_words_by_speaker,
    merge_unknown_utterances,
    consolidate_utterances,
    format_transcript,
)


# --- align_words_to_speakers tests ---

def test_align_assigns_speaker_by_midpoint():
    """Word midpoint determines speaker assignment."""
    words = [
        {"start": 0.0, "end": 1.0, "word": " Hello"},  # midpoint 0.5
    ]
    diarization = [
        {"start": 0.0, "end": 0.6, "speaker": "SPEAKER_00"},
    ]
    
    result = align_words_to_speakers(words, diarization)
    
    assert result[0]["speaker"] == "SPEAKER_00"


def test_align_returns_none_for_gap():
    """Words in gaps get None speaker."""
    words = [
        {"start": 1.0, "end": 2.0, "word": " gap"},  # midpoint 1.5
    ]
    diarization = [
        {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    
    result = align_words_to_speakers(words, diarization)
    
    assert result[0]["speaker"] is None


def test_align_empty_input():
    """Empty words list returns empty result."""
    result = align_words_to_speakers([], [])
    assert result == []


# --- group_words_by_speaker tests ---

def test_group_combines_same_speaker():
    """Consecutive same-speaker words become one utterance."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Once", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": " upon", "speaker": "SPEAKER_00"},
    ]
    
    result = group_words_by_speaker(words)
    
    assert len(result) == 1
    assert result[0]["text"] == "Once upon"


def test_group_splits_on_speaker_change():
    """Speaker change starts new utterance."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 1.5, "word": " Hi", "speaker": "SPEAKER_01"},
    ]
    
    result = group_words_by_speaker(words)
    
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_group_empty_input():
    """Empty input returns empty result."""
    result = group_words_by_speaker([])
    assert result == []


# --- merge_unknown_utterances tests ---

def test_merge_fills_sandwiched_unknown():
    """UNKNOWN between same speaker gets filled."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello"},
        {"start": 1.0, "end": 2.0, "speaker": None, "text": "there"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "friend"},
    ]
    
    result = merge_unknown_utterances(utterances)
    
    assert result[1]["speaker"] == "SPEAKER_00"


def test_merge_keeps_unknown_between_different_speakers():
    """UNKNOWN between different speakers stays None."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello"},
        {"start": 1.0, "end": 2.0, "speaker": None, "text": "um"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01", "text": "Hi"},
    ]
    
    result = merge_unknown_utterances(utterances)
    
    assert result[1]["speaker"] is None


# --- consolidate_utterances tests ---

def test_consolidate_combines_same_speaker():
    """Consecutive same-speaker utterances combine."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "there"},
    ]
    
    result = consolidate_utterances(utterances)
    
    assert len(result) == 1
    assert result[0]["text"] == "Hello there"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 2.0


def test_consolidate_keeps_different_speakers_separate():
    """Different speakers stay separate."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "Hi"},
    ]
    
    result = consolidate_utterances(utterances)
    
    assert len(result) == 2


def test_consolidate_empty_input():
    """Empty input returns empty result."""
    result = consolidate_utterances([])
    assert result == []


# --- format_transcript tests ---

def test_format_basic():
    """Format produces readable output."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello"},
    ]
    
    result = format_transcript(utterances)
    
    assert "SPEAKER_00: Hello" in result


def test_format_none_speaker_shows_unknown():
    """None speaker displays as UNKNOWN."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": None, "text": "mystery"},
    ]
    
    result = format_transcript(utterances)
    
    assert "UNKNOWN: mystery" in result
