"""Tests for align module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from align import (
    align,
    align_words_to_speakers,
    filter_zero_duration_words,
    filter_low_probability_words,
    group_words_by_speaker,
    merge_unknown_utterances,
    assign_leading_fragments,
    consolidate_utterances,
    format_transcript,
)


# --- filter_zero_duration_words tests ---

def test_filter_keeps_normal_words():
    """Words with duration pass through unchanged."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 1.0, "word": " world"},
    ]

    result = filter_zero_duration_words(words)

    assert len(result) == 2
    assert result[0]["word"] == " Hello"
    assert result[1]["word"] == " world"


def test_filter_removes_zero_duration():
    """Words with zero duration are removed."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 0.5, "word": " silly"},  # zero duration
        {"start": 0.5, "end": 1.0, "word": " world"},
    ]

    result = filter_zero_duration_words(words)

    assert len(result) == 2
    assert result[0]["word"] == " Hello"
    assert result[1]["word"] == " world"


def test_filter_empty_input():
    """Empty input returns empty result."""
    result = filter_zero_duration_words([])
    assert result == []


def test_filter_all_zero_duration():
    """All zero-duration words results in empty list."""
    words = [
        {"start": 1.0, "end": 1.0, "word": " silly"},
        {"start": 1.0, "end": 1.0, "word": " silly"},
    ]

    result = filter_zero_duration_words(words)

    assert result == []


# --- filter_low_probability_words tests ---

def test_filter_prob_keeps_high_probability():
    """Words with high probability pass through."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " world", "probability": 0.88},
    ]

    result = filter_low_probability_words(words)

    assert len(result) == 2


def test_filter_prob_removes_low_probability():
    """Words with low probability are removed."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " silly", "probability": 0.001},
        {"start": 1.0, "end": 1.5, "word": " world", "probability": 0.88},
    ]

    result = filter_low_probability_words(words)

    assert len(result) == 2
    assert result[0]["word"] == " Hello"
    assert result[1]["word"] == " world"


def test_filter_prob_custom_threshold():
    """Custom threshold is respected."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " um", "probability": 0.6},
        {"start": 0.5, "end": 1.0, "word": " hello", "probability": 0.9},
    ]

    # Default threshold 0.5 keeps both
    result_default = filter_low_probability_words(words)
    assert len(result_default) == 2

    # Higher threshold removes the 0.6
    result_strict = filter_low_probability_words(words, threshold=0.7)
    assert len(result_strict) == 1
    assert result_strict[0]["word"] == " hello"


def test_filter_prob_missing_probability():
    """Words without probability key are kept (assume valid)."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 1.0, "word": " world", "probability": 0.9},
    ]

    result = filter_low_probability_words(words)

    assert len(result) == 2


def test_filter_prob_empty_input():
    """Empty input returns empty result."""
    result = filter_low_probability_words([])
    assert result == []


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
        {"start": 0.0, "end": 0.5, "word": " Once", "speaker": "SPEAKER_00", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " upon", "speaker": "SPEAKER_00", "probability": 0.90},
    ]

    result = group_words_by_speaker(words)

    assert len(result) == 1
    assert result[0]["text"] == "Once upon"
    # Words array should be present without speaker key
    assert "words" in result[0]
    assert len(result[0]["words"]) == 2
    assert "speaker" not in result[0]["words"][0]
    assert result[0]["words"][0]["word"] == " Once"
    assert result[0]["words"][0]["probability"] == 0.95


def test_group_splits_on_speaker_change():
    """Speaker change starts new utterance."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00", "probability": 0.95},
        {"start": 1.0, "end": 1.5, "word": " Hi", "speaker": "SPEAKER_01", "probability": 0.88},
    ]

    result = group_words_by_speaker(words)

    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"
    # Each utterance should have its own words
    assert len(result[0]["words"]) == 1
    assert len(result[1]["words"]) == 1
    assert result[0]["words"][0]["word"] == " Hello"
    assert result[1]["words"][0]["word"] == " Hi"


def test_group_empty_input():
    """Empty input returns empty result."""
    result = group_words_by_speaker([])
    assert result == []


# --- merge_unknown_utterances tests ---

def test_merge_fills_sandwiched_unknown():
    """UNKNOWN between same speaker gets filled."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello", "words": [{"start": 0.0, "end": 1.0, "word": " Hello"}]},
        {"start": 1.0, "end": 2.0, "speaker": None, "text": "there", "words": [{"start": 1.0, "end": 2.0, "word": " there"}]},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "friend", "words": [{"start": 2.0, "end": 3.0, "word": " friend"}]},
    ]

    result = merge_unknown_utterances(utterances)

    assert result[1]["speaker"] == "SPEAKER_00"
    # Words should be preserved
    assert result[1]["words"] == [{"start": 1.0, "end": 2.0, "word": " there"}]


def test_merge_keeps_unknown_between_different_speakers():
    """UNKNOWN between different speakers stays None."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello", "words": [{"start": 0.0, "end": 1.0, "word": " Hello"}]},
        {"start": 1.0, "end": 2.0, "speaker": None, "text": "um", "words": [{"start": 1.0, "end": 2.0, "word": " um"}]},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01", "text": "Hi", "words": [{"start": 2.0, "end": 3.0, "word": " Hi"}]},
    ]

    result = merge_unknown_utterances(utterances)

    assert result[1]["speaker"] is None


# --- consolidate_utterances tests ---

def test_consolidate_combines_same_speaker():
    """Consecutive same-speaker utterances combine."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello", "words": [{"start": 0.0, "end": 1.0, "word": " Hello"}]},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "there", "words": [{"start": 1.0, "end": 2.0, "word": " there"}]},
    ]

    result = consolidate_utterances(utterances)

    assert len(result) == 1
    assert result[0]["text"] == "Hello there"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 2.0
    # Words should be concatenated
    assert len(result[0]["words"]) == 2
    assert result[0]["words"][0]["word"] == " Hello"
    assert result[0]["words"][1]["word"] == " there"


def test_consolidate_keeps_different_speakers_separate():
    """Different speakers stay separate."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Hello", "words": [{"start": 0.0, "end": 1.0, "word": " Hello"}]},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "Hi", "words": [{"start": 1.0, "end": 2.0, "word": " Hi"}]},
    ]

    result = consolidate_utterances(utterances)

    assert len(result) == 2
    assert len(result[0]["words"]) == 1
    assert len(result[1]["words"]) == 1


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


# --- assign_leading_fragments tests ---

def test_assign_fragment_within_gap():
    """UNKNOWN close to next utterance gets assigned to that speaker."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": None, "text": "That's", "words": [{"start": 0.0, "end": 1.0, "word": " That's"}]},
        {"start": 1.1, "end": 2.0, "speaker": "SPEAKER_00", "text": "right", "words": [{"start": 1.1, "end": 2.0, "word": " right"}]},
    ]

    result = assign_leading_fragments(utterances)

    assert result[0]["speaker"] == "SPEAKER_00"
    # Words should be preserved
    assert result[0]["words"] == [{"start": 0.0, "end": 1.0, "word": " That's"}]


def test_assign_fragment_gap_too_large():
    """UNKNOWN with large gap to next utterance stays UNKNOWN."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": None, "text": "um", "words": [{"start": 0.0, "end": 1.0, "word": " um"}]},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "hello", "words": [{"start": 2.0, "end": 3.0, "word": " hello"}]},
    ]

    result = assign_leading_fragments(utterances)

    assert result[0]["speaker"] is None


def test_assign_fragment_at_end():
    """UNKNOWN at end of list stays UNKNOWN (no next utterance)."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello", "words": [{"start": 0.0, "end": 1.0, "word": " hello"}]},
        {"start": 1.5, "end": 2.0, "speaker": None, "text": "bye", "words": [{"start": 1.5, "end": 2.0, "word": " bye"}]},
    ]

    result = assign_leading_fragments(utterances)

    assert result[1]["speaker"] is None


def test_assign_fragment_next_also_unknown():
    """UNKNOWN followed by another UNKNOWN stays UNKNOWN."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": None, "text": "um", "words": [{"start": 0.0, "end": 1.0, "word": " um"}]},
        {"start": 1.1, "end": 2.0, "speaker": None, "text": "uh", "words": [{"start": 1.1, "end": 2.0, "word": " uh"}]},
        {"start": 2.1, "end": 3.0, "speaker": "SPEAKER_00", "text": "hello", "words": [{"start": 2.1, "end": 3.0, "word": " hello"}]},
    ]

    result = assign_leading_fragments(utterances)

    # First UNKNOWN can't assign to another UNKNOWN
    assert result[0]["speaker"] is None
    # Second UNKNOWN can assign to SPEAKER_00
    assert result[1]["speaker"] == "SPEAKER_00"


def test_assign_fragment_empty_input():
    """Empty input returns empty result."""
    result = assign_leading_fragments([])
    assert result == []


def test_assign_fragment_no_unknowns():
    """List with no UNKNOWNs returns unchanged."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello", "words": [{"start": 0.0, "end": 1.0, "word": " hello"}]},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "hi", "words": [{"start": 1.0, "end": 2.0, "word": " hi"}]},
    ]

    result = assign_leading_fragments(utterances)

    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[1]["speaker"] == "SPEAKER_01"


def test_assign_fragment_custom_threshold():
    """Custom max_gap threshold is respected."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": None, "text": "um", "words": [{"start": 0.0, "end": 1.0, "word": " um"}]},
        {"start": 1.3, "end": 2.0, "speaker": "SPEAKER_00", "text": "hello", "words": [{"start": 1.3, "end": 2.0, "word": " hello"}]},
    ]

    # Default 0.5s threshold — gap is 0.3s, should assign
    result_default = assign_leading_fragments(utterances)
    assert result_default[0]["speaker"] == "SPEAKER_00"

    # Tighter 0.2s threshold — gap is 0.3s, should NOT assign
    result_tight = assign_leading_fragments(utterances, max_gap=0.2)
    assert result_tight[0]["speaker"] is None


# --- align (full pipeline) tests ---

def test_align_full_pipeline():
    """The align() wrapper runs all steps end-to-end."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " there", "probability": 0.90},
        {"start": 1.5, "end": 2.0, "word": " Hi", "probability": 0.88},
    ]

    diarization = [
        {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
        {"start": 1.4, "end": 2.5, "speaker": "SPEAKER_01"},
    ]

    result = align(words, diarization)

    # Should produce two consolidated utterances
    assert len(result) == 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["text"] == "Hello there"
    assert result[1]["speaker"] == "SPEAKER_01"
    assert result[1]["text"] == "Hi"
    # Words should be preserved with probability
    assert "words" in result[0]
    assert len(result[0]["words"]) == 2
    assert result[0]["words"][0]["word"] == " Hello"
    assert result[0]["words"][0]["probability"] == 0.95
    assert "speaker" not in result[0]["words"][0]
    assert len(result[1]["words"]) == 1
    assert result[1]["words"][0]["word"] == " Hi"
