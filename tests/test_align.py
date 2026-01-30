"""Tests for align module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from align import (
    align,
    align_words_to_speakers,
    filter_zero_duration_words,
    filter_low_probability_words,
    words_to_utterances,
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

    kept, removed = filter_zero_duration_words(words)

    assert len(kept) == 2
    assert kept[0]["word"] == " Hello"
    assert kept[1]["word"] == " world"
    assert len(removed) == 0


def test_filter_removes_zero_duration():
    """Words with zero duration are removed."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 0.5, "word": " silly"},  # zero duration
        {"start": 0.5, "end": 1.0, "word": " world"},
    ]

    kept, removed = filter_zero_duration_words(words)

    assert len(kept) == 2
    assert kept[0]["word"] == " Hello"
    assert kept[1]["word"] == " world"
    assert len(removed) == 1
    assert removed[0]["word"] == " silly"
    assert removed[0]["filter_reason"] == "zero_duration"


def test_filter_empty_input():
    """Empty input returns empty result."""
    kept, removed = filter_zero_duration_words([])
    assert kept == []
    assert removed == []


def test_filter_all_zero_duration():
    """All zero-duration words results in empty list."""
    words = [
        {"start": 1.0, "end": 1.0, "word": " silly"},
        {"start": 1.0, "end": 1.0, "word": " silly"},
    ]

    kept, removed = filter_zero_duration_words(words)

    assert kept == []
    assert len(removed) == 2
    assert all(r["filter_reason"] == "zero_duration" for r in removed)


# --- filter_low_probability_words tests ---

def test_filter_prob_keeps_high_probability():
    """Words with high probability pass through."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " world", "probability": 0.88},
    ]

    kept, removed = filter_low_probability_words(words)

    assert len(kept) == 2
    assert len(removed) == 0


def test_filter_prob_removes_low_probability():
    """Words with low probability are removed."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " silly", "probability": 0.001},
        {"start": 1.0, "end": 1.5, "word": " world", "probability": 0.88},
    ]

    kept, removed = filter_low_probability_words(words)

    assert len(kept) == 2
    assert kept[0]["word"] == " Hello"
    assert kept[1]["word"] == " world"
    assert len(removed) == 1
    assert removed[0]["word"] == " silly"
    assert "low_probability" in removed[0]["filter_reason"]


def test_filter_prob_custom_threshold():
    """Custom threshold is respected."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " um", "probability": 0.6},
        {"start": 0.5, "end": 1.0, "word": " hello", "probability": 0.9},
    ]

    # Default threshold 0.5 keeps both
    kept_default, removed_default = filter_low_probability_words(words)
    assert len(kept_default) == 2
    assert len(removed_default) == 0

    # Higher threshold removes the 0.6
    kept_strict, removed_strict = filter_low_probability_words(words, threshold=0.7)
    assert len(kept_strict) == 1
    assert kept_strict[0]["word"] == " hello"
    assert len(removed_strict) == 1
    assert removed_strict[0]["word"] == " um"
    assert "low_probability" in removed_strict[0]["filter_reason"]


def test_filter_prob_missing_probability():
    """Words without probability key are kept (assume valid)."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 1.0, "word": " world", "probability": 0.9},
    ]

    kept, removed = filter_low_probability_words(words)

    assert len(kept) == 2
    assert len(removed) == 0


def test_filter_prob_empty_input():
    """Empty input returns empty result."""
    kept, removed = filter_low_probability_words([])
    assert kept == []
    assert removed == []


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


# --- words_to_utterances tests ---

def test_words_to_utterances_basic():
    """Each word becomes a mini-utterance."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "speaker": "SPEAKER_00", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " world", "speaker": "SPEAKER_00", "probability": 0.90},
    ]

    result = words_to_utterances(words)

    assert len(result) == 2
    assert result[0]["text"] == "Hello"
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 0.5
    # Words array should be present without speaker key
    assert "words" in result[0]
    assert len(result[0]["words"]) == 1
    assert "speaker" not in result[0]["words"][0]
    assert result[0]["words"][0]["word"] == " Hello"
    assert result[0]["words"][0]["probability"] == 0.95


def test_words_to_utterances_strips_whitespace():
    """Word text is stripped of leading/trailing whitespace."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello ", "speaker": "SPEAKER_00"},
    ]

    result = words_to_utterances(words)

    assert result[0]["text"] == "Hello"


def test_words_to_utterances_empty_input():
    """Empty input returns empty result."""
    result = words_to_utterances([])
    assert result == []


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


def test_consolidate_skips_none_speaker():
    """None speaker utterances stay separate (not merged)."""
    utterances = [
        {"start": 0.0, "end": 1.0, "speaker": None, "text": "Uh", "words": [{"start": 0.0, "end": 1.0, "word": " Uh"}]},
        {"start": 1.0, "end": 2.0, "speaker": None, "text": "huh", "words": [{"start": 1.0, "end": 2.0, "word": " huh"}]},
    ]

    result = consolidate_utterances(utterances)

    # None speakers should NOT be consolidated
    assert len(result) == 2
    assert result[0]["text"] == "Uh"
    assert result[1]["text"] == "huh"


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


def test_align_with_prob_threshold():
    """The prob_threshold parameter controls word filtering."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
        {"start": 0.5, "end": 1.0, "word": " um", "probability": 0.6},
        {"start": 1.0, "end": 1.5, "word": " world", "probability": 0.88},
    ]
    diarization = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
    ]

    # Default threshold 0.5 keeps all words
    result_default = align(words, diarization)
    assert result_default[0]["text"] == "Hello um world"

    # Stricter threshold removes the 0.6 word
    result_strict = align(words, diarization, prob_threshold=0.7)
    assert result_strict[0]["text"] == "Hello world"


def test_align_return_debug():
    """The return_debug parameter returns detailed pipeline state."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
        {"start": 0.5, "end": 0.5, "word": " fake", "probability": 0.3},  # zero duration + low prob
        {"start": 0.5, "end": 1.0, "word": " there", "probability": 0.90},
        {"start": 1.5, "end": 2.0, "word": " bad", "probability": 0.2},  # low prob only
    ]
    diarization = [
        {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
        {"start": 1.4, "end": 2.5, "speaker": "SPEAKER_01"},
    ]

    result = align(words, diarization, return_debug=True)

    # Should return a dict with all debug keys
    assert isinstance(result, dict)
    assert "utterances" in result
    assert "words_after_zero_filter" in result
    assert "words_removed_zero" in result
    assert "words_after_prob_filter" in result
    assert "words_removed_prob" in result
    assert "words_labeled" in result
    assert "utterances_raw" in result

    # Check zero duration filter removed the fake word
    assert len(result["words_removed_zero"]) == 1
    assert result["words_removed_zero"][0]["word"] == " fake"
    assert result["words_removed_zero"][0]["filter_reason"] == "zero_duration"

    # Check prob filter removed the bad word
    assert len(result["words_removed_prob"]) == 1
    assert result["words_removed_prob"][0]["word"] == " bad"
    assert "low_probability" in result["words_removed_prob"][0]["filter_reason"]

    # Final utterances should only have good words
    assert len(result["utterances"]) == 1  # Only SPEAKER_00 utterance remains
    assert result["utterances"][0]["text"] == "Hello there"


def test_align_return_debug_false():
    """When return_debug=False (default), returns just utterances list."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.95},
    ]
    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ]

    result = align(words, diarization, return_debug=False)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "Hello"
