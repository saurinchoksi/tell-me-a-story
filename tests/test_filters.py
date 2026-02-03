"""Tests for filters module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from filters import (
    has_duration,
    has_content,
    min_probability,
    all_of,
    apply_filter,
    DEFAULT_FILTER,
)


# --- has_duration tests ---

def test_has_duration_true_for_normal_words():
    """Words with end > start have duration."""
    word = {"start": 0.0, "end": 0.5, "word": " Hello"}
    assert has_duration(word) is True


def test_has_duration_false_for_zero_duration():
    """Words with end == start have no duration."""
    word = {"start": 0.5, "end": 0.5, "word": " fake"}
    assert has_duration(word) is False


def test_has_duration_false_for_missing_keys():
    """Words without start/end default to 0, so no duration."""
    word = {"word": " missing"}
    assert has_duration(word) is False


# --- has_content tests ---

def test_has_content_true_for_text():
    """Words with non-empty text have content."""
    word = {"word": " Hello"}
    assert has_content(word) is True


def test_has_content_false_for_empty():
    """Words with empty text have no content."""
    word = {"word": ""}
    assert has_content(word) is False


def test_has_content_false_for_whitespace_only():
    """Words with only whitespace have no content."""
    word = {"word": "   "}
    assert has_content(word) is False


def test_has_content_false_for_missing_key():
    """Words without word key have no content."""
    word = {"start": 0.0, "end": 0.5}
    assert has_content(word) is False


# --- min_probability tests ---

def test_min_probability_keeps_high_prob():
    """Words above threshold pass."""
    check = min_probability(0.5)
    word = {"word": " Hello", "probability": 0.95}
    assert check(word) is True


def test_min_probability_rejects_low_prob():
    """Words below threshold fail."""
    check = min_probability(0.5)
    word = {"word": " bad", "probability": 0.3}
    assert check(word) is False


def test_min_probability_keeps_at_threshold():
    """Words at exactly threshold pass."""
    check = min_probability(0.5)
    word = {"word": " edge", "probability": 0.5}
    assert check(word) is True


def test_min_probability_keeps_missing_key():
    """Words without probability are kept (assumed valid)."""
    check = min_probability(0.5)
    word = {"word": " Hello"}
    assert check(word) is True


def test_min_probability_custom_threshold():
    """Custom threshold is respected."""
    check = min_probability(0.8)
    word = {"word": " um", "probability": 0.7}
    assert check(word) is False

    check_loose = min_probability(0.3)
    assert check_loose(word) is True


# --- all_of tests ---

def test_all_of_combines_predicates():
    """All predicates must pass."""
    word = {"start": 0.0, "end": 0.5, "word": " Hello"}
    combined = all_of(has_duration, has_content)
    assert combined(word) is True


def test_all_of_fails_if_any_fails():
    """Fails if any predicate fails."""
    word = {"start": 0.5, "end": 0.5, "word": " Hello"}  # no duration
    combined = all_of(has_duration, has_content)
    assert combined(word) is False


def test_all_of_single_predicate():
    """Works with single predicate."""
    word = {"start": 0.0, "end": 0.5, "word": " Hello"}
    combined = all_of(has_duration)
    assert combined(word) is True


def test_all_of_three_predicates():
    """Works with multiple predicates."""
    word = {"start": 0.0, "end": 0.5, "word": " Hello", "probability": 0.9}
    combined = all_of(has_duration, has_content, min_probability(0.5))
    assert combined(word) is True

    word_low_prob = {"start": 0.0, "end": 0.5, "word": " bad", "probability": 0.3}
    assert combined(word_low_prob) is False


# --- apply_filter tests ---

def test_apply_filter_keeps_passing():
    """Words passing predicate are kept."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 1.0, "word": " world"},
    ]
    result = apply_filter(words, has_duration)
    assert len(result) == 2


def test_apply_filter_removes_failing():
    """Words failing predicate are removed."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 0.5, "word": " fake"},  # no duration
    ]
    result = apply_filter(words, has_duration)
    assert len(result) == 1
    assert result[0]["word"] == " Hello"


def test_apply_filter_explain_mode():
    """Explain mode returns kept and rejected."""
    words = [
        {"start": 0.0, "end": 0.5, "word": " Hello"},
        {"start": 0.5, "end": 0.5, "word": " fake"},
    ]
    kept, rejected = apply_filter(words, has_duration, explain=True)
    assert len(kept) == 1
    assert len(rejected) == 1
    assert kept[0]["word"] == " Hello"
    assert rejected[0]["word"] == " fake"


def test_apply_filter_empty_input():
    """Empty input returns empty result."""
    result = apply_filter([], has_duration)
    assert result == []


def test_apply_filter_explain_empty():
    """Explain mode with empty input returns two empty lists."""
    kept, rejected = apply_filter([], has_duration, explain=True)
    assert kept == []
    assert rejected == []


# --- DEFAULT_FILTER tests ---

def test_default_filter_keeps_normal():
    """Default filter keeps words with duration and content."""
    word = {"start": 0.0, "end": 0.5, "word": " Hello"}
    assert DEFAULT_FILTER(word) is True


def test_default_filter_rejects_zero_duration():
    """Default filter rejects zero-duration words."""
    word = {"start": 0.5, "end": 0.5, "word": " fake"}
    assert DEFAULT_FILTER(word) is False


def test_default_filter_rejects_empty_content():
    """Default filter rejects empty-content words."""
    word = {"start": 0.0, "end": 0.5, "word": ""}
    assert DEFAULT_FILTER(word) is False


def test_default_filter_ignores_probability():
    """Default filter does NOT check probability."""
    word = {"start": 0.0, "end": 0.5, "word": " bad", "probability": 0.01}
    assert DEFAULT_FILTER(word) is True
