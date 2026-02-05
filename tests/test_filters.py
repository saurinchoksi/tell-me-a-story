"""Tests for filters module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from filters import min_probability, Predicate


# --- Predicate type alias test ---

def test_predicate_type_alias_exists():
    """Predicate type alias is a callable type."""
    # Just verify the import works and it's a type
    assert Predicate is not None


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
