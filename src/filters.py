"""Predicate functions for filtering transcription words.

Filters are pure functions that return True/False for inclusion.
Use apply_filter() to run a filter and optionally get explanations.
"""

from typing import Callable


# Type alias for filter predicates
Predicate = Callable[[dict], bool]


def has_duration(word: dict) -> bool:
    """Return True if word has non-zero duration (end > start).

    Zero-duration words are fabrications from Whisper hallucination.
    """
    return word.get("end", 0) > word.get("start", 0)


def has_content(word: dict) -> bool:
    """Return True if word has non-empty text after stripping whitespace."""
    text = word.get("word", "")
    return bool(text and text.strip())


def min_probability(threshold: float) -> Predicate:
    """Factory: return predicate that checks probability >= threshold.

    Words without a probability key are assumed valid (kept).
    """
    def check_prob(word: dict) -> bool:
        prob = word.get("probability")
        if prob is None:
            return True  # No probability = assume valid
        return prob >= threshold
    return check_prob


def all_of(*predicates: Predicate) -> Predicate:
    """Combine predicates with AND logic - all must pass."""
    def combined(word: dict) -> bool:
        return all(p(word) for p in predicates)
    return combined


def apply_filter(
    words: list[dict],
    predicate: Predicate,
    explain: bool = False
) -> list[dict] | tuple[list[dict], list[dict]]:
    """Apply a filter predicate to a list of words.

    Args:
        words: List of word dicts from transcription
        predicate: Function returning True to keep, False to reject
        explain: If True, return (kept, rejected) tuple for debugging

    Returns:
        If explain=False: list of words that passed the filter
        If explain=True: tuple of (kept_words, rejected_words)
    """
    if not explain:
        return [w for w in words if predicate(w)]

    kept = []
    rejected = []
    for w in words:
        if predicate(w):
            kept.append(w)
        else:
            rejected.append(w)
    return kept, rejected


# Default filter: basic sanity checks (duration + content)
# Does NOT include probability - that's optional via custom filter
DEFAULT_FILTER = all_of(has_duration, has_content)
