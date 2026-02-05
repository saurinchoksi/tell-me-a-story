"""Predicate functions for filtering transcription words.

Filters are pure functions that return True/False for inclusion.
"""

from typing import Callable


# Type alias for filter predicates
Predicate = Callable[[dict], bool]


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
