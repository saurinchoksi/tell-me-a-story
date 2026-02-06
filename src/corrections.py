"""Pure data helpers for applying normalization corrections to transcripts.

Multi-word corrections (e.g. "Bhagwad Gita" -> "Bhagavad Gita") are silently
skipped -- apply_corrections matches per-word only. Single-word matches like
"Gita" still work.
"""

import copy

_NORMALIZED_SCHEMA_VERSION = "1.1.0"


def extract_text(transcript: dict) -> str:
    """Join all words from transcript segments into a single string.

    Whisper words carry natural leading spaces (e.g. " hello"), so we
    concatenate them directly and strip the leading whitespace from the result.

    Args:
        transcript: Whisper transcript dict with segments containing words.

    Returns:
        Clean text string with words joined by their natural spacing.
    """
    parts = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            parts.append(word["word"])
    return "".join(parts).strip()


def apply_corrections(
    transcript: dict, corrections: list[dict], stage: str
) -> tuple[dict, int]:
    """Apply single-word corrections to a transcript.

    Deep-copies the transcript, then for each word in each segment, checks
    if the word matches any correction (case-insensitive, stripped). Multi-word
    corrections are silently skipped.

    On match, the word dict gains:
        _original: The stripped original word (set only on first correction)
        _corrections: List of {stage, from, to} dicts tracking each change
        word: Updated to the corrected form, preserving any leading space

    Args:
        transcript: Whisper transcript dict with segments containing words.
        corrections: List of dicts with 'transcribed' and 'correct' keys.
        stage: Label for this correction pass (e.g. "llm", "dictionary").

    Returns:
        Tuple of (modified transcript, count of words corrected).
    """
    result = copy.deepcopy(transcript)

    # Build lookup, skipping multi-word corrections
    lookup = {}
    for correction in corrections:
        transcribed = correction["transcribed"].strip()
        if " " in transcribed:
            continue
        lookup[transcribed.lower()] = correction["correct"]

    count = 0

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            raw = word["word"]
            stripped = raw.strip()
            key = stripped.lower()

            if key not in lookup:
                continue

            correct = lookup[key]

            # Preserve _original from the very first correction
            if "_original" not in word:
                word["_original"] = stripped

            # Append to corrections history
            entry = {"stage": stage, "from": stripped, "to": correct}
            if "_corrections" not in word:
                word["_corrections"] = []
            word["_corrections"].append(entry)

            # Update the word, preserving leading space
            leading_space = " " if raw.startswith(" ") else ""
            word["word"] = leading_space + correct

            count += 1

    return result, count
