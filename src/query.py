"""Query layer for combining transcription and diarization data.

This module reads computed artifacts and joins them on-demand.
Uses bisect for O(log n) speaker lookup instead of linear scan.
"""

import bisect
from typing import Callable

from filters import DEFAULT_FILTER, Predicate


def build_speaker_index(segments: list[dict]) -> list[tuple[float, float, str]]:
    """Build sorted index of (start, end, speaker) for bisect lookup.

    Args:
        segments: Diarization segments with 'start', 'end', 'speaker' keys

    Returns:
        List of (start, end, speaker) tuples sorted by start time
    """
    index = [(s["start"], s["end"], s["speaker"]) for s in segments]
    index.sort(key=lambda x: x[0])
    return index


def find_speaker(midpoint: float, index: list[tuple[float, float, str]]) -> str | None:
    """Find speaker for a timestamp using O(log n) bisect lookup.

    Args:
        midpoint: Timestamp to look up (typically word midpoint)
        index: Sorted speaker index from build_speaker_index()

    Returns:
        Speaker label or None if timestamp falls in a gap
    """
    if not index:
        return None

    # Find rightmost segment that starts at or before midpoint
    starts = [s[0] for s in index]
    pos = bisect.bisect_right(starts, midpoint) - 1

    if pos < 0:
        return None  # Before first segment

    start, end, speaker = index[pos]
    if midpoint <= end:
        return speaker
    return None  # In a gap after this segment


def assign_speakers(
    transcript: dict,
    diarization: list[dict],
    word_filter: Predicate = None
) -> list[dict]:
    """Assign speakers to transcript words.

    Args:
        transcript: Transcript dict with 'segments' containing 'words'
        diarization: List of diarization segments
        word_filter: Optional predicate to filter words (default: DEFAULT_FILTER)

    Returns:
        List of word dicts with 'speaker' field added
    """
    if word_filter is None:
        word_filter = DEFAULT_FILTER

    # Extract all words from transcript segments
    all_words = []
    for segment in transcript.get("segments", []):
        all_words.extend(segment.get("words", []))

    # Filter words
    words = [w for w in all_words if word_filter(w)]

    # Build speaker index for O(log n) lookup
    index = build_speaker_index(diarization)

    # Assign speakers by word midpoint
    labeled = []
    for word in words:
        start = word.get("start", 0)
        end = word.get("end", 0)
        midpoint = (start + end) / 2
        speaker = find_speaker(midpoint, index)

        labeled.append({
            **word,
            "speaker": speaker
        })

    return labeled


def to_utterances(labeled_words: list[dict]) -> list[dict]:
    """Convert labeled words to utterances, consolidating same-speaker runs.

    Args:
        labeled_words: Words with 'speaker' field from assign_speakers()

    Returns:
        List of utterance dicts with consolidated text and word arrays
    """
    if not labeled_words:
        return []

    utterances = []
    current = None

    for word in labeled_words:
        speaker = word.get("speaker")
        text = word.get("word", "").strip()

        # Start new utterance if speaker changes (or first word)
        # Also start new if current speaker is None (don't consolidate unknowns)
        if current is None or speaker != current["speaker"] or current["speaker"] is None:
            if current is not None:
                utterances.append(current)
            current = {
                "speaker": speaker,
                "start": word.get("start", 0),
                "end": word.get("end", 0),
                "text": text,
                "words": [word]
            }
        else:
            # Same speaker - extend current utterance
            current["end"] = word.get("end", current["end"])
            current["text"] = f"{current['text']} {text}"
            current["words"].append(word)

    # Don't forget the last utterance
    if current is not None:
        utterances.append(current)

    return utterances


def format_transcript(utterances: list[dict]) -> str:
    """Format utterances as human-readable transcript.

    Args:
        utterances: List of utterance dicts from to_utterances()

    Returns:
        Formatted string with "SPEAKER: text" lines
    """
    lines = []
    for utt in utterances:
        speaker = utt.get("speaker") or "UNKNOWN"
        text = utt.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)
