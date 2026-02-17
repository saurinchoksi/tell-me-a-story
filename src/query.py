"""Query layer for combining transcription and diarization data.

This module reads computed artifacts and joins them on-demand.
"""


def to_utterances(labeled_words: list[dict]) -> list[dict]:
    """Convert labeled words to utterances, consolidating same-speaker runs.

    Args:
        labeled_words: Words with 'speaker' field

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
