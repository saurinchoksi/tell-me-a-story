"""Enrich transcripts with metadata from other pipeline stages.

Each enrichment function deep-copies the transcript, adds underscore-prefixed
metadata to word dicts, and returns the modified copy. The original transcript
is never mutated.

Current enrichments:
    _speaker: Speaker label and coverage from diarization alignment.
"""

import copy


_ENRICHED_SCHEMA_VERSION = "1.2.0"


def _compute_speaker_coverage(word_start, word_end, diar_segments):
    """Compute which speaker best covers a word's time range.

    For each diarization segment, calculates the temporal overlap with the
    word. The speaker with the greatest total overlap wins.

    Args:
        word_start: Word start time in seconds.
        word_end: Word end time in seconds.
        diar_segments: List of diarization segment dicts with start, end,
            and speaker keys.

    Returns:
        Dict with 'label' (speaker string or None) and 'coverage' (float
        0.0-1.0 indicating what fraction of the word duration is covered
        by the best-matching speaker).
    """
    word_duration = word_end - word_start
    if word_duration <= 0:
        return {"label": None, "coverage": 0.0}

    overlap_by_speaker = {}
    for seg in diar_segments:
        overlap = max(0, min(word_end, seg["end"]) - max(word_start, seg["start"]))
        if overlap > 0:
            speaker = seg["speaker"]
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0) + overlap

    if not overlap_by_speaker:
        return {"label": None, "coverage": 0.0}

    best_speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
    coverage = min(overlap_by_speaker[best_speaker] / word_duration, 1.0)

    return {"label": best_speaker, "coverage": coverage}


def enrich_with_diarization(transcript, diarization):
    """Add speaker labels to each word in a transcript using diarization data.

    Deep-copies the transcript, then assigns a _speaker dict to every word
    based on temporal overlap with diarization segments. Does not touch
    _processing or _schema_version -- the pipeline handles those.

    Args:
        transcript: Whisper transcript dict with segments containing words.
        diarization: Diarization result dict with a 'segments' list of
            {start, end, speaker} dicts.

    Returns:
        Deep-copied transcript with _speaker metadata on each word.
    """
    result = copy.deepcopy(transcript)
    diar_segments = diarization.get("segments", [])

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            speaker_info = _compute_speaker_coverage(
                word["start"], word["end"], diar_segments
            )
            word["_speaker"] = speaker_info

    return result
