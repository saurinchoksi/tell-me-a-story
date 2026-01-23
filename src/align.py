"""Align transcription with speaker diarization."""


def align_words_to_speakers(words: list[dict], diarization: list[dict]) -> list[dict]:
    """Assign a speaker to each word based on timestamp overlap.
    
    Args:
        words: List of word dicts with 'start', 'end', 'word' keys
        diarization: List of speaker segments with 'start', 'end', 'speaker' keys
    
    Returns:
        List of word dicts, each with added 'speaker' key.
    """
    labeled_words = []
    
    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        
        speaker = None
        for seg in diarization:
            if seg["start"] <= midpoint <= seg["end"]:
                speaker = seg["speaker"]
                break
        
        labeled_words.append({
            "start": word["start"],
            "end": word["end"],
            "word": word["word"],
            "speaker": speaker
        })
    
    return labeled_words

def group_words_by_speaker(labeled_words: list[dict]) -> list[dict]:
    """Group consecutive words by the same speaker into utterances.

    Args:
        labeled_words: Words with speaker assignments (from align_words_to_speakers)

    Returns:
        List of utterance dicts with 'start', 'end', 'speaker', 'text' keys.
    """
    if not labeled_words:
        return []

    utterances = []
    current_speaker = labeled_words[0]["speaker"]
    current_words = [labeled_words[0]]

    for word in labeled_words[1:]:
        if word["speaker"] == current_speaker:
            current_words.append(word)
        else:
            utterances.append(_make_utterance(current_words, current_speaker))
            current_speaker = word["speaker"]
            current_words = [word]

    # Don't forget the last utterance
    utterances.append(_make_utterance(current_words, current_speaker))

    return utterances


def merge_unknown_utterances(utterances: list[dict]) -> list[dict]:
    """Fill in UNKNOWN utterances when sandwiched between same speaker.
    
    Args:
        utterances: List of utterance dicts from group_words_by_speaker
    
    Returns:
        New list with UNKNOWN speakers filled in where possible.
    """
    if len(utterances) < 3:
        return utterances
    
    result = []
    for i, utt in enumerate(utterances):
        if utt["speaker"] is None and 0 < i < len(utterances) - 1:
            before = utterances[i - 1]["speaker"]
            after = utterances[i + 1]["speaker"]
            if before == after and before is not None:
                result.append({
                    "start": utt["start"],
                    "end": utt["end"],
                    "speaker": before,
                    "text": utt["text"]
                })
                continue
        result.append(utt)
    
    return result


def assign_leading_fragments(utterances: list[dict], max_gap: float = 0.5) -> list[dict]:
    """Assign UNKNOWN utterances to the next speaker if close in time.
    
    When diarization misses the start of a speaker's turn, the opening
    words get labeled UNKNOWN. If the gap to the next utterance is small,
    it's likely part of that speaker's turn.
    
    Args:
        utterances: List of utterance dicts
        max_gap: Maximum gap (seconds) to consider as same turn
    
    Returns:
        New list with leading fragments assigned.
    """
    result = []
    
    for i, utt in enumerate(utterances):
        if utt["speaker"] is None and i < len(utterances) - 1:
            next_utt = utterances[i + 1]
            gap = next_utt["start"] - utt["end"]
            
            if gap <= max_gap and next_utt["speaker"] is not None:
                result.append({
                    "start": utt["start"],
                    "end": utt["end"],
                    "speaker": next_utt["speaker"],
                    "text": utt["text"]
                })
                continue
        
        result.append(utt)
    
    return result


def consolidate_utterances(utterances: list[dict]) -> list[dict]:
    """Combine consecutive utterances from the same speaker.
    
    Run this after merge_unknown_utterances to collapse fragments.
    
    Args:
        utterances: List of utterance dicts
    
    Returns:
        New list with consecutive same-speaker utterances combined.
    """
    if not utterances:
        return []
    
    result = [utterances[0].copy()]
    
    for utt in utterances[1:]:
        if utt["speaker"] == result[-1]["speaker"]:
            # Same speaker — extend the previous utterance
            result[-1]["end"] = utt["end"]
            result[-1]["text"] += " " + utt["text"]
        else:
            # Different speaker — start new utterance
            result.append(utt.copy())
    
    return result


def _make_utterance(words: list[dict], speaker: str) -> dict:
    """Create an utterance dict from a list of words."""
    text = "".join(w["word"] for w in words).strip()
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "speaker": speaker,
        "text": text
    }

def format_transcript(utterances: list[dict]) -> str:
    """Format utterances as a readable transcript.
    
    Args:
        utterances: List of utterance dicts from group_words_by_speaker
    
    Returns:
        Formatted string with timestamps and speaker labels.
    """
    lines = []
    for utt in utterances:
        speaker = utt["speaker"] or "UNKNOWN"
        start = utt["start"]
        end = utt["end"]
        text = utt["text"]
        lines.append(f"[{start:6.1f} - {end:6.1f}] {speaker}: {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from transcribe import transcribe
    from diarize import diarize

    if len(sys.argv) < 2:
        print("Usage: python src/align.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    print(f"Transcribing: {audio_file}")
    print("Using large model for better child speech recognition...")
    result = transcribe(
        audio_file,
        word_timestamps=True,
        model="mlx-community/whisper-large-v3-turbo"
    )

    print("\nDiarizing (this takes a few minutes)...")
    segments = diarize(audio_file)

    # Extract words from all transcript segments
    all_words = []
    for seg in result["segments"]:
        if seg.get("words"):
            all_words.extend(seg["words"])

    print(f"\nAligning {len(all_words)} words to {len(segments)} speaker segments...")
    labeled = align_words_to_speakers(all_words, segments)
    utterances = group_words_by_speaker(labeled)

    # Count UNKNOWNs before merge
    unknown_before = sum(1 for u in utterances if u["speaker"] is None)

    # Merge UNKNOWN utterances sandwiched between same speaker
    merged = merge_unknown_utterances(utterances)

    # Assign leading fragments to next speaker if close in time
    assigned = assign_leading_fragments(merged)

    # Consolidate consecutive same-speaker utterances
    consolidated = consolidate_utterances(assigned)

    # Count UNKNOWNs after processing
    unknown_after = sum(1 for u in consolidated if u["speaker"] is None)

    print(f"UNKNOWN utterances: {unknown_before} -> {unknown_after} after merge")
    print(f"Utterances: {len(utterances)} -> {len(consolidated)} after consolidation")
    print("\n--- Speaker-labeled transcript ---\n")
    print(format_transcript(consolidated))
