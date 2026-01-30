"""Align transcription with speaker diarization."""


def filter_zero_duration_words(words: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove words with zero duration (hallucination signal).

    Whisper sometimes hallucinates words during silence, producing
    words where start == end. Real speech always has duration.

    Args:
        words: List of word dicts with 'start', 'end', 'word' keys

    Returns:
        Tuple of (kept, removed) where:
        - kept: Words with positive duration
        - removed: Words with zero duration, each annotated with 'filter_reason'
    """
    kept, removed = [], []
    for w in words:
        if w["end"] > w["start"]:
            kept.append(w)
        else:
            removed.append({**w, "filter_reason": "zero_duration"})
    return kept, removed


def filter_low_probability_words(words: list[dict], threshold: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Remove words with low probability (hallucination signal).

    Whisper assigns probability scores to each word. Real speech
    typically has probability 0.85-1.0, while hallucinations often
    start with very low probability (< 0.1) during silence.

    Args:
        words: List of word dicts with 'probability' key
        threshold: Minimum probability to keep (default 0.5)

    Returns:
        Tuple of (kept, removed) where:
        - kept: Words with probability >= threshold
        - removed: Words with probability < threshold, each annotated with 'filter_reason'
    """
    kept, removed = [], []
    for w in words:
        prob = w.get("probability", 1.0)
        if prob >= threshold:
            kept.append(w)
        else:
            removed.append({**w, "filter_reason": f"low_probability ({prob:.2f} < {threshold})"})
    return kept, removed


def align(
    words: list[dict],
    diarization: list[dict],
    prob_threshold: float = 0.5,
    return_debug: bool = False
) -> list[dict] | dict:
    """Run the full alignment pipeline.

    Args:
        words: List of word dicts with 'start', 'end', 'word' keys
        diarization: List of speaker segments with 'start', 'end', 'speaker' keys
        prob_threshold: Minimum probability to keep words (default 0.5)
        return_debug: If True, return detailed debug info instead of just utterances

    Returns:
        If return_debug is False: List of consolidated utterances with speaker labels.
        If return_debug is True: Dict containing:
            - utterances: The final consolidated utterances
            - words_after_zero_filter: Words kept after zero duration filter
            - words_removed_zero: Words removed by zero duration filter
            - words_after_prob_filter: Words kept after probability filter
            - words_removed_prob: Words removed by probability filter
            - words_labeled: Words after speaker assignment
            - utterances_raw: Mini-utterances before consolidation
    """
    words_after_zero, words_removed_zero = filter_zero_duration_words(words)
    words_after_prob, words_removed_prob = filter_low_probability_words(words_after_zero, prob_threshold)
    labeled = align_words_to_speakers(words_after_prob, diarization)
    utterances_raw = words_to_utterances(labeled)
    consolidated = consolidate_utterances(utterances_raw)

    if return_debug:
        return {
            "utterances": consolidated,
            "words_after_zero_filter": words_after_zero,
            "words_removed_zero": words_removed_zero,
            "words_after_prob_filter": words_after_prob,
            "words_removed_prob": words_removed_prob,
            "words_labeled": labeled,
            "utterances_raw": utterances_raw,
        }

    return consolidated


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
        
        labeled_word = {
            "start": word["start"],
            "end": word["end"],
            "word": word["word"],
            "speaker": speaker
        }
        # Preserve probability if present (used for word-level filtering)
        if "probability" in word:
            labeled_word["probability"] = word["probability"]
        labeled_words.append(labeled_word)
    
    return labeled_words

def words_to_utterances(labeled_words: list[dict]) -> list[dict]:
    """Convert each labeled word to a mini-utterance.

    Args:
        labeled_words: Words with speaker assignments (from align_words_to_speakers)

    Returns:
        List of utterance dicts, one per word.
    """
    utterances = []
    for w in labeled_words:
        utterances.append({
            "start": w["start"],
            "end": w["end"],
            "speaker": w["speaker"],
            "text": w["word"].strip(),
            "words": [{k: v for k, v in w.items() if k != "speaker"}]
        })
    return utterances


def consolidate_utterances(utterances: list[dict]) -> list[dict]:
    """Combine consecutive utterances from the same known speaker.

    Only merges when speaker is not None. UNKNOWN (None) utterances stay separate.

    Args:
        utterances: List of utterance dicts

    Returns:
        New list with consecutive same-speaker utterances combined (known speakers only).
    """
    if not utterances:
        return []

    result = [utterances[0].copy()]
    # Deep copy the words list to avoid mutation
    result[0]["words"] = list(utterances[0].get("words", []))

    for utt in utterances[1:]:
        if utt["speaker"] == result[-1]["speaker"] and utt["speaker"] is not None:
            # Same known speaker — extend the previous utterance
            result[-1]["end"] = utt["end"]
            result[-1]["text"] += " " + utt["text"]
            result[-1]["words"].extend(utt.get("words", []))
        else:
            # Different speaker or None — start new utterance
            new_utt = utt.copy()
            new_utt["words"] = list(utt.get("words", []))
            result.append(new_utt)

    return result


def format_transcript(utterances: list[dict]) -> str:
    """Format utterances as a readable transcript.
    
    Args:
        utterances: List of utterance dicts (from align() or consolidate_utterances())
    
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

