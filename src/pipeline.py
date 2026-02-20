"""Run the full transcription pipeline and save computed artifacts."""

import copy
import glob
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from transcribe import transcribe, clean_transcript, MODEL as TRANSCRIPTION_MODEL, make_processing_entry as make_transcription_entry
from diarize import diarize, enrich_with_diarization, detect_unintelligible_gaps, MODEL as DIARIZATION_MODEL
from mutagen import File as MutagenFile
from normalize import llm_normalize, MODEL as LLM_MODEL
from dictionary import load_library, build_variant_map, normalize_variants, make_processing_entry as make_dictionary_entry
from corrections import extract_text, apply_corrections

_DEFAULT_LIBRARY_PATH = str(Path(__file__).parent.parent / "data" / "mahabharata.json")

logger = logging.getLogger(__name__)


def enrich_transcript(
    transcript: dict,
    diarization: dict,
    library_path: str = None,
    verbose: bool = True,
) -> tuple[dict, list[dict], dict]:
    """Run all enrichment stages on a transcript.

    Runs LLM normalization, dictionary normalization, and diarization
    enrichment in sequence. Each stage is wrapped in try/except so
    failures are recorded but don't block subsequent stages.

    Does NOT set _processing on the transcript — the caller assembles that.

    Args:
        transcript: Whisper transcript dict with segments containing words.
        diarization: Diarization result dict with a 'segments' list.
        library_path: Path to dictionary library JSON (defaults to data/mahabharata.json).
        verbose: Log progress messages.

    Returns:
        Tuple of (enriched_transcript, processing_entries, counts_dict).
        processing_entries: list of stage dicts for _processing.
        counts_dict: {"llm_count": N, "dict_count": M}.
    """
    processing = []
    llm_count = 0
    dict_count = 0

    # Pass 1: LLM normalization
    try:
        text = extract_text(transcript)
        llm_corrections, llm_entry = llm_normalize(text, model=LLM_MODEL)
        transcript, llm_count = apply_corrections(transcript, llm_corrections, "llm")
        llm_entry["corrections_applied"] = llm_count
        processing.append(llm_entry)
        if verbose:
            logger.info(f"LLM normalization: {llm_count} corrections applied")
    except Exception as e:
        logger.warning(f"LLM normalization failed: {e}")
        processing.append({
            "stage": "llm_normalization",
            "model": LLM_MODEL,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Pass 2: Dictionary normalization
    lib_path = library_path or _DEFAULT_LIBRARY_PATH
    try:
        library = load_library(lib_path)
        variant_map = build_variant_map(library)
        text = extract_text(transcript)
        dict_corrections = normalize_variants(text, variant_map)
        transcript, dict_count = apply_corrections(transcript, dict_corrections, "dictionary")
        processing.append(make_dictionary_entry(lib_path, dict_count))
        if verbose:
            logger.info(f"Dictionary normalization: {dict_count} corrections applied")
    except Exception as e:
        logger.warning(f"Dictionary normalization failed: {e}")
        processing.append({
            "stage": "dictionary_normalization",
            "library": lib_path,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Pass 3: Diarization enrichment
    try:
        transcript, diar_entry = enrich_with_diarization(transcript, diarization)
        processing.append(diar_entry)
    except Exception as e:
        logger.warning(f"Diarization enrichment failed: {e}")
        processing.append({
            "stage": "diarization_enrichment",
            "model": DIARIZATION_MODEL,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Pass 4: Unintelligible gap detection
    try:
        transcript, gap_entry = detect_unintelligible_gaps(transcript, diarization)
        processing.append(gap_entry)
        if verbose:
            logger.info(f"Gap detection: {gap_entry['gaps_found']} unintelligible gaps injected")
    except Exception as e:
        logger.warning(f"Gap detection failed: {e}")
        processing.append({
            "stage": "gap_detection",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    return transcript, processing, {"llm_count": llm_count, "dict_count": dict_count}


def get_audio_info(filepath: str) -> dict | None:
    """Return basic properties of an audio file, or None if unreadable."""
    path = Path(filepath)

    if not path.exists():
        return None

    audio = MutagenFile(path)

    if audio is None:
        return None

    return {
        "filename": path.name,
        "format": type(audio).__name__,
        "duration_seconds": audio.info.length,
        "sample_rate": audio.info.sample_rate,
        "channels": audio.info.channels,
    }


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def save_computed(session_dir: str, transcript_raw: dict, transcript: dict, diarization: dict) -> None:
    """Save computed artifacts to session directory.

    Creates:
        {session_dir}/
            transcript-raw.json
            transcript-rich.json
            diarization.json

    Args:
        session_dir: Path to session directory
        transcript_raw: Clean Whisper output before enrichment (immutable)
        transcript: Enriched transcript with corrections, speakers, and audio info
        diarization: Diarization result from diarize()
    """
    os.makedirs(session_dir, exist_ok=True)

    # Save raw transcript (pre-enrichment snapshot)
    with open(os.path.join(session_dir, "transcript-raw.json"), "w") as f:
        json.dump(transcript_raw, f, indent=2)

    # Save enriched transcript
    with open(os.path.join(session_dir, "transcript-rich.json"), "w") as f:
        json.dump(transcript, f, indent=2)

    # Save diarization
    with open(os.path.join(session_dir, "diarization.json"), "w") as f:
        json.dump(diarization, f, indent=2)


def run_pipeline(audio_path: str, verbose: bool = True, library_path: str = None) -> dict:
    """Run full pipeline on an audio file.

    Args:
        audio_path: Path to audio file
        verbose: Print progress messages
        library_path: Path to dictionary library JSON (defaults to data/mahabharata.json)

    Returns:
        Dict with 'transcript_raw', 'transcript', 'diarization' keys.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    session_id = Path(audio_path).parent.name
    if not re.fullmatch(r"\d{8}-\d{6}", session_id):
        raise ValueError(
            f"Invalid session ID '{session_id}' — expected YYYYMMDD-HHMMSS format "
            f"(from parent directory of {audio_path})"
        )

    # Get audio info
    audio_info = get_audio_info(audio_path)

    # Transcription
    if verbose:
        print(f"Transcribing: {audio_path}")
        print(f"Using: {TRANSCRIPTION_MODEL}")

    transcript_time = datetime.now(timezone.utc).isoformat()
    raw_transcript = transcribe(audio_path, model=TRANSCRIPTION_MODEL)
    transcript = clean_transcript(raw_transcript)
    transcript_raw = copy.deepcopy(transcript)

    # Diarization
    if verbose:
        print("\nDiarizing (this takes a few minutes)...")

    diarization = diarize(audio_path)

    # Enrichment (normalization + diarization)
    transcript, enrichment_processing, _ = enrich_transcript(
        transcript, diarization, library_path=library_path, verbose=verbose
    )

    # Fold audio info into enriched transcript
    transcript["audio"] = audio_info

    audio_hash = compute_file_hash(audio_path)
    processing = [make_transcription_entry(audio_hash, transcript_time)] + enrichment_processing
    transcript["_processing"] = processing

    return {
        "session_id": session_id,
        "transcript_raw": transcript_raw,
        "transcript": transcript,
        "diarization": diarization,
    }


def to_utterances(labeled_words: list[dict]) -> list[dict]:
    """Convert labeled words to utterances, consolidating same-speaker runs.

    Accepts both enriched format (speaker at word["_speaker"]["label"]) and
    flat format (speaker at word["speaker"]). Enriched format takes priority.

    Args:
        labeled_words: Words with '_speaker.label' or 'speaker' field

    Returns:
        List of utterance dicts with consolidated text and word arrays
    """
    if not labeled_words:
        return []

    utterances = []
    current = None

    for word in labeled_words:
        if "_speaker" in word:
            speaker = word["_speaker"].get("label")
        else:
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


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run transcription pipeline")
    parser.add_argument("path", help="Path to audio file, or session directory with --re-enrich")
    parser.add_argument("--re-enrich", action="store_true",
                        help="Re-enrich an existing session from transcript-raw.json")
    parser.add_argument("--library", default=None,
                        help="Path to dictionary library JSON")
    args = parser.parse_args()

    if args.re_enrich:
        # Re-enrich mode: load raw transcript + diarization, skip transcription
        session_dir = args.path
        raw_path = os.path.join(session_dir, "transcript-raw.json")
        diarization_path = os.path.join(session_dir, "diarization.json")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"No transcript-raw.json in {session_dir}")
        if not os.path.exists(diarization_path):
            raise FileNotFoundError(f"No diarization.json in {session_dir}")

        with open(raw_path) as f:
            transcript_raw = json.load(f)
        with open(diarization_path) as f:
            diarization = json.load(f)

        # Deep-copy raw so enrichment doesn't mutate it
        transcript = copy.deepcopy(transcript_raw)

        print("Running enrichment stages...")
        transcript, enrichment_processing, counts = enrich_transcript(
            transcript, diarization, library_path=args.library
        )

        # Build transcription stub from raw's generator version
        transcription_entry = {
            "stage": "transcription",
            "model": transcript_raw.get("_generator_version", "unknown"),
            "status": "prior_run",
        }

        # Fold audio info if audio file exists
        audio_matches = glob.glob(os.path.join(session_dir, "audio.*"))
        audio_path = audio_matches[0] if audio_matches else None
        if audio_path:
            audio_info = get_audio_info(audio_path)
            transcript["audio"] = audio_info
            transcription_entry["audio_hash"] = compute_file_hash(audio_path)

        processing = [transcription_entry] + enrichment_processing
        transcript["_processing"] = processing

        # Save only the enriched transcript
        with open(os.path.join(session_dir, "transcript-rich.json"), "w") as f:
            json.dump(transcript, f, indent=2)

        print(f"\nSaved to: {session_dir}/transcript-rich.json")
        print(f"  LLM corrections: {counts['llm_count']}")
        print(f"  Dictionary corrections: {counts['dict_count']}")
    else:
        # Full pipeline mode
        result = run_pipeline(args.path, library_path=args.library)

        session_id = result["session_id"]
        session_dir = f"sessions/{session_id}"

        save_computed(
            session_dir=session_dir,
            transcript_raw=result["transcript_raw"],
            transcript=result["transcript"],
            diarization=result["diarization"],
        )

        # Flatten enriched words for utterance grouping
        labeled_words = []
        for seg in result["transcript"]["segments"]:
            labeled_words.extend(seg.get("words", []))
        utterances = to_utterances(labeled_words)

        print("\n--- Speaker-labeled transcript ---\n")
        print(format_transcript(utterances))

        print(f"\nSaved session to: {session_dir}/")
        print(f"  transcript-raw.json")
        print(f"  transcript-rich.json")
        print(f"  diarization.json")

        processing = result["transcript"]["_processing"]
        for entry in processing:
            if entry["stage"] == "llm_normalization":
                print(f"\nLLM Normalization: {entry.get('corrections_applied', 0)} corrections")
            elif entry["stage"] == "dictionary_normalization":
                print(f"Dictionary Normalization: {entry.get('corrections_applied', 0)} corrections")
