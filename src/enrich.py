"""Shared enrichment orchestrator and CLI for re-enriching sessions.

Runs three enrichment stages on a transcript:
    1. LLM normalization (phonetic mishearing correction via local Ollama)
    2. Dictionary normalization (variant spelling correction via reference library)
    3. Diarization enrichment (speaker labels from pyannote alignment)

Used by pipeline.py for fresh runs and directly via CLI for re-enriching
existing sessions without re-transcribing or re-diarizing.
"""

import argparse
import copy
import json
import logging
from pathlib import Path

from normalize import normalize as llm_normalize
from dictionary import load_library, build_variant_map, normalize_variants
from corrections import extract_text, apply_corrections
from enrichment import enrich_with_diarization, _ENRICHED_SCHEMA_VERSION

_DEFAULT_LIBRARY_PATH = str(Path(__file__).parent.parent / "data" / "mahabharata.json")
_LLM_MODEL = "qwen3:8b"
_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

logger = logging.getLogger(__name__)


def strip_enrichments(transcript: dict) -> dict:
    """Remove all enrichment metadata from a transcript, restoring original words.

    Deep-copies the transcript, then:
    - Removes _processing and _schema_version (but keeps _generator_version)
    - For each word: if _original exists, restores word to leading_space + _original
    - Removes _corrections, _original, _speaker from each word

    Args:
        transcript: Enriched transcript dict.

    Returns:
        Clean transcript with enrichment metadata stripped.
    """
    result = copy.deepcopy(transcript)

    result.pop("_processing", None)
    result.pop("_schema_version", None)

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if "_original" in word:
                leading_space = " " if word["word"].startswith(" ") else ""
                word["word"] = leading_space + word["_original"]
                del word["_original"]

            word.pop("_corrections", None)
            word.pop("_speaker", None)

    return result


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

    Does NOT set _processing or _schema_version on the transcript —
    the caller assembles those.

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
        llm_corrections = llm_normalize(text, model=_LLM_MODEL)
        transcript, llm_count = apply_corrections(transcript, llm_corrections, "llm")
        processing.append({
            "stage": "llm_normalization",
            "model": _LLM_MODEL,
            "status": "success",
            "corrections_applied": llm_count,
        })
        if verbose:
            logger.info(f"LLM normalization: {llm_count} corrections applied")
    except Exception as e:
        logger.warning(f"LLM normalization failed: {e}")
        processing.append({
            "stage": "llm_normalization",
            "model": _LLM_MODEL,
            "status": "error",
            "error": str(e),
        })

    # Pass 2: Dictionary normalization
    lib_path = library_path or _DEFAULT_LIBRARY_PATH
    try:
        library = load_library(lib_path)
        variant_map = build_variant_map(library)
        text = extract_text(transcript)
        dict_corrections = normalize_variants(text, variant_map)
        transcript, dict_count = apply_corrections(transcript, dict_corrections, "dictionary")
        processing.append({
            "stage": "dictionary_normalization",
            "library": lib_path,
            "status": "success",
            "corrections_applied": dict_count,
        })
        if verbose:
            logger.info(f"Dictionary normalization: {dict_count} corrections applied")
    except Exception as e:
        logger.warning(f"Dictionary normalization failed: {e}")
        processing.append({
            "stage": "dictionary_normalization",
            "library": lib_path,
            "status": "error",
            "error": str(e),
        })

    # Pass 3: Diarization enrichment
    try:
        transcript = enrich_with_diarization(transcript, diarization)
        processing.append({
            "stage": "diarization_enrichment",
            "model": _DIARIZATION_MODEL,
            "status": "success",
        })
    except Exception as e:
        logger.warning(f"Diarization enrichment failed: {e}")
        processing.append({
            "stage": "diarization_enrichment",
            "model": _DIARIZATION_MODEL,
            "status": "error",
            "error": str(e),
        })

    return transcript, processing, {"llm_count": llm_count, "dict_count": dict_count}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Re-enrich an existing session's transcript"
    )
    parser.add_argument(
        "session_dir",
        help="Path to session directory (e.g. sessions/00000000-000001)",
    )
    parser.add_argument(
        "--library",
        default=None,
        help="Path to dictionary library JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = _parse_args()
    session_dir = Path(args.session_dir)

    transcript_path = session_dir / "transcript.json"
    diarization_path = session_dir / "diarization.json"

    if not transcript_path.exists():
        raise FileNotFoundError(f"No transcript.json in {session_dir}")
    if not diarization_path.exists():
        raise FileNotFoundError(f"No diarization.json in {session_dir}")

    with open(transcript_path) as f:
        transcript = json.load(f)
    with open(diarization_path) as f:
        diarization = json.load(f)

    # Preserve the original transcription processing entry before stripping
    original_processing = transcript.get("_processing", [])
    transcription_entry = None
    for entry in original_processing:
        if entry.get("stage") == "transcription":
            transcription_entry = entry
            break

    if transcription_entry is None:
        transcription_entry = {"stage": "transcription", "status": "unknown"}

    # Strip all enrichment metadata
    print(f"Stripping existing enrichments from {transcript_path}")
    transcript = strip_enrichments(transcript)

    # Re-run all enrichment stages
    print("Running enrichment stages...")
    transcript, enrichment_processing, counts = enrich_transcript(
        transcript, diarization, library_path=args.library
    )

    # Assemble full processing list
    transcript["_processing"] = [transcription_entry] + enrichment_processing
    transcript["_schema_version"] = _ENRICHED_SCHEMA_VERSION

    # Save
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)

    print(f"\nSaved enriched transcript to {transcript_path}")
    print(f"  LLM corrections: {counts['llm_count']}")
    print(f"  Dictionary corrections: {counts['dict_count']}")
    print(f"  Schema version: {_ENRICHED_SCHEMA_VERSION}")
    print(f"  Processing entries: {len(transcript['_processing'])}")
