"""Run the full transcription pipeline and save computed artifacts."""

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from transcribe import transcribe, clean_transcript, _SCHEMA_VERSION as TRANSCRIPT_SCHEMA
from diarize import diarize
from query import assign_speakers, to_utterances, format_transcript
from inspect_audio import get_audio_info
from normalize import normalize as llm_normalize
from dictionary import load_library, build_variant_map, normalize_variants
from corrections import extract_text, apply_corrections, _NORMALIZED_SCHEMA_VERSION

_DEFAULT_LIBRARY_PATH = str(Path(__file__).parent.parent / "data" / "mahabharata.json")
_LLM_MODEL = "qwen3:8b"
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run transcription pipeline")
    parser.add_argument("audio_file", help="Path to audio file")
    return parser.parse_args()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def create_manifest(
    session_id: str,
    audio_path: str,
    transcript_model: str,
    transcript_time: str,
    diarization_model: str,
    diarization_time: str,
) -> dict:
    """Create a session manifest with model versions and timestamps.

    Args:
        session_id: Unique session identifier
        audio_path: Path to audio file
        transcript_model: Model used for transcription
        transcript_time: ISO timestamp of transcription
        diarization_model: Model used for diarization
        diarization_time: ISO timestamp of diarization

    Returns:
        Manifest dict for JSON serialization.
    """
    return {
        "_schema_version": "1.0.0",
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "audio_file": "audio.m4a",
            "audio_hash": compute_file_hash(audio_path),
        },
        "computed": {
            "transcript": {
                "file": "transcript.json",
                "model": transcript_model,
                "created_at": transcript_time,
            },
            "diarization": {
                "file": "diarization.json",
                "model": diarization_model,
                "created_at": diarization_time,
            },
        },
    }


def save_computed(session_dir: str, audio_info: dict, transcript: dict, diarization: dict, manifest: dict) -> None:
    """Save computed artifacts to session directory.

    Creates:
        {session_dir}/
            audio-info.json
            transcript.json
            diarization.json
            manifest.json

    Args:
        session_dir: Path to session directory
        audio_info: Audio metadata from get_audio_info()
        transcript: Cleaned transcript from clean_transcript()
        diarization: Diarization result from diarize()
        manifest: Session manifest from create_manifest()
    """
    os.makedirs(session_dir, exist_ok=True)

    # Save audio info
    with open(os.path.join(session_dir, "audio-info.json"), "w") as f:
        json.dump(audio_info, f, indent=2)

    # Save transcript
    with open(os.path.join(session_dir, "transcript.json"), "w") as f:
        json.dump(transcript, f, indent=2)

    # Save diarization
    with open(os.path.join(session_dir, "diarization.json"), "w") as f:
        json.dump(diarization, f, indent=2)

    # Save manifest at session root
    with open(os.path.join(session_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def run_pipeline(audio_path: str, verbose: bool = True, library_path: str = None) -> dict:
    """Run full pipeline on an audio file.

    Args:
        audio_path: Path to audio file
        verbose: Print progress messages
        library_path: Path to dictionary library JSON (defaults to data/mahabharata.json)

    Returns:
        Dict with 'audio_info', 'transcript', 'diarization', 'manifest' keys.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    session_id = Path(audio_path).parent.name

    # Get audio info
    audio_info = get_audio_info(audio_path)

    # Transcription
    model = "mlx-community/whisper-large-v3-mlx"
    if verbose:
        print(f"Transcribing: {audio_path}")
        print(f"Using: {model}")

    transcript_time = datetime.now(timezone.utc).isoformat()
    raw_transcript = transcribe(audio_path, word_timestamps=True, model=model)
    transcript = clean_transcript(raw_transcript)

    # Normalization
    processing = [
        {"stage": "transcription", "model": model, "status": "success"}
    ]

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

    transcript["_processing"] = processing
    transcript["_schema_version"] = _NORMALIZED_SCHEMA_VERSION

    # Diarization
    if verbose:
        print("\nDiarizing (this takes a few minutes)...")

    diarization_time = datetime.now(timezone.utc).isoformat()
    diarization = diarize(audio_path)
    diarization_model = "pyannote/speaker-diarization-community-1"

    # Create manifest
    manifest = create_manifest(
        session_id=session_id,
        audio_path=audio_path,
        transcript_model=model,
        transcript_time=transcript_time,
        diarization_model=diarization_model,
        diarization_time=diarization_time,
    )

    return {
        "session_id": session_id,
        "audio_info": audio_info,
        "transcript": transcript,
        "diarization": diarization,
        "manifest": manifest,
        "llm_count": llm_count,
        "dict_count": dict_count,
    }


if __name__ == "__main__":
    args = parse_args()

    result = run_pipeline(args.audio_file)

    session_id = result["session_id"]
    session_dir = f"sessions/{session_id}"

    # Save computed artifacts
    save_computed(
        session_dir=session_dir,
        audio_info=result["audio_info"],
        transcript=result["transcript"],
        diarization=result["diarization"],
        manifest=result["manifest"],
    )

    # Use query layer to display transcript
    diarization_segments = result["diarization"]["segments"]
    labeled_words = assign_speakers(result["transcript"], diarization_segments)
    utterances = to_utterances(labeled_words)

    print("\n--- Speaker-labeled transcript ---\n")
    print(format_transcript(utterances))

    print(f"\nSaved session to: {session_dir}/")
    print(f"  manifest.json")
    print(f"  audio-info.json")
    print(f"  transcript.json")
    print(f"  diarization.json")

    print(f"\nLLM Normalization: {result['llm_count']} corrections")
    print(f"Dictionary Normalization: {result['dict_count']} corrections")
