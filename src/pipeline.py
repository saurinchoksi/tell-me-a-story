"""Run the full transcription pipeline and save computed artifacts."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from transcribe import transcribe, clean_transcript
from diarize import diarize
from inspect_audio import get_audio_info
from enrich import enrich_transcript
from enrichment import _ENRICHED_SCHEMA_VERSION


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
            "audio_file": Path(audio_path).name,
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
    raw_transcript = transcribe(audio_path, model=model)
    transcript = clean_transcript(raw_transcript)

    # Diarization
    if verbose:
        print("\nDiarizing (this takes a few minutes)...")

    diarization_time = datetime.now(timezone.utc).isoformat()
    diarization = diarize(audio_path)
    diarization_model = "pyannote/speaker-diarization-community-1"

    # Enrichment (normalization + diarization)
    transcript, enrichment_processing, _ = enrich_transcript(
        transcript, diarization, library_path=library_path, verbose=verbose
    )

    processing = [
        {"stage": "transcription", "model": model, "status": "success"}
    ] + enrichment_processing
    transcript["_processing"] = processing
    transcript["_schema_version"] = _ENRICHED_SCHEMA_VERSION

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
    }


if __name__ == "__main__":
    import argparse
    from query import to_utterances, format_transcript

    parser = argparse.ArgumentParser(description="Run transcription pipeline")
    parser.add_argument("audio_file", help="Path to audio file")
    args = parser.parse_args()

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

    # Flatten enriched words and read speaker from _speaker.label
    labeled_words = []
    for seg in result["transcript"]["segments"]:
        for word in seg.get("words", []):
            labeled_words.append({
                **word,
                "speaker": word.get("_speaker", {}).get("label"),
            })
    utterances = to_utterances(labeled_words)

    print("\n--- Speaker-labeled transcript ---\n")
    print(format_transcript(utterances))

    print(f"\nSaved session to: {session_dir}/")
    print(f"  manifest.json")
    print(f"  audio-info.json")
    print(f"  transcript.json")
    print(f"  diarization.json")

    processing = result["transcript"]["_processing"]
    for entry in processing:
        if entry["stage"] == "llm_normalization":
            print(f"\nLLM Normalization: {entry.get('corrections_applied', 0)} corrections")
        elif entry["stage"] == "dictionary_normalization":
            print(f"Dictionary Normalization: {entry.get('corrections_applied', 0)} corrections")
