"""Run the full transcription and alignment pipeline."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from transcribe import transcribe, mark_hallucinated_segments
from diarize import diarize
from align import align, format_transcript, group_words_by_speaker


def extract_words(transcript: dict) -> list[dict]:
    """Extract all words from transcript segments.
    
    Args:
        transcript: Result from transcribe() with word_timestamps=True
    
    Returns:
        Flat list of word dicts with 'start', 'end', 'word' keys.
    """
    all_words = []
    for seg in transcript["segments"]:
        if seg.get("words"):
            all_words.extend(seg["words"])
    return all_words


def run_pipeline(audio_path: str, verbose: bool = True) -> dict:
    """Run full pipeline on an audio file.

    Args:
        audio_path: Path to audio file
        verbose: Print progress messages

    Returns:
        Dict with 'transcript', 'diarization', 'words', 'utterances' keys.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if verbose:
        print(f"Transcribing: {audio_path}")
        print("Using large model for better child speech recognition...")
    
    transcript = transcribe(
        audio_path,
        word_timestamps=True,
        model="mlx-community/whisper-large-v3-turbo"
    )

    # Mark hallucinated segments before extracting words
    transcript["segments"] = mark_hallucinated_segments(transcript["segments"])

    if verbose:
        print("\nDiarizing (this takes a few minutes)...")
    
    segments = diarize(audio_path)
    
    # Extract words from transcript
    words = extract_words(transcript)
    
    if verbose:
        print(f"\nAligning {len(words)} words to {len(segments)} speaker segments...")
    
    # Run alignment pipeline
    utterances = align(words, segments)
    
    return {
        "transcript": transcript,
        "diarization": segments,
        "words": words,
        "utterances": utterances
    }


def save_session(result: dict, audio_path: str, output_dir: str = "sessions/processed") -> str:
    """Save pipeline result to JSON session file.

    Args:
        result: Output from run_pipeline()
        audio_path: Original audio file path
        output_dir: Directory for output files

    Returns:
        Path to saved JSON file.
    """
    # Extract unique speakers from utterances
    speakers_detected = list(set(
        u["speaker"] for u in result["utterances"] if u["speaker"] is not None
    ))
    speakers_detected.sort()

    # Generate story_id from audio filename
    story_id = Path(audio_path).stem

    session = {
        "_schema_version": "0.1.0",
        "meta": {
            "source_audio": audio_path,
            "transcribed_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "0.1.0"
        },
        "speakers": {
            "detected": speakers_detected,
            "names": None
        },
        "stories": [{
            "story_id": story_id,
            "utterances": result["utterances"]
        }],
        "moments": [],
        "processing": None
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{story_id}.json")
    with open(output_path, "w") as f:
        json.dump(session, f, indent=2)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    result = run_pipeline(audio_path)

    print("\n--- Speaker-labeled transcript ---\n")
    print(format_transcript(result["utterances"]))

    # Save session to JSON
    output_path = save_session(result, audio_path)
    print(f"\nSaved session to: {output_path}")
