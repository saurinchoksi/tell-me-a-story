"""Run the full transcription and alignment pipeline."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from transcribe import transcribe, mark_hallucinated_segments
from diarize import diarize
from align import align, format_transcript
from inspect_audio import get_audio_info


def _validate_prob_threshold(value: str) -> float:
    """Validate probability threshold is in [0.0, 1.0]."""
    f = float(value)
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError(f"Must be between 0.0 and 1.0, got {f}")
    return f


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run transcription pipeline")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug files to sessions/debug/{recording}/")
    parser.add_argument("--prob-threshold", type=_validate_prob_threshold, default=0.5,
                        help="Probability threshold for word filtering (0.0-1.0)")
    parser.add_argument("--show-filtered", action="store_true",
                        help="Print filtered words to console (requires --debug)")
    return parser.parse_args()


def calculate_diarization_gaps(segments: list[dict]) -> list[dict]:
    """Calculate gaps between diarization segments.

    Args:
        segments: List of diarization segments with 'start' and 'end' keys

    Returns:
        List of gap dicts with 'start', 'end', 'duration' keys.
    """
    gaps = []
    for i in range(1, len(segments)):
        gap_start = segments[i - 1]["end"]
        gap_end = segments[i]["start"]
        if gap_end > gap_start:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "duration": round(gap_end - gap_start, 3)
            })
    return gaps


def save_debug_files(
    debug_dir: str,
    audio_info: dict,
    prob_threshold: float,
    diarization: list[dict],
    transcript: dict,
    words_extracted: list[dict],
    words_filtered: list[dict],
    words_labeled: list[dict],
    utterances: list[dict]
) -> None:
    """Save debug files to the specified directory.

    Saves 7 numbered files for pipeline debugging:
    - 01-audio-info.json: Audio properties + config
    - 02-diarization.json: Raw segments + calculated gaps
    - 03-transcript.json: Raw Whisper output
    - 04-words-extracted.json: Words before filtering
    - 05-words-filtered.json: Words removed with reasons
    - 06-words-labeled.json: Words after speaker assignment
    - 07-utterances.json: Final output

    Args:
        debug_dir: Directory to save files to
        audio_info: Output from get_audio_info()
        prob_threshold: Probability threshold used for filtering
        diarization: Raw diarization segments
        transcript: Raw Whisper transcript
        words_extracted: Words before any filtering
        words_filtered: Words removed by filters (with filter_reason)
        words_labeled: Words after speaker assignment
        utterances: Final consolidated utterances
    """
    os.makedirs(debug_dir, exist_ok=True)

    # 01: Audio info + config
    audio_config = {
        "audio": audio_info,
        "config": {
            "prob_threshold": prob_threshold
        }
    }
    with open(os.path.join(debug_dir, "01-audio-info.json"), "w") as f:
        json.dump(audio_config, f, indent=2)

    # 02: Diarization + gaps
    gaps = calculate_diarization_gaps(diarization)
    diarization_data = {
        "segments": diarization,
        "gaps": gaps
    }
    with open(os.path.join(debug_dir, "02-diarization.json"), "w") as f:
        json.dump(diarization_data, f, indent=2)

    # 03: Raw transcript
    with open(os.path.join(debug_dir, "03-transcript.json"), "w") as f:
        json.dump(transcript, f, indent=2)

    # 04: Words before filtering
    with open(os.path.join(debug_dir, "04-words-extracted.json"), "w") as f:
        json.dump(words_extracted, f, indent=2)

    # 05: Filtered words with reasons (grouped by reason)
    by_reason = {}
    for w in words_filtered:
        reason_key = w.get("filter_reason", "unknown")
        # Normalize low_probability reasons to single key
        if reason_key.startswith("low_probability"):
            reason_key = "low_probability"
        by_reason[reason_key] = by_reason.get(reason_key, 0) + 1

    filtered_data = {
        "filtered_count": len(words_filtered),
        "by_reason": by_reason,
        "threshold": prob_threshold,
        "words": words_filtered
    }
    with open(os.path.join(debug_dir, "05-words-filtered.json"), "w") as f:
        json.dump(filtered_data, f, indent=2)

    # 06: Words after speaker assignment
    with open(os.path.join(debug_dir, "06-words-labeled.json"), "w") as f:
        json.dump(words_labeled, f, indent=2)

    # 07: Final utterances
    with open(os.path.join(debug_dir, "07-utterances.json"), "w") as f:
        json.dump(utterances, f, indent=2)


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


def run_pipeline(
    audio_path: str,
    verbose: bool = True,
    prob_threshold: float = 0.5,
    debug: bool = False
) -> dict:
    """Run full pipeline on an audio file.

    Args:
        audio_path: Path to audio file
        verbose: Print progress messages
        prob_threshold: Probability threshold for word filtering (default 0.5)
        debug: If True, include debug data in result

    Returns:
        Dict with 'transcript', 'diarization', 'words', 'utterances' keys.
        If debug=True, also includes 'debug' key with intermediate data.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get audio info if in debug mode
    audio_info = None
    if debug:
        audio_info = get_audio_info(audio_path)

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
    if debug:
        align_result = align(words, segments, prob_threshold=prob_threshold, return_debug=True)
        utterances = align_result["utterances"]

        # Combine filtered words from both filters
        words_filtered = align_result["words_removed_zero"] + align_result["words_removed_prob"]

        return {
            "transcript": transcript,
            "diarization": segments,
            "words": words,
            "utterances": utterances,
            "debug": {
                "audio_info": audio_info,
                "prob_threshold": prob_threshold,
                "words_extracted": words,
                "words_filtered": words_filtered,
                "words_labeled": align_result["words_labeled"],
            }
        }
    else:
        utterances = align(words, segments, prob_threshold=prob_threshold)
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
    args = parse_args()

    result = run_pipeline(
        args.audio_file,
        prob_threshold=args.prob_threshold,
        debug=args.debug
    )

    print("\n--- Speaker-labeled transcript ---\n")
    print(format_transcript(result["utterances"]))

    # Show filtered words summary if requested
    if args.show_filtered and not args.debug:
        print("\nNote: --show-filtered requires --debug to be set")
    elif args.show_filtered and "debug" in result:
        words_filtered = result["debug"]["words_filtered"]
        print(f"\n--- Filtered words ({len(words_filtered)} total) ---")

        # Group by reason
        by_reason = {}
        for w in words_filtered:
            reason = w.get("filter_reason", "unknown")
            if reason.startswith("low_probability"):
                reason = "low_probability"
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(w)

        for reason, words in by_reason.items():
            print(f"\n{reason} ({len(words)} words):")
            for w in words[:10]:  # Show first 10 per reason
                prob_str = f" (p={w.get('probability', 'N/A'):.2f})" if "probability" in w else ""
                print(f"  [{w['start']:.2f}-{w['end']:.2f}] {w['word']!r}{prob_str}")
            if len(words) > 10:
                print(f"  ... and {len(words) - 10} more")

    # Save debug files if requested
    if args.debug and "debug" in result:
        recording_name = Path(args.audio_file).stem
        debug_dir = f"sessions/debug/{recording_name}"
        save_debug_files(
            debug_dir=debug_dir,
            audio_info=result["debug"]["audio_info"],
            prob_threshold=args.prob_threshold,
            diarization=result["diarization"],
            transcript=result["transcript"],
            words_extracted=result["debug"]["words_extracted"],
            words_filtered=result["debug"]["words_filtered"],
            words_labeled=result["debug"]["words_labeled"],
            utterances=result["utterances"]
        )
        print(f"\nSaved debug files to: {debug_dir}/")

    # Save session to JSON
    output_path = save_session(result, args.audio_file)
    print(f"\nSaved session to: {output_path}")
