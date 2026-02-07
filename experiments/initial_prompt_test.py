"""Experiment: Test MLX Whisper initial_prompt for Sanskrit name recognition.

Tests whether providing vocabulary hints via initial_prompt improves
recognition of Mahabharata character names.

Note: beam_size is not yet supported in MLX Whisper.

Test matrix (3 runs):
| Run | initial_prompt     |
|-----|-------------------|
| 0   | None (baseline)   |
| 1   | Vocab list        |
| 2   | Natural sentence  |

Usage:
    python experiments/initial_prompt_test.py
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx_whisper


# =============================================================================
# Configuration
# =============================================================================

MODEL = "mlx-community/whisper-large-v3-mlx"
AUDIO_PATH = "sessions/00000000-000000/audio.m4a"

# Prompts to test
VOCAB_LIST_PROMPT = (
    "Pandavas, Kauravas, Yudhishthira, Duryodhana, Dhritarashtra, "
    "Pandu, Bhima, Arjuna, Draupadi, Karna, Krishna, Mahabharata"
)

NATURAL_SENTENCE_PROMPT = (
    "This is a story about the Mahabharata. The Pandavas and Kauravas are "
    "two families. Yudhishthira, Bhima, and Arjuna are Pandavas. "
    "Duryodhana is a Kaurava."
)

# Run configurations: (label, initial_prompt)
# Note: beam_size not supported in MLX Whisper yet
RUN_CONFIGS = [
    ("baseline", None),
    ("vocab_list", VOCAB_LIST_PROMPT),
    ("natural_sentence", NATURAL_SENTENCE_PROMPT),
]


# =============================================================================
# Name Detection Patterns
# =============================================================================

# Each entry: (canonical_name, [regex patterns including variants])
# Patterns are case-insensitive
NAME_PATTERNS = {
    "Pandavas": [r"\bpandava", r"\bpondava", r"\bfondo(?:s)?\b", r"\bpando(?:s)?\b"],
    "Kauravas": [r"\bkaurava", r"\bgoro(?:s)?\b", r"\bkoro(?:s)?\b"],
    "Yudhishthira": [r"\byudhishthira", r"\byudhisthir\b", r"\byudister"],
    "Duryodhana": [r"\bduryodhana", r"\bduryodhan\b"],
    "Dhritarashtra": [r"\bdhritarashtra", r"\bdhrashtra"],
    "Pandu": [r"\bpandu\b"],
    "Bhima": [r"\bbhima\b", r"\bbheem"],
    "Arjuna": [r"\barjuna?\b"],
    "Draupadi": [r"\bdraupadi"],
    "Karna": [r"\bkarna\b"],
    "Krishna": [r"\bkrishna"],
    "Mahabharata": [r"\bmahabharata"],
}


# =============================================================================
# Transcription
# =============================================================================


def run_transcription(
    audio_path: str,
    initial_prompt: str | None = None,
) -> tuple[dict, float]:
    """Run transcription with given parameters.

    Args:
        audio_path: Path to audio file
        initial_prompt: Optional vocabulary hint for the model

    Returns:
        Tuple of (result dict, elapsed seconds)
    """
    kwargs = {
        "path_or_hf_repo": MODEL,
        "word_timestamps": True,
        "language": "en",
        "condition_on_previous_text": True,
    }

    if initial_prompt is not None:
        kwargs["initial_prompt"] = initial_prompt

    start_time = time.time()
    result = mlx_whisper.transcribe(audio_path, **kwargs)
    elapsed = time.time() - start_time

    return result, elapsed


# =============================================================================
# Name Detection
# =============================================================================


def find_name_occurrences(transcript: dict) -> dict[str, list[dict]]:
    """Scan transcript for name matches.

    Returns:
        Dict mapping canonical names to list of match details:
        {
            "Pandavas": [
                {"matched_text": "Pandava", "segment_idx": 0, "word_idx": 5, "start": 1.2, "end": 1.5},
                ...
            ]
        }
    """
    occurrences: dict[str, list[dict]] = {name: [] for name in NAME_PATTERNS}

    full_text = transcript.get("text", "")

    # Search in full text for overview
    for canonical_name, patterns in NAME_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                occurrences[canonical_name].append({
                    "matched_text": match.group(),
                    "context": "full_text",
                    "position": match.start(),
                })

    # Also search in segments/words for detailed location
    for seg_idx, segment in enumerate(transcript.get("segments", [])):
        seg_text = segment.get("text", "")

        for canonical_name, patterns in NAME_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, seg_text, re.IGNORECASE):
                    occurrences[canonical_name].append({
                        "matched_text": match.group(),
                        "context": "segment",
                        "segment_idx": seg_idx,
                        "segment_start": segment.get("start"),
                        "segment_end": segment.get("end"),
                    })

        # Search word-level if available
        for word_idx, word in enumerate(segment.get("words", [])):
            word_text = word.get("word", "")

            for canonical_name, patterns in NAME_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, word_text, re.IGNORECASE):
                        occurrences[canonical_name].append({
                            "matched_text": word_text.strip(),
                            "context": "word",
                            "segment_idx": seg_idx,
                            "word_idx": word_idx,
                            "start": word.get("start"),
                            "end": word.get("end"),
                            "probability": word.get("probability"),
                        })

    return occurrences


def summarize_occurrences(occurrences: dict[str, list[dict]]) -> dict[str, Any]:
    """Create summary stats for name occurrences."""
    summary = {}
    for canonical_name, matches in occurrences.items():
        # Deduplicate by getting unique matched texts
        unique_matches = list(set(m["matched_text"] for m in matches))
        word_matches = [m for m in matches if m.get("context") == "word"]

        summary[canonical_name] = {
            "count": len(word_matches),  # Count word-level matches (most precise)
            "variants_found": unique_matches,
        }
    return summary


# =============================================================================
# Output Generation
# =============================================================================


def numpy_safe_serialize(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_safe_serialize(item) for item in obj]
    return obj


def save_run_result(
    results_dir: Path,
    run_number: int,
    label: str,
    initial_prompt: str | None,
    elapsed: float,
    transcript: dict,
    occurrences: dict,
) -> None:
    """Save individual run result to JSON file."""
    output = {
        "_experiment_metadata": {
            "run_number": run_number,
            "label": label,
            "initial_prompt": initial_prompt,
            "elapsed_seconds": round(elapsed, 2),
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
        },
        "_name_occurrences": summarize_occurrences(occurrences),
        "text": transcript.get("text", ""),
        "segments": transcript.get("segments", []),
    }

    output = numpy_safe_serialize(output)

    output_path = results_dir / f"run_{run_number}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {output_path}")


def generate_summary(results_dir: Path, all_results: list[dict]) -> None:
    """Generate markdown summary comparing all runs."""
    lines = [
        "# Initial Prompt Experiment Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Audio: `{AUDIO_PATH}`",
        f"Model: `{MODEL}`",
        "",
        "## Run Configuration",
        "",
        "| Run | Label | Initial Prompt | Time (s) |",
        "|-----|-------|----------------|----------|",
    ]

    for r in all_results:
        meta = r["metadata"]
        prompt_str = "None" if meta["initial_prompt"] is None else f'"{meta["initial_prompt"][:30]}..."'
        lines.append(
            f"| {meta['run_number']} | {meta['label']} | {prompt_str} | {meta['elapsed']:.1f} |"
        )

    lines.extend([
        "",
        "## Name Detection Results",
        "",
        "| Name | " + " | ".join(r["metadata"]["label"] for r in all_results) + " |",
        "|------|" + "|".join(["------"] * len(all_results)) + "|",
    ])

    # Build comparison table
    for canonical_name in NAME_PATTERNS.keys():
        row = [canonical_name]
        for r in all_results:
            summary = r["occurrences_summary"].get(canonical_name, {})
            count = summary.get("count", 0)
            variants = summary.get("variants_found", [])
            if count > 0:
                # Show count and first variant
                variant_sample = variants[0] if variants else ""
                row.append(f"{count} ({variant_sample})")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Observations",
        "",
        "_To be filled in after reviewing results._",
        "",
        "## Full Variant Details",
        "",
    ])

    # Detailed variant listing per run
    for r in all_results:
        lines.append(f"### Run {r['metadata']['run_number']}: {r['metadata']['label']}")
        lines.append("")
        for name, summary in r["occurrences_summary"].items():
            if summary.get("count", 0) > 0:
                lines.append(f"- **{name}**: {summary['variants_found']}")
        lines.append("")

    summary_path = results_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary saved: {summary_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Run the experiment."""
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    audio_path = project_root / AUDIO_PATH
    results_dir = Path(__file__).parent / "results"

    # Check audio file exists
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        print("Please ensure the test audio exists at the expected location.")
        return 1

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    print(f"Audio file: {audio_path}")
    print(f"Model: {MODEL}")
    print()

    all_results = []

    for run_number, (label, initial_prompt) in enumerate(RUN_CONFIGS):
        print(f"=== Run {run_number}: {label} ===")
        prompt_display = "None" if initial_prompt is None else f'"{initial_prompt[:50]}..."'
        print(f"  initial_prompt: {prompt_display}")
        print("  Transcribing...")

        transcript, elapsed = run_transcription(
            str(audio_path),
            initial_prompt=initial_prompt,
        )

        print(f"  Completed in {elapsed:.1f}s")

        # Find name occurrences
        occurrences = find_name_occurrences(transcript)
        occurrences_summary = summarize_occurrences(occurrences)

        # Save individual result
        save_run_result(
            results_dir,
            run_number,
            label,
            initial_prompt,
            elapsed,
            transcript,
            occurrences,
        )

        # Collect for summary
        all_results.append({
            "metadata": {
                "run_number": run_number,
                "label": label,
                "initial_prompt": initial_prompt,
                "elapsed": elapsed,
            },
            "occurrences_summary": occurrences_summary,
        })

        print()

    # Generate summary
    generate_summary(results_dir, all_results)

    print("Experiment complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
