"""Experiment 5: Whisper decode parameter sweep for quiet speech recovery.

Tests parameters that change how Whisper decides whether speech exists:
- condition_on_previous_text=False (each segment independent)
- hallucination_silence_threshold (skips silence detection)
- no_speech_threshold (sensitivity to deciding "no speech here")
- temperature (decoding confidence)

Uses the best audio from experiment 2 (spectral + bandpass + gain)
plus baseline for comparison.

Usage:
    python experiments/whisper_params_recovery.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx_whisper


# =============================================================================
# Configuration
# =============================================================================

MODEL = "mlx-community/whisper-large-v3-mlx"
AUDIO_PATH = "sessions/00000000-000002/audio.m4a"

WINDOW_START = 220.0
WINDOW_END = 260.0
TARGET_START = 241.0
TARGET_END = 243.5

VOCAB_PROMPT = (
    "Mahabharata, Yudhishthira, Pandu, Pandavas, Kauravas, "
    "Duryodhana, Dhritarashtra, Bhima, Arjuna"
)

# Audio variants to test (subset — baseline + best from experiment 2)
# We'll reuse clips if they exist, otherwise extract fresh
AUDIO_VARIANTS = [
    ("baseline", []),  # no ffmpeg filters
    ("spectral_bp_gain20", None),  # from experiment 2, pre-built
]

# Parameter variants to test
PARAM_VARIANTS = [
    {
        "name": "default",
        "params": {},
    },
    {
        "name": "no_condition",
        "params": {"condition_on_previous_text": False},
    },
    {
        "name": "no_condition__vocab",
        "params": {"condition_on_previous_text": False, "initial_prompt": VOCAB_PROMPT},
    },
    {
        "name": "low_no_speech_thresh",
        "params": {"no_speech_threshold": 0.9},  # more willing to detect speech
    },
    {
        "name": "very_low_no_speech_thresh",
        "params": {"no_speech_threshold": 0.99},  # almost always thinks there's speech
    },
    {
        "name": "hallucination_silence_1",
        "params": {"hallucination_silence_threshold": 1.0},
    },
    {
        "name": "hallucination_silence_05",
        "params": {"hallucination_silence_threshold": 0.5},
    },
    {
        "name": "low_logprob",
        "params": {"logprob_threshold": -2.0},  # more permissive decoding
    },
    {
        "name": "kitchen_sink",
        "params": {
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.99,
            "logprob_threshold": -2.0,
            "initial_prompt": VOCAB_PROMPT,
        },
    },
    {
        "name": "kitchen_sink__temp_0",
        "params": {
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.99,
            "logprob_threshold": -2.0,
            "temperature": 0.0,
            "initial_prompt": VOCAB_PROMPT,
        },
    },
]


# =============================================================================
# Audio + Transcription
# =============================================================================

def extract_window(audio_path: str, output_path: str,
                    start: float, end: float) -> None:
    """Extract a window as 16kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(end - start),
        "-ar", "16000", "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def transcribe_with_params(audio_path: str, extra_params: dict) -> dict:
    """Run MLX Whisper with custom parameters."""
    kwargs = {
        "path_or_hf_repo": MODEL,
        "word_timestamps": True,
        "verbose": False,
    }
    kwargs.update(extra_params)
    return mlx_whisper.transcribe(audio_path, **kwargs)


def extract_target_words(result: dict, target_start: float, target_end: float,
                          window_offset: float) -> list[dict]:
    """Pull words from the target time region."""
    clip_start = target_start - window_offset
    clip_end = target_end - window_offset

    words = []
    for seg in result.get("segments", []):
        for word in seg.get("words", []):
            if word["end"] >= clip_start and word["start"] <= clip_end:
                words.append({
                    "word": word["word"],
                    "start_original": round(word["start"] + window_offset, 2),
                    "end_original": round(word["end"] + window_offset, 2),
                    "probability": round(word["probability"], 4),
                })
    return words


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path("experiments/results/whisper_params_recovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    # Prepare audio clips
    print("=" * 60)
    print("Experiment 5: Whisper Parameter Sweep")
    print("=" * 60)

    # Baseline clip
    baseline_clip = str(clips_dir / "baseline.wav")
    if not os.path.exists(baseline_clip):
        print("Extracting baseline clip...")
        extract_window(AUDIO_PATH, baseline_clip, WINDOW_START, WINDOW_END)

    # Spectral clip from experiment 2
    spectral_clip = "experiments/results/spectral_recovery/clips/spectral_short_bp_gain20.wav"
    if not os.path.exists(spectral_clip):
        print(f"WARNING: {spectral_clip} not found. Run experiment 2 first.")
        print("Continuing with baseline only.")
        audio_clips = [("baseline", baseline_clip)]
    else:
        audio_clips = [
            ("baseline", baseline_clip),
            ("spectral_bp_gain20", spectral_clip),
        ]

    total_runs = len(audio_clips) * len(PARAM_VARIANTS)
    print(f"\n{len(audio_clips)} audio clips × {len(PARAM_VARIANTS)} param sets = {total_runs} runs\n")

    results = []

    for audio_name, clip_path in audio_clips:
        for pv in PARAM_VARIANTS:
            run_name = f"{audio_name}__{pv['name']}"
            params_desc = ", ".join(f"{k}={v}" for k, v in pv["params"].items()) or "defaults"

            print(f"  {run_name}...", end=" ", flush=True)
            t0 = time.time()
            result = transcribe_with_params(clip_path, pv["params"])
            elapsed = time.time() - t0

            target_words = extract_target_words(
                result, TARGET_START, TARGET_END,
                window_offset=WINDOW_START
            )

            if target_words:
                text = " ".join(w["word"].strip() for w in target_words)
                avg_p = sum(w["probability"] for w in target_words) / len(target_words)
                print(f"→ \"{text}\" (p={avg_p:.3f}) [{elapsed:.1f}s]")
            else:
                print(f"→ [nothing] [{elapsed:.1f}s]")

            results.append({
                "audio": audio_name,
                "params": pv["name"],
                "run_name": run_name,
                "param_details": pv["params"],
                "elapsed_seconds": round(elapsed, 1),
                "target_words": target_words,
                "full_text": result.get("text", ""),
            })

    # Save
    report = {
        "experiment": "whisper_params_recovery",
        "timestamp": datetime.now().isoformat(),
        "source_audio": AUDIO_PATH,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "target": {"start": TARGET_START, "end": TARGET_END},
        "expected_text": "Who's father had to go away?",
        "model": MODEL,
        "results": results,
    }

    report_path = output_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        words = r["target_words"]
        if words:
            text = " ".join(w["word"].strip() for w in words)
            avg_p = sum(w["probability"] for w in words) / len(words)
            print(f"  {r['run_name']:50s} → \"{text}\" (p={avg_p:.3f})")
        else:
            print(f"  {r['run_name']:50s} → [nothing]")

    print(f"\nFull results: {report_path}")


if __name__ == "__main__":
    main()
