"""Experiment: Recover quiet child speech from known gap.

Diarization detected SPEAKER_01 (Arti) at 241.68-242.83s in session 000002,
but Whisper produced no transcript for this region. Arti was asking
"Who's father had to go away?" (faint but audible to human ear).

Approach:
- Extract a ~40s window around the gap (Whisper needs context)
- Create variants with different audio processing:
  - Gain boost only (10dB, 15dB, 20dB)
  - High-pass filter (200Hz) + gain boost (cuts room rumble)
- Run Whisper on each variant with contextual initial_prompt
- Compare what appears in the 241-243s target window

Usage:
    python experiments/quiet_speech_recovery.py

Output:
    experiments/results/quiet_speech_recovery/
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

# Window to extract (gives Whisper ~40s of context)
WINDOW_START = 220.0
WINDOW_END = 260.0

# Target region where Arti's speech is missing
TARGET_START = 241.0
TARGET_END = 243.5

# Variants to test
VARIANTS = [
    {"name": "baseline", "gain_db": 0, "highpass_hz": None},
    {"name": "gain_10db", "gain_db": 10, "highpass_hz": None},
    {"name": "gain_15db", "gain_db": 15, "highpass_hz": None},
    {"name": "gain_20db", "gain_db": 20, "highpass_hz": None},
    {"name": "hp200_gain_15db", "gain_db": 15, "highpass_hz": 200},
    {"name": "hp200_gain_20db", "gain_db": 20, "highpass_hz": 200},
    {"name": "hp300_gain_20db", "gain_db": 20, "highpass_hz": 300},
]

# Vocabulary priming only — DO NOT include narrative that matches the audio,
# or Whisper treats the prompt as "already transcribed" and skips matching audio.
VOCAB_PROMPT = (
    "Mahabharata, Yudhishthira, Pandu, Pandavas, Kauravas, "
    "Duryodhana, Dhritarashtra, Bhima, Arjuna"
)

# Also test with no prompt at all
PROMPTS = [
    ("no_prompt", None),
    ("vocab_prompt", VOCAB_PROMPT),
]


# =============================================================================
# Audio processing
# =============================================================================

def extract_and_process(audio_path: str, output_path: str, 
                         start: float, end: float,
                         gain_db: float = 0, highpass_hz: int | None = None) -> None:
    """Extract a window from audio and apply processing with ffmpeg.
    
    Args:
        audio_path: Source audio file
        output_path: Where to write processed clip
        start: Start time in seconds
        end: End time in seconds  
        gain_db: Gain boost in decibels (0 = no change)
        highpass_hz: High-pass filter cutoff frequency (None = no filter)
    """
    duration = end - start
    
    # Build ffmpeg filter chain
    filters = []
    if highpass_hz:
        filters.append(f"highpass=f={highpass_hz}")
    if gain_db != 0:
        filters.append(f"volume={gain_db}dB")
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", audio_path,
    ]
    
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    
    cmd.extend([
        "-ar", "16000",  # Whisper expects 16kHz
        "-ac", "1",      # mono
        output_path
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


# =============================================================================
# Transcription
# =============================================================================

def transcribe_variant(audio_path: str, initial_prompt: str | None = None) -> dict:
    """Run MLX Whisper on an audio clip.
    
    Returns:
        Raw Whisper result dict with segments and words.
    """
    kwargs = {
        "path_or_hf_repo": MODEL,
        "word_timestamps": True,
        "verbose": False,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    
    return mlx_whisper.transcribe(audio_path, **kwargs)


def extract_target_words(result: dict, target_start: float, target_end: float,
                          window_offset: float) -> list[dict]:
    """Pull words from the target time region.
    
    Whisper timestamps are relative to the clip, so we need to offset
    back to the original audio timeline for comparison.
    
    Args:
        result: Whisper transcription result
        target_start: Start of target region in original audio time
        target_end: End of target region in original audio time
        window_offset: Start of the extracted window in original audio time
    
    Returns:
        List of word dicts with adjusted timestamps.
    """
    # Target region in clip-relative time
    clip_target_start = target_start - window_offset
    clip_target_end = target_end - window_offset
    
    target_words = []
    for seg in result.get("segments", []):
        for word in seg.get("words", []):
            if word["end"] >= clip_target_start and word["start"] <= clip_target_end:
                target_words.append({
                    "word": word["word"],
                    "start_original": round(word["start"] + window_offset, 2),
                    "end_original": round(word["end"] + window_offset, 2),
                    "probability": round(word["probability"], 4),
                })
    
    return target_words


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path("experiments/results/quiet_speech_recovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(AUDIO_PATH):
        print(f"Audio not found: {AUDIO_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print("Quiet Speech Recovery Experiment")
    print("=" * 60)
    print(f"Source: {AUDIO_PATH}")
    print(f"Window: {WINDOW_START}s - {WINDOW_END}s ({WINDOW_END - WINDOW_START}s)")
    print(f"Target: {TARGET_START}s - {TARGET_END}s (Arti's missing speech)")
    print(f"Expected: \"Who's father had to go away?\"")
    print(f"Variants: {len(VARIANTS)} audio x {len(PROMPTS)} prompts = {len(VARIANTS) * len(PROMPTS)} runs")
    print()
    
    results = []
    
    for variant in VARIANTS:
        clip_name = variant["name"]
        clip_path = str(clips_dir / f"{clip_name}.wav")
        
        # Extract and process audio (once per variant)
        desc_parts = []
        if variant["highpass_hz"]:
            desc_parts.append(f"highpass {variant['highpass_hz']}Hz")
        if variant["gain_db"]:
            desc_parts.append(f"+{variant['gain_db']}dB")
        if not desc_parts:
            desc_parts.append("no processing")
        
        extract_and_process(
            AUDIO_PATH, clip_path,
            start=WINDOW_START, end=WINDOW_END,
            gain_db=variant["gain_db"],
            highpass_hz=variant["highpass_hz"],
        )
        
        for prompt_name, prompt_text in PROMPTS:
            run_name = f"{clip_name}__{prompt_name}"
            
            print(f"--- {run_name} ---")
            print(f"  Audio: {', '.join(desc_parts)}")
            print(f"  Prompt: {prompt_name}")
            
            # Transcribe
            print(f"  Transcribing...")
            t0 = time.time()
            result = transcribe_variant(clip_path, initial_prompt=prompt_text)
            elapsed = time.time() - t0
            print(f"  Done ({elapsed:.1f}s)")
            
            # Extract words in target region
            target_words = extract_target_words(
                result, TARGET_START, TARGET_END, 
                window_offset=WINDOW_START
            )
            
            # Also get full transcript of target region segments for context
            clip_target_start = TARGET_START - WINDOW_START
            clip_target_end = TARGET_END - WINDOW_START
            target_segments = []
            for seg in result.get("segments", []):
                if seg["end"] >= clip_target_start and seg["start"] <= clip_target_end:
                    target_segments.append({
                        "text": seg["text"],
                        "start_original": round(seg["start"] + WINDOW_START, 2),
                        "end_original": round(seg["end"] + WINDOW_START, 2),
                    })
            
            # Report
            if target_words:
                recovered_text = " ".join(w["word"].strip() for w in target_words)
                avg_prob = sum(w["probability"] for w in target_words) / len(target_words)
                print(f"  RECOVERED: \"{recovered_text}\"")
                print(f"  Avg probability: {avg_prob:.3f}")
                for w in target_words:
                    print(f"    {w['start_original']}s-{w['end_original']}s "
                          f"p={w['probability']:.3f} \"{w['word']}\"")
            else:
                print(f"  Nothing recovered in target region.")
            print()
            
            results.append({
                "variant": clip_name,
                "prompt": prompt_name,
                "run_name": run_name,
                "gain_db": variant["gain_db"],
                "highpass_hz": variant["highpass_hz"],
                "elapsed_seconds": round(elapsed, 1),
                "target_words": target_words,
                "target_segments": target_segments,
                "full_text": result.get("text", ""),
            })
    
    # Save results
    report = {
        "experiment": "quiet_speech_recovery",
        "timestamp": datetime.now().isoformat(),
        "source_audio": AUDIO_PATH,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "target": {"start": TARGET_START, "end": TARGET_END},
        "expected_text": "Who's father had to go away?",
        "prompts": {name: text for name, text in PROMPTS},
        "model": MODEL,
        "results": results,
    }
    
    report_path = output_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        words = r["target_words"]
        if words:
            text = " ".join(w["word"].strip() for w in words)
            avg_p = sum(w["probability"] for w in words) / len(words)
            print(f"  {r['run_name']:40s} → \"{text}\" (avg p={avg_p:.3f})")
        else:
            print(f"  {r['run_name']:40s} → [nothing]")
    
    print()
    print(f"Full results: {report_path}")
    print(f"Audio clips: {clips_dir}/")


if __name__ == "__main__":
    main()
