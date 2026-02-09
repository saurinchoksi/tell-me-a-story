"""Quick check: verify the extracted clips contain the right audio region.

Runs ffprobe on the baseline clip to check duration, then re-extracts
with -ss AFTER -i (slower but frame-accurate) for comparison.

Usage:
    python experiments/verify_clip.py
"""

import subprocess
import sys


AUDIO_PATH = "sessions/00000000-000002/audio.m4a"
CLIP_DIR = "experiments/results/quiet_speech_recovery/clips"
WINDOW_START = 220.0
WINDOW_END = 260.0


def probe(path: str) -> None:
    """Print duration and format info for an audio file."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", 
         "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True
    )
    print(f"  {path}: duration = {result.stdout.strip()}s")


def extract_accurate(input_path: str, output_path: str,
                      start: float, end: float) -> None:
    """Extract with -ss AFTER -i for frame-accurate seeking."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr}")


print("=== Checking existing baseline clip ===")
probe(f"{CLIP_DIR}/baseline.wav")

print()
print("=== Re-extracting with -ss AFTER -i (frame-accurate) ===")
accurate_path = f"{CLIP_DIR}/baseline_accurate.wav"
extract_accurate(AUDIO_PATH, accurate_path, WINDOW_START, WINDOW_END)
probe(accurate_path)

print()
print("=== Extracting just the target gap region (241-244s) for listening ===")
gap_path = f"{CLIP_DIR}/gap_only.wav"
extract_accurate(AUDIO_PATH, gap_path, 240.0, 245.0)
probe(gap_path)

# Also boost the gap region
gap_boosted_path = f"{CLIP_DIR}/gap_only_boosted_20db.wav"
subprocess.run([
    "ffmpeg", "-y",
    "-i", AUDIO_PATH,
    "-ss", "240.0",
    "-t", "5.0",
    "-af", "volume=20dB",
    "-ar", "16000", "-ac", "1",
    gap_boosted_path
], capture_output=True, text=True)
probe(gap_boosted_path)

print()
print(f"Listen to these to verify:")
print(f"  Original gap:  {gap_path}")
print(f"  Boosted gap:   {gap_boosted_path}")
print(f"  Accurate clip: {accurate_path}")
