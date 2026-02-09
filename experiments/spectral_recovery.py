"""Experiment 2: Spectral subtraction + bandpass for quiet speech recovery.

Profiles room noise from a nearby silent segment, subtracts that spectral
signature from the target region, then applies a bandpass filter tuned to
child vocal frequencies. Feeds cleaned audio to Whisper.

Unlike gain boost (experiment 1), this changes the actual SNR by removing
noise rather than amplifying everything equally.

Dependencies:
    pip install noisereduce scipy

Usage:
    python experiments/spectral_recovery.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import noisereduce as nr
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

import mlx_whisper


# =============================================================================
# Configuration
# =============================================================================

MODEL = "mlx-community/whisper-large-v3-mlx"
AUDIO_PATH = "sessions/00000000-000002/audio.m4a"

# Context window for Whisper (needs enough surrounding speech)
WINDOW_START = 220.0
WINDOW_END = 260.0

# Target region where Arti's speech is missing
TARGET_START = 241.0
TARGET_END = 243.5

# Noise profile region — silent gap right before the target (236.97–238.68s)
# Close proximity means same room conditions, mic position, ambient noise
NOISE_START = 236.97
NOISE_END = 238.68

# Also try a longer noise sample for better profiling
NOISE_LONG_START = 177.47
NOISE_LONG_END = 181.84

# Child speech frequency range
# Fundamental frequency: ~250-400Hz (vs adult male ~100-150Hz)
# Including harmonics and consonants: up to ~4kHz
CHILD_BANDPASS_LOW = 200   # Hz
CHILD_BANDPASS_HIGH = 4000  # Hz

# Vocab prompt (safe — no narrative overlap)
VOCAB_PROMPT = (
    "Mahabharata, Yudhishthira, Pandu, Pandavas, Kauravas, "
    "Duryodhana, Dhritarashtra, Bhima, Arjuna"
)

PROMPTS = [
    ("no_prompt", None),
    ("vocab_prompt", VOCAB_PROMPT),
]


# =============================================================================
# Audio utilities
# =============================================================================

def extract_wav(audio_path: str, output_path: str,
                start: float, end: float) -> None:
    """Extract a region from audio as 16kHz mono WAV."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def read_wav(path: str) -> tuple[int, np.ndarray]:
    """Read WAV file, return (sample_rate, samples as float32)."""
    sr, data = wavfile.read(path)
    # Normalize to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    return sr, data


def write_wav(path: str, sr: int, data: np.ndarray) -> None:
    """Write float32 audio to WAV."""
    # Clip to prevent overflow
    data = np.clip(data, -1.0, 1.0)
    # Convert to int16 for WAV
    wavfile.write(path, sr, (data * 32767).astype(np.int16))


def bandpass_filter(data: np.ndarray, sr: int,
                     low_hz: float, high_hz: float, order: int = 5) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyquist = sr / 2
    low = low_hz / nyquist
    high = high_hz / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, data)


def measure_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Estimate SNR in dB. Returns float('inf') if noise is zero."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


# =============================================================================
# Processing variants
# =============================================================================

def build_variants(window_audio: np.ndarray, noise_short: np.ndarray,
                    noise_long: np.ndarray, sr: int) -> list[tuple[str, np.ndarray]]:
    """Build all audio variants to test.
    
    Returns list of (name, processed_audio) tuples.
    """
    variants = []
    
    # 0. Baseline (for comparison)
    variants.append(("baseline", window_audio.copy()))
    
    # 1. Spectral subtraction with nearby noise profile
    nr_short = nr.reduce_noise(
        y=window_audio, sr=sr, y_noise=noise_short,
        prop_decrease=1.0, stationary=True
    )
    variants.append(("spectral_short_noise", nr_short))
    
    # 2. Spectral subtraction with longer noise profile
    nr_long = nr.reduce_noise(
        y=window_audio, sr=sr, y_noise=noise_long,
        prop_decrease=1.0, stationary=True
    )
    variants.append(("spectral_long_noise", nr_long))
    
    # 3. Bandpass only (isolate child frequencies)
    bp = bandpass_filter(window_audio, sr, CHILD_BANDPASS_LOW, CHILD_BANDPASS_HIGH)
    variants.append(("bandpass_only", bp))
    
    # 4. Spectral subtraction (short) + bandpass
    nr_short_bp = bandpass_filter(nr_short, sr, CHILD_BANDPASS_LOW, CHILD_BANDPASS_HIGH)
    variants.append(("spectral_short_then_bandpass", nr_short_bp))
    
    # 5. Spectral subtraction (long) + bandpass
    nr_long_bp = bandpass_filter(nr_long, sr, CHILD_BANDPASS_LOW, CHILD_BANDPASS_HIGH)
    variants.append(("spectral_long_then_bandpass", nr_long_bp))
    
    # 6. Spectral subtraction (short) + bandpass + gain boost (15dB)
    gain_15 = 10 ** (15 / 20)  # ~5.6x
    nr_short_bp_gain = nr_short_bp * gain_15
    variants.append(("spectral_short_bp_gain15", nr_short_bp_gain))
    
    # 7. Spectral subtraction (short) + bandpass + gain boost (20dB)
    gain_20 = 10 ** (20 / 20)  # 10x
    nr_short_bp_gain20 = nr_short_bp * gain_20
    variants.append(("spectral_short_bp_gain20", nr_short_bp_gain20))
    
    # 8. Aggressive spectral subtraction (prop_decrease=1.0 is already max,
    #    but try with non-stationary mode which adapts to changing noise)
    nr_nonstat = nr.reduce_noise(
        y=window_audio, sr=sr, y_noise=noise_short,
        prop_decrease=1.0, stationary=False
    )
    variants.append(("spectral_nonstationary", nr_nonstat))
    
    # 9. Non-stationary + bandpass + gain
    nr_nonstat_bp = bandpass_filter(nr_nonstat, sr, CHILD_BANDPASS_LOW, CHILD_BANDPASS_HIGH)
    nr_nonstat_bp_gain = nr_nonstat_bp * gain_15
    variants.append(("spectral_nonstat_bp_gain15", nr_nonstat_bp_gain))
    
    return variants


# =============================================================================
# Transcription (reused from experiment 1)
# =============================================================================

def transcribe_variant(audio_path: str, initial_prompt: str | None = None) -> dict:
    """Run MLX Whisper on an audio clip."""
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
    """Pull words from the target time region with adjusted timestamps."""
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
    output_dir = Path("experiments/results/spectral_recovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(AUDIO_PATH):
        print(f"Audio not found: {AUDIO_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print("Experiment 2: Spectral Subtraction + Bandpass")
    print("=" * 60)
    
    # --- Step 1: Extract audio segments ---
    print("\n--- Extracting audio ---")
    
    window_path = str(clips_dir / "window_raw.wav")
    noise_short_path = str(clips_dir / "noise_short.wav")
    noise_long_path = str(clips_dir / "noise_long.wav")
    
    print(f"  Window: {WINDOW_START}–{WINDOW_END}s")
    extract_wav(AUDIO_PATH, window_path, WINDOW_START, WINDOW_END)
    
    print(f"  Noise (short): {NOISE_START}–{NOISE_END}s")
    extract_wav(AUDIO_PATH, noise_short_path, NOISE_START, NOISE_END)
    
    print(f"  Noise (long): {NOISE_LONG_START}–{NOISE_LONG_END}s")
    extract_wav(AUDIO_PATH, noise_long_path, NOISE_LONG_START, NOISE_LONG_END)
    
    # --- Step 2: Load audio ---
    print("\n--- Loading audio ---")
    sr, window_audio = read_wav(window_path)
    _, noise_short = read_wav(noise_short_path)
    _, noise_long = read_wav(noise_long_path)
    print(f"  Sample rate: {sr}Hz")
    print(f"  Window: {len(window_audio)} samples ({len(window_audio)/sr:.1f}s)")
    print(f"  Noise short: {len(noise_short)} samples ({len(noise_short)/sr:.1f}s)")
    print(f"  Noise long: {len(noise_long)} samples ({len(noise_long)/sr:.1f}s)")
    
    # Measure baseline SNR in target region
    target_samples_start = int((TARGET_START - WINDOW_START) * sr)
    target_samples_end = int((TARGET_END - WINDOW_START) * sr)
    target_region = window_audio[target_samples_start:target_samples_end]
    baseline_snr = measure_snr(target_region, noise_short)
    print(f"  Baseline SNR (target region vs noise): {baseline_snr:.1f}dB")
    
    # --- Step 3: Build variants ---
    print("\n--- Building audio variants ---")
    variants = build_variants(window_audio, noise_short, noise_long, sr)
    print(f"  {len(variants)} variants created")
    
    # Save all variant clips and measure SNR
    for name, audio in variants:
        clip_path = str(clips_dir / f"{name}.wav")
        write_wav(clip_path, sr, audio)
        
        # Measure SNR improvement in target region
        variant_target = audio[target_samples_start:target_samples_end]
        variant_snr = measure_snr(variant_target, noise_short)
        snr_delta = variant_snr - baseline_snr
        print(f"  {name}: SNR={variant_snr:.1f}dB (Δ{snr_delta:+.1f}dB)")
    
    # --- Step 4: Transcribe all variants × prompts ---
    print(f"\n--- Transcribing ({len(variants)} variants × {len(PROMPTS)} prompts = {len(variants) * len(PROMPTS)} runs) ---\n")
    
    results = []
    
    for variant_name, _ in variants:
        clip_path = str(clips_dir / f"{variant_name}.wav")
        
        for prompt_name, prompt_text in PROMPTS:
            run_name = f"{variant_name}__{prompt_name}"
            
            print(f"  {run_name}...", end=" ", flush=True)
            t0 = time.time()
            result = transcribe_variant(clip_path, initial_prompt=prompt_text)
            elapsed = time.time() - t0
            
            target_words = extract_target_words(
                result, TARGET_START, TARGET_END,
                window_offset=WINDOW_START
            )
            
            # Quick inline report
            if target_words:
                text = " ".join(w["word"].strip() for w in target_words)
                avg_p = sum(w["probability"] for w in target_words) / len(target_words)
                print(f"→ \"{text}\" (p={avg_p:.3f}) [{elapsed:.1f}s]")
            else:
                print(f"→ [nothing] [{elapsed:.1f}s]")
            
            results.append({
                "variant": variant_name,
                "prompt": prompt_name,
                "run_name": run_name,
                "elapsed_seconds": round(elapsed, 1),
                "target_words": target_words,
                "full_text": result.get("text", ""),
            })
    
    # --- Step 5: Save results ---
    report = {
        "experiment": "spectral_recovery",
        "timestamp": datetime.now().isoformat(),
        "source_audio": AUDIO_PATH,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "target": {"start": TARGET_START, "end": TARGET_END},
        "noise_profiles": {
            "short": {"start": NOISE_START, "end": NOISE_END},
            "long": {"start": NOISE_LONG_START, "end": NOISE_LONG_END},
        },
        "bandpass": {"low_hz": CHILD_BANDPASS_LOW, "high_hz": CHILD_BANDPASS_HIGH},
        "expected_text": "Who's father had to go away?",
        "baseline_snr_db": round(float(baseline_snr), 1),
        "model": MODEL,
        "prompts": {name: text for name, text in PROMPTS},
        "results": results,
    }
    
    report_path = output_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    any_recovered = False
    for r in results:
        words = r["target_words"]
        if words:
            text = " ".join(w["word"].strip() for w in words)
            avg_p = sum(w["probability"] for w in words) / len(words)
            print(f"  {r['run_name']:45s} → \"{text}\" (p={avg_p:.3f})")
            any_recovered = True
        else:
            print(f"  {r['run_name']:45s} → [nothing]")
    
    if not any_recovered:
        print("\n  No new speech recovered in any variant.")
    
    print(f"\nFull results: {report_path}")
    print(f"Audio clips: {clips_dir}/")
    print(f"\nListen to cleaned clips to hear the difference:")
    print(f"  Raw target:      {clips_dir}/baseline.wav (jump to ~21s)")
    print(f"  Best candidates: spectral_short_bp_gain15.wav, spectral_short_bp_gain20.wav")


if __name__ == "__main__":
    main()
