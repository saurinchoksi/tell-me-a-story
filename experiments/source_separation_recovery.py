"""Experiment 3: Source separation for quiet speech recovery.

Uses ML models trained to separate overlapping/mixed speakers, rather than
just filtering noise. Fundamentally different approach from experiments 1-2.

Models:
- SepFormer (SpeechBrain) — trained for 2-speaker separation
  - wsj02mix: clean speech mixtures
  - whamr: noisy + reverberant conditions
  - whamr-enhancement: speech enhancement mode
- Demucs (Meta) — music source separation, vocals extraction

Each model produces separated audio tracks. We save them for listening
and feed them to Whisper to check if Arti's speech becomes decodable.

Dependencies:
    pip install speechbrain torchaudio demucs

Usage:
    python experiments/source_separation_recovery.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
import numpy as np
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


# =============================================================================
# Audio extraction
# =============================================================================

def extract_wav(audio_path: str, output_path: str,
                start: float, end: float, sample_rate: int = 16000) -> None:
    """Extract a window as mono WAV at specified sample rate."""
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(end - start),
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


# =============================================================================
# SepFormer separation
# =============================================================================

def run_sepformer(input_wav: str, output_dir: Path, model_source: str,
                   model_name: str) -> list[str]:
    """Run SepFormer source separation and save output tracks.
    
    SepFormer expects 8kHz mono input. We resample, separate, then
    upsample outputs back to 16kHz for Whisper.
    
    Returns:
        List of output file paths (16kHz WAV).
    """
    from speechbrain.inference.separation import SepformerSeparation as separator
    
    savedir = output_dir / f"pretrained_{model_name}"
    
    print(f"    Loading {model_name}...")
    model = separator.from_hparams(
        source=model_source,
        savedir=str(savedir)
    )
    
    # Load and resample to 8kHz (SepFormer requirement)
    waveform, sr = torchaudio.load(input_wav)
    if sr != 8000:
        resampler = torchaudio.transforms.Resample(sr, 8000)
        waveform = resampler(waveform)
    
    print(f"    Separating...")
    est_sources = model.separate_batch(waveform)
    # est_sources shape: [batch, samples, num_sources]
    
    output_paths = []
    num_sources = est_sources.shape[-1]
    
    # Upsample back to 16kHz for Whisper
    upsampler = torchaudio.transforms.Resample(8000, 16000)
    
    for i in range(num_sources):
        source = est_sources[:, :, i].detach().cpu()
        source_16k = upsampler(source)
        
        out_path = str(output_dir / f"{model_name}_source{i}.wav")
        torchaudio.save(out_path, source_16k, 16000)
        output_paths.append(out_path)
        print(f"    Saved source {i}: {out_path}")
    
    return output_paths


def run_sepformer_enhancement(input_wav: str, output_dir: Path) -> list[str]:
    """Run SepFormer in speech enhancement mode.
    
    Enhancement produces a single cleaned output rather than
    multiple separated sources.
    
    Returns:
        List with single output file path.
    """
    from speechbrain.inference.separation import SepformerSeparation as separator
    
    model_name = "sepformer-whamr-enhancement"
    savedir = output_dir / f"pretrained_{model_name}"
    
    print(f"    Loading {model_name}...")
    model = separator.from_hparams(
        source=f"speechbrain/{model_name}",
        savedir=str(savedir)
    )
    
    # Load and resample to 8kHz
    waveform, sr = torchaudio.load(input_wav)
    if sr != 8000:
        resampler = torchaudio.transforms.Resample(sr, 8000)
        waveform = resampler(waveform)
    
    print(f"    Enhancing...")
    enhanced = model.separate_batch(waveform)
    
    # Upsample to 16kHz
    upsampler = torchaudio.transforms.Resample(8000, 16000)
    
    # Enhancement may return [batch, samples] or [batch, samples, 1]
    if enhanced.dim() == 3:
        enhanced = enhanced[:, :, 0]
    enhanced_16k = upsampler(enhanced.detach().cpu())
    
    out_path = str(output_dir / f"{model_name}_enhanced.wav")
    torchaudio.save(out_path, enhanced_16k, 16000)
    print(f"    Saved: {out_path}")
    
    return [out_path]


# =============================================================================
# Demucs separation
# =============================================================================

def run_demucs(input_wav: str, output_dir: Path) -> list[str]:
    """Run Demucs vocal separation via CLI.
    
    Uses --two-stems=vocals to get just vocals + other.
    
    Returns:
        List of output file paths.
    """
    print(f"    Running Demucs (--two-stems=vocals)...")
    
    # Demucs outputs to separated/htdemucs/<filename>/
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-o", str(output_dir / "demucs_raw"),
        input_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Demucs error: {result.stderr}")
        return []
    
    # Find output files
    stem_name = Path(input_wav).stem
    demucs_dir = output_dir / "demucs_raw" / "htdemucs" / stem_name
    
    output_paths = []
    for stem in ["vocals", "no_vocals"]:
        src = demucs_dir / f"{stem}.wav"
        if src.exists():
            # Resample to 16kHz mono for Whisper
            dst = str(output_dir / f"demucs_{stem}.wav")
            subprocess.run([
                "ffmpeg", "-y", "-i", str(src),
                "-ar", "16000", "-ac", "1", dst
            ], capture_output=True, text=True)
            output_paths.append(dst)
            print(f"    Saved: {dst}")
    
    return output_paths


# =============================================================================
# Whisper transcription
# =============================================================================

def transcribe_and_extract(audio_path: str, prompt: str | None,
                            target_start: float, target_end: float,
                            window_offset: float) -> tuple[list[dict], str]:
    """Transcribe and extract words from target region."""
    kwargs = {
        "path_or_hf_repo": MODEL,
        "word_timestamps": True,
        "verbose": False,
    }
    if prompt:
        kwargs["initial_prompt"] = prompt
    
    result = mlx_whisper.transcribe(audio_path, **kwargs)
    
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
    
    return words, result.get("text", "")


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path("experiments/results/source_separation")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Experiment 3: Source Separation")
    print("=" * 60)
    
    # --- Step 1: Extract source audio ---
    window_16k = str(clips_dir / "window_16k.wav")
    print(f"\nExtracting window ({WINDOW_START}–{WINDOW_END}s)...")
    extract_wav(AUDIO_PATH, window_16k, WINDOW_START, WINDOW_END, sample_rate=16000)
    
    # --- Step 2: Run separation models ---
    separation_outputs = {}  # model_name -> list of output paths
    
    # SepFormer models
    sepformer_models = [
        ("speechbrain/sepformer-wsj02mix", "sepformer_wsj02mix"),
        ("speechbrain/sepformer-whamr", "sepformer_whamr"),
    ]
    
    for source, name in sepformer_models:
        print(f"\n--- {name} ---")
        try:
            paths = run_sepformer(window_16k, clips_dir, source, name)
            separation_outputs[name] = paths
        except Exception as e:
            print(f"    FAILED: {e}")
            separation_outputs[name] = []
    
    # SepFormer enhancement
    print(f"\n--- sepformer_whamr_enhancement ---")
    try:
        paths = run_sepformer_enhancement(window_16k, clips_dir)
        separation_outputs["sepformer_whamr_enhancement"] = paths
    except Exception as e:
        print(f"    FAILED: {e}")
        separation_outputs["sepformer_whamr_enhancement"] = []
    
    # Demucs
    print(f"\n--- demucs ---")
    try:
        paths = run_demucs(window_16k, clips_dir)
        separation_outputs["demucs"] = paths
    except Exception as e:
        print(f"    FAILED: {e}")
        separation_outputs["demucs"] = []
    
    # --- Step 3: Transcribe all outputs ---
    print(f"\n--- Transcribing separated sources ---\n")
    
    results = []
    
    # Also transcribe baseline for comparison
    all_clips = [("baseline", window_16k)]
    for model_name, paths in separation_outputs.items():
        for path in paths:
            label = Path(path).stem
            all_clips.append((label, path))
    
    for label, clip_path in all_clips:
        for prompt_name, prompt_text in [("no_prompt", None), ("vocab_prompt", VOCAB_PROMPT)]:
            run_name = f"{label}__{prompt_name}"
            
            print(f"  {run_name}...", end=" ", flush=True)
            t0 = time.time()
            
            try:
                target_words, full_text = transcribe_and_extract(
                    clip_path, prompt_text,
                    TARGET_START, TARGET_END,
                    window_offset=WINDOW_START
                )
                elapsed = time.time() - t0
                
                if target_words:
                    text = " ".join(w["word"].strip() for w in target_words)
                    avg_p = sum(w["probability"] for w in target_words) / len(target_words)
                    print(f"→ \"{text}\" (p={avg_p:.3f}) [{elapsed:.1f}s]")
                else:
                    print(f"→ [nothing] [{elapsed:.1f}s]")
                
                results.append({
                    "source": label,
                    "prompt": prompt_name,
                    "run_name": run_name,
                    "elapsed_seconds": round(elapsed, 1),
                    "target_words": target_words,
                    "full_text": full_text,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"→ ERROR: {e} [{elapsed:.1f}s]")
                results.append({
                    "source": label,
                    "prompt": prompt_name,
                    "run_name": run_name,
                    "elapsed_seconds": round(elapsed, 1),
                    "target_words": [],
                    "full_text": f"ERROR: {e}",
                })
    
    # --- Step 4: Save results ---
    report = {
        "experiment": "source_separation",
        "timestamp": datetime.now().isoformat(),
        "source_audio": AUDIO_PATH,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "target": {"start": TARGET_START, "end": TARGET_END},
        "expected_text": "Who's father had to go away?",
        "model": MODEL,
        "separation_models": list(separation_outputs.keys()),
        "results": results,
    }
    
    report_path = output_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # --- Summary ---
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
    print(f"Separated audio: {clips_dir}/")
    print(f"\nLISTEN to these — even if Whisper can't decode them,")
    print(f"you might hear Arti more clearly in the separated tracks:")
    for model_name, paths in separation_outputs.items():
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
