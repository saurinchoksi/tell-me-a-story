"""Source separation v2: SpeechBrain models + MLX Whisper.

Focused re-run of Phase 1 from comprehensive_recovery.py with the
compatibility patches that were missing. The comprehensive script failed
on two issues:

1. torchaudio.list_audio_backends removed in torchaudio 2.9
2. SpeechBrain 1.0.3 passes deprecated use_auth_token to huggingface_hub 1.3.2

This script applies monkey-patches before any SpeechBrain imports, then runs
six models against the 220-260s window from session 000002. Each model's
output is transcribed with MLX Whisper (with and without vocab prompt), plus
tight clips around the target region (238-246s).

Target: Arti (SPEAKER_01) at 241.68-242.83s.
Expected: "Who's father had to go away?"

Usage:
    python experiments/source_separation_v2.py
"""

# =============================================================================
# CRITICAL: Monkey-patches MUST come before any SpeechBrain/speechbrain imports
# =============================================================================

# Fix 1: torchaudio.list_audio_backends removed in 2.9
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["sox_io"]

# Fix 2: SpeechBrain uses deprecated use_auth_token param
import huggingface_hub
_original_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _original_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_hub_download

# Also patch snapshot_download which SpeechBrain may use
if hasattr(huggingface_hub, 'snapshot_download'):
    _original_snapshot_download = huggingface_hub.snapshot_download
    def _patched_snapshot_download(*args, **kwargs):
        if 'use_auth_token' in kwargs:
            kwargs['token'] = kwargs.pop('use_auth_token')
        return _original_snapshot_download(*args, **kwargs)
    huggingface_hub.snapshot_download = _patched_snapshot_download

# =============================================================================
# Standard imports (safe to import after patches)
# =============================================================================

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import mlx_whisper


# =============================================================================
# Configuration
# =============================================================================

AUDIO_PATH = "sessions/00000000-000002/audio.m4a"
WINDOW_START = 220.0
WINDOW_END = 260.0
TARGET_START = 241.0
TARGET_END = 243.5

TIGHT_CLIP_START = 238.0
TIGHT_CLIP_END = 246.0

MODEL = "mlx-community/whisper-large-v3-mlx"
VOCAB_PROMPT = (
    "Mahabharata, Yudhishthira, Pandu, Pandavas, Kauravas, "
    "Duryodhana, Dhritarashtra, Bhima, Arjuna"
)

OUTPUT_DIR = Path("experiments/results/source_separation")
CLIPS_DIR = OUTPUT_DIR / "clips"


# =============================================================================
# Audio utilities
# =============================================================================

def extract_wav(audio_path: str, output_path: str,
                start: float, end: float, sample_rate: int = 16000) -> None:
    """Extract a window from audio as mono WAV at specified sample rate."""
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(end - start),
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def extract_tight_clip(source_wav: str, output_path: str,
                        window_offset: float) -> None:
    """Extract tight window (238-246s) from an already-clipped WAV.

    The source WAV starts at window_offset (220s). We extract the samples
    corresponding to TIGHT_CLIP_START-TIGHT_CLIP_END in original time.
    """
    waveform, sr = torchaudio.load(source_wav)
    start_sample = int((TIGHT_CLIP_START - window_offset) * sr)
    end_sample = int((TIGHT_CLIP_END - window_offset) * sr)
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[-1], end_sample)
    torchaudio.save(output_path, waveform[:, start_sample:end_sample], sr)


# =============================================================================
# SpeechBrain model runners
# =============================================================================

def run_sepformer_8k(input_wav: str, output_dir: Path,
                      model_source: str, model_name: str) -> list[tuple[str, str]]:
    """Run a SepFormer model that expects 8kHz input.

    Resamples 16kHz -> 8kHz, runs model, resamples outputs back to 16kHz.

    Returns:
        List of (label, output_path) tuples.
    """
    from speechbrain.inference.separation import SepformerSeparation as separator

    savedir = output_dir / f"pretrained_{model_name}"

    print(f"    Loading {model_name}...")
    model = separator.from_hparams(
        source=model_source,
        savedir=str(savedir),
    )

    waveform, sr = torchaudio.load(input_wav)
    if sr != 8000:
        resampler = torchaudio.transforms.Resample(sr, 8000)
        waveform = resampler(waveform)

    print(f"    Separating ({waveform.shape[-1] / 8000:.1f}s at 8kHz)...")
    est_sources = model.separate_batch(waveform)

    upsampler = torchaudio.transforms.Resample(8000, 16000)
    results = []

    # Enhancement models return [batch, samples] or [batch, samples, 1]
    # Separation models return [batch, samples, num_sources]
    if est_sources.dim() == 2:
        source_16k = upsampler(est_sources.detach().cpu())
        out_path = str(output_dir / f"{model_name}_enhanced.wav")
        torchaudio.save(out_path, source_16k, 16000)
        label = f"{model_name}_enhanced"
        results.append((label, out_path))
        print(f"    Saved: {label}")
    else:
        num_sources = est_sources.shape[-1]
        for i in range(num_sources):
            source = est_sources[:, :, i].detach().cpu()
            source_16k = upsampler(source)
            suffix = "enhanced" if num_sources == 1 else f"source{i}"
            out_path = str(output_dir / f"{model_name}_{suffix}.wav")
            torchaudio.save(out_path, source_16k, 16000)
            label = f"{model_name}_{suffix}"
            results.append((label, out_path))
            print(f"    Saved: {label}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def run_sepformer_16k(input_wav: str, output_dir: Path,
                       model_source: str, model_name: str) -> list[tuple[str, str]]:
    """Run a SepFormer model that works at 16kHz natively.

    Returns:
        List of (label, output_path) tuples.
    """
    from speechbrain.inference.separation import SepformerSeparation as separator

    savedir = output_dir / f"pretrained_{model_name}"

    print(f"    Loading {model_name}...")
    model = separator.from_hparams(
        source=model_source,
        savedir=str(savedir),
    )

    waveform, sr = torchaudio.load(input_wav)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    print(f"    Processing ({waveform.shape[-1] / 16000:.1f}s at 16kHz)...")
    est_sources = model.separate_batch(waveform)

    results = []

    if est_sources.dim() == 2:
        enhanced = est_sources.detach().cpu()
        out_path = str(output_dir / f"{model_name}_enhanced.wav")
        torchaudio.save(out_path, enhanced, 16000)
        label = f"{model_name}_enhanced"
        results.append((label, out_path))
        print(f"    Saved: {label}")
    elif est_sources.dim() == 3:
        num_sources = est_sources.shape[-1]
        if num_sources == 1:
            enhanced = est_sources[:, :, 0].detach().cpu()
            out_path = str(output_dir / f"{model_name}_enhanced.wav")
            torchaudio.save(out_path, enhanced, 16000)
            label = f"{model_name}_enhanced"
            results.append((label, out_path))
            print(f"    Saved: {label}")
        else:
            for i in range(num_sources):
                source = est_sources[:, :, i].detach().cpu()
                out_path = str(output_dir / f"{model_name}_source{i}.wav")
                torchaudio.save(out_path, source, 16000)
                label = f"{model_name}_source{i}"
                results.append((label, out_path))
                print(f"    Saved: {label}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def run_metricgan(input_wav: str, output_dir: Path) -> list[tuple[str, str]]:
    """Run MetricGAN+ speech enhancement (16kHz native).

    Uses SpectralMaskEnhancement instead of SepformerSeparation.

    Returns:
        List of (label, output_path) tuples.
    """
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    model_name = "metricgan_plus"
    savedir = output_dir / f"pretrained_{model_name}"

    print(f"    Loading MetricGAN+...")
    model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir=str(savedir),
    )

    waveform, sr = torchaudio.load(input_wav)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    print(f"    Enhancing ({waveform.shape[-1] / 16000:.1f}s)...")
    enhanced = model.enhance_batch(waveform, lengths=torch.tensor([1.0]))

    out_path = str(output_dir / f"{model_name}_enhanced.wav")
    torchaudio.save(out_path, enhanced.cpu(), 16000)
    label = f"{model_name}_enhanced"
    print(f"    Saved: {label}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return [(label, out_path)]


# =============================================================================
# Whisper transcription
# =============================================================================

def transcribe_and_extract(audio_path: str, prompt: str | None,
                            target_start: float, target_end: float,
                            window_offset: float) -> tuple[list[dict], str]:
    """Transcribe with MLX Whisper and extract words from target region."""
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
                    "start": round(word["start"] + window_offset, 2),
                    "end": round(word["end"] + window_offset, 2),
                    "probability": round(word.get("probability", 0.0), 4),
                })

    return words, result.get("text", "")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Source Separation v2: SpeechBrain + MLX Whisper")
    print("=" * 60)
    print(f"Audio: {AUDIO_PATH}")
    print(f"Window: {WINDOW_START}-{WINDOW_END}s")
    print(f"Target: {TARGET_START}-{TARGET_END}s")
    print(f"Expected: \"Who's father had to go away?\"")

    # -----------------------------------------------------------------
    # Step 1: Extract source audio (40s window at 16kHz)
    # -----------------------------------------------------------------
    window_wav = str(CLIPS_DIR / "window_16k.wav")
    print(f"\nExtracting window ({WINDOW_START}-{WINDOW_END}s) as 16kHz mono WAV...")
    extract_wav(AUDIO_PATH, window_wav, WINDOW_START, WINDOW_END, sample_rate=16000)

    # -----------------------------------------------------------------
    # Step 2: Run SpeechBrain separation/enhancement models
    # -----------------------------------------------------------------
    all_outputs: list[tuple[str, str]] = []  # (label, path)

    # --- 8kHz models (resample 16k -> 8k, run, resample 8k -> 16k) ---
    models_8k = [
        ("speechbrain/sepformer-wsj02mix", "sepformer_wsj02mix"),
        ("speechbrain/sepformer-whamr", "sepformer_whamr"),
        ("speechbrain/sepformer-whamr-enhancement", "sepformer_whamr_enhancement"),
    ]

    for source, name in models_8k:
        print(f"\n--- {name} ---")
        try:
            outputs = run_sepformer_8k(window_wav, CLIPS_DIR, source, name)
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"    FAILED: {e}")

    # --- 16kHz models (native, no resampling) ---
    models_16k = [
        ("speechbrain/sepformer-dns4-16k-enhancement", "sepformer_dns4"),
        ("speechbrain/sepformer_rescuespeech", "sepformer_rescuespeech"),
    ]

    for source, name in models_16k:
        print(f"\n--- {name} ---")
        try:
            outputs = run_sepformer_16k(window_wav, CLIPS_DIR, source, name)
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"    FAILED: {e}")

    # --- MetricGAN+ (uses SpectralMaskEnhancement, not SepformerSeparation) ---
    print(f"\n--- MetricGAN+ ---")
    try:
        outputs = run_metricgan(window_wav, CLIPS_DIR)
        all_outputs.extend(outputs)
    except Exception as e:
        print(f"    FAILED: {e}")

    print(f"\nSeparation complete: {len(all_outputs)} output tracks")
    for label, path in all_outputs:
        print(f"  {label}: {path}")

    # -----------------------------------------------------------------
    # Step 3: Transcribe all outputs with MLX Whisper
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Transcribing with MLX Whisper (full 40s window)")
    print("=" * 60)

    results = []

    # Include baseline (raw audio) for comparison
    all_clips = [("baseline", window_wav)] + list(all_outputs)

    for label, clip_path in all_clips:
        for prompt_name, prompt_text in [("no_prompt", None), ("vocab_prompt", VOCAB_PROMPT)]:
            run_name = f"{label}__{prompt_name}"
            print(f"  {run_name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                target_words, full_text = transcribe_and_extract(
                    clip_path, prompt_text,
                    TARGET_START, TARGET_END,
                    window_offset=WINDOW_START,
                )
                elapsed = time.time() - t0

                if target_words:
                    text = " ".join(w["word"].strip() for w in target_words)
                    avg_p = sum(w["probability"] for w in target_words) / len(target_words)
                    print(f"-> \"{text}\" (p={avg_p:.3f}) [{elapsed:.1f}s]")
                else:
                    print(f"-> [nothing] [{elapsed:.1f}s]")

                results.append({
                    "source": label,
                    "prompt": prompt_name,
                    "clip_type": "full_window",
                    "run_name": run_name,
                    "elapsed_seconds": round(elapsed, 1),
                    "target_words": target_words,
                    "full_text": full_text,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"-> ERROR: {e} [{elapsed:.1f}s]")
                results.append({
                    "source": label,
                    "prompt": prompt_name,
                    "clip_type": "full_window",
                    "run_name": run_name,
                    "elapsed_seconds": round(elapsed, 1),
                    "target_words": [],
                    "full_text": f"ERROR: {e}",
                })

    # -----------------------------------------------------------------
    # Step 4: Extract tight clips and transcribe those too
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Tight clips ({TIGHT_CLIP_START}-{TIGHT_CLIP_END}s) + MLX Whisper")
    print("=" * 60)

    tight_clips_dir = CLIPS_DIR / "tight"
    tight_clips_dir.mkdir(parents=True, exist_ok=True)

    for label, clip_path in all_clips:
        tight_path = str(tight_clips_dir / f"{label}_tight.wav")
        try:
            extract_tight_clip(clip_path, tight_path, WINDOW_START)
        except Exception as e:
            print(f"  Failed to extract tight clip for {label}: {e}")
            continue

        for prompt_name, prompt_text in [("no_prompt", None), ("vocab_prompt", VOCAB_PROMPT)]:
            run_name = f"{label}_tight__{prompt_name}"
            print(f"  {run_name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                target_words, full_text = transcribe_and_extract(
                    tight_path, prompt_text,
                    TARGET_START, TARGET_END,
                    window_offset=TIGHT_CLIP_START,
                )
                elapsed = time.time() - t0

                if target_words:
                    text = " ".join(w["word"].strip() for w in target_words)
                    avg_p = sum(w["probability"] for w in target_words) / len(target_words)
                    print(f"-> \"{text}\" (p={avg_p:.3f}) [{elapsed:.1f}s]")
                else:
                    print(f"-> [nothing] [{elapsed:.1f}s]")

                results.append({
                    "source": label,
                    "prompt": prompt_name,
                    "clip_type": "tight",
                    "run_name": run_name,
                    "elapsed_seconds": round(elapsed, 1),
                    "target_words": target_words,
                    "full_text": full_text,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"-> ERROR: {e} [{elapsed:.1f}s]")
                results.append({
                    "source": label,
                    "prompt": prompt_name,
                    "clip_type": "tight",
                    "run_name": run_name,
                    "elapsed_seconds": round(elapsed, 1),
                    "target_words": [],
                    "full_text": f"ERROR: {e}",
                })

    # -----------------------------------------------------------------
    # Step 5: Save results
    # -----------------------------------------------------------------
    report = {
        "experiment": "source_separation_v2",
        "timestamp": datetime.now().isoformat(),
        "source_audio": AUDIO_PATH,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "target": {"start": TARGET_START, "end": TARGET_END},
        "tight_clip": {"start": TIGHT_CLIP_START, "end": TIGHT_CLIP_END},
        "expected_text": "Who's father had to go away?",
        "whisper_model": MODEL,
        "vocab_prompt": VOCAB_PROMPT,
        "separation_models_attempted": [
            "speechbrain/sepformer-wsj02mix",
            "speechbrain/sepformer-whamr",
            "speechbrain/sepformer-whamr-enhancement",
            "speechbrain/sepformer-dns4-16k-enhancement",
            "speechbrain/sepformer_rescuespeech",
            "speechbrain/metricgan-plus-voicebank",
        ],
        "tracks_produced": [label for label, _ in all_outputs],
        "results": results,
    }

    report_path = OUTPUT_DIR / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    # Group by clip type for readability
    for clip_type in ["full_window", "tight"]:
        type_results = [r for r in results if r["clip_type"] == clip_type]
        if not type_results:
            continue
        print(f"\n--- {clip_type} ---")
        for r in type_results:
            words = r["target_words"]
            if words:
                text = " ".join(w["word"].strip() for w in words)
                avg_p = sum(w["probability"] for w in words) / len(words)
                print(f"  {r['run_name']:55s} -> \"{text}\" (p={avg_p:.3f})")
            else:
                print(f"  {r['run_name']:55s} -> [nothing]")

    print(f"\nResults saved: {report_path}")
    print(f"Audio tracks: {CLIPS_DIR}/")
    print(f"Tight clips:  {tight_clips_dir}/")
    print(f"\nListen to separated tracks -- even if Whisper fails,")
    print(f"human ears may hear Arti more clearly:")
    for label, path in all_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
