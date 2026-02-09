"""Experiment: Source separation v3 for quiet speech recovery.

Uses WORKING libraries (torchaudio built-in models + ClearVoice MossFormer2)
after SpeechBrain broke (404 errors on HuggingFace, use_auth_token deprecation).

Models:
- Conv-TasNet (torchaudio): 2-speaker separation at 8kHz
- HDemucs (torchaudio): Music source separation, vocals extraction at 44.1kHz
- MossFormer2 SE (ClearVoice): Speech enhancement at 48kHz
- MossFormer2 SS (ClearVoice): Speech separation at 16kHz

After separation, applies gain boost + dynamic range compression (DRC) to
the most promising tracks and runs MLX Whisper on everything.

Usage:
    python experiments/source_separation_v3.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import mlx_whisper


# =============================================================================
# Configuration
# =============================================================================

AUDIO_PATH = "sessions/00000000-000002/audio.m4a"

WINDOW_START = 220.0
WINDOW_END = 260.0
TARGET_START = 241.0
TARGET_END = 243.5
TIGHT_START = 238.0
TIGHT_END = 246.0

MODEL = "mlx-community/whisper-large-v3-mlx"
VOCAB_PROMPT = (
    "Mahabharata, Yudhishthira, Pandu, Pandavas, Kauravas, "
    "Duryodhana, Dhritarashtra, Bhima, Arjuna"
)

OUTPUT_DIR = Path("experiments/results/source_separation_v3")


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
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def resample_to_16k_mono(input_path: str, output_path: str) -> None:
    """Convert any audio file to 16kHz mono WAV for Whisper."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {result.stderr}")


def apply_gain_drc(input_path: str, output_path: str,
                   gain_db: float, attack: float = 0.01,
                   release: float = 0.3) -> None:
    """Apply gain boost + dynamic range compression via ffmpeg compand.

    DRC compresses the dynamic range so quiet speech gets lifted while
    loud peaks are tamed. Combined with gain boost, this can make a
    faint child voice more audible to Whisper.

    Args:
        input_path: Source WAV file.
        output_path: Destination WAV file.
        gain_db: Gain boost in dB to apply after compression.
        attack: Compressor attack time in seconds.
        release: Compressor release time in seconds.
    """
    # compand: attacks|decays  soft-knee-points  gain
    # The points define the compression curve:
    #   -70/-70  means anything at -70dB stays at -70dB
    #   -30/-15  means -30dB input -> -15dB output (2:1 compression above -30)
    #   -20/-10  means -20dB input -> -10dB output
    #   0/-5     means 0dB input -> -5dB output (limiting)
    compand_points = "-70/-70|-30/-15|-20/-10|0/-5"
    compand_filter = (
        f"compand=attacks={attack}:decays={release}"
        f":points={compand_points}"
        f":gain={gain_db}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", compand_filter,
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg compand failed: {result.stderr}")


def apply_gain_only(input_path: str, output_path: str,
                    gain_db: float) -> None:
    """Apply simple gain boost via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", f"volume={gain_db}dB",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg gain failed: {result.stderr}")


def extract_tight_window(full_window_16k: str, output_path: str) -> None:
    """Extract the tight window from the full window clip.

    The tight window is centered on the target region and shorter,
    which forces Whisper to focus on that region without as much
    surrounding context potentially masking the quiet speech.
    """
    # Tight window times relative to the full window clip
    tight_offset = TIGHT_START - WINDOW_START
    tight_duration = TIGHT_END - TIGHT_START

    cmd = [
        "ffmpeg", "-y",
        "-i", full_window_16k,
        "-ss", str(tight_offset),
        "-t", str(tight_duration),
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg tight extract failed: {result.stderr}")


# =============================================================================
# Phase 1: Source separation models
# =============================================================================

def run_convtasnet(input_16k_path: str, clips_dir: Path) -> list[str]:
    """Run Conv-TasNet 2-speaker separation (torchaudio built-in).

    Conv-TasNet operates at 8kHz. We resample input down to 8kHz,
    separate, then resample outputs back to 16kHz for Whisper.

    Returns:
        List of output WAV paths (16kHz mono).
    """
    from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX

    print("  Loading Conv-TasNet model...")
    bundle = CONVTASNET_BASE_LIBRI2MIX
    model = bundle.get_model()
    model_sr = bundle.sample_rate  # 8000

    # Load input audio
    waveform, sr = torchaudio.load(input_16k_path)
    # waveform: [channels, samples]

    # Resample to 8kHz if needed
    if sr != model_sr:
        resampler = torchaudio.transforms.Resample(sr, model_sr)
        waveform = resampler(waveform)

    # Conv-TasNet expects [batch, channels, samples]
    # Our mono audio is [1, samples], add batch dim
    input_tensor = waveform.unsqueeze(0)  # [1, 1, samples]

    print("  Running separation...")
    with torch.no_grad():
        output = model(input_tensor)
    # output: [batch, num_sources, samples]

    num_sources = output.shape[1]
    upsampler = torchaudio.transforms.Resample(model_sr, 16000)

    output_paths = []
    for i in range(num_sources):
        source = output[0, i, :].unsqueeze(0).cpu()  # [1, samples]
        source_16k = upsampler(source)

        out_path = str(clips_dir / f"convtasnet_source{i}.wav")
        torchaudio.save(out_path, source_16k, 16000)
        output_paths.append(out_path)
        print(f"    Saved: {out_path}")

    # Cleanup
    del model, output, input_tensor
    torch.mps.empty_cache()

    return output_paths


def run_hdemucs(input_16k_path: str, clips_dir: Path) -> list[str]:
    """Run HDemucs vocals extraction (torchaudio built-in).

    HDemucs is a music source separation model that operates at 44.1kHz.
    We resample up, separate into drums/bass/other/vocals, extract the
    vocals track, then resample back to 16kHz.

    While designed for music, the "vocals" source often captures human
    speech effectively, separating it from background noise.

    Returns:
        List of output WAV paths (16kHz mono).
    """
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    print("  Loading HDemucs model...")
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model_sr = bundle.sample_rate  # 44100

    # Load input audio
    waveform, sr = torchaudio.load(input_16k_path)
    # waveform: [channels, samples]

    # Resample to 44.1kHz
    if sr != model_sr:
        resampler = torchaudio.transforms.Resample(sr, model_sr)
        waveform = resampler(waveform)

    # HDemucs expects stereo [batch, channels=2, samples]
    # Duplicate mono to stereo if needed
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # [2, samples]

    input_tensor = waveform.unsqueeze(0)  # [1, 2, samples]

    print("  Running separation (this may take a minute)...")
    with torch.no_grad():
        output = model(input_tensor)
    # output: [batch, num_sources, channels, samples]
    # Sources order: drums, bass, other, vocals

    source_names = ["drums", "bass", "other", "vocals"]
    downsampler = torchaudio.transforms.Resample(model_sr, 16000)

    output_paths = []
    for i, name in enumerate(source_names):
        # Take first channel (both are identical for mono input)
        source = output[0, i, 0:1, :].cpu()  # [1, samples]
        source_16k = downsampler(source)

        out_path = str(clips_dir / f"hdemucs_{name}.wav")
        torchaudio.save(out_path, source_16k, 16000)
        output_paths.append(out_path)
        print(f"    Saved: {out_path}")

    # Cleanup
    del model, output, input_tensor
    torch.mps.empty_cache()

    return output_paths


def run_clearvoice_enhancement(input_16k_path: str,
                                clips_dir: Path) -> list[str]:
    """Run ClearVoice MossFormer2 speech enhancement.

    MossFormer2_SE_48K enhances speech by suppressing noise. Operates
    at 48kHz internally. ClearVoice handles resampling.

    Returns:
        List with single output WAV path (16kHz mono).
    """
    from clearvoice import ClearVoice

    print("  Loading MossFormer2 speech enhancement (48kHz)...")
    cv = ClearVoice(task='speech_enhancement',
                    model_names=['MossFormer2_SE_48K'])

    print("  Enhancing...")
    output_wav = cv(input_path=input_16k_path, online_write=False)
    # Returns numpy array

    # ClearVoice outputs at the model's native rate (48kHz).
    # Save as temp WAV then resample to 16kHz.
    if isinstance(output_wav, list):
        output_wav = output_wav[0]

    # Determine output sample rate from the model (48kHz)
    enhanced_sr = 48000

    # Normalize to float32 if needed
    if output_wav.dtype != np.float32:
        output_wav = output_wav.astype(np.float32)

    # Ensure 2D [channels, samples] for torchaudio
    if output_wav.ndim == 1:
        output_wav = output_wav[np.newaxis, :]

    # Save at native rate, then resample
    temp_path = str(clips_dir / "mossformer2_se_48k_raw.wav")
    tensor = torch.from_numpy(output_wav)
    torchaudio.save(temp_path, tensor, enhanced_sr)

    out_path = str(clips_dir / "mossformer2_se.wav")
    resample_to_16k_mono(temp_path, out_path)
    print(f"    Saved: {out_path}")

    # Cleanup
    del cv
    torch.mps.empty_cache()

    return [out_path]


def run_clearvoice_separation(input_16k_path: str,
                               clips_dir: Path) -> list[str]:
    """Run ClearVoice MossFormer2 speech separation (2 speakers).

    MossFormer2_SS_16K separates two overlapping speakers at 16kHz.
    Returns one WAV per estimated speaker.

    Returns:
        List of output WAV paths (16kHz mono).
    """
    from clearvoice import ClearVoice

    print("  Loading MossFormer2 speech separation (16kHz)...")
    cv = ClearVoice(task='speech_separation',
                    model_names=['MossFormer2_SS_16K'])

    print("  Separating...")
    output_wavs = cv(input_path=input_16k_path, online_write=False)
    # Returns list of numpy arrays (one per speaker)

    if not isinstance(output_wavs, list):
        output_wavs = [output_wavs]

    output_paths = []
    for i, wav in enumerate(output_wavs):
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        if wav.ndim == 1:
            wav = wav[np.newaxis, :]

        out_path = str(clips_dir / f"mossformer2_ss_source{i}.wav")
        tensor = torch.from_numpy(wav)
        torchaudio.save(out_path, tensor, 16000)
        output_paths.append(out_path)
        print(f"    Saved: {out_path}")

    # Cleanup
    del cv
    torch.mps.empty_cache()

    return output_paths


# =============================================================================
# Phase 2: Post-processing (gain boost + DRC)
# =============================================================================

def build_post_processed_variants(separation_outputs: dict[str, list[str]],
                                   clips_dir: Path) -> dict[str, list[str]]:
    """Apply gain boost and DRC to the most promising separated tracks.

    "Most promising" = vocals tracks from HDemucs, all ClearVoice outputs,
    and both Conv-TasNet sources (since we don't know which is the child).

    Returns:
        Dict of model_name -> list of post-processed WAV paths.
    """
    # Tracks worth post-processing
    candidates = []
    for model_name, paths in separation_outputs.items():
        for path in paths:
            stem = Path(path).stem
            # Include: hdemucs vocals, all mossformer outputs, all convtasnet
            if any(kw in stem for kw in ["vocals", "mossformer2", "convtasnet"]):
                candidates.append((model_name, path, stem))

    post_outputs = {}

    for model_name, path, stem in candidates:
        variant_paths = []

        # Gain boost variants
        for gain_db in [15, 20]:
            out_name = f"{stem}_gain{gain_db}db"
            out_path = str(clips_dir / f"{out_name}.wav")
            try:
                apply_gain_only(path, out_path, gain_db)
                variant_paths.append(out_path)
                print(f"    {out_name}")
            except Exception as e:
                print(f"    {out_name}: FAILED ({e})")

        # DRC + gain variants
        for gain_db in [15, 20]:
            out_name = f"{stem}_drc_gain{gain_db}db"
            out_path = str(clips_dir / f"{out_name}.wav")
            try:
                apply_gain_drc(path, out_path, gain_db)
                variant_paths.append(out_path)
                print(f"    {out_name}")
            except Exception as e:
                print(f"    {out_name}: FAILED ({e})")

        key = f"{model_name}_postprocessed"
        if key not in post_outputs:
            post_outputs[key] = []
        post_outputs[key].extend(variant_paths)

    return post_outputs


# =============================================================================
# Phase 3: Whisper transcription
# =============================================================================

def transcribe_and_extract(audio_path: str, prompt: str | None,
                           target_start: float, target_end: float,
                           window_offset: float) -> tuple[list[dict], str]:
    """Transcribe audio and extract words from target time region.

    Args:
        audio_path: Path to WAV file to transcribe.
        prompt: Optional initial prompt for vocabulary priming.
        target_start: Start of target region in ORIGINAL audio time.
        target_end: End of target region in ORIGINAL audio time.
        window_offset: Start of the extracted window in ORIGINAL audio time.

    Returns:
        Tuple of (target_words, full_text).
    """
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


def run_transcription_sweep(all_clips: list[tuple[str, str]],
                            clips_dir: Path) -> list[dict]:
    """Run Whisper on every clip with both full and tight windows,
    with and without vocab prompt.

    For each clip we generate:
    - full window + no prompt
    - full window + vocab prompt
    - tight window + no prompt
    - tight window + vocab prompt

    Returns:
        List of result dicts.
    """
    results = []
    total = len(all_clips) * 4  # 2 windows x 2 prompts
    run_idx = 0

    for label, clip_path in all_clips:
        # Prepare tight window version of this clip
        tight_path = str(clips_dir / f"{Path(clip_path).stem}_tight.wav")
        try:
            extract_tight_window(clip_path, tight_path)
        except Exception as e:
            print(f"  WARNING: Could not extract tight window for {label}: {e}")
            tight_path = None

        window_configs = [
            ("full", clip_path, WINDOW_START),
        ]
        if tight_path and os.path.exists(tight_path):
            window_configs.append(("tight", tight_path, TIGHT_START))

        prompt_configs = [
            ("no_prompt", None),
            ("vocab_prompt", VOCAB_PROMPT),
        ]

        for window_name, wav_path, window_offset in window_configs:
            for prompt_name, prompt_text in prompt_configs:
                run_idx += 1
                run_name = f"{label}__{window_name}__{prompt_name}"

                print(f"  [{run_idx}/{total}] {run_name}...",
                      end=" ", flush=True)
                t0 = time.time()

                try:
                    target_words, full_text = transcribe_and_extract(
                        wav_path, prompt_text,
                        TARGET_START, TARGET_END,
                        window_offset=window_offset
                    )
                    elapsed = time.time() - t0

                    if target_words:
                        text = " ".join(w["word"].strip()
                                        for w in target_words)
                        avg_p = (sum(w["probability"]
                                     for w in target_words)
                                 / len(target_words))
                        print(f"-> \"{text}\" (p={avg_p:.3f}) "
                              f"[{elapsed:.1f}s]")
                    else:
                        print(f"-> [nothing] [{elapsed:.1f}s]")

                    results.append({
                        "source": label,
                        "window": window_name,
                        "prompt": prompt_name,
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
                        "window": window_name,
                        "prompt": prompt_name,
                        "run_name": run_name,
                        "elapsed_seconds": round(elapsed, 1),
                        "target_words": [],
                        "full_text": f"ERROR: {e}",
                    })

    return results


# =============================================================================
# Phase 4: Summary
# =============================================================================

def print_summary(results: list[dict]) -> None:
    """Print a formatted results summary sorted by recovery quality."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Expected: \"Who's father had to go away?\"")
    print(f"Target region: {TARGET_START}s - {TARGET_END}s")
    print()

    # Separate into recovered vs not
    recovered = []
    empty = []
    for r in results:
        words = r["target_words"]
        if words:
            text = " ".join(w["word"].strip() for w in words)
            avg_p = (sum(w["probability"] for w in words)
                     / len(words))
            recovered.append((r["run_name"], text, avg_p, words))
        else:
            empty.append(r["run_name"])

    if recovered:
        # Sort by average probability descending
        recovered.sort(key=lambda x: x[2], reverse=True)

        print("RECOVERED SPEECH (sorted by confidence):")
        print("-" * 70)
        for run_name, text, avg_p, words in recovered:
            print(f"  {run_name}")
            print(f"    Text: \"{text}\"")
            print(f"    Avg probability: {avg_p:.3f}")
            for w in words:
                print(f"      {w['start_original']:.2f}s-"
                      f"{w['end_original']:.2f}s  "
                      f"p={w['probability']:.3f}  "
                      f"\"{w['word']}\"")
            print()
    else:
        print("  No speech recovered in any variant.")
        print()

    if empty:
        print(f"NO RECOVERY ({len(empty)} runs):")
        print("-" * 70)
        for name in empty:
            print(f"  {name}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_dir = OUTPUT_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)

    if not os.path.exists(AUDIO_PATH):
        print(f"Audio not found: {AUDIO_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("Source Separation v3: torchaudio + ClearVoice")
    print("=" * 70)
    print(f"Source: {AUDIO_PATH}")
    print(f"Full window: {WINDOW_START}s - {WINDOW_END}s "
          f"({WINDOW_END - WINDOW_START}s)")
    print(f"Tight window: {TIGHT_START}s - {TIGHT_END}s "
          f"({TIGHT_END - TIGHT_START}s)")
    print(f"Target: {TARGET_START}s - {TARGET_END}s")
    print(f"Expected: \"Who's father had to go away?\"")
    print()

    # =====================================================================
    # Step 1: Extract source audio at 16kHz (Whisper standard)
    # =====================================================================
    print("--- Step 1: Extract source audio ---")

    window_16k = str(clips_dir / "window_16k.wav")
    print(f"  Extracting {WINDOW_START}-{WINDOW_END}s at 16kHz...")
    extract_wav(AUDIO_PATH, window_16k,
                WINDOW_START, WINDOW_END, sample_rate=16000)

    # =====================================================================
    # Phase 1: Run all separation/enhancement models
    # =====================================================================
    print("\n--- Phase 1: Source Separation Models ---\n")

    separation_outputs = {}  # model_name -> list of output WAV paths

    # --- Conv-TasNet (torchaudio, 8kHz 2-speaker separation) ---
    print("  [1/4] Conv-TasNet (torchaudio)")
    try:
        paths = run_convtasnet(window_16k, clips_dir)
        separation_outputs["convtasnet"] = paths
    except Exception as e:
        print(f"    FAILED: {e}")
        separation_outputs["convtasnet"] = []

    # --- HDemucs (torchaudio, 44.1kHz music source separation) ---
    print("\n  [2/4] HDemucs (torchaudio)")
    try:
        paths = run_hdemucs(window_16k, clips_dir)
        separation_outputs["hdemucs"] = paths
    except Exception as e:
        print(f"    FAILED: {e}")
        separation_outputs["hdemucs"] = []

    # --- ClearVoice MossFormer2 speech enhancement ---
    print("\n  [3/4] MossFormer2 Speech Enhancement (ClearVoice)")
    try:
        paths = run_clearvoice_enhancement(window_16k, clips_dir)
        separation_outputs["mossformer2_se"] = paths
    except Exception as e:
        print(f"    FAILED: {e}")
        separation_outputs["mossformer2_se"] = []

    # --- ClearVoice MossFormer2 speech separation ---
    print("\n  [4/4] MossFormer2 Speech Separation (ClearVoice)")
    try:
        paths = run_clearvoice_separation(window_16k, clips_dir)
        separation_outputs["mossformer2_ss"] = paths
    except Exception as e:
        print(f"    FAILED: {e}")
        separation_outputs["mossformer2_ss"] = []

    # Report Phase 1 results
    total_tracks = sum(len(v) for v in separation_outputs.values())
    print(f"\n  Phase 1 complete: {total_tracks} separated tracks from "
          f"{sum(1 for v in separation_outputs.values() if v)} models")

    # =====================================================================
    # Phase 2: Post-processing (gain + DRC)
    # =====================================================================
    print("\n--- Phase 2: Post-processing (gain boost + DRC) ---\n")

    post_outputs = build_post_processed_variants(
        separation_outputs, clips_dir)

    total_post = sum(len(v) for v in post_outputs.values())
    print(f"\n  Phase 2 complete: {total_post} post-processed variants")

    # =====================================================================
    # Phase 3: Whisper transcription sweep
    # =====================================================================
    print("\n--- Phase 3: Whisper Transcription ---\n")

    # Build clip list: baseline + all separation outputs + post-processed
    all_clips = [("baseline", window_16k)]

    for model_name, paths in separation_outputs.items():
        for path in paths:
            label = Path(path).stem
            all_clips.append((label, path))

    for model_name, paths in post_outputs.items():
        for path in paths:
            label = Path(path).stem
            all_clips.append((label, path))

    print(f"  {len(all_clips)} clips x 2 windows x 2 prompts = "
          f"{len(all_clips) * 4} transcription runs\n")

    results = run_transcription_sweep(all_clips, clips_dir)

    # =====================================================================
    # Phase 4: Results
    # =====================================================================
    print_summary(results)

    # Save full report
    report = {
        "experiment": "source_separation_v3",
        "timestamp": datetime.now().isoformat(),
        "source_audio": AUDIO_PATH,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "tight_window": {"start": TIGHT_START, "end": TIGHT_END},
        "target": {"start": TARGET_START, "end": TARGET_END},
        "expected_text": "Who's father had to go away?",
        "model": MODEL,
        "separation_models": {
            name: [str(p) for p in paths]
            for name, paths in separation_outputs.items()
        },
        "post_processed": {
            name: [str(p) for p in paths]
            for name, paths in post_outputs.items()
        },
        "results": results,
    }

    report_path = OUTPUT_DIR / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Full results: {report_path}")
    print(f"Audio clips: {clips_dir}/")
    print()
    print("LISTEN to separated tracks — even if Whisper can't decode them,")
    print("you might hear Arti more clearly:")
    for model_name, paths in separation_outputs.items():
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
