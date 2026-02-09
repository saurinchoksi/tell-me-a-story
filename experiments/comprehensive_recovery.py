"""Comprehensive quiet speech recovery: four-phase experiment.

Diarization detected Arti (SPEAKER_01) at 241.68-242.83s in session 000002,
but Whisper produced no transcript. Expected: "Who's father had to go away?"
Previous experiments (gain boost, spectral subtraction, Whisper param sweep)
all failed.

This script tries everything in sequence, each phase building on what we learn:

Phase 1: Source Separation (SpeechBrain)
    Six models with different strengths: 2-speaker separation, noisy/reverberant
    conditions, speech enhancement, DNS4 denoising, RescueSpeech for degraded
    audio, and MetricGAN+ perceptual enhancement.

Phase 2: Dynamic Range Compression (ffmpeg compand)
    Hearing-aid-style compression that boosts quiet passages while limiting loud
    ones. Applied to both raw audio and best separated tracks from Phase 1.

Phase 3: Alternative ASR
    Every audio variant tested against multiple ASR backends:
    - MLX Whisper large-v3 (baseline)
    - HuggingFace Whisper large-v3-turbo (different decoder)
    - wav2vec2-base-960h (CTC model — completely different architecture)
    Both full 40s window and tight 8s window around the target.

Phase 4: Harmonic Enhancement
    Detect child voice F0 (~300-400Hz), build a comb filter that amplifies
    harmonics, apply to best separated track, re-run through ASR.

Usage:
    python experiments/comprehensive_recovery.py
"""

# === CRITICAL: Monkey-patch torchaudio BEFORE any SpeechBrain imports ===
# torchaudio.list_audio_backends was removed in torchaudio 2.9.
# SpeechBrain crashes without it. Must be patched before import.
import torchaudio
torchaudio.list_audio_backends = lambda: ["sox_io"]

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import mlx_whisper
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


# =============================================================================
# Configuration
# =============================================================================

AUDIO_PATH = "sessions/00000000-000002/audio.m4a"
WINDOW_START = 220.0
WINDOW_END = 260.0
TARGET_START = 241.0
TARGET_END = 243.5
MODEL = "mlx-community/whisper-large-v3-mlx"
VOCAB_PROMPT = (
    "Mahabharata, Yudhishthira, Pandu, Pandavas, Kauravas, "
    "Duryodhana, Dhritarashtra, Bhima, Arjuna"
)

# Tight window around target for focused transcription (238-246s in original,
# which is 18-26s into the 220-260 clip)
TIGHT_CLIP_START = 238.0
TIGHT_CLIP_END = 246.0

# Output directories
OUTPUT_DIR = Path("experiments/results/comprehensive_recovery")
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


def read_wav(path: str) -> tuple[int, np.ndarray]:
    """Read WAV file, return (sample_rate, float32 samples in [-1, 1])."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    return sr, data


def write_wav(path: str, sr: int, data: np.ndarray) -> None:
    """Write float32 audio to 16-bit WAV."""
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(path, sr, (data * 32767).astype(np.int16))


def extract_tight_clip(source_wav: str, output_path: str,
                        window_offset: float) -> None:
    """Extract the tight window (238-246s) from an already-clipped WAV.

    The source WAV starts at window_offset (220s). We need samples
    corresponding to TIGHT_CLIP_START-TIGHT_CLIP_END in original time.
    """
    sr, data = read_wav(source_wav)
    start_sample = int((TIGHT_CLIP_START - window_offset) * sr)
    end_sample = int((TIGHT_CLIP_END - window_offset) * sr)
    start_sample = max(0, start_sample)
    end_sample = min(len(data), end_sample)
    write_wav(output_path, sr, data[start_sample:end_sample])


# =============================================================================
# Phase 1: Source Separation (SpeechBrain)
# =============================================================================

def run_sepformer_8k(input_wav: str, output_dir: Path,
                      model_source: str, model_name: str) -> list[tuple[str, str]]:
    """Run a SepFormer model that expects 8kHz input.

    Resamples 16kHz -> 8kHz, runs separation, resamples outputs back to 16kHz.

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

    # Enhancement models may return [batch, samples] or [batch, samples, 1]
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

    # Clean up model to free memory
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def run_sepformer_16k(input_wav: str, output_dir: Path,
                       model_source: str, model_name: str) -> list[tuple[str, str]]:
    """Run a SepFormer model that works at 16kHz natively (no resampling).

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

    # Enhancement models may return [batch, samples] or [batch, samples, 1]
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
            # Single enhanced output
            enhanced = est_sources[:, :, 0].detach().cpu()
            out_path = str(output_dir / f"{model_name}_enhanced.wav")
            torchaudio.save(out_path, enhanced, 16000)
            label = f"{model_name}_enhanced"
            results.append((label, out_path))
            print(f"    Saved: {label}")
        else:
            # Multiple separated sources
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


def phase1_source_separation(input_wav: str) -> list[tuple[str, str]]:
    """Phase 1: Run all SpeechBrain source separation/enhancement models.

    Returns:
        List of (label, audio_path) tuples for all output tracks.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Source Separation (SpeechBrain)")
    print("=" * 60)

    clips_dir = CLIPS_DIR / "phase1_separation"
    clips_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    # --- 8kHz models (need resample down/up) ---
    models_8k = [
        ("speechbrain/sepformer-wsj02mix", "sepformer_wsj02mix"),
        ("speechbrain/sepformer-whamr", "sepformer_whamr"),
        ("speechbrain/sepformer-whamr-enhancement", "sepformer_whamr_enhancement"),
    ]

    for source, name in models_8k:
        print(f"\n--- {name} ---")
        try:
            outputs = run_sepformer_8k(input_wav, clips_dir, source, name)
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"    FAILED: {e}")

    # --- 16kHz models (native) ---
    models_16k = [
        ("speechbrain/sepformer-dns4-16k-enhancement", "sepformer_dns4"),
        ("speechbrain/sepformer_rescuespeech", "sepformer_rescuespeech"),
    ]

    for source, name in models_16k:
        print(f"\n--- {name} ---")
        try:
            outputs = run_sepformer_16k(input_wav, clips_dir, source, name)
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"    FAILED: {e}")

    # --- MetricGAN+ ---
    print(f"\n--- MetricGAN+ ---")
    try:
        outputs = run_metricgan(input_wav, clips_dir)
        all_outputs.extend(outputs)
    except Exception as e:
        print(f"    FAILED: {e}")

    print(f"\nPhase 1 complete: {len(all_outputs)} audio tracks produced")
    return all_outputs


# =============================================================================
# Phase 2: Dynamic Range Compression (ffmpeg)
# =============================================================================

def apply_compand(input_wav: str, output_path: str, label: str) -> str:
    """Apply hearing-aid-style compression via ffmpeg compand filter.

    Uses fast attack (0.01s) to catch transients, moderate release (0.3s),
    and a transfer function that boosts quiet signals while limiting loud ones.

    The compand parameters:
    - attacks=0.01: 10ms attack — responds quickly to quiet speech appearing
    - decays=0.3: 300ms release — smooth transition back
    - points: transfer curve mapping input dB to output dB
      -80/-50: signals at -80dB boosted to -50dB (30dB boost for very quiet)
      -50/-30: signals at -50dB boosted to -30dB (20dB boost for quiet)
      -30/-20: moderate boost for medium signals
      -20/-10: mild boost
      0/-5: slight limiting for loud signals
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-af", (
            "compand="
            "attacks=0.01:decays=0.3:"
            "points=-80/-50|-50/-30|-30/-20|-20/-10|0/-5:"
            "gain=5"
        ),
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg compand failed: {result.stderr}")
    return output_path


def phase2_dynamic_compression(input_wav: str,
                                phase1_outputs: list[tuple[str, str]]
                                ) -> list[tuple[str, str]]:
    """Phase 2: Apply dynamic range compression to raw and best separated tracks.

    Returns:
        List of (label, audio_path) tuples for compressed audio.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Dynamic Range Compression (ffmpeg compand)")
    print("=" * 60)

    clips_dir = CLIPS_DIR / "phase2_compressed"
    clips_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    # Compress the raw window
    print("\n--- Compressing raw audio ---")
    raw_compressed = str(clips_dir / "raw_compressed.wav")
    try:
        apply_compand(input_wav, raw_compressed, "raw_compressed")
        all_outputs.append(("raw_compressed", raw_compressed))
        print(f"    Saved: raw_compressed")
    except Exception as e:
        print(f"    FAILED: {e}")

    # Compress all Phase 1 outputs
    for label, path in phase1_outputs:
        compressed_label = f"{label}_compressed"
        out_path = str(clips_dir / f"{compressed_label}.wav")
        print(f"--- Compressing {label} ---")
        try:
            apply_compand(path, out_path, compressed_label)
            all_outputs.append((compressed_label, out_path))
            print(f"    Saved: {compressed_label}")
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nPhase 2 complete: {len(all_outputs)} compressed tracks produced")
    return all_outputs


# =============================================================================
# Phase 3: Alternative ASR
# =============================================================================

def transcribe_mlx_whisper(audio_path: str, prompt: str | None,
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


def transcribe_hf_whisper(audio_path: str, prompt: str | None,
                           target_start: float, target_end: float,
                           window_offset: float) -> tuple[list[dict], str]:
    """Transcribe with HuggingFace Whisper pipeline (large-v3-turbo).

    Uses the transformers library instead of MLX — different decoder
    implementation may handle edge cases differently.
    """
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch.float32,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )

    generate_kwargs = {}
    if prompt:
        # HF pipeline uses generate_kwargs for prompt
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3-turbo")
        prompt_ids = tokenizer.get_prompt_ids(prompt, return_tensors="pt")
        generate_kwargs["prompt_ids"] = prompt_ids

    result = pipe(
        audio_path,
        return_timestamps="word",
        generate_kwargs=generate_kwargs if generate_kwargs else None,
    )

    clip_start = target_start - window_offset
    clip_end = target_end - window_offset

    words = []
    full_text = result.get("text", "")

    # HF pipeline returns chunks with timestamps
    for chunk in result.get("chunks", []):
        timestamps = chunk.get("timestamp", (None, None))
        if timestamps and timestamps[0] is not None and timestamps[1] is not None:
            w_start, w_end = timestamps
            if w_end >= clip_start and w_start <= clip_end:
                words.append({
                    "word": chunk["text"],
                    "start": round(w_start + window_offset, 2),
                    "end": round(w_end + window_offset, 2),
                    "probability": 0.0,  # HF pipeline doesn't expose this
                })

    return words, full_text


def transcribe_wav2vec2(audio_path: str,
                         target_start: float, target_end: float,
                         window_offset: float) -> tuple[list[dict], str]:
    """Transcribe with wav2vec2-base-960h (CTC model).

    Completely different architecture from Whisper — encoder-only with
    CTC decoding. Different failure modes mean it might succeed where
    Whisper fails.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Process
    input_values = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_values

    with torch.no_grad():
        logits = wav2vec_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    full_text = processor.batch_decode(predicted_ids)[0]

    # wav2vec2 doesn't provide word-level timestamps easily,
    # but we can get them via CTC alignment
    words = []

    # Use CTC token-level predictions to estimate word boundaries
    audio_duration = waveform.shape[-1] / 16000
    tokens_per_second = logits.shape[1] / audio_duration

    clip_start = target_start - window_offset
    clip_end = target_end - window_offset

    # Decode with timestamps by finding non-blank, non-repeat tokens
    predicted = predicted_ids[0].tolist()
    blank_id = processor.tokenizer.pad_token_id

    current_word = ""
    word_start_idx = None

    for idx, token_id in enumerate(predicted):
        if token_id == blank_id:
            if current_word:
                # End of word
                t_start = (word_start_idx / tokens_per_second)
                t_end = (idx / tokens_per_second)
                if t_end >= clip_start and t_start <= clip_end:
                    words.append({
                        "word": current_word,
                        "start": round(t_start + window_offset, 2),
                        "end": round(t_end + window_offset, 2),
                        "probability": 0.0,
                    })
                current_word = ""
                word_start_idx = None
            continue

        char = processor.tokenizer.decode([token_id])
        if char == "|":
            # Word separator
            if current_word:
                t_start = (word_start_idx / tokens_per_second)
                t_end = (idx / tokens_per_second)
                if t_end >= clip_start and t_start <= clip_end:
                    words.append({
                        "word": current_word,
                        "start": round(t_start + window_offset, 2),
                        "end": round(t_end + window_offset, 2),
                        "probability": 0.0,
                    })
                current_word = ""
                word_start_idx = None
        else:
            # Skip repeated characters (CTC property)
            if idx > 0 and token_id == predicted[idx - 1]:
                continue
            if word_start_idx is None:
                word_start_idx = idx
            current_word += char

    # Flush last word
    if current_word and word_start_idx is not None:
        t_start = (word_start_idx / tokens_per_second)
        t_end = (len(predicted) / tokens_per_second)
        if t_end >= clip_start and t_start <= clip_end:
            words.append({
                "word": current_word,
                "start": round(t_start + window_offset, 2),
                "end": round(t_end + window_offset, 2),
                "probability": 0.0,
            })

    return words, full_text


def check_transformers_available() -> bool:
    """Check if transformers is installed, install if not."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        print("    transformers not installed. Attempting install...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "transformers", "accelerate"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("    Installed transformers successfully.")
            return True
        else:
            print(f"    Failed to install transformers: {result.stderr[:200]}")
            return False


def phase3_alternative_asr(all_audio_variants: list[tuple[str, str]],
                            tight_clips: list[tuple[str, str]]
                            ) -> list[dict]:
    """Phase 3: Run all audio variants through multiple ASR backends.

    Tests both the full 40s window and tight 8s window for each variant.

    Returns:
        List of result dicts.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Alternative ASR Models")
    print("=" * 60)

    results = []
    has_transformers = check_transformers_available()

    # Build list of ASR backends to try
    asr_backends = [
        ("mlx_whisper", "MLX Whisper large-v3"),
    ]
    if has_transformers:
        asr_backends.append(("hf_whisper", "HF Whisper large-v3-turbo"))
        asr_backends.append(("wav2vec2", "wav2vec2-base-960h"))

    print(f"\nASR backends available: {len(asr_backends)}")
    for name, desc in asr_backends:
        print(f"  - {desc}")

    # For each audio variant, run all ASR backends
    # Full window clips use WINDOW_START as offset
    # Tight clips use TIGHT_CLIP_START as offset
    clip_sets = [
        ("full_window", all_audio_variants, WINDOW_START),
        ("tight_window", tight_clips, TIGHT_CLIP_START),
    ]

    for window_name, clips, window_offset in clip_sets:
        print(f"\n--- {window_name} ({len(clips)} clips) ---\n")

        for audio_label, audio_path in clips:
            if not os.path.exists(audio_path):
                print(f"  SKIP {audio_label}: file not found")
                continue

            for asr_name, asr_desc in asr_backends:
                # Test with and without vocab prompt (except wav2vec2 which
                # doesn't support prompts)
                prompt_configs = [("no_prompt", None)]
                if asr_name != "wav2vec2":
                    prompt_configs.append(("vocab_prompt", VOCAB_PROMPT))

                for prompt_name, prompt_text in prompt_configs:
                    run_name = f"{audio_label}__{asr_name}__{prompt_name}__{window_name}"

                    print(f"  {run_name}...", end=" ", flush=True)
                    t0 = time.time()

                    try:
                        if asr_name == "mlx_whisper":
                            words, full_text = transcribe_mlx_whisper(
                                audio_path, prompt_text,
                                TARGET_START, TARGET_END, window_offset,
                            )
                        elif asr_name == "hf_whisper":
                            words, full_text = transcribe_hf_whisper(
                                audio_path, prompt_text,
                                TARGET_START, TARGET_END, window_offset,
                            )
                        elif asr_name == "wav2vec2":
                            words, full_text = transcribe_wav2vec2(
                                audio_path,
                                TARGET_START, TARGET_END, window_offset,
                            )
                        else:
                            words, full_text = [], ""

                        elapsed = time.time() - t0

                        if words:
                            text = " ".join(w["word"].strip() for w in words)
                            print(f"-> \"{text}\" [{elapsed:.1f}s]")
                        else:
                            print(f"-> [nothing] [{elapsed:.1f}s]")

                        results.append({
                            "audio": audio_label,
                            "asr": asr_name,
                            "prompt": prompt_name,
                            "window": window_name,
                            "run_name": run_name,
                            "elapsed_seconds": round(elapsed, 1),
                            "target_words": words,
                            "full_text": full_text[:500],  # Truncate for JSON
                        })

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"-> ERROR: {e} [{elapsed:.1f}s]")
                        results.append({
                            "audio": audio_label,
                            "asr": asr_name,
                            "prompt": prompt_name,
                            "window": window_name,
                            "run_name": run_name,
                            "elapsed_seconds": round(elapsed, 1),
                            "target_words": [],
                            "full_text": f"ERROR: {e}",
                        })

    print(f"\nPhase 3 complete: {len(results)} transcription runs")
    return results


# =============================================================================
# Phase 4: Harmonic Enhancement
# =============================================================================

def estimate_f0(audio: np.ndarray, sr: int, low_hz: float = 200.0,
                high_hz: float = 500.0) -> float | None:
    """Estimate fundamental frequency via autocorrelation.

    Looks for the strongest periodicity in the expected child voice F0 range.

    Returns:
        Estimated F0 in Hz, or None if no clear pitch detected.
    """
    # Focus on the target region
    min_lag = int(sr / high_hz)
    max_lag = int(sr / low_hz)

    # Autocorrelation
    n = len(audio)
    if n < max_lag * 2:
        return None

    # Normalize
    audio = audio - np.mean(audio)
    norm = np.sqrt(np.sum(audio ** 2))
    if norm < 1e-10:
        return None
    audio = audio / norm

    # Compute autocorrelation for relevant lags
    correlations = np.zeros(max_lag - min_lag + 1)
    for i, lag in enumerate(range(min_lag, max_lag + 1)):
        correlations[i] = np.sum(audio[:n - lag] * audio[lag:])

    # Find strongest peak
    best_idx = np.argmax(correlations)
    best_lag = min_lag + best_idx
    best_corr = correlations[best_idx]

    if best_corr < 0.1:
        # No clear periodicity
        return None

    f0 = sr / best_lag
    return f0


def harmonic_comb_filter(audio: np.ndarray, sr: int, f0: float,
                          num_harmonics: int = 8,
                          boost_db: float = 6.0,
                          bandwidth_hz: float = 30.0) -> np.ndarray:
    """Apply a comb filter that boosts harmonics of the given F0.

    Constructs narrow bandpass filters at f0, 2*f0, 3*f0, etc.
    and adds the boosted harmonic content back to the signal.

    Args:
        audio: Input audio (float32, mono)
        sr: Sample rate
        f0: Fundamental frequency in Hz
        num_harmonics: Number of harmonics to boost (including fundamental)
        boost_db: How much to boost each harmonic band (dB)
        bandwidth_hz: Width of each bandpass filter around each harmonic

    Returns:
        Enhanced audio.
    """
    nyquist = sr / 2.0
    boost_linear = 10 ** (boost_db / 20.0)

    enhanced = audio.copy()

    for h in range(1, num_harmonics + 1):
        center = f0 * h
        if center >= nyquist - bandwidth_hz:
            break

        low = (center - bandwidth_hz / 2) / nyquist
        high = (center + bandwidth_hz / 2) / nyquist

        # Clamp to valid range
        low = max(low, 0.001)
        high = min(high, 0.999)

        if low >= high:
            continue

        try:
            sos = butter(4, [low, high], btype='band', output='sos')
            harmonic_band = sosfilt(sos, audio)
            enhanced += harmonic_band * (boost_linear - 1.0)
        except Exception:
            # Filter design can fail for very narrow bands near Nyquist
            continue

    return enhanced


def phase4_harmonic_enhancement(input_wav: str,
                                 phase1_outputs: list[tuple[str, str]]
                                 ) -> list[tuple[str, str]]:
    """Phase 4: Detect child voice F0 and apply harmonic comb filter.

    Tries to amplify the harmonic structure of the child's voice specifically,
    making it more prominent relative to background noise and adult speech.

    Returns:
        List of (label, audio_path) tuples for harmonically enhanced audio.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Harmonic Enhancement")
    print("=" * 60)

    clips_dir = CLIPS_DIR / "phase4_harmonic"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Determine which audio files to enhance: raw + all phase 1 outputs
    sources = [("raw", input_wav)] + list(phase1_outputs)
    all_outputs = []

    for source_label, source_path in sources:
        print(f"\n--- Analyzing {source_label} ---")

        try:
            sr, audio = read_wav(source_path)

            # Extract just the target region for F0 estimation
            target_start_sample = int((TARGET_START - WINDOW_START) * sr)
            target_end_sample = int((TARGET_END - WINDOW_START) * sr)
            target_region = audio[target_start_sample:target_end_sample]

            f0 = estimate_f0(target_region, sr)

            if f0 is not None:
                print(f"    Detected F0: {f0:.1f}Hz")
            else:
                # Use a reasonable default for child voice
                f0 = 350.0
                print(f"    No clear F0 detected, using default: {f0:.1f}Hz")

            # Apply harmonic enhancement with a few boost levels
            for boost_db in [6, 12, 18]:
                enhanced = harmonic_comb_filter(
                    audio, sr, f0, num_harmonics=8, boost_db=boost_db,
                )
                label = f"{source_label}_harmonic_{int(f0)}hz_{boost_db}db"
                out_path = str(clips_dir / f"{label}.wav")
                write_wav(out_path, sr, enhanced)
                all_outputs.append((label, out_path))
                print(f"    Saved: {label}")

        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nPhase 4 complete: {len(all_outputs)} harmonically enhanced tracks")
    return all_outputs


# =============================================================================
# Results summary
# =============================================================================

def print_summary(results: list[dict]) -> None:
    """Print a clear summary table of all results."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Run':<65s} {'Words':<5s}")
    print("-" * 70)

    hits = []
    misses = 0

    for r in results:
        words = r.get("target_words", [])
        if words:
            text = " ".join(w["word"].strip() for w in words)
            # Truncate long text for table display
            if len(text) > 40:
                text = text[:37] + "..."
            run = r["run_name"]
            if len(run) > 64:
                run = run[:61] + "..."
            print(f"  {run:<63s} \"{text}\"")
            hits.append(r)
        else:
            misses += 1

    if not hits:
        print("  (no words recovered in any combination)")

    print("-" * 70)
    print(f"  TOTAL: {len(hits)} hits / {len(results)} runs ({misses} empty)")

    if hits:
        print("\n  BEST CANDIDATES (words found in target region):")
        for r in hits:
            words = r["target_words"]
            text = " ".join(w["word"].strip() for w in words)
            probs = [w.get("probability", 0) for w in words]
            avg_p = sum(probs) / len(probs) if probs else 0
            print(f"    [{r['asr']}] {r['audio']} ({r['window']})")
            print(f"      Text: \"{text}\"")
            if avg_p > 0:
                print(f"      Avg probability: {avg_p:.3f}")
            print()


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(AUDIO_PATH):
        print(f"Audio not found: {AUDIO_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("COMPREHENSIVE QUIET SPEECH RECOVERY")
    print("=" * 70)
    print(f"Source:   {AUDIO_PATH}")
    print(f"Window:   {WINDOW_START}s - {WINDOW_END}s ({WINDOW_END - WINDOW_START:.0f}s)")
    print(f"Target:   {TARGET_START}s - {TARGET_END}s (Arti's missing speech)")
    print(f"Expected: \"Who's father had to go away?\"")
    print(f"Output:   {OUTPUT_DIR}/")

    start_time = time.time()

    # --- Extract source audio ---
    print("\n--- Extracting source audio ---")
    window_wav = str(CLIPS_DIR / "window_16k.wav")
    extract_wav(AUDIO_PATH, window_wav, WINDOW_START, WINDOW_END, 16000)
    print(f"  Full window: {window_wav}")

    # --- Phase 1: Source Separation ---
    phase1_outputs = phase1_source_separation(window_wav)

    # --- Phase 2: Dynamic Range Compression ---
    phase2_outputs = phase2_dynamic_compression(window_wav, phase1_outputs)

    # --- Phase 4: Harmonic Enhancement (before Phase 3 so we can include
    #     enhanced audio in the ASR sweep) ---
    phase4_outputs = phase4_harmonic_enhancement(window_wav, phase1_outputs)

    # --- Build complete audio variant list for Phase 3 ---
    all_variants = [("raw", window_wav)]
    all_variants.extend(phase1_outputs)
    all_variants.extend(phase2_outputs)
    all_variants.extend(phase4_outputs)

    print(f"\n--- Total audio variants for ASR: {len(all_variants)} ---")

    # Also create tight clips (8s around target) for each variant
    tight_clips_dir = CLIPS_DIR / "tight_clips"
    tight_clips_dir.mkdir(parents=True, exist_ok=True)
    tight_clips = []

    for label, path in all_variants:
        tight_label = f"{label}_tight"
        tight_path = str(tight_clips_dir / f"{tight_label}.wav")
        try:
            extract_tight_clip(path, tight_path, WINDOW_START)
            tight_clips.append((tight_label, tight_path))
        except Exception as e:
            print(f"  Failed to create tight clip for {label}: {e}")

    print(f"  Tight clips created: {len(tight_clips)}")

    # --- Phase 3: Alternative ASR ---
    all_results = phase3_alternative_asr(all_variants, tight_clips)

    elapsed_total = time.time() - start_time

    # --- Save comprehensive results ---
    report = {
        "experiment": "comprehensive_recovery",
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_seconds": round(elapsed_total, 1),
        "source_audio": AUDIO_PATH,
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "tight_window": {"start": TIGHT_CLIP_START, "end": TIGHT_CLIP_END},
        "target": {"start": TARGET_START, "end": TARGET_END},
        "expected_text": "Who's father had to go away?",
        "mlx_model": MODEL,
        "vocab_prompt": VOCAB_PROMPT,
        "phases": {
            "phase1_separation": {
                "models_tried": [
                    "sepformer-wsj02mix", "sepformer-whamr",
                    "sepformer-whamr-enhancement",
                    "sepformer-dns4-16k-enhancement",
                    "sepformer_rescuespeech",
                    "metricgan-plus-voicebank",
                ],
                "outputs": len(phase1_outputs),
            },
            "phase2_compression": {
                "method": "ffmpeg compand",
                "outputs": len(phase2_outputs),
            },
            "phase4_harmonic": {
                "method": "harmonic comb filter",
                "outputs": len(phase4_outputs),
            },
            "phase3_asr": {
                "backends": ["mlx_whisper", "hf_whisper", "wav2vec2"],
                "total_runs": len(all_results),
            },
        },
        "audio_variants": [
            {"label": label, "path": path} for label, path in all_variants
        ],
        "results": all_results,
    }

    report_path = OUTPUT_DIR / "results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Summary ---
    print_summary(all_results)

    print(f"\nTotal time: {elapsed_total / 60:.1f} minutes")
    print(f"Full results: {report_path}")
    print(f"Audio clips:  {CLIPS_DIR}/")
    print(f"\nLISTEN to separated tracks even if ASR fails:")
    for label, path in phase1_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
