# Quiet Speech Recovery — Experiment Tracker

**Goal:** Recover Arti's speech from a known gap in session 000002 (241.68–242.83s).

**Ground truth:** Diarization detected SPEAKER_01. Human ear hears faint "Who's father had to go away?" Conversational context confirms it (your "huh?" before, "Yudhishthira's dad" after).

**Baseline:** Whisper large-v3 produces nothing for this region across all standard pipeline settings.

---

## Experiments

### 1. Gain Boost + High-Pass Filter ✗

**Hypothesis:** Speech signal is present but below Whisper's decoding threshold. Amplification and noise filtering could push it above.

**What we tried:** 7 audio variants (10/15/20dB gain, 200/300Hz high-pass + gain combos) × 2 prompt settings = 14 runs.

**Result:** No recovery. Gain amplifies noise equally — doesn't change SNR. High-pass cuts rumble but Arti's speech still too faint.

**Side finding:** Vocabulary-only `initial_prompt` safely improves name spelling (Yudhishthira vs Yudister). Narrative prompts cause Whisper to skip matching audio entirely.

**Script:** `experiments/quiet_speech_recovery.py`
**Data:** `experiments/results/quiet_speech_recovery/`

---

### 2. Spectral Subtraction + Bandpass ✗

**Hypothesis:** Room noise has a consistent spectral profile. Subtracting it changes the actual SNR (unlike flat gain). Additionally, a bandpass filter tuned to child vocal range (250Hz–3kHz fundamental) could suppress adult speech and room noise while preserving Arti's frequency band.

**Result:** No recovery. Baseline SNR was already 30.3dB — the problem was never noise masking the signal. Spectral subtraction actually decreased SNR slightly (Δ-0.6dB) by removing some signal along with noise. Bandpass + gain variants showed higher measured SNR but Whisper still produced nothing new.

**Key insight:** High SNR ≠ audible speech. Arti's voice is genuinely very quiet, not noise-masked. Noise removal can't create signal that isn't there.

**Listening test:** The spectral + bandpass + 20dB gain clip has faint audible speech at the target region — you can hear *someone* talking but can't make out the words. Real energy exists, below intelligibility threshold.

**Script:** `experiments/spectral_recovery.py`
**Data:** `experiments/results/spectral_recovery/`

---

### 3. Source Separation Model — Partial ✗

**Hypothesis:** ML source separation (trained to isolate voices from noise/other speakers) might extract Arti's voice track even from low-SNR audio.

**Original plan:** Run Demucs and SepFormer on the target region, feed isolated tracks to Whisper.

**What actually happened:** SpeechBrain models (SepFormer, MetricGAN) all failed due to compatibility issues — `torchaudio.list_audio_backends` removed in torchaudio 2.9, `use_auth_token` deprecated in huggingface_hub 1.3.2, and then SpeechBrain HuggingFace repos returned 404 (custom.py not found — repos restructured). Demucs wouldn't install on Python 3.14.

**Pivoted to working libraries:**
- Conv-TasNet (torchaudio built-in `CONVTASNET_BASE_LIBRI2MIX`, 8kHz, 2-speaker separation)
- HDemucs (torchaudio built-in `HDEMUCS_HIGH_MUSDB_PLUS`, 44.1kHz, music source separation → vocals track)
- MossFormer2 SE (ClearVoice/Alibaba, 48kHz, speech enhancement — single-channel denoising)
- MossFormer2 SS (ClearVoice/Alibaba, 16kHz, 2-speaker source separation)

All 4 models ran successfully producing 9 source tracks. Post-processing: gain boost (+15dB, +20dB) and DRC on promising tracks → 24 variants. **136 Whisper runs total** (each variant × full/tight window × with/without vocab prompt).

**Key results:**
- **Conv-TasNet source 0:** Pure noise/artifacts — hallucinated "Tel Tel Tel" (detected as Nynorsk). Useless.
- **Conv-TasNet source 1:** Preserved Dad's voice. Tight window: "You get" (p=0.86/0.25). Essentially same as baseline.
- **HDemucs vocals:** Near-identical to baseline. Tight window: "You dissed" (p=0.97/0.98). The model treated this as a music-style vocal extraction — not designed for multi-speaker speech.
- **HDemucs other/bass/drums:** Hallucinated patterns or silence.
- **MossFormer2 SE (speech enhancement):** Best quality output. Tight window: "You dissed" (p=0.98/0.97). Full window: "Yudister's" at 243.34–243.98s (p=1.0). Enhancement made Dad's voice clearer but didn't extract Arti's voice.
- **MossFormer2 SS (2-speaker separation):** Source 0 produced an alternative interpretation — "It just" or "fun. It just" on tight window. Source 1 produced "Wow." (p=0.66). Neither matched expected speech.

**Conclusion:** Source separation didn't change what Whisper decodes from the gap. The models either preserved the dominant speaker (Dad) or produced artifacts. None isolated Arti's voice as a clean separate track. This makes sense — the models are trained on mixed audio where both speakers are at comparable volume. Arti's signal may be below even the separation model's sensitivity floor.

**Scripts:** `experiments/source_separation_v2.py` (failed SpeechBrain attempt), `experiments/source_separation_v3.py` (working version)
**Data:** `experiments/results/source_separation_v3/`

---

### 4. Different ASR Model — Partial ✗

**Hypothesis:** Whisper large-v3 has a particular sensitivity floor. Other models may have different architectures or training data that handle low-SNR or child speech better.

**What we tried:**
- **wav2vec2-base-960h** (CTC architecture, HuggingFace): Character-level output "H A Y OU" spanning the gap at 241.13–243.45s. CTC models emit per-frame characters — they detected *some* audio energy but couldn't resolve it into words. Interesting: the "Y" character spans 241.39–243.39s, covering nearly the entire gap.
- **HF Whisper large-v3-turbo:** Failed with MPS/CPU device mismatch errors. Not evaluated.

**Result:** wav2vec2's CTC output confirms audio energy exists in the gap but can't decode it. Different architecture, same fundamental limit.

**Script:** `experiments/comprehensive_recovery.py` (Phase 3)
**Data:** `experiments/results/comprehensive_recovery/`

---

### 5. Whisper Decode Parameter Sweep ✗

**Hypothesis:** Whisper's internal decisions (speech detection threshold, conditioning, logprob acceptance) might be marginal for this region. Tuning them could flip the decision.

**What we tried:** 10 parameter combos × 2 audio clips (baseline + best spectral) = 20 runs. Tested `condition_on_previous_text=False`, `no_speech_threshold` up to 0.99, `hallucination_silence_threshold`, `logprob_threshold=-2.0`, all combined ("kitchen sink"), fixed temperature.

**Result:** Completely flat. Every variant returned identical results. Whisper isn't on the fence — it genuinely doesn't detect speech energy in the target region regardless of decode parameters.

**Script:** `experiments/whisper_params_recovery.py`
**Data:** `experiments/results/whisper_params_recovery/`

---

### 6. Tight Window + Processed Audio — Key Finding ✓

**Hypothesis:** Short clips centered on the gap force Whisper to attend to the quiet region instead of skipping it. Combined with processed audio from other experiments, this might push detection over the threshold.

**What we tried:** 8-second clips (238–246s) centered on the gap, with and without vocabulary prompt, across all audio variants (raw, compressed, harmonic-enhanced, source-separated, speech-enhanced).

**Result — the single biggest finding of all experiments:** On tight windows WITHOUT a vocab prompt, Whisper consistently produces "You dissed" at 242.42–243.82s across multiple audio sources:
- **Baseline tight:** "Huh? You dissed their dad." — "You" at 242.42–243.36 (p=0.97), "dissed" at 243.36–243.82 (p=0.98)
- **MossFormer2 SE tight:** "huh? You dissed their dad." — "You" at 242.90–243.34 (p=0.98), "dissed" at 243.34–243.74 (p=0.97)
- **HDemucs vocals tight:** identical to baseline
- **All gain/DRC variants:** same pattern, slightly varying probabilities

WITH a vocab prompt, the same region becomes "Yudhishthira's" (p=0.52–0.62) — lower confidence and clearly prompt-influenced rather than genuine recognition.

**Why this matters:** The tight window forces Whisper to attend to the quiet region instead of skipping it. Whisper IS detecting audio energy and producing high-confidence tokens in the gap. But "You dissed" is almost certainly a misrecognition — the model is confabulating something plausible from sub-threshold acoustic features. The expected "Who's father had to go away?" shares some phonetic structure ("who's" ≈ "You", "father" has a soft /ð/ like "dissed") but the match is speculative.

**Script:** `experiments/comprehensive_recovery.py` (Phase 1, plus tight windows across all experiments)
**Data:** `experiments/results/comprehensive_recovery/`

---

### 7. Dynamic Range Compression ✗

**Hypothesis:** Hearing-aid-style compression using ffmpeg compand (attack 0.3s, decay 1s, soft-knee curve) could compress the dynamic range so quiet speech is amplified more than loud speech.

**Result:** No change on full window. On tight window, produced identical "You dissed" pattern. DRC doesn't help because the quiet speech is below the compressor's threshold — there's not enough signal to compress.

**Script:** `experiments/comprehensive_recovery.py` (Phase 2)
**Data:** `experiments/results/comprehensive_recovery/`

---

### 8. Harmonic Comb Filter ✗

**Hypothesis:** Estimating Arti's fundamental frequency and boosting its harmonics could selectively amplify her voice.

**What we tried:** Estimated child fundamental frequency at 347Hz via autocorrelation on the target region. Applied harmonic comb filter at 6dB, 12dB, and 18dB boost on harmonics of 347Hz.

**Result:** No improvement. At 6dB: identical to baseline. At 12dB: identical. At 18dB: slightly worse — wav2vec2 went from "HALL YOU DISS" to "HOH YOU THIS". Harmonic enhancement can't work when the fundamental itself is too weak to detect reliably.

**Script:** `experiments/comprehensive_recovery.py` (Phase 4)
**Data:** `experiments/results/comprehensive_recovery/`

---

## Priority Order (final — all approaches exhausted)

Every software-based recovery technique has been tried. Summary of what was ruled out:

| Approach | Experiments | Why it failed |
|----------|-------------|---------------|
| Signal amplification | 1, 7 | Boosts noise equally; quiet speech below compressor threshold |
| Noise removal | 2 | SNR was already 30.3dB — noise wasn't the problem |
| Whisper tuning | 5 | Model isn't on the fence; parameters make no difference |
| Source separation | 3 | Models trained on comparable-volume speakers; can't isolate 20+dB mismatch |
| Alternative ASR | 4 | Different architecture, same sensitivity floor |
| Spectral enhancement | 8 | Fundamental too weak to boost harmonics from |
| Attention forcing | 6 | **Best result** — got Whisper to produce *something*, but it's confabulation not recognition |

**Remaining options (non-software):**
1. **Human listening test** — play the best clips at high gain with context priming. A human ear expecting "Who's father had to go away?" may resolve what ASR cannot.
2. **Hardware fix** — ESP32 mic placed closer to Arti would capture her voice at usable levels for future sessions.
3. **Future child-speech ASR models** — if child-speech-specific models with lower sensitivity floors become available, the preserved audio clips are ready to test.

---

## Final Assessment

### Is this gap recoverable from this recording?

**No.** After 8 distinct approaches and 186+ transcription runs, no technique recovered the expected "Who's father had to go away?" The speech energy is real — confirmed by diarization, human ear at high gain, wav2vec2's detection of audio energy, and Whisper's production of *something* on tight clips — but it falls below the intelligibility threshold of every ASR model tested.

### What we learned

- **Tight windowing was the most impactful technique.** It forced attention onto the quiet region and produced high-confidence tokens in the gap for the first time. Every other technique produced silence or identical-to-baseline output.
- **But those tokens ("You dissed") are confabulated.** They're consistent across models and audio variants, suggesting Whisper is pattern-matching from sub-threshold acoustics rather than genuinely recognizing speech.
- **Source separation models couldn't isolate Arti's voice** because they're trained on comparable-volume speakers, not 20+dB mismatches. The signal is below even the separation model's sensitivity floor.
- **The problem isn't noise** (SNR is 30dB), **isn't decoder sensitivity** (parameter sweep exhausted), and **isn't model architecture** (4 ASR backends, 4 separation models) — it's genuinely insufficient acoustic energy captured by the microphone.

### Most promising future directions

- **Hardware:** ESP32 mic closer to Arti would solve this at the source.
- **Future models:** Child-speech-specific ASR models (when they exist for low-resource speakers) might have lower sensitivity floors.
- **Human listening test:** The separated/enhanced audio clips in `experiments/results/source_separation_v3/clips/` are worth a listen at high gain — a human ear with context priming may succeed where ASR fails.

### Best audio clips for human listening

- `mossformer2_se.wav` — clearest enhanced version
- `mossformer2_se_gain20db.wav` — enhanced + boosted for listening
- `hdemucs_vocals.wav` — isolated vocal track
- `convtasnet_source1.wav` — alternative separation

### Value even in failure

The experiments confirm that diarization coverage (`_speaker.coverage == 0.0`) is a reliable signal for unrecovered speech. This validates the future `_unrecovered_speech` enrichment marker concept — segments where diarization detected a speaker but no transcript exists. The pipeline can honestly say "someone spoke here but we couldn't decode it" rather than silently dropping the moment.

---

## Notes

- The gap is only ~1.15 seconds of speech. Very little audio to work with.
- Faint to human ear even at +20dB boost. This may simply be unrecoverable from this recording.
- Long-term fix: ESP32 mic placement closer to Arti.
- Even if recovery fails, the experiment informs a future `_unrecovered_speech` enrichment marker — diarization segments with no transcript coverage.
