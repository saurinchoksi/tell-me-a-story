# Audio-Native Model Evaluation — Experiment Plan

## Context

The TMAS pipeline currently runs: audio → Whisper transcription → LLM normalization → dictionary normalization → speaker diarization. Each stage is deterministic and sequential. The pipeline has no way to evaluate its own output — that requires a human who can listen to the audio and compare it to the transcript.

Through manual validation, confirmed failure patterns include:
- **Hallucinations clustering near speaker transitions** — fabricated segments where Whisper generates plausible-sounding text that wasn't spoken
- **Quiet/distant speech dropped or garbled** — particularly Arti's voice at distance
- **Speaker attribution errors** — pyannote misassigning segments near transitions
- **Name confusion** — uncommon Sanskrit names and child voice at distance

The question: could a model that natively understands audio close this evaluation loop — functioning as a second listener that can compare what was said against what the transcript claims?

## Why Now

Open-source audio-native multimodal models now exist that can run locally:

- **Phi-4 Multimodal** (Microsoft, 5.6B params) — processes text, vision, and audio in a single neural network. Beats Whisper v3 on ASR benchmarks (6.14% WER). Open weights, runs on consumer GPU. 40s audio limit per query, 30min for summarization. Designed for edge deployment.
- **Qwen2-Audio** (Alibaba, 7B params) — accepts audio + text input, generates text output. Supports voice chat and audio analysis modes. Open source via HuggingFace. Best with clips under 30s.
- **Ultravox** (Fixie AI, various sizes) — extends any open-weight LLM with a multimodal audio projector. No separate ASR stage. Trained on Llama, Gemma, Qwen, GLM backbones. v0.7 beats GPT-4o on audio reasoning. Trainable on custom data.

All run locally. No cloud dependency. Compatible with TMAS privacy constraints.

## Experiment Progression

Start simple. Observe before testing hypotheses. Each level builds on what's learned from the previous one.

---

### Experiment 0: "What do you hear?"

**Goal:** Establish the floor — what does the model perceive with zero guidance?

**Method:** Send raw audio segments to the model with an open-ended prompt. No transcript, no context, no leading questions.

**Prompt:** "Listen to this audio. What do you hear? Describe what's happening."

**Segments to test (3):**
1. One clean segment where Whisper transcription is validated as correct
2. One segment that's a known mess (hallucination, garbled speech, etc.)
3. One in-between segment with minor issues

**What to observe:**
- Does it notice two speakers? Parent and child?
- Does it pick up any words, names, narrative context?
- How does it handle the quiet/distant child voice?
- Does its description correlate with audio quality?

---

### Experiment 1: "Transcribe this."

**Goal:** Direct transcription comparison against Whisper output and validated ground truth.

**Method:** Same audio segments, but now explicitly ask for a transcript.

**Prompt:** "Transcribe the speech in this audio as accurately as possible."

**What to observe:**
- Where do Whisper and the audio-native model agree vs. diverge?
- When they disagree, which is closer to validated ground truth?
- Do they fail in the same places? (If yes → genuinely hard audio. If no → architecture differences matter.)
- How does it handle Sanskrit names without domain context?

---

### Experiment 2: "Here's a transcript. Is it accurate?"

**Goal:** Test the model as a transcript reviewer — the actual pipeline-relevant use case.

**Method:** Send audio + Whisper transcript together. Ask for evaluation.

**Prompt:** "Here is an audio recording and a transcript of it. Does the transcript accurately represent what's being said? Note any discrepancies."

**Segments to test:**
- Mix of validated-correct and validated-incorrect transcripts
- Include at least one hallucinated segment

**What to observe:**
- Can it distinguish accurate transcripts from inaccurate ones? (Binary classification accuracy)
- Does it identify specific errors, or just give a vague "mostly accurate"?
- False positive rate — does it flag correct transcripts as wrong?

---

### Experiment 3: "What's wrong with this transcript?"

**Goal:** Test targeted error identification on transcripts known to have problems.

**Method:** Send audio + a transcript you know has errors. Ask it to find them.

**Prompt:** "This transcript has some errors. Listen to the audio and identify what the transcript gets wrong."

**Segments to test:** Only use segments with confirmed, categorized errors from validation notes.

**What to observe:**
- Does it catch the same errors you found manually?
- Does it catch errors you missed?
- Does it hallucinate errors that aren't there?
- Which error categories does it handle best/worst? (hallucinations, dropped speech, name confusion, speaker attribution)

---

### Experiment 4: Domain-Specific Probes

**Goal:** Test specific capabilities relevant to TMAS failure patterns.

**Sub-experiments:**

**4a — Name Recognition:**
Send segments where Whisper got Sanskrit names wrong. Ask: "What name is being spoken in this audio?" Compare against Whisper + LLM normalization pipeline.

**4b — Speaker Change Detection:**
Send segments spanning speaker transitions. Ask: "How many speakers are in this clip? When does the speaker change?" Compare against pyannote output.

**4c — Quiet Speech Recovery:**
Send segments where Arti was speaking quietly and Whisper dropped content. Ask: "Is there any speech in this audio that might be hard to hear? Transcribe everything you can detect."

**4d — Confidence Calibration:**
For the verification task (Experiment 2), ask the model to rate its confidence per segment. Cross-reference with validation notes. Does the model know when it's uncertain?

---

## Practical Notes

- **Start with Phi-4 Multimodal** — smallest footprint (5.6B), designed for edge, closest to eventual Nano deployment target
- **Segment selection matters** — pick from validated sessions where you have ground truth. Don't guess.
- **Record everything** — model outputs, prompts used, segments tested, observations. This is research data.
- **40-second limit on Phi-4** — fine for segment-level evaluation, not full recordings
- **No conclusions yet** — these experiments are for developing intuition about what these models can and can't do. Let patterns emerge.

## Success Criteria

Not "does this work?" but rather:
- Do I understand what these models can perceive?
- Are there failure patterns where they add signal beyond what Whisper + heuristics give me?
- Is transcript verification (Experiment 2) viable enough to pursue further?
- What would I need to build to integrate this into the pipeline?
