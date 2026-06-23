# The ingest pipeline

What happens when a new recording comes in, start to finish.

Everything runs **on-device** (Apple-Silicon / MLX). No audio or transcript leaves the machine, and only one model runs on the GPU at a time. Drop an audio file in `inbox/` and run `python src/process_inbox.py`; these stages do the rest. Every artifact lands in `sessions/<id>/`.

Orchestration lives in `src/process_inbox.py`: Phase 1 initializes all inbox files, Phase 2 runs the pipeline + embeddings + speaker ID + the monitors per session. The full transcription pipeline is `src/pipeline.py` `run_pipeline()`.

---

## 1. Intake — `init_session.py`
A recording dropped in `inbox/` gets a session folder. Duplicates are skipped by audio content hash, so the same recording can't be processed twice.
→ a session folder under `sessions/`

## 2. Transcribe — `transcribe.py` (MLX Whisper, large)
Audio becomes words, each stamped with the moment it was spoken. `clean_transcript()` drops zero-duration and garbage tokens at this step rather than inventing them away.
→ `transcript-raw.json`

## 3. Diarize — `diarize.py` (Pyannote)
Marks which stretches of audio are which voice (parent vs. child), converting to 16 kHz WAV via ffmpeg first. The diarization model is freed afterward so the next model gets a clean GPU.
→ `diarization.json`

## 4. Speaker embeddings — `embeddings.py` (Pyannote wespeaker)
Turns each speaker's voice into a 256-dimensional, L2-normalized signature, averaged per speaker. This is the basis for recognizing the same person across recordings.
→ `embeddings.json`

## 5. Enrich — `pipeline.py` `enrich_transcript()`
Five passes turn raw words into the working transcript:
1. **Label speakers** — attach who-said-it to every word by temporal overlap with diarization (`_speaker`).
2. **Mark the gaps** — where a speaker is detected but no words decoded, inject `[unintelligible]` rather than guess or delete (honest transcripts).
3. **LLM normalization** (Qwen3-8B) — correct phonetic mishearings of proper nouns. Runs in a subprocess so its GPU memory is isolated from pyannote.
4. **Reference library** — optional dictionary correction against a known-name list; off by default (content-agnostic default).
5. **Split into stories + recognize the world** (Qwen3.5-4B) — break the recording into its separate tales and name each one's world (e.g. the Mahabharata, Thomas the Tank Engine), persisted as `_stories`.
→ `transcript-rich.json` (the working transcript + the story map)

## 6. Identify speakers — `identify.py`
Matches the voice signatures against people already known to the system; recognized voices get named. Runs only when there are saved profiles to match against.
→ `identifications.json`

## 7. Monitor — `scan_session(..., run_offline=True)`
Three name checks read the finished transcript and **flag** likely name errors for review. They never edit the transcript — they only point.
- **M9a — family-name mis-transcription** (`family_names.py`): a name that sounds like a real family member, matched by sound against a private roster. Un-gated on purpose: the transcriber often garbles a family name into an ordinary-sounding word, and catching that is exactly the job.
- **M9b — inconsistent spelling** (`name_consistency.py`): the same name written several ways in one recording.
- **M9c — canon misspelling** (`story_names/`): a character from a recognized story spelled wrong (a Mahabharata hero heard a beat off). It runs the check across several shuffles of the name list and keeps only the confident, sound-matched catches.
→ `detections.json` (what the Monitor shows)

---

## Notes
- **One model at a time.** Whisper → Pyannote → Qwen3-8B → Qwen3.5 → Gemma, never two on the GPU at once. That's why the run is staged, not parallel.
- **The monitors flag, never edit.** The transcript is saved and usable *before* the flagging step runs, so a stumble there can't lose the recording (the detector step has its own error handling for exactly this).
- **Re-running is harmless** — the same audio is recognized as a duplicate and skipped.
- The data-file shapes are documented in `CLAUDE.md` ("Data File Schemas"); the detector framework in its "Failure-Mode Detection" section.
