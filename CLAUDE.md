# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local-first pipeline for capturing bedtime stories. Records parent-child storytelling sessions, transcribes with speaker labels, and extracts story elements. Local-only by design—no cloud APIs for family content.

## Design Principles

- **Zero Latency** (Bret Victor): Immediate connection between intent and result
- **Hard Fun** (Papert): Building this is the learning
- **Calm Tech** (Weiser): Technology that enables rather than extracts
- **The Arti Test**: Would I actually use this with my daughter? If no, don't build it
- **Slow Discovery**: Capture generously, extract tentatively. Let structure emerge from use.
- **Honest Transcripts**: Mark unclear speech as `[unintelligible]` rather than deleting or inventing. Preserves truth that something was said; can be filled in later.

## Commands

```bash
# Activate environment
source venv/bin/activate

# Run full pipeline (transcribes, diarizes, saves JSON)
python src/pipeline.py sessions/<session-id>/audio.m4a

# Run individual modules
python src/transcribe.py <audio_file>
python src/diarize.py <audio_file>
python src/enrich.py sessions/<session-id>  # Re-enrich existing session

# Tests
pytest                    # All tests
pytest -m "not slow"      # Fast unit tests only
pytest -m slow            # Integration tests (requires audio files)
pytest tests/test_align.py::test_function_name  # Single test
```

## Architecture

Audio flows through stages:

1. **transcribe.py** — MLX Whisper (large model) produces word-level timestamps
   - `clean_transcript()` — removes zero-duration words and empty segments

2. **diarize.py** — Pyannote identifies speaker segments (converts to 16kHz WAV via ffmpeg)

3. **enrich.py** — Orchestrates three enrichment passes on the transcript:
   - **LLM normalization** (`normalize.py`) — Ollama/Qwen3 corrects phonetic mishearings of Sanskrit names
   - **Dictionary normalization** (`dictionary.py`) — Reference library corrects known variant spellings
   - **Diarization enrichment** (`enrichment.py`) — Adds `_speaker` labels to each word by temporal overlap
   - Corrections are applied via `corrections.py`, which preserves `_original` and `_corrections` audit trails

4. **query.py** — Read-time layer that joins transcript + diarization on demand
   - `assign_speakers()` — O(log n) bisect lookup for speaker assignment
   - `to_utterances()` — consolidates same-speaker word runs
   - Accepts filter predicates from `filters.py` (silence gap, near-zero probability, duplicates)

5. **pipeline.py** — Orchestrates all stages:
   - `run_pipeline()` — runs transcription, diarization, enrichment
   - `save_computed()` — writes JSON artifacts to `sessions/{session-id}/`

## Hallucination Handling

Two-layer approach with different purposes:

1. **Garbage removal** (in `transcribe.py`/`clean_transcript()`): Removes zero-duration words and empty segments at transcription time. These are fabrications that never existed.

2. **Filter predicates** (in `filters.py`): Query-time predicates for further refinement:
   - `silence_gap()` — no speaker + zero coverage (hallucination in silence)
   - `near_zero_probability()` — Whisper confidence essentially zero
   - `find_duplicate_segments()` — repeated text at 30-second seek boundaries
   - `min_probability()` — factory for configurable probability thresholds

Key insight: `no_speech_prob` is NOT useful for our case. It stays low during hallucination because real speech IS happening (quiet child voice) — the model just can't decode it.

## Session Directory Structure

Each session is stored at `sessions/{session-id}/`:

```
sessions/00000000-000000/
  audio.m4a
  audio-info.json
  transcript.json
  diarization.json
  validation-notes.json
  manifest.json
```

Simple type names. The folder provides session context, so IDs in filenames are redundant.

Word-level timestamps in transcript enable future caption sync (audio plays, words highlight).

## Key Details

- Requires `HF_TOKEN` env var for pyannote model access
- Pyannote struggles with soft/child speech—alignment heuristics compensate
- Test audio: `sessions/00000000-000000/audio.m4a`
- Private data in `sessions/` is gitignored

## Development Principles

From `docs/principles.md`: Simple over clever, patterns over tools, understand deeply before scaling with AI.

## Current State

- Full pipeline complete: audio → transcribe → diarize → enrich → save JSON
- 93+ automated tests (fast unit + slow integration)
- Schema version 1.2.0 with inline corrections and speaker labels at word level
- Mahabharata reference library (56 entries, variants vs. aliases distinction)
- Session initialization from inbox with duplicate detection

## Future Work

- Story element extraction (characters, worlds, plot beats)
- Searchable story bible
- Mobile browser interface for session selection with synced captions
- Hardware: ESP32 capture devices, Jetson Orin Nano deployment

## Journal

Daily build log in `journal/` tracks decisions, experiments, and learnings.

## SYNC.md Handoff Protocol

When completing a task from SYNC.md:

1. **Move** the task from "For Code" section to "From Code" section
2. **Add Outcome:** What actually happened, what was built/changed
3. **Add Surfaced:** Questions that came up, surprises, decisions you made that Desktop might want to revisit, loose threads, anything Desktop needs to know to continue intelligently

**Example:**

```markdown
## From Code (for Desktop to process)

### Completed: [Task name from original]
**Completed:** 2026-01-30

**Outcome:**
[What you built, what changed, results]

**Surfaced:**
[Questions, surprises, things Desktop should know]
```

**Do not** write to SYNC-LOG.md — Desktop manages that file.
