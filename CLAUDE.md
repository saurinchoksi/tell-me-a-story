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

# Run full pipeline
python src/pipeline.py sessions/audio/<filename>.m4a

# Run individual modules
python src/transcribe.py <audio_file>
python src/diarize.py <audio_file>

# Tests
pytest                    # All tests
pytest -m "not slow"      # Fast unit tests only
pytest -m slow            # Integration tests (requires audio files)
pytest tests/test_align.py::test_function_name  # Single test
```

## Architecture

Audio flows through stages:

1. **transcribe.py** — MLX Whisper (large model) produces word-level timestamps
   - `mark_hallucinated_segments()` — detects segments where model guessed (temperature=1.0 or compression_ratio>2.5), marks as `[unintelligible]`

2. **diarize.py** — Pyannote identifies speaker segments (converts to 16kHz WAV via ffmpeg)

3. **align.py** — Merges transcription with diarization:
   - `filter_zero_duration_words()` — removes fabricated words (end == start)
   - `filter_low_probability_words()` — removes low-confidence fabrications
   - `align_words_to_speakers()` — assigns words by midpoint timestamp
   - `group_words_by_speaker()` — combines into utterances
   - `merge_unknown_utterances()` — fills UNKNOWN gaps between same speaker
   - `assign_leading_fragments()` — assigns turn-starts to next speaker (≤0.5s gap)
   - `consolidate_utterances()` — merges consecutive same-speaker runs
   - `align()` — wrapper that runs the full alignment pipeline

**pipeline.py** orchestrates all stages and formats the final transcript.

## Hallucination Handling

Two-layer approach with different purposes:

1. **Segment filter** (in transcribe.py): Marks real-but-unclear speech as `[unintelligible]`
   - Triggers on: temperature == 1.0 OR compression_ratio > 2.5
   - Purpose: Honest metadata — speech happened, we couldn't decode it

2. **Word filters** (in align.py): Removes fabricated content entirely
   - Triggers on: zero duration OR probability < 0.5
   - Purpose: Delete lies — these words never existed

Key insight: `no_speech_prob` is NOT useful for our case. It stays low during hallucination because real speech IS happening (quiet child voice) — the model just can't decode it.

## Output Format

JSON sessions will be saved to `sessions/processed/` with structure:
- `meta`: source audio, duration, speakers detected, timestamps
- `stories[]`: array of story segments (currently always one per file)
- Each story has `utterances[]` with speaker labels and word-level timestamps

Markdown transcripts (human-readable) planned as a future addition.

## Key Details

- Requires `HF_TOKEN` env var for pyannote model access
- Pyannote struggles with soft/child speech—alignment heuristics compensate
- Test audio: `sessions/audio/00000000-000000.m4a`
- Private data in `sessions/` is gitignored

## Development Principles

From `docs/principles.md`: Simple over clever, patterns over tools, understand deeply before scaling with AI.

## Current State

- Alignment pipeline complete with hallucination filtering
- 43 fast tests + 6 slow integration tests passing
- Next: JSON saving to `sessions/processed/`
- Future: Story element extraction, searchable story bible

## Journal

Daily build log in `journal/` tracks decisions, experiments, and learnings.
