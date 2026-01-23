# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local-first pipeline for capturing bedtime stories. Records parent-child storytelling sessions, transcribes with speaker labels, and extracts story elements. Local-only by design—no cloud APIs for family content.

## Commands

```bash
# Activate environment
source venv/bin/activate

# Run full pipeline
python src/pipeline.py stories/audio/<filename>.m4a

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

Audio flows through three stages:

1. **transcribe.py** — MLX Whisper (large model) produces word-level timestamps
2. **diarize.py** — Pyannote identifies speaker segments (converts to 16kHz WAV via ffmpeg)
3. **align.py** — Merges transcription with diarization:
   - `align_words_to_speakers()` — assigns words by midpoint timestamp
   - `group_words_by_speaker()` — combines into utterances
   - `merge_unknown_utterances()` — fills UNKNOWN gaps between same speaker
   - `assign_leading_fragments()` — assigns turn-starts to next speaker (≤0.5s gap)
   - `consolidate_utterances()` — merges consecutive same-speaker runs

**pipeline.py** orchestrates all stages and formats the final transcript.

## Key Details

- Requires `HF_TOKEN` env var for pyannote model access
- Pyannote struggles with soft/child speech—alignment heuristics compensate
- Test audio: `stories/audio/00000000-000000.m4a`
- Private data in `stories/` is gitignored

## Development Principles

From `docs/principles.md`: Simple over clever, patterns over tools, understand deeply before scaling with AI.

## Journal

Daily build log in `journal/` tracks decisions, experiments, and learnings.
