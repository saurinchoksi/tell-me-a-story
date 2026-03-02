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

# Re-enrich existing session (skips transcription/diarization)
python src/pipeline.py --re-enrich sessions/<session-id>

# Run individual modules
python src/transcribe.py <audio_file>
python src/diarize.py <audio_file>

# Tests
pytest                    # All tests
pytest -m "not slow"      # Fast unit tests only
pytest -m slow            # Integration tests (requires audio files)
pytest tests/test_align.py::test_function_name  # Single test

# Full dev environment (API on :5002 + Vite on :5174)
cd ui && npm run dev

# Or run separately:
python api/app.py         # API only (port 5002)
cd ui && npm run dev:vite  # Vite only (port 5174)

# Lint frontend
cd ui && npm run lint

# Process inbox (init + pipeline + embeddings for all inbox audio)
python src/process_inbox.py
```

## Architecture

Audio flows through stages:

1. **transcribe.py** — MLX Whisper (large model) produces word-level timestamps
   - `clean_transcript()` — removes zero-duration words and empty segments

2. **diarize.py** — Pyannote identifies speaker segments (converts to 16kHz WAV via ffmpeg). ML-only; no enrichment logic.

3. **speaker.py** — Pure data transforms: speaker labeling + gap detection. No torch/pyannote imports.

4. **pipeline.py** — Orchestrates all stages:
   - `run_pipeline()` — runs transcription, diarization, enrichment
   - `enrich_transcript()` — runs four enrichment passes:
     - **Diarization enrichment** (`speaker.py`) — Adds `_speaker` labels to each word by temporal overlap
     - **Gap detection** (`speaker.py`) — Injects `[unintelligible]` segments where speaker detected but no transcript
     - **LLM normalization** (`normalize.py`) — MLX-LM/Qwen3-8B corrects phonetic mishearings of Sanskrit names (spawned subprocess for GPU isolation)
     - **Dictionary normalization** (`dictionary.py`) — Reference library corrects known variant spellings
     - Corrections are applied via `corrections.py`, which preserves `_original` and `_corrections` audit trails
   - `save_computed()` — writes `transcript-raw.json`, `transcript-rich.json`, `diarization.json`
   - `to_utterances()` — consolidates same-speaker word runs into utterances
   - `format_transcript()` — renders utterances as `SPEAKER: text` lines
   - `--re-enrich` — re-enriches from `transcript-raw.json` without re-transcribing
   - Accepts filter predicates from `filters.py` (silence gap, near-zero probability, duplicates)

## Hallucination Handling

Two-layer approach with different purposes:

1. **Garbage removal** (in `transcribe.py`/`clean_transcript()`): Removes zero-duration words and empty segments at transcription time. These are fabrications that never existed.

2. **Filter predicates** (in `filters.py`): Query-time predicates for further refinement:
   - `silence_gap()` — no speaker + zero coverage (hallucination in silence)
   - `near_zero_probability()` — Whisper confidence essentially zero
   - `find_duplicate_segments()` — repeated text at 30-second seek boundaries

Key insight: `no_speech_prob` is NOT useful for our case. It stays low during hallucination because real speech IS happening (quiet child voice) — the model just can't decode it.

## Speaker Identification

Separate from the main pipeline, run on demand:

5. **embeddings.py** — Pyannote wespeaker model extracts 256-dim speaker vectors (L2-normalized, averaged per speaker)
6. **identify.py** — Cosine similarity matching: embeddings → profiles. Confidence tiers: ≥0.75 "identified", ≥0.45 "suggested", <0.45 "unknown"
7. **profiles.py** — Cross-session speaker identity store at `data/speaker_profiles.json`. Centroid computed from embeddings only (voice variants excluded)

Session onboarding:

8. **init_session.py** — Creates session folders from inbox audio (duplicate detection via content hash)
9. **process_inbox.py** — Batch processor: init → pipeline → embeddings for all inbox files

## API + Frontend

- **`api/`** — Flask app serving session data, profiles, and audio on port 5002
  - `app.py` — `create_app()` factory with injectable paths for test isolation
  - `helpers.py` — `validate_session_id()`, `get_session_dir()`, `discover_sessions()`
  - `routes/sessions.py` — `GET /api/sessions`, `GET /api/sessions/:id`, `POST /api/sessions/:id/identify`
  - `routes/profiles.py` — `GET /api/profiles`, `GET /api/profiles/:id`, `POST /api/profiles`, `PUT /api/profiles/:id`, `DELETE /api/profiles/:id`, `POST /api/profiles/:id/refresh-centroid`, `DELETE /api/profiles/:id/embeddings/:session_id`
  - `routes/speakers.py` — `POST /api/sessions/:id/confirm-speakers` (batch confirm/reassign/create)
  - `routes/audio.py` — `GET /api/sessions/:id/audio` (Flask handles range requests)
- **`ui/`** — React + TypeScript + Vite on port 5174
  - Vite proxies `/api` → localhost:5002 (configured in `ui/vite.config.ts`)
  - Routes: `/sessions`, `/sessions/:id/speakers`, `/profiles`, `/profiles/:id`
  - `AudioPlayer` component: play/pause, scrub, external `seekTo` prop for caption sync
  - `api/client.ts` — typed fetch wrapper matching all API endpoints
  - CSS custom properties in `App.css` establish design tokens

## Session Directory Structure

Each session is stored at `sessions/{session-id}/`:

```
sessions/00000000-000000/
  audio.m4a
  transcript-raw.json
  transcript-rich.json
  diarization.json
  embeddings.json
  identifications.json
  validation-notes.json
```

Simple type names. The folder provides session context, so IDs in filenames are redundant.

Word-level timestamps in transcript enable future caption sync (audio plays, words highlight).

## Import Convention

`src/` modules use **bare imports** (e.g. `from profiles import load_profiles`, not `from src.profiles import ...`). Both `api/app.py` and `tools/transcript_validator/server.py` add `src/` to `sys.path` at startup. `api/app.py` also adds PROJECT_ROOT so the `api` package is importable from any working directory. Tests also rely on this — pytest discovers `src/` via the working directory.

## Writing Tests

Every test file manually inserts `src/` into `sys.path` (no conftest.py magic). API tests also insert PROJECT_ROOT so the `api` package resolves. Flask test clients use the `create_app()` factory with `tmp_path` for full isolation.

Boilerplate for new test files:

```python
# src/ module tests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# API tests (need both src/ and project root)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from api.app import create_app

# Flask test client fixture
@pytest.fixture
def client(tmp_path):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=tmp_path, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
```

## Key Details

- Requires `HF_TOKEN` env var for pyannote model access
- Pyannote struggles with soft/child speech—alignment heuristics compensate
- Test audio: `sessions/00000000-000000/audio.m4a`
- Private data in `sessions/` is gitignored
- `tools/transcript_validator/` — standalone Flask app for manual transcript review with filter visualization
- Reference library: `data/mahabharata.json` (56 entries, variants vs. aliases distinction)

## Development Principles

From `docs/principles.md`: Simple over clever, patterns over tools, understand deeply before scaling with AI.

## Current State

- Full pipeline: audio → transcribe → diarize → enrich → save JSON
- Speaker identification: embeddings → profiles → cosine matching
- Flask API + React/TypeScript frontend with speaker review UI and profile gallery
- 301 automated tests (289 fast unit + 12 slow integration)
- Schema version 1.2.0 with inline corrections and speaker labels at word level
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
