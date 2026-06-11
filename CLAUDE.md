# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local-only pipeline for capturing bedtime stories. Records parent-child storytelling sessions, transcribes with speaker labels, and extracts story elements. Local-only by design—no cloud APIs for family content.

## Design Principles

- **Zero Latency** (Bret Victor): Immediate connection between intent and result
- **Hard Fun** (Papert): Building this is the learning
- **Calm Tech** (Weiser): Technology that enables rather than extracts
- **The Daughter Test**: Would I actually use this with my daughter? If no, don't build it
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

# Process inbox (init + pipeline + embeddings + detectors for all inbox audio)
python src/process_inbox.py

# Run failure-mode detectors (writes sessions/<id>/detections.json)
python src/detect.py                          # all sessions with a transcript
python src/detect.py <session-id> [...]       # specific sessions
python src/detect.py --detector m9a-family-names
```

## Environment

- Apple Silicon Mac (MLX models require Metal GPU)
- Python 3.14 (venv in project root)
- Node.js (for `ui/`)
- FFmpeg (audio conversion in `diarize.py`)
- `HF_TOKEN` env var for pyannote model access

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
     - **LLM normalization** (`normalize.py`) — MLX-LM/Qwen3-8B corrects phonetic mishearings of proper nouns (subprocess isolates GPU memory from pyannote to prevent OOM). Generic by default; content-specific prompts passed explicitly.
     - **Dictionary normalization** (`dictionary.py`) — Reference library corrects known variant spellings. Skipped when no library path provided (content-agnostic default).
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
9. **process_inbox.py** — Batch processor: init → pipeline → embeddings → detectors for all inbox files

## Failure-Mode Detection (Monitor)

Validated detectors run over session transcripts and emit flags — **detection only, the transcript is never modified**. This is the production ("monitor") side of the EMP's offline evals: a detector graduates here once validated there.

- **`src/detectors/`** — the framework. `base.py` has the `Detector` contract (`id`, `label`, `failure_mode`, `version`, `run(session_dir)`) and `write_detections()` (read-merge-write of `detections.json`; each detector overwrites only its own section). `__init__.py` holds the explicit `DETECTORS` registry — add new detectors there.
- **`src/detectors/family_names.py`** — first detector (`m9a-family-names`): Double Metaphone phonetic matching of word tokens against the family-name roster + exact-alias layer + capitalization gate. Ported verbatim from the sealed EMP probe `emp/src/detect_m9a.py` (1.00/1.00 on validated sessions) — don't change the matching logic without re-validating.
- **Roster:** `data/name_roster.json` — **gitignored** (real family names). Schema documented in the committed `data/name_roster.example.json` (fake names): `people` (canonicals, phonetic layer) + `aliases` (cleaned tokens, exact layer). The loader rejects aliases the phonetic layer already covers.
- **Runner:** `python src/detect.py` (see Commands) force-runs; `process_inbox.py` also runs all registered detectors on each new session.
- **Freshness:** each section stamps a `transcript_fingerprint` (sha256 of `transcript-rich.json`). The detection API routes call `ensure_fresh_detections()` (`src/detectors/base.py`) on every view — any section missing or fingerprinted against an older transcript is re-run inline (detectors are pure code, ~ms/session). Viewing the Monitor after `--re-enrich` re-scans automatically; no manual step.

## API + Frontend

- **`api/`** — Flask app serving session data, profiles, and audio on port 5002
  - `app.py` — `create_app()` factory with injectable paths for test isolation
  - `helpers.py` — `validate_session_id()`, `get_session_dir()`, `discover_sessions()`
  - `routes/sessions.py` — `GET /api/sessions`, `GET /api/sessions/:id`, `POST /api/sessions/:id/identify`
  - `routes/profiles.py` — `GET /api/profiles`, `GET /api/profiles/:id`, `POST /api/profiles`, `PUT /api/profiles/:id`, `DELETE /api/profiles/:id`, `POST /api/profiles/:id/refresh-centroid`, `DELETE /api/profiles/:id/embeddings/:session_id`
  - `routes/speakers.py` — `POST /api/sessions/:id/confirm-speakers` (batch confirm/reassign/create)
  - `routes/audio.py` — `GET /api/sessions/:id/audio` (Flask handles range requests)
  - `routes/detections.py` — `GET /api/detections` (rollup: every transcribed session × every registered detector), `GET /api/sessions/:id/detections` (flags joined server-side to transcript segments). Both auto-refresh stale/missing sections via `ensure_fresh_detections()` before responding. Registry injectable via `create_app(detectors=...)` so tests don't need the real roster.
- **`ui/`** — React + TypeScript + Vite on port 5174
  - Vite proxies `/api` → localhost:5002 (configured in `ui/vite.config.ts`)
  - Routes: `/sessions`, `/sessions/:id/speakers`, `/sessions/:id/detections`, `/monitor`, `/profiles`, `/profiles/:id`
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
  detections.json
```

Simple type names. The folder provides session context, so IDs in filenames are redundant.

Word-level timestamps in transcript enable future caption sync (audio plays, words highlight).

## Data File Schemas

The field names that matter when reading session data (the gotchas below have cost real time — `emp/src/count.py` is a worked read of all of these):

- **`axial-labels.json`** — EMP failure-mode coding; **present only on coded sessions**. Shape: `{"labels": [ {segmentId, codes, createdAt, updatedAt}, ... ]}`.
  - `codes` is a **list** of mode tags (`"M1"`–`"M10"` or `"NotA"`) — *not* a single `mode` string. A segment may carry several (e.g. `["M1","M9"]`); count each.
  - There is **no `text` and no `notes`** field here. To get a segment's words, join by `segmentId` to `transcript-rich.json`.
  - `segmentId` is usually the integer index into `transcript-rich.json`'s `segments`, but injected gap segments use a string like `"gap_782.052"` (the gap's start time).
- **`transcript-rich.json`** — enriched transcript. Shape: `{text, segments, language, audio, _processing, ...}`. `segments` is a list; each has `id` (== its list index), `start`, `end`, **`text`** (the words — they live here, not in axial-labels), `words` (word-level timestamps), and `_speaker`.
- **`transcript-raw.json`** — same shape, pre-enrichment (before normalization/gap-injection); `--re-enrich` rebuilds rich from raw.
- **`validation-notes.json`** — open-coding notes. Shape: `{"notes": [ ... ]}` — timestamped, segment-attached free-text observations (the human's prose, distinct from the `codes` in axial-labels).
- **`diarization.json`** — pyannote speaker segments (speaker label + start/end).
- **`detections.json`** — failure-mode detector output (read-only monitor; never edits the transcript). Shape: `{_about, detectors: {<detector-id>: {label, failure_mode, detector_version, run_at, transcript_fingerprint, n_word_tokens, n_flags, flags: [...]}}}`. `transcript_fingerprint` ties the section to the transcript version it scanned; the API re-runs sections whose fingerprint is stale. Each flag carries `segment_id`/`word_index` (join to `transcript-rich.json` for text), the flagged `token`, `match_type` (`phonetic`|`alias`), and `matched_canonicals`. Flag tokens can echo mis-rendered family names — the file lives only under gitignored `sessions/`.
- **`data/mahabharata.json`** — proper-noun reference. Shape: `{_version, _description, entries: [...]}`; each entry has a `canonical` spelling plus `variants` / `aliases` lists. Build a name set from canonical + variants + aliases.

**Session IDs are date-stamped (`YYYYMMDD-HHMMSS`)**, not hex — there are no hex-named session dirs. The five EMP-coded sessions are mapped to their story names at the top of `tell-me-a-story/emp/emp.md`'s "Count result" section. 

**EMP tooling and outputs live under `emp/`:** `emp/src/` holds the analysis scripts (run from the repo root, e.g. `python emp/src/count.py`); `emp/results/` holds the committed outputs (`pivot.html` and the two sweep summaries). The pivot is an **interactive triage doc** — `python emp/src/count.py --serve` serves it with editable, file-backed notes saved to `emp/results/pivot-notes.json`. Each mode carries root-cause / fix-locus / a `detection` dropdown (code-signal · code-text · llm-judge · hybrid · ears · none) / sub-types / free-text thoughts; Mode 9 also renders three derived sub-rows (M9a/M9b/M9c) from the `_m9_cases` name→case map. **Real family names stay out of this public repo:** M9a's name variants live in the gitignored `emp/results/pivot-notes.private.json`, which `count.py` reads for counting only (never rendered to HTML). Per-session visual HTMLs go to `emp/results/visuals/<id>/` (gitignored — transcript text). Portfolio write-up drafts live in `emp/writeup/` (scrubbed of family names — safe to commit), with one codified exception (2026-06-10): a draft that mirrors a case study already published on saurinchoksi.com may carry exactly what the live page carries — currently `name-detection-eval.html`, which includes the child's first name. The website page is the boundary; nothing appears in a draft that isn't already public on the site.

## Import Convention

`src/` modules use **bare imports** (e.g. `from profiles import load_profiles`, not `from src.profiles import ...`). `api/app.py` adds `src/` to `sys.path` at startup. `api/app.py` also adds PROJECT_ROOT so the `api` package is importable from any working directory. Tests also rely on this — pytest discovers `src/` via the working directory.

## Writing Tests

pytest is configured with `pythonpath = [".", "src"]` in `pyproject.toml`, so bare imports work automatically — no `sys.path` hacks needed. Flask test clients use the `create_app()` factory with `tmp_path` for full isolation.

Boilerplate for new test files:

```python
# src/ module tests — just import directly
from corrections import apply_corrections

# API tests — just import directly
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

- Pyannote struggles with soft/child speech—alignment heuristics compensate
- Test audio: `sessions/00000000-000000/audio.m4a`
- Private data in `sessions/` is gitignored. Derived JSON artifacts (transcript-rich, validation-notes, axial-labels) are mirrored to the sibling private repo `tell-me-a-story-data/` via its `sync.sh` script — audio stays local-only. A launchd job (`~/Library/LaunchAgents/com.choksi.tmas-data-sync.plist`) runs the sync daily at noon; run it on-demand with `cd ../tell-me-a-story-data && ./sync.sh`.
- Reference library: `data/mahabharata.json` (56 entries, variants vs. aliases distinction). Opt-in via `library_path` — not loaded by default.
- **Inbox:** unprocessed sessions wait in `inbox/` at the **project root** — i.e. `tell-me-a-story/inbox/` (e.g. `inbox/New Recording 60.m4a`). `init_session.py` and `process_inbox.py` read from here. This is the only TMAS inbox — **not** the workspace-level `dev/inbox/`.
- Supported audio formats: `.m4a`, `.mp3`, `.wav` (defined in `init_session.py`)

## Coding Conventions

**Fail Loud.** No silent fallbacks. Use `utt["words"]` not `utt.get("words", [])`. If assumptions break, fail immediately — don't hide bugs behind default values. (See `docs/principles.md` for full list.)

**Changelog entries** (`changelog.md`): One entry is a `## YYYY-MM-DD — Title` heading and one short plain-English paragraph. Newest first. An entry should be readable in about 15 seconds.

- *Title* — state what happened, not what area was touched ("Per-stage timing added to the pipeline", not "Pipeline statistics"). Sentence case. Plain words: no jargon, no file/function names, no ticket IDs. One headline per title — no "X + Y" pile-ups.
- *Body* — two or three sentences, one idea each. Lead with the change; add why it was needed and what it turned up only when they earn their place. Don't repeat a domain noun more than twice — vary it or restructure. Keep a real lesson, but as one plain sentence; don't force one if there isn't one.

Exact diffs live in git history; decision context lives in Linear.

## Linear Handoff Protocol

Linear handles task tracking for most TMAS work. **The EMP (Evals Mini Project) is an exception: it is doc-canonical via `/Users/choksi/dev/tell-me-a-story/emp/emp.md`** — no parallel Linear maintenance during in-progress work. Create a Linear ticket for EMP work only at ship-time, when the write-up is ready to ship.

For all other TMAS work:

**Status flow:** Backlog → In Progress → In Review → Done

**When picking up a task:**
- Move ticket to In Progress
- Read the full description — Context, Goal, Intent, Desired Result, References

**When implementation is complete — in this order:**
1. Add a comment to the ticket with:
   - **Outcome:** What was built, what changed, key results
   - **Surfaced:** Surprises, decisions made, open questions, anything worth flagging in review
2. Move ticket to **In Review**

The comment is the briefing for Choksi's review — write it first so the review has full context. A ticket moved to In Review without this comment is incomplete. **Choksi** reviews and closes the ticket; there is no Desktop hand-off. **Never** move a ticket straight from In Progress to Done — always go through In Review first.

**Ticket description format:**
```
## Context
[Situation and background]

## Goal
[What we want]

## Intent
[Direction and approach — not step-by-step]

## Desired Result
[What done looks like]

## References
[File paths, related tickets]
```
