"""Shared helpers for the API — validation, path resolution, session discovery."""

import json
import re
from pathlib import Path

# The all-zeros sample/test fixture (sessions/00000000-000000) is a valid session ID for path
# resolution and tests, but it is not a real recording — it is excluded from the UI lists so it
# doesn't show up as a phantom "November 30, 1899" row in the Monitor / Sessions views.
FIXTURE_SESSION_ID = "00000000-000000"


def validate_session_id(session_id: str) -> bool:
    """Check that a session ID matches the YYYYMMDD-HHMMSS convention."""
    return bool(re.fullmatch(r"\d{8}-\d{6}", session_id))


def validate_path(base_dir: Path, subpath: Path) -> Path:
    """Prevent path traversal attacks."""
    full_path = (base_dir / subpath).resolve()
    if not str(full_path).startswith(str(base_dir.resolve())):
        raise ValueError("Invalid path")
    return full_path


def get_session_dir(sessions_dir: Path, session_id: str) -> Path:
    """Validate a session ID and return the session directory path.

    Raises:
        ValueError: If the session ID format is invalid or path traversal is detected.
        FileNotFoundError: If the session directory does not exist.
    """
    if not validate_session_id(session_id):
        raise ValueError(f"Invalid session ID format: {session_id}")

    validate_path(sessions_dir, Path(session_id))
    session_dir = sessions_dir / session_id

    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")

    return session_dir


def _read_session_metadata(session_dir: Path) -> dict:
    """Return the parsed session-metadata.json, or {} if no file exists.

    A missing file is a legitimate empty state. A corrupt file is not —
    json.JSONDecodeError propagates (fail loud).
    """
    metadata_path = session_dir / "session-metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as f:
        return json.load(f)


def _derive_story_label(stories: list) -> dict | None:
    """Build a glanceable summary from a transcript's `_stories` array.

    Returns {'label', 'n_stories', 'worlds', 'titles'} so a list view can
    recognize a recording by its content instead of just its date — or None
    when the session has no stories (untranscribed, or a transcript that
    predates story segmentation). `label` is a compact one-liner for tight
    spaces (the Monitor); `worlds` (distinct recognized worlds, e.g. "Thomas
    the Tank Engine") and `titles` let a roomier view render chips + a title.

    Soft `.get()` reads on purpose: this is optional display metadata over data
    that may predate a field, so absence is a legitimate empty state — not a
    pipeline assumption to fail loud on.
    """
    if not stories:
        return None

    titles = [s["title"].strip() for s in stories if isinstance(s.get("title"), str) and s["title"].strip()]
    worlds: list[str] = []
    for s in stories:
        w = (s.get("world") or "").strip()
        if w and w not in worlds:
            worlds.append(w)

    n = len(stories)
    n_world_stories = sum(1 for s in stories if (s.get("world") or "").strip())
    n_original = n - n_world_stories

    if worlds:
        label = " · ".join(worlds)
        if n_original > 0:
            label += f" + {n_original} original{'s' if n_original != 1 else ''}"
    elif titles:
        label = titles[0]
        if n > 1:
            label += f" + {n - 1} more"
    else:
        label = f"{n} stor{'ies' if n != 1 else 'y'}"

    return {"label": label, "n_stories": n, "worlds": worlds, "titles": titles}


def _read_transcript_facts(session_dir: Path) -> dict:
    """Read transcript-rich.json once for the facts the sessions list needs.

    Returns {'duration_seconds': float|None, 'failed_stages': list[str],
    'stories': dict|None}. All empty/None when the session was never
    transcribed (or the transcript predates the relevant block). Parsing
    transcript-rich.json (~200-600KB) is acceptable at this app's scale; if
    session counts grow, have the pipeline cache these into
    session-metadata.json instead. A corrupt transcript propagates
    json.JSONDecodeError (fail loud).
    """
    transcript_path = session_dir / "transcript-rich.json"
    if not transcript_path.exists():
        return {"duration_seconds": None, "failed_stages": [], "stories": None}
    with open(transcript_path) as f:
        data = json.load(f)

    audio = data.get("audio")
    duration = audio.get("duration_seconds") if isinstance(audio, dict) else None

    # The _processing stage set is not fixed across pipeline versions —
    # iterate whatever is present rather than assuming a stage list.
    processing = data.get("_processing")
    failed_stages = []
    if isinstance(processing, list):
        failed_stages = [
            entry["stage"]
            for entry in processing
            if entry.get("status") == "error"
        ]

    stories = _derive_story_label(data.get("_stories") or [])

    return {"duration_seconds": duration, "failed_stages": failed_stages, "stories": stories}


def _read_note_count(session_dir: Path) -> int:
    """Return the number of validation notes, or 0 if no file exists.

    A missing validation-notes.json is a legitimate empty state. A corrupt
    file propagates json.JSONDecodeError (fail loud).
    """
    notes_path = session_dir / "validation-notes.json"
    if not notes_path.exists():
        return 0
    with open(notes_path) as f:
        data = json.load(f)
    return len(data["notes"])


def discover_sessions(sessions_dir: Path) -> list[dict]:
    """Iterate session directories and report which artifacts exist.

    Returns a list sorted by session ID (newest first) with boolean flags
    for each known artifact type.
    """
    if not sessions_dir.exists():
        return []

    sessions = []
    for entry in sessions_dir.iterdir():
        if not entry.is_dir():
            continue
        if not validate_session_id(entry.name) or entry.name == FIXTURE_SESSION_ID:
            continue

        metadata = _read_session_metadata(entry)
        facts = _read_transcript_facts(entry)
        sessions.append({
            "id": entry.name,
            "has_audio": next(entry.glob("audio.*"), None) is not None,
            "has_transcript": (entry / "transcript-rich.json").exists(),
            "has_diarization": (entry / "diarization.json").exists(),
            "has_embeddings": (entry / "embeddings.json").exists(),
            "has_identifications": (entry / "identifications.json").exists(),
            "note": metadata.get("note", ""),
            "validation_status": metadata.get("validationStatus", "not_started"),
            "duration_seconds": facts["duration_seconds"],
            "note_count": _read_note_count(entry),
            "failed_stages": facts["failed_stages"],
            "stories": facts["stories"],
        })

    sessions.sort(key=lambda s: s["id"], reverse=True)
    return sessions
