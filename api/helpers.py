"""Shared helpers for the API — validation, path resolution, session discovery."""

import json
import re
from pathlib import Path


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


def _read_duration_seconds(session_dir: Path) -> float | None:
    """Return the recording's duration in seconds, or None if unavailable.

    None when the session was never transcribed, or its transcript predates
    the audio block. Parsing transcript-rich.json (~200-600KB) for one float
    is acceptable at this app's scale; if session counts grow, have the
    pipeline cache duration into session-metadata.json instead. A corrupt
    transcript propagates json.JSONDecodeError (fail loud).
    """
    transcript_path = session_dir / "transcript-rich.json"
    if not transcript_path.exists():
        return None
    with open(transcript_path) as f:
        data = json.load(f)
    audio = data.get("audio")
    if not isinstance(audio, dict):
        return None
    return audio.get("duration_seconds")


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
        if not validate_session_id(entry.name):
            continue

        metadata = _read_session_metadata(entry)
        sessions.append({
            "id": entry.name,
            "has_audio": next(entry.glob("audio.*"), None) is not None,
            "has_transcript": (entry / "transcript-rich.json").exists(),
            "has_diarization": (entry / "diarization.json").exists(),
            "has_embeddings": (entry / "embeddings.json").exists(),
            "has_identifications": (entry / "identifications.json").exists(),
            "note": metadata.get("note", ""),
            "validation_status": metadata.get("validationStatus", "not_started"),
            "duration_seconds": _read_duration_seconds(entry),
            "note_count": _read_note_count(entry),
        })

    sessions.sort(key=lambda s: s["id"], reverse=True)
    return sessions
