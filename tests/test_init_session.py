"""Tests for init_session module."""

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from init_session import (
    get_creation_time,
    generate_session_id,
    file_hash,
    init_session,
    main,
)


def _make_audio_file(directory, name, content=b"audio data"):
    """Create a small file in directory and return its Path."""
    path = Path(directory) / name
    path.write_bytes(content)
    return path


# --- get_creation_time tests ---


def test_get_creation_time_returns_datetime():
    """Return value is a datetime instance."""
    with tempfile.TemporaryDirectory() as tmp:
        f = _make_audio_file(tmp, "test.m4a")
        result = get_creation_time(f)
        assert isinstance(result, datetime)


# --- generate_session_id tests ---


def test_generate_session_id_format():
    """Known datetime produces expected YYYYMMDD-HHMMSS string."""
    dt = datetime(2026, 2, 15, 20, 30, 45)
    assert generate_session_id(dt) == "20260215-203045"


def test_generate_session_id_zero_pads():
    """Single-digit month, day, hour, minute, second are zero-padded."""
    dt = datetime(2026, 1, 5, 3, 7, 9)
    assert generate_session_id(dt) == "20260105-030709"


# --- file_hash tests ---


def test_file_hash_deterministic():
    """Same content in different files produces the same hash."""
    with tempfile.TemporaryDirectory() as tmp:
        f1 = _make_audio_file(tmp, "a.m4a", b"identical content")
        f2 = _make_audio_file(tmp, "b.m4a", b"identical content")
        assert file_hash(f1) == file_hash(f2)


def test_file_hash_differs_for_different_content():
    """Different content produces different hashes."""
    with tempfile.TemporaryDirectory() as tmp:
        f1 = _make_audio_file(tmp, "a.m4a", b"content one")
        f2 = _make_audio_file(tmp, "b.m4a", b"content two")
        assert file_hash(f1) != file_hash(f2)


# --- init_session tests ---


def test_init_session_creates_session_dir():
    """New file with no existing session creates dir and moves file."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        audio = _make_audio_file(inbox, "recording.m4a")

        with (
            patch("init_session.SESSIONS_DIR", sessions),
            patch("init_session.INBOX_DIR", inbox),
            patch("init_session.DUPLICATES_DIR", inbox / "duplicates"),
            patch("init_session.get_creation_time", return_value=datetime(2026, 3, 1, 14, 0, 0)),
        ):
            result = init_session(audio)

        assert result is not None
        session_id, filename = result
        assert session_id == "20260301-140000"
        assert filename == "audio.m4a"
        assert (sessions / session_id / "audio.m4a").exists()
        assert not audio.exists()  # moved out of inbox


def test_init_session_normalizes_extension():
    """Uppercase .M4A extension is lowered in destination."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        audio = _make_audio_file(inbox, "REC.M4A")

        with (
            patch("init_session.SESSIONS_DIR", sessions),
            patch("init_session.INBOX_DIR", inbox),
            patch("init_session.DUPLICATES_DIR", inbox / "duplicates"),
            patch("init_session.get_creation_time", return_value=datetime(2026, 4, 1, 10, 0, 0)),
        ):
            result = init_session(audio)

        assert result is not None
        _, filename = result
        assert filename == "audio.m4a"


def test_init_session_duplicate_detected():
    """File with matching hash in existing session is moved to duplicates."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        duplicates = inbox / "duplicates"

        # Create existing session with same content
        session_dir = sessions / "20260501-120000"
        session_dir.mkdir()
        _make_audio_file(session_dir, "audio.m4a", b"same bytes")

        # Create inbox file with same content
        audio = _make_audio_file(inbox, "copy.m4a", b"same bytes")

        with (
            patch("init_session.SESSIONS_DIR", sessions),
            patch("init_session.INBOX_DIR", inbox),
            patch("init_session.DUPLICATES_DIR", duplicates),
            patch("init_session.get_creation_time", return_value=datetime(2026, 5, 1, 12, 0, 0)),
        ):
            result = init_session(audio)

        assert result is None
        assert not audio.exists()  # removed from inbox
        assert (duplicates / "20260501-120000_copy.m4a").exists()


def test_init_session_collision_different_hash():
    """Same timestamp but different content leaves file in inbox."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        session_dir = sessions / "20260601-090000"
        session_dir.mkdir()
        _make_audio_file(session_dir, "audio.m4a", b"original")

        audio = _make_audio_file(inbox, "different.m4a", b"different content")

        with (
            patch("init_session.SESSIONS_DIR", sessions),
            patch("init_session.INBOX_DIR", inbox),
            patch("init_session.DUPLICATES_DIR", inbox / "duplicates"),
            patch("init_session.get_creation_time", return_value=datetime(2026, 6, 1, 9, 0, 0)),
        ):
            result = init_session(audio)

        assert result is None
        assert audio.exists()  # still in inbox


def test_init_session_collision_no_existing_audio():
    """Session dir exists but has no audio file — treated as collision."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        # Empty session dir
        (sessions / "20260701-100000").mkdir()
        audio = _make_audio_file(inbox, "new.m4a")

        with (
            patch("init_session.SESSIONS_DIR", sessions),
            patch("init_session.INBOX_DIR", inbox),
            patch("init_session.DUPLICATES_DIR", inbox / "duplicates"),
            patch("init_session.get_creation_time", return_value=datetime(2026, 7, 1, 10, 0, 0)),
        ):
            result = init_session(audio)

        assert result is None
        assert audio.exists()  # left in inbox


def test_init_session_move_failure_cleans_up():
    """If shutil.move raises, the new session dir is removed."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        audio = _make_audio_file(inbox, "fail.m4a")

        with (
            patch("init_session.SESSIONS_DIR", sessions),
            patch("init_session.INBOX_DIR", inbox),
            patch("init_session.DUPLICATES_DIR", inbox / "duplicates"),
            patch("init_session.get_creation_time", return_value=datetime(2026, 8, 1, 12, 0, 0)),
            patch("init_session.shutil.move", side_effect=OSError("disk full")),
        ):
            with pytest.raises(OSError, match="disk full"):
                init_session(audio)

        assert not (sessions / "20260801-120000").exists()


# --- main tests ---


def test_main_creates_inbox_if_missing(capsys):
    """Inbox directory is created when it does not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        assert not inbox.exists()

        with patch("init_session.INBOX_DIR", inbox):
            main()

        assert inbox.exists()
        output = capsys.readouterr().out
        assert "Created inbox directory" in output


def test_main_empty_inbox(capsys):
    """Empty inbox prints informational message."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()

        with patch("init_session.INBOX_DIR", inbox):
            main()

        output = capsys.readouterr().out
        assert "No audio files found" in output


def test_main_skips_unsupported_formats(capsys):
    """Non-audio files in inbox are ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        _make_audio_file(inbox, "notes.txt", b"not audio")

        with patch("init_session.INBOX_DIR", inbox):
            main()

        output = capsys.readouterr().out
        assert "No audio files found" in output


def test_main_processes_supported_files(capsys):
    """Supported audio file is processed through init_session."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "inbox"
        inbox.mkdir()
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        _make_audio_file(inbox, "story.m4a")

        with (
            patch("init_session.INBOX_DIR", inbox),
            patch("init_session.SESSIONS_DIR", sessions),
            patch("init_session.DUPLICATES_DIR", inbox / "duplicates"),
            patch("init_session.get_creation_time", return_value=datetime(2026, 9, 1, 20, 0, 0)),
        ):
            main()

        output = capsys.readouterr().out
        assert "Initialized 1 sessions" in output
        assert (sessions / "20260901-200000" / "audio.m4a").exists()
