"""Tests for process_inbox module."""

import contextlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from process_inbox import process_inbox, _print_summary


def _make_inbox(tmp_dir, filenames):
    """Create dummy files in a temp inbox directory and return the inbox Path."""
    inbox = Path(tmp_dir) / "inbox"
    inbox.mkdir()
    for name in filenames:
        (inbox / name).write_bytes(b"fake audio")
    return inbox


def _patch_dirs(stack, inbox_dir, sessions_dir):
    """Patch module-level directory constants onto an ExitStack."""
    stack.enter_context(patch("process_inbox.INBOX_DIR", inbox_dir))
    stack.enter_context(patch("process_inbox.SESSIONS_DIR", sessions_dir))


# --- process_inbox tests ---


def test_process_inbox_empty_inbox(capsys):
    """Empty inbox prints message and never calls pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, [])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_pipeline = MagicMock()
        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            process_inbox()

        output = capsys.readouterr().out
        assert "Inbox is empty" in output
        mock_pipeline.assert_not_called()


def test_process_inbox_init_and_pipeline_success(capsys):
    """Successful init and pipeline run records session as created."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        # Create the session dir that init would create
        session_dir = sessions / "20260301-120000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260301-120000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value={
            "transcript_raw": {"segments": []},
            "transcript": {"segments": []},
            "diarization": {"segments": []},
        })
        mock_save = MagicMock()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox()

        mock_pipeline.assert_called_once()
        mock_save.assert_called_once()
        output = capsys.readouterr().out
        assert "20260301-120000" in output


def test_process_inbox_init_returns_none(capsys):
    """Init returning None records file as skipped; pipeline not called."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["dup.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_init = MagicMock(return_value=None)
        mock_pipeline = MagicMock()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", MagicMock()))
            process_inbox()

        mock_pipeline.assert_not_called()
        output = capsys.readouterr().out
        assert "No new sessions to process" in output


def test_process_inbox_init_raises(capsys):
    """Init raising an exception records failure; processing continues."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["bad.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_init = MagicMock(side_effect=Exception("corrupt file"))
        mock_pipeline = MagicMock()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", MagicMock()))
            process_inbox()

        mock_pipeline.assert_not_called()
        output = capsys.readouterr().out
        assert "Init failed" in output
        assert "corrupt file" in output


def test_process_inbox_pipeline_raises(capsys):
    """Init succeeds but pipeline raises records session as failed."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260401-100000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        mock_init = MagicMock(return_value=("20260401-100000", "audio.m4a"))
        mock_pipeline = MagicMock(side_effect=RuntimeError("GPU OOM"))

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", MagicMock()))
            process_inbox()

        output = capsys.readouterr().out
        assert "Pipeline failed" in output
        assert "GPU OOM" in output


def test_process_inbox_mixed_results(capsys):
    """Multiple files: one succeeds, one skipped, one fails — all recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["a.m4a", "b.m4a", "c.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()
        session_dir = sessions / "20260501-100000"
        session_dir.mkdir()
        (session_dir / "audio.m4a").write_bytes(b"fake audio")

        def _init_side_effect(path):
            if path.name == "a.m4a":
                return ("20260501-100000", "audio.m4a")  # success
            elif path.name == "b.m4a":
                return None  # skipped (duplicate)
            else:
                raise Exception("bad file")  # failure

        mock_pipeline = MagicMock(return_value={
            "transcript_raw": {"segments": []},
            "transcript": {"segments": []},
            "diarization": {"segments": []},
        })
        mock_save = MagicMock()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", side_effect=_init_side_effect))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox()

        output = capsys.readouterr().out
        # All three categories appear in summary with correct counts
        assert "Created (1):" in output
        assert "Skipped" in output
        assert "Failed (1):" in output


# --- _print_summary tests ---


def test_print_summary_all_populated(capsys):
    """All three lists populated prints all sections."""
    _print_summary(["sess-1"], ["dup.m4a"], [("bad.m4a", "error")])
    output = capsys.readouterr().out
    assert "Created" in output
    assert "Skipped" in output
    assert "Failed" in output


def test_print_summary_all_empty(capsys):
    """Empty lists prints nothing-processed message."""
    _print_summary([], [], [])
    output = capsys.readouterr().out
    assert "(nothing processed)" in output


def test_print_summary_only_created(capsys):
    """Only created list populated prints just that section."""
    _print_summary(["sess-1"], [], [])
    output = capsys.readouterr().out
    assert "Created" in output
    assert "Skipped" not in output
    assert "Failed" not in output


def test_print_summary_only_skipped(capsys):
    """Only skipped list populated prints just that section."""
    _print_summary([], ["dup.m4a"], [])
    output = capsys.readouterr().out
    assert "Skipped" in output
    assert "Created" not in output
    assert "Failed" not in output


def test_print_summary_only_failed(capsys):
    """Only failed list populated prints just that section."""
    _print_summary([], [], [("bad.m4a", "oops")])
    output = capsys.readouterr().out
    assert "Failed" in output
    assert "Created" not in output
    assert "Skipped" not in output


def test_process_inbox_missing_inbox_dir():
    """Missing inbox directory raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = Path(tmp) / "nonexistent"
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            with pytest.raises(FileNotFoundError):
                process_inbox()


def test_process_inbox_save_computed_raises(capsys):
    """save_computed failure is recorded as a pipeline failure."""
    with tempfile.TemporaryDirectory() as tmp:
        inbox = _make_inbox(tmp, ["story.m4a"])
        sessions = Path(tmp) / "sessions"
        sessions.mkdir()

        mock_init = MagicMock(return_value=("20260601-100000", "audio.m4a"))
        mock_pipeline = MagicMock(return_value={
            "transcript_raw": {"segments": []},
            "transcript": {"segments": []},
            "diarization": {"segments": []},
        })
        mock_save = MagicMock(side_effect=OSError("disk full"))

        with contextlib.ExitStack() as stack:
            _patch_dirs(stack, inbox, sessions)
            stack.enter_context(patch("process_inbox.init_session", mock_init))
            stack.enter_context(patch("process_inbox.run_pipeline", mock_pipeline))
            stack.enter_context(patch("process_inbox.save_computed", mock_save))
            process_inbox()

        output = capsys.readouterr().out
        assert "Pipeline failed" in output
        assert "disk full" in output
