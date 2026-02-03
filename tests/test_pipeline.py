"""Tests for pipeline module."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import compute_file_hash, create_manifest, save_computed


# --- compute_file_hash tests ---

def test_compute_file_hash_returns_sha256():
    """Hash has correct prefix and length."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name

    try:
        result = compute_file_hash(temp_path)
        assert result.startswith("sha256:")
        # SHA256 hex is 64 chars + 7 for "sha256:"
        assert len(result) == 71
    finally:
        os.unlink(temp_path)


def test_compute_file_hash_deterministic():
    """Same content produces same hash."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("deterministic")
        temp_path = f.name

    try:
        hash1 = compute_file_hash(temp_path)
        hash2 = compute_file_hash(temp_path)
        assert hash1 == hash2
    finally:
        os.unlink(temp_path)


# --- create_manifest tests ---

def test_create_manifest_structure():
    """Manifest has required keys."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.m4a') as f:
        f.write("fake audio")
        temp_path = f.name

    try:
        manifest = create_manifest(
            session_id="test-session",
            audio_path=temp_path,
            transcript_model="test-model",
            transcript_time="2026-01-01T00:00:00Z",
            diarization_model="test-diarize",
            diarization_time="2026-01-01T00:01:00Z",
        )

        assert manifest["_schema_version"] == "1.0.0"
        assert manifest["session_id"] == "test-session"
        assert "created_at" in manifest
        assert "source" in manifest
        assert "computed" in manifest
    finally:
        os.unlink(temp_path)


def test_create_manifest_computed_files():
    """Manifest has correct computed file paths with simple naming."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.m4a') as f:
        f.write("fake audio")
        temp_path = f.name

    try:
        manifest = create_manifest(
            session_id="test",
            audio_path=temp_path,
            transcript_model="whisper",
            transcript_time="2026-01-01T00:00:00Z",
            diarization_model="pyannote",
            diarization_time="2026-01-01T00:01:00Z",
        )

        assert manifest["source"]["audio_file"] == "audio.m4a"
        assert manifest["computed"]["transcript"]["file"] == "transcript.json"
        assert manifest["computed"]["transcript"]["model"] == "whisper"
        assert manifest["computed"]["diarization"]["file"] == "diarization.json"
        assert manifest["computed"]["diarization"]["model"] == "pyannote"
    finally:
        os.unlink(temp_path)


# --- save_computed tests ---

def test_save_computed_creates_directory():
    """save_computed creates session directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        save_computed(
            session_dir=session_dir,
            audio_info={"duration": 10.0},
            transcript={"text": "hello", "segments": []},
            diarization={"segments": []},
            manifest={"_schema_version": "1.0.0"},
        )

        assert os.path.isdir(session_dir)


def test_save_computed_creates_files():
    """save_computed creates all expected files with simple naming."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        save_computed(
            session_dir=session_dir,
            audio_info={"duration": 10.0},
            transcript={"text": "hello", "segments": []},
            diarization={"segments": []},
            manifest={"_schema_version": "1.0.0"},
        )

        assert os.path.isfile(os.path.join(session_dir, "manifest.json"))
        assert os.path.isfile(os.path.join(session_dir, "audio-info.json"))
        assert os.path.isfile(os.path.join(session_dir, "transcript.json"))
        assert os.path.isfile(os.path.join(session_dir, "diarization.json"))


def test_save_computed_content():
    """save_computed writes correct JSON content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        audio_info = {"duration": 10.5, "sample_rate": 44100}
        transcript = {"text": "hello world", "segments": []}
        diarization = {"segments": [{"start": 0, "end": 1, "speaker": "A"}]}
        manifest = {"_schema_version": "1.0.0", "session_id": "test"}

        save_computed(
            session_dir=session_dir,
            audio_info=audio_info,
            transcript=transcript,
            diarization=diarization,
            manifest=manifest,
        )

        with open(os.path.join(session_dir, "audio-info.json")) as f:
            assert json.load(f) == audio_info

        with open(os.path.join(session_dir, "transcript.json")) as f:
            assert json.load(f) == transcript

        with open(os.path.join(session_dir, "diarization.json")) as f:
            assert json.load(f) == diarization

        with open(os.path.join(session_dir, "manifest.json")) as f:
            assert json.load(f) == manifest
