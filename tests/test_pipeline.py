"""Tests for pipeline module."""

import contextlib
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import compute_file_hash, create_manifest, save_computed, run_pipeline


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


# --- run_pipeline normalization tests ---

_FAKE_TRANSCRIPT = {
    "text": "hello world",
    "segments": [{
        "text": "hello world",
        "words": [
            {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " world", "start": 0.5, "end": 1.0, "probability": 0.8},
        ]
    }]
}


def _apply_pipeline_mocks(stack, **overrides):
    """Enter common mock patches onto an ExitStack and return a dict of active mocks.

    Any key in overrides replaces the default mock for that target name
    (use the short name after 'pipeline.', e.g. llm_normalize=MagicMock(...)).
    """
    defaults = {
        "transcribe": MagicMock(return_value=copy.deepcopy(_FAKE_TRANSCRIPT)),
        "clean_transcript": MagicMock(side_effect=lambda x: x),
        "diarize": MagicMock(return_value={"segments": []}),
        "get_audio_info": MagicMock(return_value={"duration": 10.0}),
        "compute_file_hash": MagicMock(return_value="sha256:fake"),
        "os.path.exists": MagicMock(return_value=True),
        "extract_text": MagicMock(return_value="some text"),
        "load_library": MagicMock(return_value={"entries": []}),
        "build_variant_map": MagicMock(return_value={}),
        "enrich_with_diarization": MagicMock(side_effect=lambda t, d: t),
    }
    defaults.update(overrides)

    active = {}
    for short_name, mock_obj in defaults.items():
        target = f"pipeline.{short_name}"
        active[short_name] = stack.enter_context(patch(target, mock_obj))
    return active


def test_pipeline_both_normalizations_succeed():
    """Both LLM and dictionary normalization succeed and counts are recorded."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            llm_normalize=MagicMock(
                return_value=[{"transcribed": "hello", "correct": "Hello"}]
            ),
            normalize_variants=MagicMock(
                return_value=[{"transcribed": "world", "correct": "World"}]
            ),
            apply_corrections=MagicMock(
                side_effect=[
                    (copy.deepcopy(fake_transcript), 5),
                    (copy.deepcopy(fake_transcript), 3),
                ]
            ),
        )
        result = run_pipeline("/fake/session/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 4

    assert processing[0]["stage"] == "transcription"
    assert processing[0]["status"] == "success"

    assert processing[1]["stage"] == "llm_normalization"
    assert processing[1]["status"] == "success"
    assert processing[1]["corrections_applied"] == 5

    assert processing[2]["stage"] == "dictionary_normalization"
    assert processing[2]["status"] == "success"
    assert processing[2]["corrections_applied"] == 3

    assert result["transcript"]["_schema_version"] == "1.2.0"
    assert result["llm_count"] == 5
    assert result["dict_count"] == 3


def test_pipeline_llm_fails_dictionary_continues():
    """LLM normalization failure does not prevent dictionary normalization."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            llm_normalize=MagicMock(
                side_effect=RuntimeError("Ollama not running")
            ),
            normalize_variants=MagicMock(
                return_value=[{"transcribed": "x", "correct": "Y"}]
            ),
            apply_corrections=MagicMock(
                return_value=(copy.deepcopy(fake_transcript), 2)
            ),
        )
        result = run_pipeline("/fake/session/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 4

    llm_entry = processing[1]
    assert llm_entry["stage"] == "llm_normalization"
    assert llm_entry["status"] == "error"
    assert "Ollama not running" in llm_entry["error"]

    dict_entry = processing[2]
    assert dict_entry["stage"] == "dictionary_normalization"
    assert dict_entry["status"] == "success"
    assert dict_entry["corrections_applied"] == 2

    assert result["llm_count"] == 0
    assert result["dict_count"] == 2


def test_pipeline_empty_corrections():
    """Empty correction lists result in zero counts but success status."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            llm_normalize=MagicMock(return_value=[]),
            normalize_variants=MagicMock(return_value=[]),
            apply_corrections=MagicMock(
                side_effect=[
                    (copy.deepcopy(fake_transcript), 0),
                    (copy.deepcopy(fake_transcript), 0),
                ]
            ),
        )
        result = run_pipeline("/fake/session/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 4

    assert processing[1]["stage"] == "llm_normalization"
    assert processing[1]["status"] == "success"
    assert processing[1]["corrections_applied"] == 0

    assert processing[2]["stage"] == "dictionary_normalization"
    assert processing[2]["status"] == "success"
    assert processing[2]["corrections_applied"] == 0

    # Words should be unchanged
    words = result["transcript"]["segments"][0]["words"]
    assert words[0]["word"] == " hello"
    assert words[1]["word"] == " world"


def test_pipeline_diarization_enrichment_succeeds():
    """Diarization enrichment stage is recorded in processing."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            llm_normalize=MagicMock(return_value=[]),
            normalize_variants=MagicMock(return_value=[]),
            apply_corrections=MagicMock(
                side_effect=[
                    (copy.deepcopy(fake_transcript), 0),
                    (copy.deepcopy(fake_transcript), 0),
                ]
            ),
        )
        result = run_pipeline("/fake/session/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 4

    enrichment_entry = processing[3]
    assert enrichment_entry["stage"] == "diarization_enrichment"
    assert enrichment_entry["status"] == "success"
    assert enrichment_entry["model"] == "pyannote/speaker-diarization-community-1"


def test_pipeline_diarization_enrichment_fails_gracefully():
    """Diarization enrichment failure is recorded but pipeline continues."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            llm_normalize=MagicMock(return_value=[]),
            normalize_variants=MagicMock(return_value=[]),
            apply_corrections=MagicMock(
                side_effect=[
                    (copy.deepcopy(fake_transcript), 0),
                    (copy.deepcopy(fake_transcript), 0),
                ]
            ),
            enrich_with_diarization=MagicMock(
                side_effect=RuntimeError("Enrichment crashed")
            ),
        )
        result = run_pipeline("/fake/session/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 4

    enrichment_entry = processing[3]
    assert enrichment_entry["stage"] == "diarization_enrichment"
    assert enrichment_entry["status"] == "error"
    assert "Enrichment crashed" in enrichment_entry["error"]

    # Pipeline still returns valid result
    assert "transcript" in result
    assert "diarization" in result
