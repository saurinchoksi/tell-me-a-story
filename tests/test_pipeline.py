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

from pipeline import compute_file_hash, save_computed, run_pipeline


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


# --- save_computed tests ---

def test_save_computed_creates_directory():
    """save_computed creates session directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        save_computed(
            session_dir=session_dir,
            transcript_raw={"text": "hello", "segments": []},
            transcript={"text": "hello", "segments": []},
            diarization={"segments": []},
        )

        assert os.path.isdir(session_dir)


def test_save_computed_creates_files():
    """save_computed creates all expected files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        save_computed(
            session_dir=session_dir,
            transcript_raw={"text": "hello", "segments": []},
            transcript={"text": "hello", "segments": []},
            diarization={"segments": []},
        )

        assert os.path.isfile(os.path.join(session_dir, "transcript-raw.json"))
        assert os.path.isfile(os.path.join(session_dir, "transcript-rich.json"))
        assert os.path.isfile(os.path.join(session_dir, "diarization.json"))


def test_save_computed_content():
    """save_computed writes correct JSON content."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_dir = os.path.join(temp_dir, "test-session")

        transcript_raw = {"text": "hello world", "segments": []}
        transcript = {"text": "hello world", "segments": [], "audio": {"duration": 10.5}}
        diarization = {"segments": [{"start": 0, "end": 1, "speaker": "A"}]}

        save_computed(
            session_dir=session_dir,
            transcript_raw=transcript_raw,
            transcript=transcript,
            diarization=diarization,
        )

        with open(os.path.join(session_dir, "transcript-raw.json")) as f:
            assert json.load(f) == transcript_raw

        with open(os.path.join(session_dir, "transcript-rich.json")) as f:
            assert json.load(f) == transcript

        with open(os.path.join(session_dir, "diarization.json")) as f:
            assert json.load(f) == diarization


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
    (use the short name after 'pipeline.', e.g. enrich_transcript=MagicMock(...)).
    """
    defaults = {
        "transcribe": MagicMock(return_value=copy.deepcopy(_FAKE_TRANSCRIPT)),
        "clean_transcript": MagicMock(side_effect=lambda x: x),
        "diarize": MagicMock(return_value={"segments": []}),
        "get_audio_info": MagicMock(return_value={"duration": 10.0}),
        "compute_file_hash": MagicMock(return_value="sha256:fake"),
        "os.path.exists": MagicMock(return_value=True),
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

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "success", "corrections_applied": 5},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 3},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"llm_count": 5, "dict_count": 3})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 5

    assert processing[0]["stage"] == "transcription"
    assert processing[0]["status"] == "success"
    assert processing[0]["audio_hash"] == "sha256:fake"
    assert "timestamp" in processing[0]

    assert processing[1]["stage"] == "diarization_enrichment"
    assert processing[1]["status"] == "success"

    assert processing[2]["stage"] == "gap_detection"
    assert processing[2]["status"] == "success"

    assert processing[3]["stage"] == "llm_normalization"
    assert processing[3]["status"] == "success"
    assert processing[3]["corrections_applied"] == 5

    assert processing[4]["stage"] == "dictionary_normalization"
    assert processing[4]["status"] == "success"
    assert processing[4]["corrections_applied"] == 3

    assert "llm_count" not in result
    assert "dict_count" not in result


def test_pipeline_llm_fails_dictionary_continues():
    """LLM normalization failure does not prevent dictionary normalization."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "error", "error": "Ollama not running"},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 2},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"llm_count": 0, "dict_count": 2})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 5

    llm_entry = processing[3]
    assert llm_entry["stage"] == "llm_normalization"
    assert llm_entry["status"] == "error"
    assert "Ollama not running" in llm_entry["error"]

    dict_entry = processing[4]
    assert dict_entry["stage"] == "dictionary_normalization"
    assert dict_entry["status"] == "success"
    assert dict_entry["corrections_applied"] == 2

    assert "llm_count" not in result
    assert "dict_count" not in result


def test_pipeline_empty_corrections():
    """Empty correction lists result in zero counts but success status."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "success", "corrections_applied": 0},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(copy.deepcopy(fake_transcript), enrichment_processing, {"llm_count": 0, "dict_count": 0})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 5

    assert processing[3]["stage"] == "llm_normalization"
    assert processing[3]["status"] == "success"
    assert processing[3]["corrections_applied"] == 0

    assert processing[4]["stage"] == "dictionary_normalization"
    assert processing[4]["status"] == "success"
    assert processing[4]["corrections_applied"] == 0

    # Words should be unchanged
    words = result["transcript"]["segments"][0]["words"]
    assert words[0]["word"] == " hello"
    assert words[1]["word"] == " world"


def test_pipeline_diarization_enrichment_succeeds():
    """Diarization enrichment stage is recorded in processing."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "success", "corrections_applied": 0},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"llm_count": 0, "dict_count": 0})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 5

    enrichment_entry = processing[1]
    assert enrichment_entry["stage"] == "diarization_enrichment"
    assert enrichment_entry["status"] == "success"
    assert enrichment_entry["model"] == "pyannote/speaker-diarization-community-1"


def test_pipeline_diarization_enrichment_fails_gracefully():
    """Diarization enrichment failure is recorded but pipeline continues."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)

    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "error", "error": "Enrichment crashed"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "success", "corrections_applied": 0},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]

    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"llm_count": 0, "dict_count": 0})
            ),
        )
        result = run_pipeline("/fake/20260101-120000/audio.m4a", verbose=False)

    processing = result["transcript"]["_processing"]
    assert len(processing) == 5

    enrichment_entry = processing[1]
    assert enrichment_entry["stage"] == "diarization_enrichment"
    assert enrichment_entry["status"] == "error"
    assert "Enrichment crashed" in enrichment_entry["error"]

    # Pipeline still returns valid result
    assert "transcript" in result
    assert "diarization" in result


# --- session_id validation tests ---

import pytest


def test_pipeline_rejects_invalid_session_id():
    """Non-YYYYMMDD-HHMMSS parent directory raises ValueError."""
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(stack)
        with pytest.raises(ValueError, match="Invalid session ID 'random-folder'"):
            run_pipeline("/fake/random-folder/audio.m4a", verbose=False)


def test_pipeline_rejects_uuid_session_id():
    """UUID-style directory name is not a valid session ID."""
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(stack)
        with pytest.raises(ValueError, match="Invalid session ID"):
            run_pipeline("/fake/a1b2c3d4-e5f6-7890-abcd-ef1234567890/audio.m4a", verbose=False)


def test_pipeline_accepts_valid_session_id():
    """Standard YYYYMMDD-HHMMSS session ID passes validation."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)
    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "success", "corrections_applied": 0},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"llm_count": 0, "dict_count": 0})
            ),
        )
        result = run_pipeline("/sessions/20260217-143000/audio.m4a", verbose=False)
    assert result["session_id"] == "20260217-143000"


def test_pipeline_accepts_zeroed_session_id():
    """Test session 00000000-000000 passes validation."""
    fake_transcript = copy.deepcopy(_FAKE_TRANSCRIPT)
    enrichment_processing = [
        {"stage": "diarization_enrichment", "model": "pyannote/speaker-diarization-community-1", "status": "success"},
        {"stage": "gap_detection", "gaps_found": 0, "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "success", "corrections_applied": 0},
        {"stage": "dictionary_normalization", "library": "data/mahabharata.json", "status": "success", "corrections_applied": 0},
    ]
    with contextlib.ExitStack() as stack:
        _apply_pipeline_mocks(
            stack,
            enrich_transcript=MagicMock(
                return_value=(fake_transcript, enrichment_processing, {"llm_count": 0, "dict_count": 0})
            ),
        )
        result = run_pipeline("/sessions/00000000-000000/audio.m4a", verbose=False)
    assert result["session_id"] == "00000000-000000"
