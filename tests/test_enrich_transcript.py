"""Tests for enrich_transcript function (in pipeline module)."""

import copy
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import enrich_transcript


# --- Test fixtures ---

_CLEAN_TRANSCRIPT = {
    "text": "hello world",
    "segments": [{
        "text": "hello world",
        "words": [
            {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " world", "start": 0.5, "end": 1.0, "probability": 0.8},
        ],
    }],
}

_FAKE_DIARIZATION = {
    "segments": [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ],
}


# --- enrich_transcript tests ---

def test_enrich_all_stages_succeed():
    """All 4 enrichment stages succeed, producing 4 processing entries."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    llm_entry = {"stage": "llm_normalization", "model": "qwen3:8b",
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    diar_entry = {"stage": "diarization_enrichment",
                  "model": "pyannote/speaker-diarization-community-1",
                  "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0,
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)) as mock_llm, \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", side_effect=[
             (copy.deepcopy(transcript), 2),
             (copy.deepcopy(transcript), 1),
         ]), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    assert len(processing) == 4
    assert processing[0]["stage"] == "llm_normalization"
    assert processing[0]["status"] == "success"
    assert "timestamp" in processing[0]
    assert processing[0]["corrections_applied"] == 2
    assert processing[1]["stage"] == "dictionary_normalization"
    assert processing[1]["status"] == "success"
    assert "timestamp" in processing[1]
    assert processing[2]["stage"] == "diarization_enrichment"
    assert processing[2]["status"] == "success"
    assert "timestamp" in processing[2]
    assert processing[3]["stage"] == "gap_detection"
    assert processing[3]["status"] == "success"
    assert counts["llm_count"] == 2
    assert counts["dict_count"] == 1


def test_enrich_llm_fails_others_continue():
    """LLM failure is recorded but dict + diarization + gap detection still run."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    diar_entry = {"stage": "diarization_enrichment",
                  "model": "pyannote/speaker-diarization-community-1",
                  "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0,
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}

    with patch("pipeline.llm_normalize", side_effect=RuntimeError("Ollama down")), \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(
             copy.deepcopy(transcript), 3,
         )), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    assert len(processing) == 4
    assert processing[0]["stage"] == "llm_normalization"
    assert processing[0]["status"] == "error"
    assert "Ollama down" in processing[0]["error"]
    assert "timestamp" in processing[0]
    assert processing[1]["stage"] == "dictionary_normalization"
    assert processing[1]["status"] == "success"
    assert "timestamp" in processing[1]
    assert processing[2]["stage"] == "diarization_enrichment"
    assert processing[2]["status"] == "success"
    assert "timestamp" in processing[2]
    assert processing[3]["stage"] == "gap_detection"
    assert processing[3]["status"] == "success"
    assert counts["llm_count"] == 0
    assert counts["dict_count"] == 3


def test_enrich_does_not_set_processing_on_transcript():
    """enrich_transcript does not set _processing on the returned transcript."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    llm_entry = {"stage": "llm_normalization", "model": "qwen3:8b",
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    diar_entry = {"stage": "diarization_enrichment",
                  "model": "pyannote/speaker-diarization-community-1",
                  "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0,
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)), \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", side_effect=[
             (copy.deepcopy(transcript), 0),
             (copy.deepcopy(transcript), 0),
         ]), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    assert "_processing" not in result
