"""Tests for enrich module."""

import copy
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enrich import strip_enrichments, enrich_transcript


# --- Test fixtures ---

_ENRICHED_TRANSCRIPT = {
    "_schema_version": "1.2.0",
    "_generator_version": "0.1.0",
    "_processing": [
        {"stage": "transcription", "model": "whisper", "status": "success"},
        {"stage": "llm_normalization", "model": "qwen3:8b", "status": "success"},
    ],
    "text": "hello Arjuna",
    "segments": [{
        "text": "hello Arjuna",
        "words": [
            {
                "word": " hello",
                "start": 0.0,
                "end": 0.5,
                "probability": 0.9,
            },
            {
                "word": " Arjuna",
                "start": 0.5,
                "end": 1.0,
                "probability": 0.8,
                "_original": "arjun",
                "_corrections": [
                    {"stage": "llm", "from": "arjun", "to": "Arjuna"},
                ],
                "_speaker": {"label": "SPEAKER_00", "coverage": 0.95},
            },
        ],
    }],
}

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


# --- strip_enrichments tests ---

def test_strip_reverts_corrected_words():
    """Corrected words are restored to their original form."""
    result = strip_enrichments(_ENRICHED_TRANSCRIPT)
    words = result["segments"][0]["words"]
    # " Arjuna" should revert to " arjun" (leading space + _original)
    assert words[1]["word"] == " arjun"


def test_strip_removes_enrichment_keys_from_words():
    """_speaker, _corrections, _original are removed from words."""
    result = strip_enrichments(_ENRICHED_TRANSCRIPT)
    word = result["segments"][0]["words"][1]
    assert "_speaker" not in word
    assert "_corrections" not in word
    assert "_original" not in word


def test_strip_removes_processing_and_schema_version():
    """_processing and _schema_version are removed from transcript."""
    result = strip_enrichments(_ENRICHED_TRANSCRIPT)
    assert "_processing" not in result
    assert "_schema_version" not in result


def test_strip_preserves_generator_version():
    """_generator_version is kept (set by transcription, not enrichment)."""
    result = strip_enrichments(_ENRICHED_TRANSCRIPT)
    assert result["_generator_version"] == "0.1.0"


def test_strip_preserves_uncorrected_words():
    """Words without corrections are left untouched."""
    result = strip_enrichments(_ENRICHED_TRANSCRIPT)
    words = result["segments"][0]["words"]
    assert words[0]["word"] == " hello"
    assert words[0]["start"] == 0.0


def test_strip_deep_copies():
    """Input transcript is not mutated."""
    original = copy.deepcopy(_ENRICHED_TRANSCRIPT)
    strip_enrichments(_ENRICHED_TRANSCRIPT)
    assert _ENRICHED_TRANSCRIPT == original


def test_strip_handles_clean_transcript():
    """Clean transcript with no enrichment keys passes through safely."""
    result = strip_enrichments(_CLEAN_TRANSCRIPT)
    words = result["segments"][0]["words"]
    assert words[0]["word"] == " hello"
    assert words[1]["word"] == " world"
    assert "_processing" not in result
    assert "_schema_version" not in result


# --- enrich_transcript tests ---

def test_enrich_all_stages_succeed():
    """All 3 enrichment stages succeed, producing 3 processing entries."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("enrich.llm_normalize", return_value=[]) as mock_llm, \
         patch("enrich.extract_text", return_value="hello world"), \
         patch("enrich.apply_corrections", side_effect=[
             (copy.deepcopy(transcript), 2),
             (copy.deepcopy(transcript), 1),
         ]), \
         patch("enrich.load_library", return_value={"entries": []}), \
         patch("enrich.build_variant_map", return_value={}), \
         patch("enrich.normalize_variants", return_value=[]), \
         patch("enrich.enrich_with_diarization", side_effect=lambda t, d: t):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    assert len(processing) == 3
    assert processing[0]["stage"] == "llm_normalization"
    assert processing[0]["status"] == "success"
    assert processing[1]["stage"] == "dictionary_normalization"
    assert processing[1]["status"] == "success"
    assert processing[2]["stage"] == "diarization_enrichment"
    assert processing[2]["status"] == "success"
    assert counts["llm_count"] == 2
    assert counts["dict_count"] == 1


def test_enrich_llm_fails_others_continue():
    """LLM failure is recorded but dict + diarization still run."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("enrich.llm_normalize", side_effect=RuntimeError("Ollama down")), \
         patch("enrich.extract_text", return_value="hello world"), \
         patch("enrich.apply_corrections", return_value=(
             copy.deepcopy(transcript), 3,
         )), \
         patch("enrich.load_library", return_value={"entries": []}), \
         patch("enrich.build_variant_map", return_value={}), \
         patch("enrich.normalize_variants", return_value=[]), \
         patch("enrich.enrich_with_diarization", side_effect=lambda t, d: t):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    assert len(processing) == 3
    assert processing[0]["stage"] == "llm_normalization"
    assert processing[0]["status"] == "error"
    assert "Ollama down" in processing[0]["error"]
    assert processing[1]["stage"] == "dictionary_normalization"
    assert processing[1]["status"] == "success"
    assert processing[2]["stage"] == "diarization_enrichment"
    assert processing[2]["status"] == "success"
    assert counts["llm_count"] == 0
    assert counts["dict_count"] == 3


def test_enrich_does_not_set_processing_on_transcript():
    """enrich_transcript does not set _processing on the returned transcript."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("enrich.llm_normalize", return_value=[]), \
         patch("enrich.extract_text", return_value="hello world"), \
         patch("enrich.apply_corrections", side_effect=[
             (copy.deepcopy(transcript), 0),
             (copy.deepcopy(transcript), 0),
         ]), \
         patch("enrich.load_library", return_value={"entries": []}), \
         patch("enrich.build_variant_map", return_value={}), \
         patch("enrich.normalize_variants", return_value=[]), \
         patch("enrich.enrich_with_diarization", side_effect=lambda t, d: t):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    assert "_processing" not in result
