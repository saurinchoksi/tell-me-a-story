"""Tests for corrections module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corrections import extract_text, apply_corrections


# --- extract_text tests ---


def test_extract_text_joins_words():
    """Words with leading spaces are joined and leading whitespace stripped."""
    transcript = {
        "segments": [
            {
                "words": [
                    {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": " world", "start": 0.5, "end": 1.0, "probability": 0.8},
                ]
            }
        ]
    }
    assert extract_text(transcript) == "hello world"


# --- apply_corrections tests ---


def _make_transcript(*words):
    """Helper: build a minimal transcript from word strings."""
    return {
        "segments": [
            {
                "words": [
                    {"word": w, "start": 0.0, "end": 0.5, "probability": 0.9}
                    for w in words
                ]
            }
        ]
    }


def test_apply_single_correction():
    """Single correction updates word, sets _original, records history."""
    transcript = _make_transcript(" fondos")
    corrections = [{"transcribed": "fondos", "correct": "Pandavas"}]

    result, count = apply_corrections(transcript, corrections, "llm")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Pandavas"
    assert word["_original"] == "fondos"
    assert word["_corrections"] == [
        {"stage": "llm", "from": "fondos", "to": "Pandavas"}
    ]
    assert count == 1


def test_apply_chained_corrections():
    """Second correction preserves original _original from first pass."""
    transcript = _make_transcript(" fondos")

    # Pass 1: LLM correction
    result, _ = apply_corrections(
        transcript, [{"transcribed": "fondos", "correct": "Pandvas"}], "llm"
    )

    # Pass 2: dictionary correction on the result
    result, _ = apply_corrections(
        result, [{"transcribed": "Pandvas", "correct": "Pandavas"}], "dictionary"
    )

    word = result["segments"][0]["words"][0]
    assert word["_original"] == "fondos"
    assert len(word["_corrections"]) == 2
    assert word["_corrections"][0] == {
        "stage": "llm",
        "from": "fondos",
        "to": "Pandvas",
    }
    assert word["_corrections"][1] == {
        "stage": "dictionary",
        "from": "Pandvas",
        "to": "Pandavas",
    }
    assert word["word"] == " Pandavas"


def test_apply_case_insensitive():
    """Correction matching is case-insensitive."""
    transcript = _make_transcript(" FONDOS")
    corrections = [{"transcribed": "fondos", "correct": "Pandavas"}]

    result, count = apply_corrections(transcript, corrections, "llm")

    word = result["segments"][0]["words"][0]
    assert word["word"] == " Pandavas"
    assert count == 1


def test_apply_no_match():
    """Unmatched words are left untouched with no extra keys."""
    transcript = _make_transcript(" hello")
    corrections = [{"transcribed": "fondos", "correct": "Pandavas"}]

    result, count = apply_corrections(transcript, corrections, "llm")

    word = result["segments"][0]["words"][0]
    assert count == 0
    assert "_original" not in word
    assert "_corrections" not in word


def test_apply_empty_corrections():
    """Empty corrections list leaves transcript unchanged."""
    transcript = _make_transcript(" hello", " world")

    result, count = apply_corrections(transcript, [], "llm")

    assert count == 0
    assert result["segments"][0]["words"][0]["word"] == " hello"
    assert result["segments"][0]["words"][1]["word"] == " world"
