"""Unit tests for the M9b name-consistency detector (fake names only)."""

import json

import pytest

from detectors import phonetics
from detectors.name_consistency import NameConsistencyDetector


def make_session(tmp_path, words, extra_segments=None):
    """One-segment transcript built from word strings, plus optional extra segments."""
    session_dir = tmp_path / "20260101-120000"
    session_dir.mkdir(exist_ok=True)
    segments = [{
        "id": 0,
        "start": 0.0,
        "end": float(len(words)),
        "text": " ".join(w.strip() for w in words),
        "words": [
            {"word": f" {w}", "start": float(i), "end": float(i) + 0.4}
            for i, w in enumerate(words)
        ],
    }]
    if extra_segments:
        segments.extend(extra_segments)
    (session_dir / "transcript-rich.json").write_text(json.dumps({"segments": segments}))
    return session_dir


def run_on(tmp_path, words, extra_segments=None):
    det = NameConsistencyDetector()
    return det.run(make_session(tmp_path, words, extra_segments))


# --- Core behavior -------------------------------------------------------------

def test_inconsistent_cluster_all_occurrences_flagged(tmp_path):
    # "Zerk" and "Zerg" are one phonetic cluster spelled two ways → both flagged.
    result = run_on(tmp_path, ["Zerk", "met", "Zerg", "today"])
    assert len(result["flags"]) == 2
    tokens = sorted(f["token"] for f in result["flags"])
    assert tokens == ["Zerg", "Zerk"]
    flag = result["flags"][0]
    assert flag["cluster_spellings"] == ["Zerg", "Zerk"]
    assert flag["n_cluster_occurrences"] == 2


def test_consistent_name_not_flagged(tmp_path):
    # Same name spelled the same way every time → inconsistency signal absent.
    result = run_on(tmp_path, ["Bibi", "and", "Bibi", "played"])
    assert result["flags"] == []


def test_three_spellings_cluster(tmp_path):
    result = run_on(tmp_path, ["Jarko", "Jarco", "Jarkko", "raced"])
    assert len(result["flags"]) == 3
    # cluster_spellings is sorted alphabetically; all three share a phonetic code
    assert result["flags"][0]["cluster_spellings"] == ["Jarco", "Jarkko", "Jarko"]


def test_cap_gate_drops_lowercase(tmp_path):
    # A lowercase token sharing the cluster's sound is not a name candidate, so it
    # neither gets flagged nor counts toward the cluster.
    result = run_on(tmp_path, ["Zerk", "zerg", "ran"])
    assert result["flags"] == []  # only one capitalized spelling → consistent


def test_single_capitalized_name_not_flagged(tmp_path):
    result = run_on(tmp_path, ["Marco", "went", "home"])
    assert result["flags"] == []


def test_gap_segment_without_words_skipped(tmp_path):
    gap = {"id": "gap_99.0", "start": 99.0, "end": 100.0, "text": "[unintelligible]"}
    result = run_on(tmp_path, ["Zerk", "Zerg"], extra_segments=[gap])
    assert len(result["flags"]) == 2  # gap contributed nothing, no crash


def test_two_distinct_names_not_merged(tmp_path):
    # Phonetically distinct names must not cluster, even when each is consistent.
    result = run_on(tmp_path, ["Marco", "and", "Tilly", "and", "Marco"])
    assert result["flags"] == []


def test_n_word_tokens_counts_all(tmp_path):
    result = run_on(tmp_path, ["Zerk", "Zerg", "went"])
    assert result["n_word_tokens"] == 3


def test_flag_anchor_fields_present(tmp_path):
    # Shares the M9a anchor fields so the API segment-join + audio playback work.
    flag = run_on(tmp_path, ["Zerk", "Zerg"])["flags"][0]
    for k in ("segment_id", "word_index", "start", "end", "token", "cleaned",
              "dm_codes", "cluster_id", "cluster_spellings", "n_cluster_occurrences"):
        assert k in flag


# --- Contract ------------------------------------------------------------------

def test_config_fingerprint_is_none(tmp_path):
    # No external config — freshness keys on the transcript alone.
    assert NameConsistencyDetector().config_fingerprint() is None


def test_missing_transcript_fails_loud(tmp_path):
    session_dir = tmp_path / "20260101-130000"
    session_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="transcript-rich.json"):
        NameConsistencyDetector().run(session_dir)


# --- Phonetics module move did not break M9a's helpers -------------------------

def test_phonetics_helpers_intact():
    # The shared module must keep the validated behavior the M9a probe relied on:
    # "th" names carry both a theta primary and a plain-t secondary code.
    assert phonetics.codes("martha") == {"MR0", "MRT"}
    assert phonetics.codes("marta") == {"MRT"}
    assert phonetics.clean(" Marta's,") == "marta"
    assert phonetics.is_capitalized("Marta") and not phonetics.is_capitalized("marta")
