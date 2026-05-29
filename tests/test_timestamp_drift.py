"""Unit tests for the pure logic in scripts/timestamp_drift_analysis.py.

The acoustic/diarization signals and the category decision tree are the parts
most likely to break under a future edit; the full per-session run (which loads
audio) is the integration test, driven by the script itself.

scripts/ isn't on pytest's pythonpath (which is [".", "src"]), so add it here.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from timestamp_drift_analysis import (  # noqa: E402
    CHUNK_SEC,
    classify_word,
    is_filler_word,
    m10_filler_ids,
    other_failure_codes,
    session_floor,
    speaker_coverage,
    window_loudness,
)

# A 5-second db track: floor at -45 dBFS everywhere, with a loud burst
# (-15 dBFS) from 2.0s to 3.0s. One chunk = CHUNK_SEC seconds.
N = int(5.0 / CHUNK_SEC)
DB = np.full(N, -45.0)
DB[int(2.0 / CHUNK_SEC):int(3.0 / CHUNK_SEC)] = -15.0


def test_window_loudness_picks_the_loudest_chunk():
    # a window over the burst sees the loud chunk
    assert window_loudness(DB, 2.4, 2.6) == -15.0
    # a window over the floor sees only the floor
    assert window_loudness(DB, 0.2, 0.8) == -45.0
    # a window that only clips the edge of the burst still catches it (max, not mean)
    assert window_loudness(DB, 1.98, 2.04) == -15.0
    # clamps to the array rather than indexing past the end
    assert window_loudness(DB, 4.9, 99.0) == -45.0


def test_speaker_coverage_fractions():
    diar = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    ]
    # window fully inside one speaker
    by_spk, total = speaker_coverage(0.2, 0.6, diar)
    assert by_spk == {"SPEAKER_00": 1.0}
    assert total == 1.0
    # window straddling the boundary splits 50/50
    by_spk, total = speaker_coverage(0.5, 1.5, diar)
    assert round(by_spk["SPEAKER_00"], 3) == 0.5
    assert round(by_spk["SPEAKER_01"], 3) == 0.5
    assert round(total, 3) == 1.0
    # window in a true silence (no diarization) -> nobody
    assert speaker_coverage(3.0, 4.0, diar) == ({}, 0.0)
    # degenerate window
    assert speaker_coverage(2.0, 2.0, diar) == ({}, 0.0)


# threshold for these: loud >= -33 dBFS (floor -45 + margin 12)
THR = -33.0
# classify_word(loud_db, threshold_db, total_frac, loud_nearby)


def test_classify_ok_when_loud():
    # loud enough to hold a word -> it is plainly there
    v = classify_word(-15.0, THR, 0.9, loud_nearby=True)
    assert v["category"] == "ok"
    assert v["score"] == 0.0


def test_classify_drifted_when_quiet_empty_but_real_word_is_next_door():
    # quiet here, nobody here, BUT loud audio just beside it -> the clean drift case
    v = classify_word(-45.0, THR, 0.0, loud_nearby=True)
    assert v["category"] == "drifted"
    assert v["score"] >= 0.6


def test_classify_isolated_when_quiet_empty_and_nothing_loud_nearby():
    # quiet, nobody, and no loud neighbour -> no word at all (≈hallucination),
    # held OUT of the headline floor.
    v = classify_word(-45.0, THR, 0.0, loud_nearby=False)
    assert v["category"] == "isolated"
    assert 0.4 <= v["score"] <= 0.6


def test_classify_quiet_ambiguous_is_held_out_of_the_floor():
    # quiet BUT a speaker is present -> maybe real quiet speech (#13), ambiguous
    v = classify_word(-44.0, THR, 0.8, loud_nearby=False)
    assert v["category"] == "quiet_ambiguous"
    assert 0.3 <= v["score"] <= 0.5


def test_classify_loud_word_is_ok_even_with_no_diarization():
    # loud but no speaker detected: the word is plainly there -> a diarization
    # gap, not a drifted timestamp.
    v = classify_word(-12.0, THR, 0.0, loud_nearby=True)
    assert v["category"] == "ok"


def test_session_floor_is_a_margin_above_the_percentile():
    floor, thr = session_floor(DB, 20, 12.0)
    assert floor == -45.0          # p20 of a mostly -45 track
    assert thr == -33.0


def test_m10_filler_ids_folds_only_in_range_filler_segments():
    # Moon Story has a configured "Hmm." stretch over segs 99-136. Only filler
    # segments (text == "Hmm.") inside the range are folded; a real-content line
    # inside the range, and a "Hmm." just outside it, are NOT.
    transcript = {"segments": [
        {"id": 99, "text": "Hmm.", "start": 0, "end": 1},          # in range, filler -> folded
        {"id": 110, "text": "The moon ran away.", "start": 1, "end": 2},  # in range, real -> kept
        {"id": 120, "text": "Hmm.", "start": 2, "end": 3},          # in range, filler -> folded
        {"id": 62, "text": "Hmm.", "start": 3, "end": 4},           # OUT of range -> kept
    ]}
    ids = m10_filler_ids(transcript, "20251207-195607")
    assert ids == {99, 120}


def test_m10_filler_ids_empty_for_session_without_stretches():
    transcript = {"segments": [{"id": 1, "text": "Hmm.", "start": 0, "end": 1}]}
    assert m10_filler_ids(transcript, "00000000-000000") == set()


def test_other_failure_codes_ignores_nota_and_m7():
    # NotA ("fine") and M7 itself are NOT "another mode" — a segment with only
    # those stays eligible as a genuine M7 candidate.
    assert other_failure_codes(["M2"]) == ["M2"]
    assert other_failure_codes(["M4", "M9"]) == ["M4", "M9"]
    assert other_failure_codes(["NotA"]) == []
    assert other_failure_codes(["M7"]) == []
    assert other_failure_codes(["NotA", "M2"]) == ["M2"]
    assert other_failure_codes([]) == []


def test_is_filler_word_normalizes_punctuation_and_case():
    # the residuals Choksi flagged, plus the dash-prefixed "-hmm." an earlier
    # regex missed — all should read as filler (review aid; never excludes)
    for w in ["Hmm.", "-hmm.", "Mm", "Huh?", "um,", "Yeah.", "Right.", "Uh", "OH"]:
        assert is_filler_word(w), w
    # real content words must not be flagged
    for w in ["moon", "and", "the", "milk", "ducky", ""]:
        assert not is_filler_word(w), w
