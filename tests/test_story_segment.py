"""Tests for the segmenter's disk-free input builder.

The module split (segment_session -> load_segments / segment_segments) newly exposed
load_segments_from_list, which must replicate load_segments' meta-list exactly so the
in-memory pipeline path and the disk-based detector fallback segment identically. The
Gemma walk itself is ported verbatim from the sealed EMP probe and validated there.
"""
import pytest

from story_segment import load_segments_from_list


def test_pos_gap_flag_and_gap_before():
    segs = load_segments_from_list([
        {"id": 0, "start": 0.0, "end": 1.0, "text": " hi", "words": []},
        {"id": "gap_1.5", "start": 1.0, "end": 1.2, "text": "[unintelligible]", "words": []},
        {"id": 1, "start": 2.0, "end": 3.0, "text": " bye", "words": []},
    ])
    assert [s["pos"] for s in segs] == [0, 1, 2]
    assert [s["id"] for s in segs] == [0, "gap_1.5", 1]
    assert [s["is_gap"] for s in segs] == [False, True, False]  # string id => gap segment
    assert segs[0]["gap_before"] is None                        # first segment
    assert segs[2]["gap_before"] == pytest.approx(0.8)          # 2.0 - 1.2


def test_text_stripped_and_missing_fields_tolerated():
    segs = load_segments_from_list([{"id": 0, "text": "  spaced  "}])  # no start/end/words
    assert segs[0]["text"] == "spaced"
    assert segs[0]["start"] is None and segs[0]["gap_before"] is None
