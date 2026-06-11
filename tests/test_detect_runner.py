"""Tests for detections.json persistence (write_detections section merging)."""

import json

import pytest

from detectors.base import Detector, build_section, write_detections


class FakeDetector(Detector):
    id = "fake-detector"
    label = "Fake detector"
    failure_mode = "M0"
    version = "0.1.0"


RESULT = {"n_word_tokens": 100, "flags": [{"segment_id": 0, "token": "Xyz"}]}


def test_write_creates_file_with_section_shape(tmp_path):
    section = write_detections(tmp_path, FakeDetector(), RESULT)
    data = json.loads((tmp_path / "detections.json").read_text())
    assert set(data) == {"_about", "detectors"}
    assert data["detectors"]["fake-detector"] == section
    assert section["label"] == "Fake detector"
    assert section["failure_mode"] == "M0"
    assert section["detector_version"] == "0.1.0"
    assert section["n_word_tokens"] == 100
    assert section["n_flags"] == 1
    assert section["flags"] == RESULT["flags"]
    assert "run_at" in section


def test_rerun_overwrites_only_own_section(tmp_path):
    # Pre-seed a section from another detector
    (tmp_path / "detections.json").write_text(json.dumps({
        "_about": "x",
        "detectors": {"other-detector": {"n_flags": 7, "flags": []}},
    }))
    write_detections(tmp_path, FakeDetector(), RESULT)
    data = json.loads((tmp_path / "detections.json").read_text())
    assert data["detectors"]["other-detector"]["n_flags"] == 7
    assert data["detectors"]["fake-detector"]["n_flags"] == 1

    # Re-run with new results replaces only the fake-detector section
    write_detections(tmp_path, FakeDetector(), {"n_word_tokens": 100, "flags": []})
    data = json.loads((tmp_path / "detections.json").read_text())
    assert data["detectors"]["fake-detector"]["n_flags"] == 0
    assert data["detectors"]["other-detector"]["n_flags"] == 7


def test_corrupt_detections_file_fails_loud(tmp_path):
    (tmp_path / "detections.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        write_detections(tmp_path, FakeDetector(), RESULT)


def test_build_section_counts_flags():
    section = build_section(FakeDetector(), {"n_word_tokens": 5, "flags": [{}, {}]})
    assert section["n_flags"] == 2
