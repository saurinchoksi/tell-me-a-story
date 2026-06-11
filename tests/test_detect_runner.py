"""Tests for detections.json persistence and the freshness gate."""

import json

import pytest

from detectors.base import (
    Detector,
    build_section,
    ensure_fresh_detections,
    transcript_fingerprint,
    write_detections,
)


class FakeDetector(Detector):
    id = "fake-detector"
    label = "Fake detector"
    failure_mode = "M0"
    version = "0.1.0"

    def __init__(self):
        self.run_count = 0

    def run(self, session_dir):
        self.run_count += 1
        return {"n_word_tokens": 100, "flags": [{"segment_id": 0, "token": "Xyz"}]}


RESULT = {"n_word_tokens": 100, "flags": [{"segment_id": 0, "token": "Xyz"}]}


@pytest.fixture
def session_dir(tmp_path):
    """Session dir with a minimal transcript (fingerprint source)."""
    (tmp_path / "transcript-rich.json").write_text(json.dumps({"segments": []}))
    return tmp_path


def test_write_creates_file_with_section_shape(session_dir):
    section = write_detections(session_dir, FakeDetector(), RESULT)
    data = json.loads((session_dir / "detections.json").read_text())
    assert set(data) == {"_about", "detectors"}
    assert data["detectors"]["fake-detector"] == section
    assert section["label"] == "Fake detector"
    assert section["failure_mode"] == "M0"
    assert section["detector_version"] == "0.1.0"
    assert section["n_word_tokens"] == 100
    assert section["n_flags"] == 1
    assert section["flags"] == RESULT["flags"]
    assert "run_at" in section
    assert section["transcript_fingerprint"] == transcript_fingerprint(session_dir)
    assert section["config_fingerprint"] is None  # FakeDetector has no config


def test_write_requires_transcript(tmp_path):
    with pytest.raises(FileNotFoundError, match="transcript-rich.json"):
        write_detections(tmp_path, FakeDetector(), RESULT)


def test_rerun_overwrites_only_own_section(session_dir):
    # Pre-seed a section from another detector
    (session_dir / "detections.json").write_text(json.dumps({
        "_about": "x",
        "detectors": {"other-detector": {"n_flags": 7, "flags": []}},
    }))
    write_detections(session_dir, FakeDetector(), RESULT)
    data = json.loads((session_dir / "detections.json").read_text())
    assert data["detectors"]["other-detector"]["n_flags"] == 7
    assert data["detectors"]["fake-detector"]["n_flags"] == 1

    # Re-run with new results replaces only the fake-detector section
    write_detections(session_dir, FakeDetector(), {"n_word_tokens": 100, "flags": []})
    data = json.loads((session_dir / "detections.json").read_text())
    assert data["detectors"]["fake-detector"]["n_flags"] == 0
    assert data["detectors"]["other-detector"]["n_flags"] == 7


def test_corrupt_detections_file_fails_loud(session_dir):
    (session_dir / "detections.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        write_detections(session_dir, FakeDetector(), RESULT)


def test_build_section_counts_flags():
    section = build_section(FakeDetector(), {"n_word_tokens": 5, "flags": [{}, {}]},
                            "sha256:x", None)
    assert section["n_flags"] == 2
    assert section["transcript_fingerprint"] == "sha256:x"


def test_save_leaves_no_temp_file(session_dir):
    write_detections(session_dir, FakeDetector(), RESULT)
    assert (session_dir / "detections.json").exists()
    assert not (session_dir / "detections.json.tmp").exists()


# --- ensure_fresh_detections (the freshness gate) -------------------------------

def test_ensure_fresh_runs_when_never_scanned(session_dir):
    det = FakeDetector()
    data = ensure_fresh_detections(session_dir, [det])
    assert det.run_count == 1
    assert data["detectors"]["fake-detector"]["n_flags"] == 1
    # ...and persisted
    on_disk = json.loads((session_dir / "detections.json").read_text())
    assert on_disk["detectors"]["fake-detector"]["n_flags"] == 1


def test_ensure_fresh_skips_when_fingerprint_matches(session_dir):
    det = FakeDetector()
    first = ensure_fresh_detections(session_dir, [det])["detectors"]["fake-detector"]
    second = ensure_fresh_detections(session_dir, [det])["detectors"]["fake-detector"]
    assert det.run_count == 1          # second view did not re-run
    assert second["run_at"] == first["run_at"]


def test_ensure_fresh_reruns_when_transcript_changes(session_dir):
    det = FakeDetector()
    ensure_fresh_detections(session_dir, [det])
    old_fp = transcript_fingerprint(session_dir)

    (session_dir / "transcript-rich.json").write_text(
        json.dumps({"segments": [], "changed": True})
    )
    data = ensure_fresh_detections(session_dir, [det])
    assert det.run_count == 2
    new_fp = data["detectors"]["fake-detector"]["transcript_fingerprint"]
    assert new_fp == transcript_fingerprint(session_dir)
    assert new_fp != old_fp


def test_ensure_fresh_preserves_unregistered_sections(session_dir):
    (session_dir / "detections.json").write_text(json.dumps({
        "_about": "x",
        "detectors": {"other-detector": {"n_flags": 7, "flags": []}},
    }))
    data = ensure_fresh_detections(session_dir, [FakeDetector()])
    assert data["detectors"]["other-detector"]["n_flags"] == 7
    assert data["detectors"]["fake-detector"]["n_flags"] == 1


def test_ensure_fresh_requires_transcript(tmp_path):
    with pytest.raises(FileNotFoundError, match="transcript-rich.json"):
        ensure_fresh_detections(tmp_path, [FakeDetector()])


def test_ensure_fresh_corrupt_detections_fails_loud(session_dir):
    (session_dir / "detections.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        ensure_fresh_detections(session_dir, [FakeDetector()])


def test_ensure_fresh_reruns_when_config_changes(session_dir):
    """A detector's output is a function of (transcript, config) — a config
    edit must trigger a re-run even with the transcript unchanged."""

    class ConfigDetector(FakeDetector):
        def __init__(self):
            super().__init__()
            self.config = "v1"

        def config_fingerprint(self):
            return f"sha256:{self.config}"

    det = ConfigDetector()
    ensure_fresh_detections(session_dir, [det])
    ensure_fresh_detections(session_dir, [det])
    assert det.run_count == 1          # stable while config unchanged

    det.config = "v2"
    data = ensure_fresh_detections(session_dir, [det])
    assert det.run_count == 2
    assert data["detectors"]["fake-detector"]["config_fingerprint"] == "sha256:v2"
