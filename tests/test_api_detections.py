"""Tests for the detections (monitor) API routes.

The routes auto-refresh: viewing them runs any detector whose section is
missing or whose transcript fingerprint is stale. Tests inject a fake
detector via create_app(detectors=...) so they never need the gitignored
name roster.
"""

import json

import pytest

from api.app import create_app
from detectors.base import Detector, transcript_fingerprint

TRANSCRIPT = {
    "audio": {"duration_seconds": 60.0},
    "segments": [
        {
            "id": 0,
            "start": 1.0,
            "end": 4.0,
            "text": " And then Martha smiled.",
            "_speaker": {"label": "SPEAKER_00", "source": "dominant"},
            "words": [{"word": " And"}, {"word": " then"},
                      {"word": " Martha"}, {"word": " smiled."}],
        },
    ],
}

# Canned section pre-written for the "scanned" session. Same detector id as
# the fake so the freshness gate applies; the token differs from the fake's
# output ("Martha" canned vs "Marda" fresh) so tests can tell which served.
CANNED_FLAG = {
    "segment_id": 0,
    "word_index": 2,
    "start": 2.0,
    "end": 2.4,
    "token": "Martha",
    "cleaned": "martha",
    "dm_codes": ["MR0", "MRT"],
    "match_type": "phonetic",
    "matched_person_ids": ["child"],
    "matched_canonicals": ["Marta"],
}


class _FakeDetector(Detector):
    id = "fake-detector"
    label = "Fake detector"
    failure_mode = "M0"
    version = "0.1.0"

    def __init__(self):
        self.run_count = 0

    def run(self, session_dir):
        self.run_count += 1
        return {"n_word_tokens": 4,
                "flags": [{**CANNED_FLAG, "token": "Marda", "cleaned": "marda"}]}


def _canned_detections(session_dir):
    return {
        "_about": "test",
        "detectors": {
            "fake-detector": {
                "label": "Fake detector",
                "failure_mode": "M0",
                "detector_version": "0.1.0",
                "run_at": "2026-06-11T12:00:00+00:00",
                "transcript_fingerprint": transcript_fingerprint(session_dir),
                "config_fingerprint": None,
                "n_word_tokens": 4,
                "n_flags": 1,
                "flags": [CANNED_FLAG],
            },
        },
    }


@pytest.fixture
def fake_detector():
    return _FakeDetector()


@pytest.fixture
def sessions_dir(tmp_path):
    # Scanned session: transcript + detections with a MATCHING fingerprint
    scanned = tmp_path / "sessions" / "20260101-120000"
    scanned.mkdir(parents=True)
    (scanned / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))
    (scanned / "detections.json").write_text(json.dumps(_canned_detections(scanned)))

    # Unscanned session: transcript only — auto-scanned on first view
    unscanned = tmp_path / "sessions" / "20260102-120000"
    unscanned.mkdir()
    (unscanned / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))

    # Untranscribed session: excluded from the rollup entirely
    raw = tmp_path / "sessions" / "20260103-120000"
    raw.mkdir()

    return tmp_path / "sessions"


@pytest.fixture
def client(sessions_dir, tmp_path, fake_detector):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path,
                     detectors=[fake_detector])
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --- Rollup -------------------------------------------------------------------

def test_rollup_lists_injected_detectors(client):
    resp = client.get("/api/detections")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["detectors"] == [{"id": "fake-detector", "label": "Fake detector",
                                  "failure_mode": "M0", "version": "0.1.0"}]


def test_rollup_auto_scans_unscanned_sessions(client, fake_detector, sessions_dir):
    data = client.get("/api/detections").get_json()
    by_id = {s["session_id"]: s for s in data["sessions"]}
    # Untranscribed session excluded; newest first
    assert list(by_id) == ["20260102-120000", "20260101-120000"]
    # Only the unscanned session ran (the scanned one's fingerprint matched)
    assert fake_detector.run_count == 1
    assert by_id["20260102-120000"]["results"]["fake-detector"]["n_flags"] == 1
    assert by_id["20260101-120000"]["results"]["fake-detector"]["n_flags"] == 1
    assert by_id["20260101-120000"]["duration_seconds"] == 60.0
    assert data["totals"]["fake-detector"] == 2
    # ...and the scan was persisted
    on_disk = json.loads(
        (sessions_dir / "20260102-120000" / "detections.json").read_text()
    )
    assert on_disk["detectors"]["fake-detector"]["n_flags"] == 1


def test_rollup_second_view_runs_nothing(client, fake_detector):
    client.get("/api/detections")
    runs_after_first = fake_detector.run_count
    client.get("/api/detections")
    assert fake_detector.run_count == runs_after_first


def test_rollup_reruns_when_transcript_changes(client, fake_detector, sessions_dir):
    client.get("/api/detections")
    assert fake_detector.run_count == 1

    transcript = (sessions_dir / "20260101-120000" / "transcript-rich.json")
    transcript.write_text(json.dumps({**TRANSCRIPT, "_changed": True}))
    data = client.get("/api/detections").get_json()
    assert fake_detector.run_count == 2
    # The canned section was replaced by a fresh run
    by_id = {s["session_id"]: s for s in data["sessions"]}
    assert by_id["20260101-120000"]["results"]["fake-detector"]["run_at"] != \
        "2026-06-11T12:00:00+00:00"


def test_rollup_empty_sessions_dir(tmp_path, fake_detector):
    app = create_app(sessions_dir=tmp_path / "empty",
                     profiles_path=str(tmp_path / "profiles.json"),
                     detectors=[fake_detector])
    app.config["TESTING"] = True
    with app.test_client() as c:
        data = c.get("/api/detections").get_json()
    assert data["sessions"] == []
    assert [d["id"] for d in data["detectors"]] == ["fake-detector"]


def test_rollup_corrupt_detections_is_500(client, sessions_dir):
    (sessions_dir / "20260102-120000" / "detections.json").write_text("{nope")
    resp = client.get("/api/detections")
    assert resp.status_code == 500
    assert "Corrupt detections.json" in resp.get_json()["error"]


class _BrokenDetector(_FakeDetector):
    """Simulates detector setup failure (e.g. missing roster)."""

    def run(self, session_dir):
        raise FileNotFoundError("Family-name roster not found at data/name_roster.json.")


def _broken_client(sessions_dir, tmp_path):
    app = create_app(sessions_dir=sessions_dir,
                     profiles_path=str(tmp_path / "profiles.json"),
                     detectors=[_BrokenDetector()])
    app.config["TESTING"] = True
    return app.test_client()


def test_rollup_detector_failure_is_legible_500(sessions_dir, tmp_path):
    # The unscanned session forces a run; the broken detector raises
    resp = _broken_client(sessions_dir, tmp_path).get("/api/detections")
    assert resp.status_code == 500
    error = resp.get_json()["error"]
    assert "Detector failed on 20260102-120000" in error
    assert "roster not found" in error


def test_session_detector_failure_is_legible_500(sessions_dir, tmp_path):
    resp = _broken_client(sessions_dir, tmp_path).get(
        "/api/sessions/20260102-120000/detections")
    assert resp.status_code == 500
    assert "roster not found" in resp.get_json()["error"]


# --- Per-session detail ---------------------------------------------------------

def test_session_detections_bad_id(client):
    assert client.get("/api/sessions/not-a-session/detections").status_code == 400


def test_session_detections_unknown_session(client):
    assert client.get("/api/sessions/20269999-000000/detections").status_code == 404


def test_session_detections_auto_scans_on_view(client, fake_detector):
    resp = client.get("/api/sessions/20260102-120000/detections")
    assert resp.status_code == 200
    assert fake_detector.run_count == 1
    flag = resp.get_json()["detectors"]["fake-detector"]["flags"][0]
    assert flag["token"] == "Marda"            # fresh run, not canned
    assert flag["segment_text"] == "And then Martha smiled."   # joined


def test_session_detections_fresh_section_not_rerun(client, fake_detector):
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    assert fake_detector.run_count == 0        # fingerprint matched — no run
    section = data["detectors"]["fake-detector"]
    assert section["run_at"] == "2026-06-11T12:00:00+00:00"
    assert section["flags"][0]["token"] == "Martha"   # canned section served


def test_session_detections_joins_segment_context(client):
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    section = data["detectors"]["fake-detector"]
    assert section["n_flags"] == 1
    flag = section["flags"][0]
    assert flag["segment_text"] == "And then Martha smiled."
    assert flag["segment_start"] == 1.0
    assert flag["segment_end"] == 4.0
    assert flag["segment_speaker"] == "SPEAKER_00"


def test_session_detections_no_transcript_no_detections(client):
    resp = client.get("/api/sessions/20260103-120000/detections")
    assert resp.status_code == 200
    assert resp.get_json() == {"session_id": "20260103-120000", "detectors": {}}


def test_session_detections_missing_transcript_joins_null(client, sessions_dir,
                                                          fake_detector):
    (sessions_dir / "20260101-120000" / "transcript-rich.json").unlink()
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    assert fake_detector.run_count == 0        # nothing to scan — passive read
    flag = data["detectors"]["fake-detector"]["flags"][0]
    assert flag["segment_text"] is None
    assert flag["segment_speaker"] is None
    assert flag["token"] == "Martha"           # the flag itself is still real data
