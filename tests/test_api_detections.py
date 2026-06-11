"""Tests for the detections (monitor) API routes."""

import json

import pytest

from api.app import create_app

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

DETECTIONS = {
    "_about": "test",
    "detectors": {
        "m9a-family-names": {
            "label": "Family-name mistranscription",
            "failure_mode": "M9a",
            "detector_version": "1.0.0",
            "run_at": "2026-06-11T12:00:00+00:00",
            "n_word_tokens": 4,
            "n_flags": 1,
            "flags": [{
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
            }],
        },
    },
}


@pytest.fixture
def sessions_dir(tmp_path):
    # Scanned session: transcript + detections
    scanned = tmp_path / "sessions" / "20260101-120000"
    scanned.mkdir(parents=True)
    (scanned / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))
    (scanned / "detections.json").write_text(json.dumps(DETECTIONS))

    # Unscanned session: transcript only
    unscanned = tmp_path / "sessions" / "20260102-120000"
    unscanned.mkdir()
    (unscanned / "transcript-rich.json").write_text(json.dumps(TRANSCRIPT))

    # Untranscribed session: excluded from the rollup entirely
    raw = tmp_path / "sessions" / "20260103-120000"
    raw.mkdir()

    return tmp_path / "sessions"


@pytest.fixture
def client(sessions_dir, tmp_path):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --- Rollup -------------------------------------------------------------------

def test_rollup_lists_registry_detectors(client):
    resp = client.get("/api/detections")
    assert resp.status_code == 200
    data = resp.get_json()
    ids = [d["id"] for d in data["detectors"]]
    assert "m9a-family-names" in ids
    det = data["detectors"][ids.index("m9a-family-names")]
    assert set(det) == {"id", "label", "failure_mode", "version"}


def test_rollup_distinguishes_scanned_from_unscanned(client):
    data = client.get("/api/detections").get_json()
    by_id = {s["session_id"]: s for s in data["sessions"]}
    # Untranscribed session excluded; newest first
    assert list(by_id) == ["20260102-120000", "20260101-120000"]
    assert by_id["20260101-120000"]["results"]["m9a-family-names"]["n_flags"] == 1
    assert by_id["20260102-120000"]["results"] == {}
    assert by_id["20260101-120000"]["duration_seconds"] == 60.0
    assert data["totals"]["m9a-family-names"] == 1


def test_rollup_empty_sessions_dir(tmp_path):
    app = create_app(sessions_dir=tmp_path / "empty",
                     profiles_path=str(tmp_path / "profiles.json"))
    app.config["TESTING"] = True
    with app.test_client() as c:
        data = c.get("/api/detections").get_json()
    assert data["sessions"] == []
    assert [d["id"] for d in data["detectors"]] != []  # registry still listed


def test_rollup_corrupt_detections_is_500(client, sessions_dir):
    (sessions_dir / "20260102-120000" / "detections.json").write_text("{nope")
    resp = client.get("/api/detections")
    assert resp.status_code == 500
    assert "Corrupt detections.json" in resp.get_json()["error"]


# --- Per-session detail ---------------------------------------------------------

def test_session_detections_bad_id(client):
    assert client.get("/api/sessions/not-a-session/detections").status_code == 400


def test_session_detections_unknown_session(client):
    assert client.get("/api/sessions/20269999-000000/detections").status_code == 404


def test_session_detections_never_scanned(client):
    resp = client.get("/api/sessions/20260102-120000/detections")
    assert resp.status_code == 200
    assert resp.get_json() == {"session_id": "20260102-120000", "detectors": {}}


def test_session_detections_joins_segment_context(client):
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    section = data["detectors"]["m9a-family-names"]
    assert section["n_flags"] == 1
    flag = section["flags"][0]
    assert flag["token"] == "Martha"
    assert flag["segment_text"] == "And then Martha smiled."
    assert flag["segment_start"] == 1.0
    assert flag["segment_end"] == 4.0
    assert flag["segment_speaker"] == "SPEAKER_00"


def test_session_detections_missing_transcript_joins_null(client, sessions_dir):
    (sessions_dir / "20260101-120000" / "transcript-rich.json").unlink()
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    flag = data["detectors"]["m9a-family-names"]["flags"][0]
    assert flag["segment_text"] is None
    assert flag["segment_speaker"] is None
    assert flag["token"] == "Martha"  # the flag itself is still real data
