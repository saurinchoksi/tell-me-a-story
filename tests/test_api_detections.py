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
    (scanned / "audio.m4a").write_bytes(b"fake audio")  # drives has_audio=True

    # Unscanned session: transcript only, no audio — auto-scanned on first view
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


def test_rollup_is_read_only(client, fake_detector):
    data = client.get("/api/detections").get_json()
    by_id = {s["session_id"]: s for s in data["sessions"]}
    # Untranscribed session excluded; newest first
    assert list(by_id) == ["20260102-120000", "20260101-120000"]
    assert fake_detector.run_count == 0                  # viewing never runs a detector
    assert by_id["20260101-120000"]["results"]["fake-detector"]["n_flags"] == 1  # canned
    assert by_id["20260102-120000"]["results"] == {}     # unscanned: empty, not auto-run
    assert by_id["20260101-120000"]["duration_seconds"] == 60.0
    assert by_id["20260101-120000"]["stale"] is False
    assert data["totals"]["fake-detector"] == 1


def test_rollup_second_view_runs_nothing(client, fake_detector):
    client.get("/api/detections")
    runs_after_first = fake_detector.run_count
    client.get("/api/detections")
    assert fake_detector.run_count == runs_after_first


def test_rollup_marks_stale_on_transcript_change(client, fake_detector, sessions_dir):
    (sessions_dir / "20260101-120000" / "transcript-rich.json").write_text(
        json.dumps({**TRANSCRIPT, "_changed": True}))
    data = client.get("/api/detections").get_json()
    assert fake_detector.run_count == 0     # surfaced, not recomputed on view
    by_id = {s["session_id"]: s for s in data["sessions"]}
    assert by_id["20260101-120000"]["stale"] is True


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


# Detector failures now surface during a SCAN (POST), never a view (GET).

def test_scan_all_detector_failure_is_legible_500(sessions_dir, tmp_path):
    resp = _broken_client(sessions_dir, tmp_path).post("/api/detections/scan")
    assert resp.status_code == 500
    error = resp.get_json()["error"]
    assert "Scan failed on 20260102-120000" in error and "roster not found" in error


def test_scan_one_detector_failure_is_legible_500(sessions_dir, tmp_path):
    resp = _broken_client(sessions_dir, tmp_path).post(
        "/api/sessions/20260102-120000/detections/scan")
    assert resp.status_code == 500
    assert "roster not found" in resp.get_json()["error"]


# --- Scan (POST) --------------------------------------------------------------

def test_scan_one_runs_and_returns_detail(client, fake_detector):
    resp = client.post("/api/sessions/20260102-120000/detections/scan")
    assert resp.status_code == 200
    assert fake_detector.run_count == 1
    flag = resp.get_json()["detectors"]["fake-detector"]["flags"][0]
    assert flag["token"] == "Marda"                            # fresh run
    assert flag["segment_text"] == "And then Martha smiled."   # joined


def test_scan_one_force_reruns_fresh_session(client, fake_detector):
    client.post("/api/sessions/20260101-120000/detections/scan")
    assert fake_detector.run_count == 1   # force=True re-runs even the fresh canned session


def test_scan_all_runs_missing_skips_fresh(client, fake_detector):
    client.post("/api/detections/scan")
    # unscanned session ran; the canned (fresh) one was skipped
    assert fake_detector.run_count == 1


def test_scan_one_no_transcript_400(client):
    resp = client.post("/api/sessions/20260103-120000/detections/scan")
    assert resp.status_code == 400


# --- Per-session detail ---------------------------------------------------------

def test_session_detections_bad_id(client):
    assert client.get("/api/sessions/not-a-session/detections").status_code == 400


def test_session_detections_unknown_session(client):
    assert client.get("/api/sessions/20269999-000000/detections").status_code == 404


def test_session_detections_unscanned_reads_empty(client, fake_detector):
    resp = client.get("/api/sessions/20260102-120000/detections")
    assert resp.status_code == 200
    assert fake_detector.run_count == 0                       # viewing never runs
    assert resp.get_json()["detectors"] == {}                # never scanned → empty


def test_session_detections_reads_canned_with_stale_flag(client, fake_detector):
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    assert fake_detector.run_count == 0
    section = data["detectors"]["fake-detector"]
    assert section["run_at"] == "2026-06-11T12:00:00+00:00"   # canned, read as-is
    assert section["flags"][0]["token"] == "Martha"
    assert section["stale"] is False


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
    assert resp.get_json() == {"session_id": "20260103-120000",
                               "has_audio": False, "detectors": {}}


def test_session_detections_reports_has_audio(client):
    # 20260101-120000 has an audio.m4a fixture; 20260102-120000 does not
    with_audio = client.get("/api/sessions/20260101-120000/detections").get_json()
    without_audio = client.get("/api/sessions/20260102-120000/detections").get_json()
    assert with_audio["has_audio"] is True
    assert without_audio["has_audio"] is False


def test_session_detections_missing_transcript_joins_null(client, sessions_dir,
                                                          fake_detector):
    (sessions_dir / "20260101-120000" / "transcript-rich.json").unlink()
    data = client.get("/api/sessions/20260101-120000/detections").get_json()
    assert fake_detector.run_count == 0        # nothing to scan — passive read
    flag = data["detectors"]["fake-detector"]["flags"][0]
    assert flag["segment_text"] is None
    assert flag["segment_speaker"] is None
    assert flag["token"] == "Martha"           # the flag itself is still real data


# --- canon de-dup (m9b defers to m9c) -----------------------------------------

def test_canon_dedup_suppresses_m9b_flag_owned_by_m9c():
    # A canon name the reader (m9c) flagged is dropped from m9b, so it shows only
    # under M9c; an improvised name m9c did not claim stays in m9b.
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 2, "flags": [
            {"cleaned": "pondavas", "token": "Pondavas", "cluster_id": "pondavas"},  # canon — m9c owns it
            {"cleaned": "jammus", "token": "Jammus", "cluster_id": "jammus"},         # improvised — m9b keeps
        ]},
        "m9c-canon": {"n_flags": 1, "flags": [
            {"cleaned": "pondavas", "wrong_cleaned": ["pondavas"], "canonical": "Pandavas"},
        ]},
    }
    _apply_canon_dedup(sections)
    m9b = sections["m9b-name-consistency"]
    assert m9b["n_flags"] == 1
    assert [f["cleaned"] for f in m9b["flags"]] == ["jammus"]


def test_canon_dedup_drops_whole_cluster_when_m9c_owns_one_token():
    # Regression for the split-name bug: M9b's unit is the CLUSTER. If M9c owns even ONE token of a
    # cluster, the whole cluster defers to M9c — so a canon name never appears half in M9b (the
    # stranded "james"/"jamis" remnant after M9c took the other James spellings).
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 4, "flags": [
            {"cleaned": "jameis", "token": "Jameis", "cluster_id": "jameis"},  # m9c owns this token
            {"cleaned": "james", "token": "James", "cluster_id": "jameis"},     # same cluster -> also defers
            {"cleaned": "jamis", "token": "Jamis", "cluster_id": "jameis"},     # same cluster -> also defers
            {"cleaned": "jiraki", "token": "Jiraki", "cluster_id": "jiraki"},   # separate cluster -> stays
        ]},
        "m9c-canon": {"n_flags": 1, "flags": [
            {"cleaned": "jameis", "wrong_cleaned": ["jameis"], "canonical": "James"},
        ]},
    }
    _apply_canon_dedup(sections)
    m9b = sections["m9b-name-consistency"]
    assert m9b["n_flags"] == 1
    assert [f["cleaned"] for f in m9b["flags"]] == ["jiraki"]  # the whole James cluster deferred to M9c


def test_canon_dedup_noop_when_m9c_found_nothing():
    # When the canon reader found no canon (e.g. it didn't recognize the world),
    # m9b keeps its catches — they are the only thing flagging those names.
    from api.routes.detections import _apply_canon_dedup
    sections = {
        "m9b-name-consistency": {"n_flags": 2, "flags": [
            {"cleaned": "bishma", "token": "Bishma"},
            {"cleaned": "garn", "token": "Garn"},
        ]},
        "m9c-canon": {"n_flags": 0, "flags": []},
    }
    _apply_canon_dedup(sections)
    assert sections["m9b-name-consistency"]["n_flags"] == 2
