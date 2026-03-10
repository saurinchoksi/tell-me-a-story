"""Tests for API session endpoints."""

import json

import pytest

from api.app import create_app


@pytest.fixture
def sessions_dir(tmp_path):
    """Create a temp sessions directory with two session fixtures."""
    # Session with full artifacts
    s1 = tmp_path / "20260101-120000"
    s1.mkdir()
    (s1 / "audio.m4a").write_bytes(b"fake audio data")
    (s1 / "transcript-rich.json").write_text(json.dumps({
        "segments": [{"text": "hello"}]
    }))
    (s1 / "diarization.json").write_text(json.dumps({
        "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]
    }))
    (s1 / "embeddings.json").write_text(json.dumps({
        "speakers": {
            "SPEAKER_00": {"vector": [0.1] * 256, "num_segments": 3}
        }
    }))

    # Session with only audio (no pipeline artifacts yet)
    s2 = tmp_path / "20260102-180000"
    s2.mkdir()
    (s2 / "audio.m4a").write_bytes(b"fake audio 2")

    return tmp_path


@pytest.fixture
def client(sessions_dir, tmp_path):
    """Flask test client with isolated paths."""
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# --- GET /api/sessions ---

def test_list_sessions(client):
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sessions" in data
    assert len(data["sessions"]) == 2


def test_list_sessions_sorted_newest_first(client):
    resp = client.get("/api/sessions")
    ids = [s["id"] for s in resp.get_json()["sessions"]]
    assert ids == ["20260102-180000", "20260101-120000"]


def test_list_sessions_artifact_flags(client):
    resp = client.get("/api/sessions")
    sessions = {s["id"]: s for s in resp.get_json()["sessions"]}

    full = sessions["20260101-120000"]
    assert full["has_audio"] is True
    assert full["has_transcript"] is True
    assert full["has_diarization"] is True
    assert full["has_embeddings"] is True
    assert full["has_identifications"] is False

    minimal = sessions["20260102-180000"]
    assert minimal["has_audio"] is True
    assert minimal["has_transcript"] is False


def test_list_sessions_empty_dir(tmp_path):
    app = create_app(sessions_dir=tmp_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/api/sessions")
        assert resp.get_json()["sessions"] == []


# --- GET /api/sessions/:id ---

def test_get_session_full(client):
    resp = client.get("/api/sessions/20260101-120000")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["id"] == "20260101-120000"
    assert data["has_audio"] is True
    assert data["transcript"] is not None
    assert data["diarization"] is not None
    assert data["embeddings"] is not None
    assert data["identifications"] is None


def test_get_session_minimal(client):
    resp = client.get("/api/sessions/20260102-180000")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["transcript"] is None
    assert data["diarization"] is None


def test_get_session_not_found(client):
    resp = client.get("/api/sessions/20260199-000000")
    assert resp.status_code == 404


def test_get_session_bad_format(client):
    resp = client.get("/api/sessions/not-a-session")
    assert resp.status_code == 400


def test_get_session_path_traversal(client):
    # Flask normalizes ../ before routing, so this may 404 (no matching route)
    # or 400 (regex rejects the ID). Either way, traversal is blocked.
    resp = client.get("/api/sessions/..%2F..%2Fetc")
    assert resp.status_code in (400, 404)


# --- POST /api/sessions/:id/identify ---

def test_identify_no_embeddings(client):
    """Session without embeddings should return 400."""
    resp = client.post("/api/sessions/20260102-180000/identify")
    assert resp.status_code == 400
    assert "embeddings" in resp.get_json()["error"].lower()


def test_identify_session_not_found(client):
    resp = client.post("/api/sessions/20260199-000000/identify")
    assert resp.status_code == 404


def test_identify_cold_start(client):
    """With embeddings but no profiles, all speakers should be unknown."""
    resp = client.post("/api/sessions/20260101-120000/identify")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["session_id"] == "20260101-120000"
    assert data["profiles_used"] == 0
    assert len(data["identifications"]) == 1
    assert data["identifications"][0]["status"] == "unknown"


def test_identify_writes_file(client, sessions_dir):
    """Identify should persist results to identifications.json."""
    session_dir = sessions_dir / "20260101-120000"
    assert not (session_dir / "identifications.json").exists()

    client.post("/api/sessions/20260101-120000/identify")

    assert (session_dir / "identifications.json").exists()
    with open(session_dir / "identifications.json") as f:
        saved = json.load(f)
    assert saved["session_id"] == "20260101-120000"
