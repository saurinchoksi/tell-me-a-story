"""Tests for API profile endpoints."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.app import create_app


@pytest.fixture
def client(tmp_path):
    """Flask test client with isolated profiles path."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=sessions_dir, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def profiles_path(tmp_path):
    """Return the profiles path for direct file inspection."""
    return str(tmp_path / "profiles.json")


# --- GET /api/profiles ---

def test_list_profiles_empty(client):
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["profiles"] == []


def test_list_profiles_strips_vectors(tmp_path):
    """Embedding vectors should be replaced with counts in list view."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(json.dumps({"profiles": [{
        "id": "spk_abc123",
        "name": "Saurin",
        "role": "parent",
        "created": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-01T00:00:00+00:00",
        "embeddings": [
            {"vector": [0.1] * 256, "session_id": "s1"},
            {"vector": [0.2] * 256, "session_id": "s2"},
        ],
        "centroid": [0.15] * 256,
        "voice_variants": [{"id": "var_abc"}],
    }]}))

    app = create_app(sessions_dir=sessions_dir, profiles_path=str(profiles_path))
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/api/profiles")
        profile = resp.get_json()["profiles"][0]
        # Vectors replaced with count
        assert profile["embeddings"] == 2
        assert profile["voice_variants"] == 1
        # Centroid stripped entirely
        assert "centroid" not in profile


# --- POST /api/profiles ---

def test_create_profile(client):
    resp = client.post("/api/profiles", json={"name": "Arti", "role": "child"})
    assert resp.status_code == 201
    data = resp.get_json()
    assert "profile_id" in data
    assert data["profile_id"].startswith("spk_")


def test_create_profile_persists(client):
    client.post("/api/profiles", json={"name": "Arti", "role": "child"})
    resp = client.get("/api/profiles")
    profiles = resp.get_json()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["name"] == "Arti"


def test_create_profile_missing_fields(client):
    resp = client.post("/api/profiles", json={"name": "Arti"})
    assert resp.status_code == 400

    resp = client.post("/api/profiles", json={"role": "child"})
    assert resp.status_code == 400


def test_create_profile_no_body(client):
    resp = client.post("/api/profiles", content_type="application/json")
    assert resp.status_code == 400


# --- PUT /api/profiles/:id ---

def test_update_profile_name(client):
    resp = client.post("/api/profiles", json={"name": "Arti", "role": "child"})
    profile_id = resp.get_json()["profile_id"]

    resp = client.put(f"/api/profiles/{profile_id}", json={"name": "Artika"})
    assert resp.status_code == 200

    resp = client.get("/api/profiles")
    assert resp.get_json()["profiles"][0]["name"] == "Artika"


def test_update_profile_not_found(client):
    resp = client.put("/api/profiles/spk_nonexistent", json={"name": "X"})
    assert resp.status_code == 404


def test_update_profile_no_fields(client):
    resp = client.post("/api/profiles", json={"name": "Arti", "role": "child"})
    profile_id = resp.get_json()["profile_id"]

    resp = client.put(f"/api/profiles/{profile_id}", json={})
    assert resp.status_code == 400
