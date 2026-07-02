"""Tests for the name-correction queue API (the namefix bless loop)."""
import json

import pytest

from api.app import create_app


@pytest.fixture
def client(tmp_path):
    profiles_path = str(tmp_path / "profiles.json")
    app = create_app(sessions_dir=tmp_path, profiles_path=profiles_path)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


SID = "20260101-120000"


def make_session(tmp_path, pending=None):
    sdir = tmp_path / SID
    sdir.mkdir()
    transcript = {"text": "", "segments": [
        {"id": 0, "text": " the Bandos were good",
         "words": [
             {"word": " the", "start": 0.0, "end": 0.2, "probability": 0.9},
             {"word": " Bandos", "start": 0.2, "end": 0.6, "probability": 0.9},
             {"word": " were", "start": 0.6, "end": 0.8, "probability": 0.9},
             {"word": " good", "start": 0.8, "end": 1.0, "probability": 0.9},
         ]},
        {"id": 2, "text": " Bandos.",
         "words": [{"word": " Bandos.", "start": 2.0, "end": 2.4, "probability": 0.9}]},
    ]}
    (sdir / "transcript-rich.json").write_text(json.dumps(transcript))
    if pending is not None:
        (sdir / "pending-name-corrections.json").write_text(json.dumps(pending))
    return sdir


def pending_fixture():
    return {
        "_about": "test", "namefix_version": "1.0.0",
        "config_fingerprint": "cfg", "transcript_fingerprint": "tfp",
        "run_at": "2026-07-01T00:00:00Z",
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
        "pending": [{
            "world": "Mahabharata", "story_id": 0, "heard": "Bandos",
            "heard_cleaned": "bandos", "suggestion": "Pandavas", "canonical": "Pandavas",
            "method": "exact", "action": "queued",
            "occurrences": [
                {"segment_id": 0, "word_index": 1, "start": 0.2, "token": "Bandos"},
                {"segment_id": 2, "word_index": 0, "start": 2.0, "token": "Bandos."},
            ],
        }],
        "_rejected": [],
    }


def test_rollup_groups_by_world_and_name(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.get("/api/name-corrections")
    assert r.status_code == 200
    data = r.get_json()
    assert data["n_pending_groups"] == 1
    assert data["worlds"][0]["world"] == "Mahabharata"
    name = data["worlds"][0]["names"][0]
    assert name["heard_cleaned"] == "bandos" and name["canonical"] == "Pandavas"
    assert name["sessions"][0]["session_id"] == SID
    assert len(name["sessions"][0]["occurrences"]) == 2


def test_rollup_empty_when_no_pending(client, tmp_path):
    make_session(tmp_path, pending=None)
    r = client.get("/api/name-corrections")
    assert r.status_code == 200
    assert r.get_json()["n_pending_groups"] == 0


def test_session_get_404_without_file(client, tmp_path):
    make_session(tmp_path, pending=None)
    assert client.get(f"/api/sessions/{SID}/name-corrections").status_code == 404


def test_bless_applies_all_occurrences_and_records(client, tmp_path, monkeypatch):
    sdir = make_session(tmp_path, pending_fixture())
    import worlddict
    blessed = {}
    monkeypatch.setattr(worlddict, "bless",
                        lambda world, heard, canonical, provenance="": blessed.update(
                            {"world": world, "heard": heard, "canonical": canonical}))
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"heard_cleaned": "bandos"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["applied_occurrences"] == 2 and body["n_pending"] == 0

    t = json.loads((sdir / "transcript-rich.json").read_text())
    w1 = t["segments"][0]["words"][1]
    w2 = t["segments"][1]["words"][0]
    assert w1["word"] == " Pandavas" and w1["_original"] == "Bandos"
    assert w2["word"] == " Pandavas."          # trailing punctuation preserved
    assert t["segments"][0]["text"] == " the Pandavas were good"   # text healed
    assert blessed == {"world": "Mahabharata", "heard": "Bandos", "canonical": "Pandavas"}

    p = json.loads((sdir / "pending-name-corrections.json").read_text())
    assert p["pending"] == [] and p["_blessed"][0]["heard_cleaned"] == "bandos"


def test_bless_can_override_spelling(client, tmp_path, monkeypatch):
    sdir = make_session(tmp_path, pending_fixture())
    import worlddict
    monkeypatch.setattr(worlddict, "bless", lambda *a, **k: None)
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"heard_cleaned": "bandos", "canonical": "The Pandavas"})
    assert r.status_code == 200 and r.get_json()["canonical"] == "The Pandavas"


def test_reject_drops_group_leaves_transcript(client, tmp_path):
    sdir = make_session(tmp_path, pending_fixture())
    before = (sdir / "transcript-rich.json").read_text()
    r = client.post(f"/api/sessions/{SID}/name-corrections/reject",
                    json={"heard_cleaned": "bandos"})
    assert r.status_code == 200 and r.get_json()["n_pending"] == 0
    assert (sdir / "transcript-rich.json").read_text() == before
    p = json.loads((sdir / "pending-name-corrections.json").read_text())
    assert p["_rejected"][0]["heard_cleaned"] == "bandos"


def test_bless_unknown_group_404(client, tmp_path):
    make_session(tmp_path, pending_fixture())
    r = client.post(f"/api/sessions/{SID}/name-corrections/bless",
                    json={"heard_cleaned": "nope"})
    assert r.status_code == 404
