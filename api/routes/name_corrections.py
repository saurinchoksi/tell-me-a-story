"""Name-correction queue endpoints — the human bless loop for the namefix stage.

The namefix stage auto-applies only bulletproof audio-verified fixes; everything else lands
in `sessions/<id>/pending-name-corrections.json` awaiting a human verdict. These routes serve
that queue and act on verdicts:

  GET  /api/name-corrections
        Rollup across all sessions, grouped by world -> (heard -> suggestion) so a batch of
        recordings reviews as one queue per NAME, not file-by-file (the settled hybrid design).
  POST /api/sessions/<id>/name-corrections/bless   {heard_cleaned, canonical}
        The human confirms: apply the group's occurrences to the transcript SURGICALLY
        (in-place word rewrites; segment ids + word counts asserted unchanged), record the
        pair in the per-world dictionary (worlddict.bless — deterministic forever after),
        drop the group from pending.
  POST /api/sessions/<id>/name-corrections/reject  {heard_cleaned}
        The suggestion is wrong: drop the group from pending into the file's _rejected audit
        list. The transcript is untouched (it was never modified for a queued item).

Viewing is read-only; only the two POST verbs mutate anything.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir, validate_session_id

bp = Blueprint("name_corrections", __name__)

PENDING_FILE = "pending-name-corrections.json"


def _enrich_occurrences(session_dir: Path, groups: list) -> None:
    """Join each occurrence to its segment's CURRENT sentence (built from words, the source
    of truth) so the reviewer sees the word in context — several names can fly by in one
    span, and a bare token + play button isn't enough to know what's being judged. Adds
    `segment_text`, `word_offset`/`word_len` (char span of the judged word inside it)."""
    rich_path = session_dir / "transcript-rich.json"
    if not rich_path.exists():
        return
    transcript = json.loads(rich_path.read_text())
    seg_by_id = {s["id"]: s for s in transcript.get("segments", [])}
    for g in groups:
        for o in g.get("occurrences", []):
            seg = seg_by_id.get(o.get("segment_id"))
            words = (seg or {}).get("words") or []
            wi = o.get("word_index")
            if seg is None or wi is None or wi >= len(words):
                continue
            parts = [w["word"] for w in words]
            text = "".join(parts)
            off = len("".join(parts[:wi]))
            tok = parts[wi]
            lead_ws = len(tok) - len(tok.lstrip())
            o["segment_text"] = text.strip()
            # offset within the STRIPPED text: subtract the segment's own leading whitespace
            seg_lead = len(text) - len(text.lstrip())
            o["word_offset"] = off + lead_ws - seg_lead
            o["word_len"] = len(tok.strip())


def _read_pending(session_dir: Path) -> dict | None:
    path = session_dir / PENDING_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _write_pending(session_dir: Path, data: dict) -> None:
    path = session_dir / PENDING_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


@bp.route("/name-corrections", methods=["GET"])
def name_corrections_rollup():
    """The consolidated queue: every session's pending groups, grouped by world then by
    (heard -> suggestion) so the same name across a 30-file batch is ONE review item."""
    sessions_dir = current_app.config["SESSIONS_DIR"]
    worlds: dict = {}
    n_pending = 0
    if sessions_dir.exists():
        for entry in sorted(sessions_dir.iterdir()):
            if not entry.is_dir() or not validate_session_id(entry.name):
                continue
            data = _read_pending(entry)
            if not data:
                continue
            _enrich_occurrences(entry, data.get("pending", []))
            for g in data.get("pending", []):
                n_pending += 1
                wkey = g.get("world") or "(unrecognized)"
                nkey = f"{g.get('heard_cleaned')}→{g.get('canonical')}"
                slot = worlds.setdefault(wkey, {}).setdefault(nkey, {
                    "heard": g.get("heard"),
                    "heard_cleaned": g.get("heard_cleaned"),
                    "canonical": g.get("canonical"),
                    "method": g.get("method"),
                    "sessions": [],
                })
                slot["sessions"].append({
                    "session_id": entry.name,
                    "story_id": g.get("story_id"),
                    "occurrences": g.get("occurrences", []),
                })
    payload = {
        "n_pending_groups": n_pending,
        "worlds": [
            {"world": w, "names": sorted(names.values(), key=lambda x: x["heard_cleaned"] or "")}
            for w, names in sorted(worlds.items())
        ],
    }
    return jsonify(payload)


@bp.route("/sessions/<session_id>/name-corrections", methods=["GET"])
def session_name_corrections(session_id):
    """One session's pending file, verbatim (404 if the stage never queued anything)."""
    if not validate_session_id(session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    if not session_dir.exists():
        return jsonify({"error": "Session not found"}), 404
    data = _read_pending(session_dir)
    if data is None:
        return jsonify({"error": "No pending name corrections"}), 404
    _enrich_occurrences(session_dir, data.get("pending", []))
    return jsonify(data)


def _apply_group(session_dir: Path, group: dict) -> int:
    """Surgically apply one blessed group's occurrences to transcript-rich.json.

    Rewrites each occurrence token in place via the corrections machinery's conventions
    (preserve leading space + trailing punctuation, set _original once, append to
    _corrections, heal the segment text) — then asserts segment ids and word counts
    unchanged before writing. Returns the number of words corrected."""
    import re
    rich_path = session_dir / "transcript-rich.json"
    transcript = json.loads(rich_path.read_text())
    seg_by_id = {s["id"]: s for s in transcript["segments"]}
    before_ids = [s["id"] for s in transcript["segments"]]
    before_counts = [len(s.get("words") or []) for s in transcript["segments"]]

    trailing = re.compile(r'^(.*?)([.,;:!?"\)\]\}\']+\.{0,3})$')
    n = 0
    touched_segs = set()
    for occ in group.get("occurrences", []):
        seg = seg_by_id.get(occ.get("segment_id"))
        words = (seg or {}).get("words") or []
        wi = occ.get("word_index")
        if seg is None or wi is None or wi >= len(words):
            continue
        w = words[wi]
        raw = w["word"]
        stripped = raw.strip()
        m = trailing.match(stripped)
        bare, punct = (m.group(1), m.group(2)) if m else (stripped, "")
        if "_original" not in w:
            w["_original"] = stripped
        w.setdefault("_corrections", []).append(
            {"stage": "namefix_bless", "from": bare, "to": group["canonical"]})
        leading = " " if raw.startswith(" ") else ""
        w["word"] = leading + group["canonical"] + punct
        touched_segs.add(seg["id"])
        n += 1
    for sid in touched_segs:
        seg = seg_by_id[sid]
        if seg.get("words"):
            seg["text"] = "".join(x["word"] for x in seg["words"])

    after_ids = [s["id"] for s in transcript["segments"]]
    after_counts = [len(s.get("words") or []) for s in transcript["segments"]]
    assert after_ids == before_ids, "segment ids changed — abort"
    assert after_counts == before_counts, "word counts changed — abort"

    tmp = rich_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    tmp.replace(rich_path)
    return n


def _pop_group(data: dict, heard_cleaned: str) -> dict | None:
    for i, g in enumerate(data.get("pending", [])):
        if g.get("heard_cleaned") == heard_cleaned:
            return data["pending"].pop(i)
    return None


@bp.route("/sessions/<session_id>/name-corrections/bless", methods=["POST"])
def bless_name_correction(session_id):
    """Human confirms a queued correction: apply it, remember it in the world dictionary.

    Optional body field `occurrences`: a list of {segment_id, word_index} pairs — bless
    only THOSE spots (the same-sound-two-referents case, e.g. one "Bheem" meaning Bhima
    and another meaning Arjuna). Unselected occurrences stay pending under the group.
    A partial bless does NOT write the world dictionary (the spelling isn't a universal
    rule for this world if its own occurrences disagree)."""
    if not validate_session_id(session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    body = request.get_json(silent=True) or {}
    heard_cleaned = (body.get("heard_cleaned") or "").strip()
    if not heard_cleaned:
        return jsonify({"error": "heard_cleaned is required"}), 400

    data = _read_pending(session_dir)
    if data is None:
        return jsonify({"error": "No pending name corrections"}), 404
    group = _pop_group(data, heard_cleaned)
    if group is None:
        return jsonify({"error": f"No pending group for {heard_cleaned!r}"}), 404
    # allow the human to override the suggested spelling in the same gesture
    canonical = (body.get("canonical") or group["canonical"]).strip()
    group["canonical"] = canonical

    # per-occurrence selection: split the group; the remainder goes back to pending
    wanted = body.get("occurrences")
    partial = False
    if isinstance(wanted, list) and wanted:
        keys = {(o.get("segment_id"), o.get("word_index")) for o in wanted}
        selected = [o for o in group["occurrences"]
                    if (o["segment_id"], o["word_index"]) in keys]
        rest = [o for o in group["occurrences"]
                if (o["segment_id"], o["word_index"]) not in keys]
        if not selected:
            data["pending"].append(group)  # nothing matched — put it back untouched
            _write_pending(session_dir, data)
            return jsonify({"error": "no matching occurrences"}), 400
        if rest:
            partial = True
            data["pending"].append({**group, "canonical": group["canonical"],
                                    "occurrences": rest})
        group = {**group, "occurrences": selected}

    n = _apply_group(session_dir, group)

    if group.get("world") and not partial:
        import worlddict  # src/ on sys.path via api.app
        worlddict.bless(group["world"], group.get("heard") or heard_cleaned, canonical,
                        provenance=f"bless:{session_id}")

    data.setdefault("_blessed", []).append({
        "heard_cleaned": heard_cleaned, "canonical": canonical,
        "applied_occurrences": n,
        "at": datetime.now(timezone.utc).isoformat(),
    })
    _write_pending(session_dir, data)
    return jsonify({"status": "blessed", "applied_occurrences": n,
                    "canonical": canonical, "n_pending": len(data["pending"])})


@bp.route("/sessions/<session_id>/name-corrections/reject", methods=["POST"])
def reject_name_correction(session_id):
    """Human says the suggestion is wrong: drop it from the queue (transcript untouched)."""
    if not validate_session_id(session_id):
        return jsonify({"error": "Invalid session ID"}), 400
    session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    body = request.get_json(silent=True) or {}
    heard_cleaned = (body.get("heard_cleaned") or "").strip()
    if not heard_cleaned:
        return jsonify({"error": "heard_cleaned is required"}), 400

    data = _read_pending(session_dir)
    if data is None:
        return jsonify({"error": "No pending name corrections"}), 404
    group = _pop_group(data, heard_cleaned)
    if group is None:
        return jsonify({"error": f"No pending group for {heard_cleaned!r}"}), 404

    data.setdefault("_rejected", []).append({
        **{k: group.get(k) for k in ("heard", "heard_cleaned", "canonical", "world")},
        "at": datetime.now(timezone.utc).isoformat(),
    })
    _write_pending(session_dir, data)
    return jsonify({"status": "rejected", "n_pending": len(data["pending"])})
