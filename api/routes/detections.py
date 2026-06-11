"""Detection (monitor) endpoints — failure-mode detector results.

Detectors write sessions/<id>/detections.json (see src/detectors/); these
routes only read. The rollup distinguishes "0 flags" (scanned, clean) from
"never scanned" (no section) — a monitoring view must not conflate them.
"""

import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify

from api.helpers import _read_transcript_facts, get_session_dir, validate_session_id

bp = Blueprint("detections", __name__)


def _registry():
    from detectors import DETECTORS  # deferred; src/ is on sys.path via api.app
    return DETECTORS


def _read_detections(session_dir: Path) -> dict | None:
    """Parsed detections.json, or None if never scanned. Corrupt file
    propagates json.JSONDecodeError (fail loud)."""
    path = session_dir / "detections.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@bp.route("/detections")
def detections_rollup():
    """System-wide rollup: every session with a transcript × every detector."""
    sessions_dir = current_app.config["SESSIONS_DIR"]
    detectors = [
        {"id": d.id, "label": d.label, "failure_mode": d.failure_mode, "version": d.version}
        for d in _registry()
    ]
    totals = {d["id"]: 0 for d in detectors}
    sessions = []
    if sessions_dir.exists():
        for entry in sorted(sessions_dir.iterdir(), reverse=True):
            if not entry.is_dir() or not validate_session_id(entry.name):
                continue
            if not (entry / "transcript-rich.json").exists():
                continue
            try:
                data = _read_detections(entry)
            except json.JSONDecodeError as e:
                return jsonify({"error": f"Corrupt detections.json in {entry.name}: {e}"}), 500
            results = {}
            if data is not None:
                for det_id, section in data["detectors"].items():
                    results[det_id] = {
                        "n_flags": section["n_flags"],
                        "run_at": section["run_at"],
                        "detector_version": section["detector_version"],
                    }
                    totals[det_id] = totals.get(det_id, 0) + section["n_flags"]
            facts = _read_transcript_facts(entry)
            sessions.append({
                "session_id": entry.name,
                "duration_seconds": facts["duration_seconds"],
                "results": results,
            })
    return jsonify({"detectors": detectors, "sessions": sessions, "totals": totals})


@bp.route("/sessions/<session_id>/detections")
def session_detections(session_id: str):
    """Full flag detail for one session, joined to transcript segments."""
    try:
        session_dir = get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    data = _read_detections(session_dir)
    if data is None:
        return jsonify({"session_id": session_id, "detectors": {}})

    # Join flags to segments server-side so the UI never ships the transcript.
    seg_by_id = {}
    transcript_path = session_dir / "transcript-rich.json"
    if transcript_path.exists():
        with open(transcript_path) as f:
            seg_by_id = {seg["id"]: seg for seg in json.load(f)["segments"]}

    detectors = {}
    for det_id, section in data["detectors"].items():
        flags = []
        for flag in section["flags"]:
            seg = seg_by_id.get(flag["segment_id"])
            speaker = (seg.get("_speaker") or {}) if seg else {}
            flags.append({
                **flag,
                "segment_text": seg["text"].strip() if seg else None,
                "segment_start": seg["start"] if seg else None,
                "segment_end": seg["end"] if seg else None,
                "segment_speaker": speaker.get("label"),
            })
        detectors[det_id] = {**section, "flags": flags}

    return jsonify({"session_id": session_id, "detectors": detectors})
