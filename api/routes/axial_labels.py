"""Axial label endpoints — read and write per-segment axial codes for a session.

A label is a categorical code chosen from the 8 EMP failure modes (M1..M8) or
NotA (none-of-the-above). One label per segment. Used to count failure-mode
frequencies for EMP step 5.
"""

import json

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir

bp = Blueprint("axial_labels", __name__)


@bp.route("/sessions/<session_id>/axial-labels")
def get_axial_labels(session_id: str):
    """Return axial labels for a session, or empty list if none exist."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    labels_path = session_dir / "axial-labels.json"
    if not labels_path.exists():
        return jsonify({"labels": []})

    try:
        with open(labels_path) as f:
            data = json.load(f)
        return jsonify({"labels": data.get("labels", [])})
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Corrupt axial-labels.json: {e}"}), 500


@bp.route("/sessions/<session_id>/axial-labels", methods=["POST"])
def save_axial_labels(session_id: str):
    """Save axial labels for a session (full array replacement)."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    body = request.get_json(silent=True)
    if body is None or "labels" not in body:
        return jsonify({"error": "Request body must contain 'labels' array"}), 400

    if not isinstance(body["labels"], list):
        return jsonify({"error": "'labels' must be an array"}), 400

    labels_path = session_dir / "axial-labels.json"
    with open(labels_path, "w") as f:
        json.dump({"labels": body["labels"]}, f, indent=2)

    return jsonify({"saved": len(body["labels"])})
