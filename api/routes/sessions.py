"""Session endpoints — list, detail, and speaker identification."""

import json

from flask import Blueprint, current_app, jsonify

from api.helpers import get_session_dir, discover_sessions

bp = Blueprint("sessions", __name__)


@bp.route("/sessions")
def list_sessions():
    """List all sessions with artifact availability flags."""
    sessions = discover_sessions(current_app.config["SESSIONS_DIR"])
    return jsonify({"sessions": sessions})


@bp.route("/sessions/<session_id>")
def get_session(session_id: str):
    """Return full session detail — transcript, diarization, embeddings, identifications."""
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    def load_json(filename):
        path = session_dir / filename
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    try:
        return jsonify({
            "id": session_id,
            "has_audio": next(session_dir.glob("audio.*"), None) is not None,
            "transcript": load_json("transcript-rich.json"),
            "diarization": load_json("diarization.json"),
            "embeddings": load_json("embeddings.json"),
            "identifications": load_json("identifications.json"),
        })
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Malformed JSON in session: {e}"}), 500


@bp.route("/sessions/<session_id>/identify", methods=["POST"])
def identify_session_speakers(session_id: str):
    """Run speaker identification for a session.

    Requires embeddings.json to exist. Uses deferred import to avoid
    loading numpy/torch at API startup time.
    """
    try:
        session_dir = get_session_dir(
            current_app.config["SESSIONS_DIR"], session_id
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    embeddings_path = session_dir / "embeddings.json"
    if not embeddings_path.exists():
        return jsonify({"error": "No embeddings found for this session. Run the pipeline first."}), 400

    # Deferred import — identify.py pulls in numpy
    from identify import identify_speakers, save_identifications

    profiles_path = current_app.config.get("PROFILES_PATH")
    results = identify_speakers(
        embeddings_path,
        profiles_path=profiles_path,
    )

    output_path = session_dir / "identifications.json"
    save_identifications(results, output_path)

    return jsonify(results)
