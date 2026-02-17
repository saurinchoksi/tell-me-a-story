"""
Transcription Validator Server
Serves the validation UI and handles transcript/notes operations.
"""

from flask import Flask, send_file, jsonify, request
from pathlib import Path
import json
import sys

app = Flask(__name__)

# Path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent  # tools/transcript_validator → project root
SESSIONS_DIR = PROJECT_ROOT / "sessions"

# Add src/ to path for filter imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from filters import silence_gap, near_zero_probability, find_duplicate_segments


def get_session_paths(stem: str) -> dict:
    """Get paths for a session."""
    session_dir = SESSIONS_DIR / stem
    return {
        "session_dir": session_dir,
        "audio": session_dir / "audio.m4a",
        "transcript": session_dir / "transcript-rich.json",
        "notes": session_dir / "validation-notes.json",
    }


def validate_path(base_dir: Path, subpath: Path) -> Path:
    """Prevent path traversal attacks."""
    full_path = (base_dir / subpath).resolve()
    if not str(full_path).startswith(str(base_dir.resolve())):
        raise ValueError("Invalid path")
    return full_path


@app.route("/")
def index():
    """Serve the validator HTML."""
    return send_file(Path(__file__).parent / "validator.html")


@app.route("/validator.css")
def serve_css():
    """Serve the validator CSS."""
    return send_file(Path(__file__).parent / "validator.css", mimetype="text/css")


@app.route("/validator.js")
def serve_js():
    """Serve the validator JS."""
    return send_file(Path(__file__).parent / "validator.js", mimetype="application/javascript")


@app.route("/files")
def list_files():
    """List sessions with both transcript AND matching audio."""
    valid_stems = []

    if SESSIONS_DIR.exists():
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue

            stem = session_dir.name
            paths = get_session_paths(stem)

            if paths["transcript"].exists() and paths["audio"].exists():
                valid_stems.append(stem)

    return jsonify({"files": sorted(valid_stems)})


@app.route("/audio/<stem>")
def get_audio(stem: str):
    """Stream audio file."""
    try:
        validate_path(SESSIONS_DIR, Path(stem))
        paths = get_session_paths(stem)

        if not paths["audio"].exists():
            return jsonify({"error": "Audio not found"}), 404
        return send_file(paths["audio"], mimetype="audio/mp4")
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/transcript/<stem>")
def get_transcript(stem: str):
    """Return transcript content, optionally filtered for hallucinations."""
    try:
        validate_path(SESSIONS_DIR, Path(stem))
        paths = get_session_paths(stem)

        if not paths["transcript"].exists():
            return jsonify({"error": "Transcript not found"}), 404

        with open(paths["transcript"]) as f:
            transcript = json.load(f)

        # Apply silence-gap hallucination filter if requested
        if request.args.get("silence_gap") == "1":
            transcript["segments"] = [
                seg for seg in transcript.get("segments", [])
                if not (
                    len(seg.get("words", [])) == 1
                    and silence_gap(seg["words"][0])
                )
            ]

        # Apply near-zero probability filter if requested
        if request.args.get("near_zero") == "1":
            transcript["segments"] = [
                seg for seg in transcript.get("segments", [])
                if not (
                    len(seg.get("words", [])) == 1
                    and near_zero_probability(seg["words"][0])
                )
            ]

        # Apply duplicate segment filter if requested
        if request.args.get("duplicates") == "1":
            dup_ids = find_duplicate_segments(transcript.get("segments", []))
            transcript["segments"] = [
                seg for seg in transcript.get("segments", [])
                if seg.get("id") not in dup_ids
            ]

        return jsonify(transcript)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Malformed transcript JSON"}), 500


@app.route("/notes/<stem>", methods=["GET"])
def get_notes(stem: str):
    """Get validation notes."""
    try:
        validate_path(SESSIONS_DIR, Path(stem))
        paths = get_session_paths(stem)

        if not paths["notes"].exists():
            return jsonify({"notes": []})
        with open(paths["notes"]) as f:
            return jsonify(json.load(f))
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/notes/<stem>", methods=["POST"])
def save_notes(stem: str):
    """Save validation notes."""
    try:
        validate_path(SESSIONS_DIR, Path(stem))
        paths = get_session_paths(stem)

        paths["session_dir"].mkdir(parents=True, exist_ok=True)

        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
        with open(paths["notes"], "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True})
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400
    except (IOError, OSError) as e:
        return jsonify({"error": f"Failed to write notes: {e}"}), 500


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Sessions dir: {SESSIONS_DIR}")
    app.run(host="localhost", port=5001, debug=True)
