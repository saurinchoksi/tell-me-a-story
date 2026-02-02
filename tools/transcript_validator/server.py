"""
Transcription Validator Server
Serves the validation UI and handles transcript/notes operations.
"""

from flask import Flask, send_file, jsonify, request
from pathlib import Path
import json

app = Flask(__name__)

# Path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent  # tools/transcript_validator → project root
AUDIO_DIR = PROJECT_ROOT / "sessions" / "audio"
DEBUG_DIR = PROJECT_ROOT / "sessions" / "debug"
NOTES_DIR = Path(__file__).parent / "notes"

# Ensure notes directory exists
NOTES_DIR.mkdir(exist_ok=True)


def validate_path(base_dir: Path, filename: str) -> Path:
    """Prevent path traversal attacks."""
    # Resolve the full path
    full_path = (base_dir / filename).resolve()
    # Ensure it's still under base_dir
    if not str(full_path).startswith(str(base_dir.resolve())):
        raise ValueError("Invalid path")
    return full_path


@app.route("/")
def index():
    """Serve the validator HTML."""
    return send_file(Path(__file__).parent / "validator.html")


@app.route("/files")
def list_files():
    """List sessions with both debug transcript AND matching audio."""
    # Find debug folders with 03-transcript.json
    debug_stems = set()
    if DEBUG_DIR.exists():
        for d in DEBUG_DIR.iterdir():
            if d.is_dir() and (d / "03-transcript.json").exists():
                debug_stems.add(d.name)

    # Find audio files
    audio_stems = set()
    if AUDIO_DIR.exists():
        for f in AUDIO_DIR.glob("*.m4a"):
            audio_stems.add(f.stem)

    # Return intersection
    valid_stems = sorted(debug_stems & audio_stems)
    return jsonify({"files": valid_stems})


@app.route("/audio/<stem>")
def get_audio(stem: str):
    """Stream audio file."""
    try:
        audio_path = validate_path(AUDIO_DIR, f"{stem}.m4a")
        if not audio_path.exists():
            return jsonify({"error": "Audio not found"}), 404
        return send_file(audio_path, mimetype="audio/mp4")
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/transcript/<stem>")
def get_transcript(stem: str):
    """Return 03-transcript.json content."""
    try:
        transcript_path = validate_path(DEBUG_DIR / stem, "03-transcript.json")
        if not transcript_path.exists():
            return jsonify({"error": "Transcript not found"}), 404
        with open(transcript_path) as f:
            return jsonify(json.load(f))
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "Malformed transcript JSON"}), 500


@app.route("/notes/<stem>", methods=["GET"])
def get_notes(stem: str):
    """Get validation notes."""
    try:
        notes_path = validate_path(NOTES_DIR, f"{stem}.json")
        if not notes_path.exists():
            return jsonify({"notes": []})
        with open(notes_path) as f:
            return jsonify(json.load(f))
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400


@app.route("/notes/<stem>", methods=["POST"])
def save_notes(stem: str):
    """Save validation notes."""
    try:
        notes_path = validate_path(NOTES_DIR, f"{stem}.json")
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400
        with open(notes_path, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True})
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400
    except (IOError, OSError) as e:
        return jsonify({"error": f"Failed to write notes: {e}"}), 500


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Debug dir: {DEBUG_DIR}")
    print(f"Notes dir: {NOTES_DIR}")
    app.run(host="localhost", port=5001, debug=True)
