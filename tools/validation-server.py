#!/usr/bin/env python3
"""
Validation Server for Tell Me A Story

A lightweight Flask server for validating transcription quality by playing
audio alongside transcripts. Enables human review of speaker diarization
and word-level timing.

Usage:
    python tools/validation-server.py [--port PORT]

Endpoints:
    GET  /                  Serve validation player HTML
    GET  /files             List available session files (with both audio + transcript)
    GET  /audio/<stem>      Stream audio file
    GET  /transcript/<stem> Get transcript JSON
    GET  /notes/<stem>      Get validation notes
    POST /notes/<stem>      Save validation notes
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "sessions" / "audio"
TRANSCRIPT_DIR = PROJECT_ROOT / "sessions" / "processed"
NOTES_DIR = Path(__file__).parent / "validation-notes"

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the validation player HTML."""
    return send_from_directory(Path(__file__).parent, "validation-player.html")


@app.route("/files")
def list_files():
    """
    List session files that have both audio and transcript.

    Returns:
        JSON with list of file stems (without extensions)
    """
    # Find all audio files
    audio_stems = set()
    if AUDIO_DIR.exists():
        for audio_file in AUDIO_DIR.glob("*.m4a"):
            audio_stems.add(audio_file.stem)

    # Find all transcript files
    transcript_stems = set()
    if TRANSCRIPT_DIR.exists():
        for transcript_file in TRANSCRIPT_DIR.glob("*.json"):
            transcript_stems.add(transcript_file.stem)

    # Only include files that have BOTH audio and transcript
    valid_stems = sorted(audio_stems & transcript_stems)

    return jsonify({"files": valid_stems})


@app.route("/audio/<path:stem>")
def get_audio(stem: str):
    """
    Stream audio file for playback.

    Uses send_file() which handles HTTP range requests automatically,
    enabling seeking in the audio player.

    Args:
        stem: Filename without extension (may contain spaces)

    Returns:
        Audio file stream with proper MIME type
    """
    audio_path = AUDIO_DIR / f"{stem}.m4a"

    if not audio_path.exists():
        return jsonify({"error": f"Audio file not found: {stem}"}), 404

    return send_file(
        audio_path,
        mimetype="audio/mp4",
        as_attachment=False
    )


@app.route("/transcript/<path:stem>")
def get_transcript(stem: str):
    """
    Get transcript JSON for a session.

    Args:
        stem: Filename without extension (may contain spaces)

    Returns:
        Transcript JSON content
    """
    transcript_path = TRANSCRIPT_DIR / f"{stem}.json"

    if not transcript_path.exists():
        return jsonify({"error": f"Transcript not found: {stem}"}), 404

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    return jsonify(transcript)


@app.route("/notes/<path:stem>", methods=["GET"])
def get_notes(stem: str):
    """
    Get validation notes for a session.

    Args:
        stem: Filename without extension (may contain spaces)

    Returns:
        JSON with notes array, empty array if no notes exist
    """
    notes_path = NOTES_DIR / f"{stem}.json"

    if not notes_path.exists():
        return jsonify({"notes": []})

    with open(notes_path, "r", encoding="utf-8") as f:
        notes = json.load(f)

    return jsonify(notes)


@app.route("/notes/<path:stem>", methods=["POST"])
def save_notes(stem: str):
    """
    Save validation notes for a session.

    Expects JSON body with notes array:
        {"notes": [...]}

    Args:
        stem: Filename without extension (may contain spaces)

    Returns:
        Success confirmation
    """
    # Ensure notes directory exists
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    notes_path = NOTES_DIR / f"{stem}.json"

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    with open(notes_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return jsonify({"status": "saved", "path": str(notes_path)})


def main():
    """Parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="Validation server for reviewing transcription quality"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)"
    )
    args = parser.parse_args()

    print(f"Starting validation server on http://localhost:{args.port}")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Transcript directory: {TRANSCRIPT_DIR}")
    print(f"Notes directory: {NOTES_DIR}")

    app.run(host="localhost", port=args.port, debug=True)


if __name__ == "__main__":
    main()
