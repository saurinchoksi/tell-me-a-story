"""Shared detector contract and detections.json persistence.

Detectors are read-only monitors: they scan a session's transcript and emit
flags. The transcript itself is never modified.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DETECTIONS_FILENAME = "detections.json"
ABOUT = (
    "Failure-mode detector output. Detection only — the transcript is never "
    "modified. Sections keyed by detector id; each run overwrites only its own section."
)


class Detector:
    """Base contract for failure-mode detectors.

    Subclasses set the class attributes and implement
    run(session_dir) -> {"n_word_tokens": int, "flags": list[dict]}.
    """

    id: str
    label: str
    failure_mode: str
    version: str

    def run(self, session_dir: Path) -> dict:
        raise NotImplementedError


def load_transcript(session_dir: Path) -> dict:
    path = Path(session_dir) / "transcript-rich.json"
    if not path.exists():
        raise FileNotFoundError(f"No transcript-rich.json in {session_dir}")
    return json.loads(path.read_text())


def build_section(detector: Detector, result: dict) -> dict:
    """Stamp detector metadata onto a run result — one place, so every
    detector's detections.json section has the same shape."""
    return {
        "label": detector.label,
        "failure_mode": detector.failure_mode,
        "detector_version": detector.version,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_word_tokens": result["n_word_tokens"],
        "n_flags": len(result["flags"]),
        "flags": result["flags"],
    }


def write_detections(session_dir: Path, detector: Detector, result: dict) -> dict:
    """Read-merge-write detections.json, replacing only this detector's section."""
    path = Path(session_dir) / DETECTIONS_FILENAME
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"_about": ABOUT, "detectors": {}}
    section = build_section(detector, result)
    data["detectors"][detector.id] = section
    path.write_text(json.dumps(data, indent=2))
    return section
