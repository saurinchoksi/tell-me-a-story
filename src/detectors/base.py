"""Shared detector contract and detections.json persistence.

Detectors are read-only monitors: they scan a session's transcript and emit
flags. The transcript itself is never modified.

Each detections.json section carries a fingerprint of the transcript it
scanned. `ensure_fresh_detections()` is the freshness gate: a section whose
fingerprint no longer matches the current transcript (e.g. after --re-enrich)
is re-run automatically — results can be cached but never silently stale.
"""

import hashlib
import json
import os
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

    def config_fingerprint(self) -> str | None:
        """Content hash of the detector's configuration (e.g. a roster file),
        or None for a configless detector. A detector's output is a function
        of (transcript, config) — both feed the staleness check, so config
        edits trigger re-runs just like transcript changes do."""
        return None


def load_transcript(session_dir: Path) -> dict:
    path = Path(session_dir) / "transcript-rich.json"
    if not path.exists():
        raise FileNotFoundError(f"No transcript-rich.json in {session_dir}")
    return json.loads(path.read_text())


def transcript_fingerprint(session_dir: Path) -> str:
    """Content hash of the transcript a detector run is tied to."""
    path = Path(session_dir) / "transcript-rich.json"
    if not path.exists():
        raise FileNotFoundError(f"No transcript-rich.json in {session_dir}")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_section(detector: Detector, result: dict, fingerprint: str,
                  config_fingerprint: str | None) -> dict:
    """Stamp detector metadata onto a run result — one place, so every
    detector's detections.json section has the same shape."""
    return {
        "label": detector.label,
        "failure_mode": detector.failure_mode,
        "detector_version": detector.version,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "transcript_fingerprint": fingerprint,
        "config_fingerprint": config_fingerprint,
        "n_word_tokens": result["n_word_tokens"],
        "n_flags": len(result["flags"]),
        "flags": result["flags"],
    }


def _load_detections(session_dir: Path) -> dict:
    path = Path(session_dir) / DETECTIONS_FILENAME
    if path.exists():
        return json.loads(path.read_text())
    return {"_about": ABOUT, "detectors": {}}


def _save_detections(session_dir: Path, data: dict) -> None:
    """Write via temp file + atomic rename so a crash mid-write can't leave
    corrupt JSON behind. There is deliberately no cross-process lock:
    concurrent writers (e.g. detect.py during a Monitor view) last-write-wins
    on the whole file — acceptable for a single-user local tool; revisit if
    the registry grows past a couple of detectors."""
    path = Path(session_dir) / DETECTIONS_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def write_detections(session_dir: Path, detector: Detector, result: dict) -> dict:
    """Read-merge-write detections.json, replacing only this detector's section."""
    data = _load_detections(session_dir)
    section = build_section(detector, result, transcript_fingerprint(session_dir),
                            detector.config_fingerprint())
    data["detectors"][detector.id] = section
    _save_detections(session_dir, data)
    return section


def ensure_fresh_detections(session_dir: Path, detectors: list) -> dict:
    """Run any detector whose section is missing or was produced from a
    different transcript version OR detector configuration; return the
    (now fresh) detections data.

    Sections from unregistered detectors are left untouched. Corrupt
    detections.json propagates json.JSONDecodeError (fail loud).
    """
    fingerprint = transcript_fingerprint(session_dir)
    data = _load_detections(session_dir)
    dirty = False
    for det in detectors:
        config_fp = det.config_fingerprint()
        section = data["detectors"].get(det.id)
        if (
            section is not None
            and section.get("transcript_fingerprint") == fingerprint
            and section.get("config_fingerprint") == config_fp
        ):
            continue
        data["detectors"][det.id] = build_section(det, det.run(session_dir),
                                                  fingerprint, config_fp)
        dirty = True
    if dirty:
        _save_detections(session_dir, data)
    return data
