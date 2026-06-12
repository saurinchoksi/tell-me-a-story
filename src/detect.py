#!/usr/bin/env python3
"""Run failure-mode detectors over sessions, writing sessions/<id>/detections.json.

Detection only — transcripts are never modified. Console output is counts only;
flagged tokens (which can echo mis-rendered names) live in the gitignored
detections.json and the Monitor screen.

Usage:
    python src/detect.py                          # all sessions with a transcript
    python src/detect.py 20260414-213156 ...      # specific sessions
    python src/detect.py --detector m9a-family-names
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detectors import DETECTORS, get_detector
from detectors.base import write_detections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = PROJECT_ROOT / "sessions"
SESSION_ID_RE = re.compile(r"^\d{8}-\d{6}$")


def discover_session_ids() -> list[str]:
    """All session dirs with a transcript, newest first. Sessions without a
    transcript aren't an error in batch mode — there's nothing to scan yet."""
    return sorted(
        (
            d.name
            for d in SESSIONS_DIR.iterdir()
            if d.is_dir()
            and SESSION_ID_RE.match(d.name)
            and (d / "transcript-rich.json").exists()
        ),
        reverse=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Run failure-mode detectors over sessions.")
    ap.add_argument("sessions", nargs="*", help="session ids under sessions/ (default: all)")
    ap.add_argument("--detector", help="run only this detector id")
    ap.add_argument("--judge", action="store_true",
                    help="apply the offline LLM judge to m9b-name-consistency "
                         "(recovers dictionary-word names; needs the mlx-vlm venv)")
    args = ap.parse_args()

    detectors = [get_detector(args.detector)] if args.detector else DETECTORS

    # The judge is an offline upgrade for the M9b detector only.
    m9b_judge = None
    if args.judge:
        from detectors.name_consistency_judge import make_judge
        m9b_judge = make_judge()

    if args.sessions:
        session_ids = args.sessions
        for sid in session_ids:
            if not (SESSIONS_DIR / sid).is_dir():
                raise FileNotFoundError(f"No session directory: sessions/{sid}")
    else:
        session_ids = discover_session_ids()
        if not session_ids:
            print("No sessions with a transcript found.")
            return

    print(f"Running {len(detectors)} detector(s) over {len(session_ids)} session(s):\n")
    for sid in session_ids:
        session_dir = SESSIONS_DIR / sid
        for det in detectors:
            if m9b_judge and det.id == "m9b-name-consistency":
                result = det.run(session_dir, judge=m9b_judge)
            else:
                result = det.run(session_dir)
            section = write_detections(session_dir, det, result)
            judged = "  +judge" if (m9b_judge and det.id == "m9b-name-consistency") else ""
            print(
                f"  {sid}  {det.id:<20} "
                f"{section['n_flags']:>3} flags / {section['n_word_tokens']} tokens{judged}"
            )


if __name__ == "__main__":
    main()
