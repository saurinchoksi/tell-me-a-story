"""Pre-populate Mode 9 axial labels for a session's known name variants.

Walks transcript-rich.json, finds words matching the given variant list, and
writes one M9 label entry per parent segment to axial-labels.json. Refuses to
overwrite an axial-labels.json that already has labels unless --force is passed
(prevents destroying manually-applied tags).

Examples:
    python scripts/populate_mode9.py 20260117-202237 \\
        Duryodhan Yudhisthir Pondavas Dhrashtra

    python scripts/populate_mode9.py 20251210-203654 \\
        Artie "Artie's" Arthie "Arthie's"
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.helpers import get_session_dir  # noqa: E402

WORD_RE = re.compile(r"\W*([\w\-'’]+)\W*")


def clean(word: str) -> str:
    m = WORD_RE.fullmatch(word.strip())
    return m.group(1) if m else word.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-populate Mode 9 axial labels for a session's name variants.",
    )
    parser.add_argument("session_id", help="Session ID (YYYYMMDD-HHMMSS)")
    parser.add_argument(
        "variants",
        nargs="+",
        help="Name variants to tag as M9 / #2 instances",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite axial-labels.json even if it has existing labels",
    )
    args = parser.parse_args()

    variants = set(args.variants)
    session_dir = get_session_dir(ROOT / "sessions", args.session_id)
    transcript_path = session_dir / "transcript-rich.json"
    labels_path = session_dir / "axial-labels.json"

    if labels_path.exists():
        with open(labels_path) as f:
            existing = json.load(f)
        if existing.get("labels"):
            if not args.force:
                print(
                    f"Refusing to overwrite {labels_path} — file already has "
                    f"{len(existing['labels'])} labels. Pass --force to overwrite.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # --force was passed and there's real data — back it up before clobbering.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = labels_path.parent / f"axial-labels.json.bak-{stamp}"
            shutil.copy2(labels_path, backup)
            print(f"Backed up existing labels to: {backup}", file=sys.stderr)

    with open(transcript_path) as f:
        tx = json.load(f)

    seg_to_variants: dict[int | str, list[str]] = {}
    for seg in tx["segments"]:
        if seg.get("_source") == "diarization_gap":
            continue
        sid = seg["id"]
        for w in seg["words"]:
            cleaned = clean(w["word"])
            if cleaned in variants:
                seg_to_variants.setdefault(sid, []).append(cleaned)

    now = datetime.now(timezone.utc).isoformat()
    labels = [
        {"segmentId": sid, "codes": ["M9"], "createdAt": now, "updatedAt": now}
        for sid in seg_to_variants
    ]

    with open(labels_path, "w") as f:
        json.dump({"labels": labels}, f, indent=2)

    total_instances = sum(len(v) for v in seg_to_variants.values())
    observed = sorted({v for vs in seg_to_variants.values() for v in vs})
    print(f"Session:                  {args.session_id}")
    print(f"Variants targeted:        {sorted(variants)}")
    print(f"Variants observed:        {observed}")
    print(f"Segments tagged with M9:  {len(labels)}")
    print(f"Total variant instances:  {total_instances}")
    print(f"Wrote: {labels_path}")


if __name__ == "__main__":
    main()
