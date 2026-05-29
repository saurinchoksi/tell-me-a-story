"""Fold broken-Whisper "filler-loop" stretches into single Mode 10 events.

A broken-Whisper stretch is a region where Whisper's decoder gets stuck and
writes the same filler word ("Hmm.", "Right.", ...) across many consecutive
segments over a quiet patch, while the real speech underneath goes uncaptured.
Counting each repeated segment as its own failure overstates the count (one
broken event read as ~27 failures) and understates its severity. See TMAS-49
and emp.md closing-review finding #1 (2026-05-28).

This folds each configured stretch to a single event:
  - the FIRST filler segment in the range gets codes = ["M10"]
  - every OTHER filler segment in the range has its axial entry removed — it
    becomes unlabeled, subsumed into the one M10 event (this is what fixes the
    overcount: the repeated per-segment M1/M2 stop inflating those modes)
  - real-content segments inside the range (text != the filler word) are left
    untouched, e.g. the Moon story lines interleaved in the "Hmm." region
  - a "[filler-loop region]" Span Note is ensured on the boundary segment so the
    region stays documented after the per-segment codes are gone (preserves the
    "what was happening here" detail the redundant chips can't)

Merge-based and reversible: every modified file is backed up to a timestamped
copy before any write, and the script is dry-run by default (pass --write to
commit). Run with the validator closed, then refresh the browser tab — the UI's
debounced auto-save can otherwise clobber the rewrite.

Stretches (surfaced during the count, all verified by ear 2026-05-28):
  Moon Story  20251207-195607 : Hmm. 99-136, Yeah. 159-173, How? 223-235
  New Rec 60  20260129-204404 : Right. 280-334

Usage:
    python emp/src/populate_mode10.py            # dry-run: print the diff
    python emp/src/populate_mode10.py --write     # commit (validator closed)
"""

import argparse
import collections
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.helpers import get_session_dir  # noqa: E402

# Each stretch: the session, the looped filler word (exact transcript text once
# stripped), and the inclusive segmentId range it spans.
STRETCHES = [
    {"session": "20251207-195607", "word": "Hmm.",   "lo": 99,  "hi": 136},
    {"session": "20251207-195607", "word": "Yeah.",  "lo": 159, "hi": 173},
    {"session": "20251207-195607", "word": "How?",   "lo": 223, "hi": 235},
    {"session": "20260129-204404", "word": "Right.", "lo": 280, "hi": 334},
]

SPAN_TAG = "[filler-loop region]"


def code_freq(labels) -> collections.Counter:
    f: collections.Counter = collections.Counter()
    for label in labels:
        for code in label.get("codes", []):
            f[code] += 1
    return f


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def backup(path: Path, tag: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = path.parent / f"{path.name}.pre-{tag}-{stamp}"
    shutil.copy2(path, dest)
    if dest.stat().st_size != path.stat().st_size:
        print(f"Backup size mismatch for {path} — aborting.", file=sys.stderr)
        sys.exit(3)
    return dest


def find_filler_segments(transcript, word: str, lo: int, hi: int) -> list[int]:
    """segmentIds in [lo, hi] whose stripped text == word, sorted ascending."""
    ids = []
    for seg in transcript["segments"]:
        sid = seg.get("id")
        if not isinstance(sid, int):
            continue
        if lo <= sid <= hi and (seg.get("text") or "").strip() == word:
            ids.append(sid)
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold broken-Whisper filler-loop stretches into single M10 events."
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Commit changes (default: dry-run, prints the diff only).",
    )
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()

    # Group stretches by session so each session's files are read/written once.
    by_session: dict[str, list[dict]] = collections.defaultdict(list)
    for stretch in STRETCHES:
        by_session[stretch["session"]].append(stretch)

    for session_id, stretches in by_session.items():
        session_dir = get_session_dir(ROOT / "sessions", session_id)
        tx = load_json(session_dir / "transcript-rich.json")
        seg_start = {seg["id"]: seg.get("start") for seg in tx["segments"]}

        labels_path = session_dir / "axial-labels.json"
        labels = load_json(labels_path)["labels"]
        by_id = {label["segmentId"]: dict(label) for label in labels}
        pre_freq = code_freq(by_id.values())

        notes_path = session_dir / "validation-notes.json"
        notes_doc = load_json(notes_path) if notes_path.exists() else {"notes": []}
        notes = notes_doc["notes"]

        print(f"\n=== {session_id} ===")
        removed_total = 0
        new_notes: list[dict] = []
        for st in stretches:
            word, lo, hi = st["word"], st["lo"], st["hi"]
            filler = find_filler_segments(tx, word, lo, hi)
            if not filler:
                print(f"  !! no '{word}' segments in {lo}-{hi} — check config",
                      file=sys.stderr)
                sys.exit(2)
            boundary, inner = filler[0], filler[1:]

            old_boundary = by_id.get(boundary, {}).get("codes")
            removed = []
            for sid in inner:
                if sid in by_id:
                    removed.append((sid, by_id[sid]["codes"]))
                    del by_id[sid]
            removed_total += len(removed)

            # The boundary segment carries the single M10 event for the stretch.
            entry = by_id.get(boundary) or {
                "segmentId": boundary, "codes": [], "createdAt": now,
            }
            entry["codes"] = ["M10"]
            entry["updatedAt"] = now
            by_id[boundary] = entry

            print(f"  {word:7} segs {lo}-{hi}: {len(filler)} filler segments — "
                  f"boundary {boundary}: {old_boundary} -> ['M10']; "
                  f"removed {len(removed)} inner")
            for sid, codes in removed:
                print(f"        - seg {sid}: {codes}")

            # Ensure a [filler-loop region] Span Note documents the stretch.
            already = any(
                n.get("segmentId") == boundary and SPAN_TAG in (n.get("text") or "")
                for n in notes
            )
            if already:
                print(f"        (Span Note already present on seg {boundary})")
            else:
                new_notes.append({
                    "id": f"m10-{boundary}-{word.strip('.?!').lower()}",
                    "segmentId": boundary,
                    "wordIndex": None,
                    "wordText": None,
                    "wordStart": None,
                    "timestamp": seg_start.get(boundary),
                    "text": (f"{SPAN_TAG} segs {lo}-{hi}. Whisper looped on "
                             f'"{word}" across this run of segments; counted as '
                             "one failure (M10), not per-segment. What was said "
                             "underneath varies — sometimes silence, sometimes "
                             "real speech the loop masks and loses (see M3 / "
                             "gap analysis)."),
                    "createdAt": now,
                })

        merged = list(by_id.values())
        post_freq = code_freq(merged)

        print(f"  code freq:  {dict(sorted(pre_freq.items()))}")
        print(f"           -> {dict(sorted(post_freq.items()))}")
        print(f"  entries:    {len(labels)} -> {len(merged)} (removed {removed_total})")
        if new_notes:
            print(f"  Span Notes: + {len(new_notes)} on segs "
                  f"{[n['segmentId'] for n in new_notes]}")

        # Invariant: M10 grew by exactly the number of stretches in this session.
        expected_m10 = pre_freq.get("M10", 0) + len(stretches)
        if post_freq.get("M10", 0) != expected_m10:
            print(f"  !! M10 count off: expected {expected_m10}, "
                  f"got {post_freq.get('M10', 0)} — aborting.", file=sys.stderr)
            sys.exit(2)

        if not args.write:
            continue

        backup(labels_path, "m10")
        with open(labels_path, "w") as fh:
            json.dump({"labels": merged}, fh, indent=2)
        print(f"  wrote {labels_path}")
        if new_notes:
            if notes_path.exists():
                backup(notes_path, "m10")
            notes_doc["notes"] = notes + new_notes
            with open(notes_path, "w") as fh:
                json.dump(notes_doc, fh, indent=2)
            print(f"  wrote {notes_path}")

    if not args.write:
        print("\nDry run — no files changed. Re-run with --write to commit.")
    else:
        print("\nDone. Refresh the validator browser tab to reload the rewrite.")


if __name__ == "__main__":
    main()
