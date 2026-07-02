#!/usr/bin/env python3
"""Revert the removed world-blind LLM normalizer's edits on a session — back to honest words.

The old normalizer (removed 2026-07-01) rewrote word tokens with confident wrong guesses
(Fondos->Bhishma on the Mahabharata held-out; Zorro->Marley and the invented "Jakrash" on
the KPop session). Its audit trail survives on each word (`_original` = the pre-correction
token, punctuation included, minus leading whitespace; `_corrections` = the stage log), so
the damage is exactly invertible: restore `leading_ws + _original` for every word whose
trail contains an llm-stage correction, strip the correction metadata (as if the pass never
ran), and heal each touched segment's `text` from its words.

Surgical: token rewrites only — segment ids, order, and per-segment word counts are asserted
unchanged before writing (axial labels bind to (segment id, word index)). Backup first.

Dry-run by default. Usage:
    python src/revert_normalization.py <session-id>            # report what would change
    python src/revert_normalization.py <session-id> --write    # revert
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--write", action="store_true", help="apply (default: dry-run)")
    args = ap.parse_args()

    session_dir = get_session_dir(ROOT / "sessions", args.session_id)
    rich_path = session_dir / "transcript-rich.json"
    transcript = json.loads(rich_path.read_text())

    before_ids = [s["id"] for s in transcript["segments"]]
    before_counts = [len(s.get("words") or []) for s in transcript["segments"]]

    n, changed, touched = 0, [], set()
    for seg in transcript["segments"]:
        for w in seg.get("words") or []:
            trails = w.get("_corrections") or []
            if not any("llm" in (c.get("stage") or "") for c in trails):
                continue
            if "_original" not in w:  # fail loud: an llm trail without its original is corrupt
                raise SystemExit(f"word {w.get('word')!r} in seg {seg['id']} has an llm "
                                 f"correction but no _original — cannot revert safely")
            cur = w["word"]
            leading = " " if cur.startswith(" ") else ""
            honest = leading + w["_original"]
            if honest != cur:
                changed.append((seg["id"], cur.strip(), w["_original"]))
            w["word"] = honest
            # strip the metadata — the pass never happened
            del w["_original"]
            del w["_corrections"]
            touched.add(seg["id"])
            n += 1
    for seg in transcript["segments"]:
        if seg["id"] in touched and seg.get("words"):
            seg["text"] = "".join(x["word"] for x in seg["words"])

    print(f"[revert-normalization] {args.session_id}: {n} llm-corrected words "
          f"({len(changed)} actually change, rest are no-op self-corrections)")
    for sid, cur, orig in changed:
        print(f"    seg {sid}: {cur!r} -> {orig!r}")

    after_ids = [s["id"] for s in transcript["segments"]]
    after_counts = [len(s.get("words") or []) for s in transcript["segments"]]
    assert after_ids == before_ids, "segment ids changed — ABORT"
    assert after_counts == before_counts, "word counts changed — ABORT"

    if not args.write:
        print("  dry-run: nothing written. Re-run with --write to revert.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = rich_path.with_suffix(f".json.pre-normrevert-{stamp}")
    shutil.copy2(rich_path, backup)
    tmp = rich_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    tmp.replace(rich_path)
    print(f"  invariants OK; backup {backup.name}; wrote transcript-rich.json")
    print(f"  NOTE: detections now stale — re-scan: python src/detect.py {args.session_id}")


if __name__ == "__main__":
    main()
