#!/usr/bin/env python3
"""Run the namefix stage on an EXISTING session — dry-run by default, surgical on --write.

For sessions processed before namefix shipped (or after a manual revert), this runs the full
chain (Qwen world+cast subprocess, then Whisper re-decode subprocess) against the session's
current transcript-rich.json and either reports what it WOULD do (default) or applies it.

--write safety rails (the realign_session pattern):
  - backup transcript-rich.json -> transcript-rich.json.pre-namefix-<UTC> first
  - corrections apply via corrections.apply_corrections (in-place token rewrites)
  - INVARIANTS asserted before saving: segment id list/order/count byte-identical and every
    segment's word count unchanged — axial-labels bind to (segment id, word index), so a
    violation aborts without writing
  - pending-name-corrections.json written (the bless queue), detections become stale by
    fingerprint (re-scan with: python src/detect.py <session-id>)

Usage:
    python src/namefix_session.py <session-id>            # dry-run (prints decisions)
    python src/namefix_session.py <session-id> --write    # apply
"""
import argparse
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
from corrections import apply_corrections  # noqa: E402
from model_cache import cached_or_run, fingerprint  # noqa: E402
import namefix  # noqa: E402


def score_vs_key(session_dir: Path, result: dict) -> None:
    """If the session has a by-ear key in validation-notes.json, score every auto/queued
    decision against it (join by segment_id+word_index) so the human can see the confusion
    BEFORE applying. Sessions without a key just skip this."""
    notes_path = session_dir / "validation-notes.json"
    if not notes_path.exists():
        return
    from detectors.phonetics import clean, codes
    key = {}
    for n in json.loads(notes_path.read_text()).get("notes", []):
        ans = (n.get("text") or "").strip()
        if ans and n.get("segmentId") is not None and n.get("wordIndex") is not None:
            key[(n["segmentId"], n["wordIndex"])] = ans
    if not key:
        return
    tally = {"auto_right": 0, "auto_wrong": 0, "queued_right": 0, "queued_wrong": 0}
    wrongs = []
    for action, groups in (("auto", result["auto"]), ("queued", result["pending"])):
        for g in groups:
            for o in g["occurrences"]:
                ans = key.get((o["segment_id"], o["word_index"]))
                if ans is None:
                    continue
                right = (clean(g["canonical"]) == clean(ans)
                         or bool(codes(clean(g["canonical"])) & codes(clean(ans))))
                tally[f"{action}_{'right' if right else 'wrong'}"] += 1
                if action == "auto" and not right:
                    wrongs.append(f"{o['token']!r}@seg{o['segment_id']} -> "
                                  f"{g['canonical']} (key: {ans})")
    print(f"\n  SCORED vs by-ear key: auto right {tally['auto_right']} | "
          f"auto WRONG {tally['auto_wrong']} | queued right {tally['queued_right']} | "
          f"queued wrong {tally['queued_wrong']}")
    for w in wrongs:
        print(f"    AUTO-WRONG: {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--write", action="store_true", help="apply (default: dry-run)")
    args = ap.parse_args()

    session_dir = get_session_dir(ROOT / "sessions", args.session_id)
    rich_path = session_dir / "transcript-rich.json"
    audio_path = next(iter(session_dir.glob("audio.*")), None)
    if audio_path is None:
        raise SystemExit(f"no audio file in {session_dir}")
    transcript = json.loads(rich_path.read_text())
    if not transcript.get("_stories"):
        raise SystemExit("no _stories on this transcript — run segmentation first")

    print(f"[namefix-session] {args.session_id}  mode={'WRITE' if args.write else 'dry-run'}")
    # Same cache as the pipeline stage: a dry-run computes and caches; a subsequent --write
    # applies EXACTLY the reviewed decisions (no second model run, no Metal wobble between
    # what you saw and what gets written). A transcript/config change recomputes.
    result, was_cached = cached_or_run(
        session_dir, "namefix",
        fingerprint(namefix.transcript_fingerprint(transcript)),
        fingerprint(namefix.config_fingerprint()),
        lambda: namefix.run_namefix(transcript, str(audio_path)))
    if was_cached:
        print("  (decisions served from the namefix cache — identical to the reviewed run)")

    for w in result["worlds"]:
        print(f"  story {w['story_id']} \"{w['title']}\": recognized={w['recognized_world']!r} "
              f"({w['n_candidates']} candidates)")
    print(f"\n  AUTO ({sum(len(g['occurrences']) for g in result['auto'])} occurrences, "
          f"{len(result['auto'])} names):")
    for g in sorted(result["auto"], key=lambda x: -len(x["occurrences"])):
        print(f"    {g['heard']:16} -> {g['canonical']:16} [{g['method']}] "
              f"x{len(g['occurrences'])}")
    print(f"\n  QUEUED ({len(result['pending'])} names):")
    for g in result["pending"]:
        print(f"    {g['heard']:16} -> {g['canonical']:16} [{g['method']}] "
              f"x{len(g['occurrences'])}")

    score_vs_key(session_dir, result)

    if not args.write:
        print("\n  dry-run: nothing written. Re-run with --write to apply the SAME decisions.")
        return

    # ---- apply, surgically ----
    before_ids = [s["id"] for s in transcript["segments"]]
    before_counts = [len(s.get("words") or []) for s in transcript["segments"]]

    corrections = namefix.auto_to_corrections(result["auto"])
    fixed, n = apply_corrections(transcript, corrections, "namefix")

    after_ids = [s["id"] for s in fixed["segments"]]
    after_counts = [len(s.get("words") or []) for s in fixed["segments"]]
    assert after_ids == before_ids, "segment ids changed — ABORT (axial labels would unbind)"
    assert after_counts == before_counts, "word counts changed — ABORT"
    print(f"\n  invariants OK: {len(after_ids)} segment ids + word counts unchanged")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = rich_path.with_suffix(f".json.pre-namefix-{stamp}")
    shutil.copy2(rich_path, backup)
    print(f"  backup: {backup.name}")

    tmp = rich_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fixed, ensure_ascii=False, indent=2))
    tmp.replace(rich_path)
    namefix.write_pending(session_dir, result, fixed)
    print(f"  wrote transcript-rich.json ({n} words corrected) + pending-name-corrections.json")
    print(f"  NOTE: detections are now stale — re-scan: python src/detect.py {args.session_id}")


if __name__ == "__main__":
    main()
