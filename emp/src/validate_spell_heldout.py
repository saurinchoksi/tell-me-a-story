#!/usr/bin/env python3
"""Validate the spell-it canon audit on the two HELD-OUT real sessions, against the by-ear key.

Segments each held-out session ONCE, then runs BOTH the production worksheet (run_v2) and the
experimental spell-it (run_spell) on the same cards, and scores both with the real scorer
(score_canon) against the human by-ear key. Apples-to-apples — same world, same cards, only the
audit differs. This is the real-data trust gate for the synthetic 0.20 -> 0.57 result.

    ./venv/bin/python emp/src/validate_spell_heldout.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from story_segment import make_reader, segment_session              # noqa: E402
from detectors.story_names._audit import story_name_cards, run_v2   # noqa: E402
from detectors.story_names._names import story_segments             # noqa: E402
from audit_spell import run_spell                                   # noqa: E402
from score_canon_heldout import score_canon, load_items            # noqa: E402

HELDOUT = ["20260211-210718", "20260414-213156"]


def regions_from(stories, pos_of):
    out = []
    for i, st in enumerate(stories):
        sp, ep = pos_of.get(st["start_id"]), pos_of.get(st["end_id"])
        if sp is None or ep is None:
            continue
        if ep < sp:
            sp, ep = ep, sp
        out.append({"idx": i, "start_pos": sp, "end_pos": ep, "world": st.get("world", "")})
    return out


def audit_both(gen, sid):
    """Segment once; run both audits on the same cards. Returns (old_flags, new_flags, worlds)."""
    rich = json.loads((ROOT / "sessions" / sid / "transcript-rich.json").read_text())
    pos_of = {s["id"]: i for i, s in enumerate(rich["segments"])}
    seg_result, _ = segment_session(ROOT / "sessions" / sid, gen)
    old_flags, new_flags, worlds = [], [], []
    for r in regions_from(seg_result["stories"], pos_of):
        segs = story_segments(rich, r, pos_of)
        cards = story_name_cards(segs, recover=True)
        worlds.append(r["world"])
        old_flags += [f for f in run_v2(gen, r["world"], segs, cards, [])[0] if f.get("case") == "M9c"]
        new_flags += run_spell(gen, r["world"], segs, cards)
    return old_flags, new_flags, worlds


def prec(r):
    real, n = r["matrix"]["M9c"]["flagged"], r["flagged"]
    return f"{real}/{n} = {real / n:.2f}" if n else f"{real}/{n} (no flags)"


def main():
    gen = make_reader()
    for sid in HELDOUT:
        items = load_items(sid) or {}
        old, new, worlds = audit_both(gen, sid)
        ro, rn = score_canon(items, old), score_canon(items, new)
        print(f"\n{'=' * 70}\n{sid}   worlds={worlds}\n{'=' * 70}", flush=True)
        print(f"  OLD worksheet : recall {ro['caught']}/{ro['gold_m9c']}   precision {prec(ro)}")
        print(f"  NEW spell-it  : recall {rn['caught']}/{rn['gold_m9c']}   precision {prec(rn)}")
        if rn["hits"]:
            print(f"     new catches: {rn['hits']}")
        if rn["misses"]:
            print(f"     still missed: {rn['misses']}")
        for name, g in rn["false_pos"]:
            print(f"     NEW false flag: {name!r} (gold {g})")


if __name__ == "__main__":
    main()
