#!/usr/bin/env python3
"""Diagnostic: for chosen worlds, show WHY a planted name was missed — the model's verdict
on each name card, and whether the dictionary gate dropped a real canon flag. Separates a
model miss (verdict != canon_wrong) from a gate over-drop (flagged, then gated out).

    ./venv/bin/python emp/src/diag_canon_spread.py "Star Wars" "Steven Universe"
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from story_segment import make_reader                                      # noqa: E402
from detectors.story_names._audit import (story_name_cards, _run_card_pass,  # noqa: E402
                                          gate_canon_flags)
from detectors.story_names._names import proper_name_candidates           # noqa: E402
from detectors.phonetics import clean                                     # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit                # noqa: E402

WANT = sys.argv[1:] or ["Star Wars", "Steven Universe"]
bench = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())
stories = {s["world"]: s for s in bench["stories"]}
gen = make_reader()

for world in WANT:
    st = stories[world]
    segs = build_segs(st["lines"])
    singles = proper_name_candidates(segs)
    cards = story_name_cards(segs, recover=True)
    pre, verdicts = _run_card_pass(gen, world, cards, [])     # before the gate
    post = gate_canon_flags([dict(f) for f in pre], singles)  # after the gate
    pre_caught = m9c_caught(pre)
    post_caught = m9c_caught(post)

    print(f"\n{'=' * 70}\n{world}\n{'=' * 70}")
    print("  model verdicts per name card:")
    for v in verdicts:
        print(f"    {','.join(v['spellings'])[:30]:32} -> {v['verdict']:13} "
              f"canon={v['canonical']!r}")
    print("  planted -> outcome:")
    for p in st["planted"]:
        in_pre = hit(p["heard"], pre_caught)
        in_post = hit(p["heard"], post_caught)
        if in_post:
            tag = "CAUGHT"
        elif in_pre:
            tag = "GATE DROPPED IT (model flagged, gate removed)"
        else:
            tag = "model miss (never flagged canon_wrong)"
        print(f"    {p['heard']!r:20} -> {p['correct']!r:22} {tag}")
