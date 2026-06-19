#!/usr/bin/env python3
"""Compare the production worksheet audit (run_v2) against the experimental spell-it audit
(run_spell) on BOTH axes:

  - recall  : the 7-world spread (40 planted misspellings) — does spell-it catch more?
  - precision: false flags on (a) non-planted names in the spread, and (b) made-up control
    stories run with the recognized world AND force-fed "Star Wars" — does spell-it invent
    canon corrections for invented names?

One model load. Read-only. Experiment — neither path is wired to production.

    ./venv/bin/python emp/src/bench_spell.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from story_segment import make_reader, pass2_name                     # noqa: E402
from detectors.story_names._audit import story_name_cards, run_v2     # noqa: E402
from detectors.phonetics import clean                                 # noqa: E402
from audit_spell import run_spell                                     # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit, forms     # noqa: E402

SPREAD = ROOT / "emp" / "results" / "canon-spread" / "stories.json"
CONTROLS = ROOT / "emp" / "results" / "canon-spread" / "controls.json"


def planted_forms(planted):
    s = set()
    for p in planted:
        s |= {clean(t) for t in p["heard"].split() if clean(t)}
        s.add(clean(p["heard"]))
    return s


def n_recall(flags, planted):
    caught = m9c_caught(flags)
    return sum(1 for p in planted if hit(p["heard"], caught))


def n_false(flags, pforms):
    """M9c flags landing on a token that is NOT a planted misspelling = a false positive."""
    n = 0
    for f in flags:
        if f.get("case") != "M9c":
            continue
        if not (forms(f.get("wrong_cleaned", [])) & pforms):
            n += 1
    return n


def lines_text(segs):
    return "\n".join(f'[{s["id"]}] "{s["text"]}"' for s in segs if s["text"])


def run():
    gen = make_reader()
    spread = json.loads(SPREAD.read_text())["stories"]
    controls = json.loads(CONTROLS.read_text())["stories"]

    print("=" * 74)
    print("RECALL on the spread (correct world handed in)        old(run_v2)   new(spell)")
    print("=" * 74)
    rec_old = rec_new = fp_old = fp_new = tot = 0
    for st in spread:
        segs = build_segs(st["lines"])
        cards = story_name_cards(segs, recover=True)
        world, planted = st["world"], st["planted"]
        old = run_v2(gen, world, segs, cards, [])[0]
        new = run_spell(gen, world, segs, cards)
        ro, rn = n_recall(old, planted), n_recall(new, planted)
        pforms = planted_forms(planted)
        fo, fn = n_false(old, pforms), n_false(new, pforms)
        rec_old += ro; rec_new += rn; fp_old += fo; fp_new += fn; tot += len(planted)
        print(f"  {world[:36]:38} {ro}/{len(planted):<3}->{'':2}{rn}/{len(planted):<5}  "
              f"(extra non-planted flags: old {fo}, new {fn})", flush=True)
    print("-" * 74)
    print(f"  {'TOTAL recall':38} {rec_old}/{tot} = {rec_old / tot:.2f}    "
          f"{rec_new}/{tot} = {rec_new / tot:.2f}")
    print(f"  spread non-planted false flags                old {fp_old}    new {fp_new}")

    print("\n" + "=" * 74)
    print("PRECISION on made-up controls (every flag here is a FALSE alarm)")
    print("=" * 74)
    cf_old = cf_new = cf_old_f = cf_new_f = 0
    for st in controls:
        segs = build_segs(st["lines"])
        cards = story_name_cards(segs, recover=True)
        _, pred = pass2_name(gen, lines_text(segs), [])
        # (a) the recognized world (production path)
        o_p = n_false(run_v2(gen, pred, segs, cards, [])[0], set())
        n_p = n_false(run_spell(gen, pred, segs, cards), set())
        # (b) force-fed a real world — the over-correction stress
        o_f = n_false(run_v2(gen, "Star Wars", segs, cards, [])[0], set())
        n_f = n_false(run_spell(gen, "Star Wars", segs, cards), set())
        cf_old += o_p; cf_new += n_p; cf_old_f += o_f; cf_new_f += n_f
        tag = pred if pred else "(made-up)"
        print(f"  control story  recognized->{tag[:22]:24} "
              f"flags: old {o_p} / new {n_p}    forced-SW: old {o_f} / new {n_f}", flush=True)
    print("-" * 74)
    print(f"  controls, recognized world : old {cf_old}  new {cf_new}   (want 0)")
    print(f"  controls, forced Star Wars : old {cf_old_f}  new {cf_new_f}   (over-correction stress; want 0)")

    print("\n" + "=" * 74)
    print(f"VERDICT  recall {rec_old}/{tot} -> {rec_new}/{tot}   "
          f"| spread false flags {fp_old} -> {fp_new}   "
          f"| control false flags {cf_old + cf_old_f} -> {cf_new + cf_new_f}")
    print("=" * 74)


if __name__ == "__main__":
    run()
