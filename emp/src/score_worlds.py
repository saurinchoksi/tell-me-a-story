#!/usr/bin/env python3
"""Score a world-classifier candidate against the 100-story benchmark.  READ-ONLY.

This is the ONLY side that reads the gold labels in `stories.json`
(`world_gold`/`world_aliases`/`bucket`/`difficulty`). The runner (`bench_worlds.py`)
never does — that firewall keeps the eval honest, the same split as
`score_names.py` vs `audit_names.py`.

Outcome per story (the prediction is the model's `world` string; empty = "made up"):
  gold is a real world (bucket canon|hybrid):
    names THIS world (matches world_gold or an alias)   -> correct
    says made-up / empty                                -> MISSED CANON  (the Feb 11 error)
    names something else                                -> wrong-world
  gold is made-up (bucket made_up):
    says made-up / empty                                -> correct
    names any world                                     -> FALSE WORLD

Headline = overall accuracy, with MISSED-CANON and FALSE-WORLD called out (they cost
differently), plus a breakdown by difficulty and by world. Off-diagonals are printed.

    ./venv/bin/python emp/src/score_worlds.py                 # score every pred/*.json
    ./venv/bin/python emp/src/score_worlds.py current general_abstain
    ./venv/bin/python emp/src/score_worlds.py current.sampled  # the ablation arm
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "emp" / "results" / "worlds-bench"
STORIES = BENCH / "stories.json"
PRED_DIR = BENCH / "pred"

# Phrases that mean "this is an original / made-up story, no known world."
NONE_MARKERS = ("made up", "madeup", "make believe", "makebelieve", "original",
                "invent", "imaginar", "fiction", "no world", "not a real",
                "not real", "no known", "none", "unknown", "n a", "generic")
OUTCOMES = ["correct", "missed_canon", "wrong_world", "false_world"]


def normalize(s):
    """Lowercase, letters/digits to spaces, collapse, drop a leading 'the'."""
    n = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    n = re.sub(r"\s+", " ", n)
    if n.startswith("the "):
        n = n[4:]
    return n


def claims_none(pred):
    """Did the model say 'made up / no known world' (or return empty)?"""
    n = normalize(pred)
    if n == "":
        return True
    return any(m in n for m in NONE_MARKERS)


def identifies(pred, story):
    """The alias/world this prediction names for `story`, or None. A match is a
    two-way substring on the normalized forms (so 'the star wars universe' matches
    'star wars'); aliases shorter than 3 chars are ignored to avoid junk hits."""
    normp = normalize(pred)
    if not normp:
        return None
    candidates = [story["world_gold"]] + list(story.get("world_aliases", []))
    for alias in candidates:
        a = normalize(alias)
        if len(a) >= 3 and (a in normp or normp in a):
            return alias
    return None


def classify(story, pred):
    gold_is_world = story["bucket"] in ("canon", "hybrid")
    if gold_is_world:
        if identifies(pred, story):
            return "correct"
        if claims_none(pred):
            return "missed_canon"
        return "wrong_world"
    # made_up
    return "correct" if claims_none(pred) else "false_world"


def load_gold():
    data = json.loads(STORIES.read_text())
    return {s["id"]: s for s in data["stories"]}


def score(gold, pred_path):
    pred = json.loads(pred_path.read_text())
    by_id = {r["id"]: r["world_pred"] for r in pred["results"]}
    rows = []
    for sid, p in by_id.items():
        story = gold.get(sid)
        if story is None:
            continue
        outcome = classify(story, p)
        rows.append({
            "id": sid, "bucket": story["bucket"], "difficulty": story["difficulty"],
            "world_gold": story["world_gold"], "pred": p, "outcome": outcome,
        })
    return pred, rows


def _pct(n, d):
    return f"{n}/{d} = {n / d:.2f}" if d else f"{n}/{d} = —"


def report(pred, rows):
    n = len(rows)
    by_outcome = Counter(r["outcome"] for r in rows)
    correct = by_outcome["correct"]
    world_rows = [r for r in rows if r["bucket"] in ("canon", "hybrid")]
    made_rows = [r for r in rows if r["bucket"] == "made_up"]

    print(f"\n{'#' * 74}")
    print(f"## {pred['candidate']!r}  ({pred.get('input_mode', '?')} input, {n} stories)")
    print('#' * 74)
    print(f"  OVERALL accuracy        {_pct(correct, n)}")
    print(f"  ! MISSED CANON (said made-up on a real world)  "
          f"{by_outcome['missed_canon']}   <- the costly Feb-11 error")
    print(f"  ! FALSE WORLD  (named a world on a made-up story) {by_outcome['false_world']}")
    print(f"    wrong-world (named the wrong real world)        {by_outcome['wrong_world']}")

    # by bucket
    print("\n  by bucket:")
    for b in ("canon", "hybrid", "made_up"):
        rs = [r for r in rows if r["bucket"] == b]
        if rs:
            c = sum(1 for r in rs if r["outcome"] == "correct")
            print(f"    {b:8s} {_pct(c, len(rs))}")

    # by difficulty (world buckets only — 'difficulty' on made-up is about the trap)
    print("\n  by difficulty (canon+hybrid — does it hold as the tells thin out?):")
    for d in ("obvious", "medium", "subtle"):
        rs = [r for r in world_rows if r["difficulty"] == d]
        if rs:
            c = sum(1 for r in rs if r["outcome"] == "correct")
            print(f"    {d:8s} {_pct(c, len(rs))}")
    print("  made-up by difficulty (trap resistance):")
    for d in ("obvious", "medium", "subtle"):
        rs = [r for r in made_rows if r["difficulty"] == d]
        if rs:
            c = sum(1 for r in rs if r["outcome"] == "correct")
            print(f"    {d:8s} {_pct(c, len(rs))}")

    # by world (which canons does it actually know?)
    print("\n  by world (canon+hybrid):")
    per_world = defaultdict(list)
    for r in world_rows:
        per_world[r["world_gold"]].append(r)
    for world in sorted(per_world):
        rs = per_world[world]
        c = sum(1 for r in rs if r["outcome"] == "correct")
        print(f"    {world:34s} {_pct(c, len(rs))}")

    # off-diagonals
    print("\n  --- off-diagonals (every non-correct story) ---")
    order = {"missed_canon": 0, "false_world": 1, "wrong_world": 2}
    for r in sorted([r for r in rows if r["outcome"] != "correct"],
                    key=lambda r: (order.get(r["outcome"], 9), r["bucket"], r["id"])):
        gold = r["world_gold"] or "(made-up)"
        print(f"    {r['outcome']:13s} {r['id']:6s} {r['bucket']:8s}/{r['difficulty']:7s} "
              f"gold={gold!r}  pred={r['pred']!r}")

    return {"candidate": pred["candidate"], "mode": pred.get("input_mode", "?"),
            "n": n, "acc": correct / n if n else 0.0,
            "missed_canon": by_outcome["missed_canon"],
            "false_world": by_outcome["false_world"],
            "wrong_world": by_outcome["wrong_world"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="*",
                    help="candidate names or pred filenames (default: all pred/*.json)")
    args = ap.parse_args()

    if not STORIES.exists():
        sys.exit(f"no benchmark at {STORIES} — author stories.json first")
    gold = load_gold()

    if args.preds:
        paths = []
        for p in args.preds:
            cand = PRED_DIR / (p if p.endswith(".json") else f"{p}.json")
            if not cand.exists():
                sys.exit(f"no prediction file {cand}")
            paths.append(cand)
    else:
        paths = sorted(PRED_DIR.glob("*.json"))
        if not paths:
            sys.exit(f"no prediction files in {PRED_DIR} — run bench_worlds.py first")

    print("=" * 74)
    print(f"WORLD-CLASSIFIER BENCHMARK — scored vs {len(gold)} gold stories")
    print("=" * 74)
    summaries = []
    for path in paths:
        pred, rows = score(gold, path)
        summaries.append(report(pred, rows))

    print(f"\n{'=' * 74}\nSIDE-BY-SIDE\n{'=' * 74}")
    print(f"  {'candidate':22s} {'mode':8s} {'acc':>6s}  {'missed':>6s} {'false':>6s} {'wrong':>6s}")
    for s in summaries:
        print(f"  {s['candidate']:22s} {s['mode']:8s} {s['acc']:6.2f}  "
              f"{s['missed_canon']:6d} {s['false_world']:6d} {s['wrong_world']:6d}")


if __name__ == "__main__":
    main()
