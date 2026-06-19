#!/usr/bin/env python3
"""Per-error coverage: which model's cast catches which name error (fuzzy match). Answers
"do we need the other models, or is Qwen3.5 enough?" by showing exactly which errors are caught
ONLY because Gemma/Qwen3-4B contributed a name Qwen3.5's cast lacked. No GPU.

    ./venv/bin/python emp/src/analyze_coverage.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from detectors.story_names._audit import story_name_cards            # noqa: E402
from detectors.story_names._names import proper_name_candidates      # noqa: E402
from score_canon_heldout import score_canon, load_items             # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit           # noqa: E402
from bench_cast import cast_index                                    # noqa: E402
from fuzzy_canon import flags_for                                    # noqa: E402

SID, CE, MR = "20260211-210718", 1, 0.55
SRC_ORDER = ["gemma4", "qwen3-4b", "qwen35"]
CASTDIR = ROOT / "emp" / "results" / "canon-spread" / "casts"


def ff(cards, singles, cast):
    return flags_for(cards, singles, *cast_index(cast), CE, MR)


def main():
    sources = {p.stem: json.loads(p.read_text())["casts"] for p in sorted(CASTDIR.glob("*.json"))}
    order = [t for t in SRC_ORDER if t in sources]

    # ---- real Mahabharata ----
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    mcards = story_name_cards(rich["segments"], recover=True)
    msingles = proper_name_candidates(rich["segments"])
    items = load_items(SID) or {}
    hits = {t: set(score_canon(items, ff(mcards, msingles, sources[t].get("Mahabharata", [])))["hits"]) for t in order}
    print("REAL MAHABHARATA — which models catch each error:")
    for e in sorted(set().union(*hits.values())):
        print(f"  {e:16} {'  '.join('Y' if e in hits[t] else '·' for t in order)}    ({', '.join(order)})")
    q35 = hits.get("qwen35", set())
    others = set().union(*(hits[t] for t in order if t != "qwen35"))
    print(f"  >> caught by others but NOT Qwen3.5: {sorted(others - q35) or 'none'}")
    print(f"  >> caught ONLY by Qwen3.5:           {sorted(q35 - others) or 'none'}")

    # ---- spread ----
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    need_others, only_q35 = [], []
    for st in spread:
        segs = build_segs(st["lines"])
        cards = story_name_cards(segs, recover=True)
        singles = proper_name_candidates(segs)
        cg = {t: m9c_caught(ff(cards, singles, sources[t].get(st["world"], []))) for t in order}
        for pl in st["planted"]:
            q = hit(pl["heard"], cg.get("qwen35", set()))
            oth = any(hit(pl["heard"], cg[t]) for t in order if t != "qwen35")
            if oth and not q:
                need_others.append((st["world"], pl["heard"], pl["correct"]))
            elif q and not oth:
                only_q35.append((st["world"], pl["heard"], pl["correct"]))
    print("\nSPREAD — errors the OTHER models catch but Qwen3.5 misses (the case FOR keeping them):")
    for w, h, c in need_others:
        print(f"   {w[:26]:28} {h!r} -> {c}")
    print(f"   ({len(need_others)} total)")
    print("\nSPREAD — errors ONLY Qwen3.5 catches:")
    for w, h, c in only_q35:
        print(f"   {w[:26]:28} {h!r} -> {c}")
    print(f"   ({len(only_q35)} total)")


if __name__ == "__main__":
    main()
