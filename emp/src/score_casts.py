#!/usr/bin/env python3
"""Score every cached cast list (emp/results/canon-spread/casts/*.json) — individually AND as a
union — with fuzzy matching, on the real Mahabharata and the spread. No GPU.

    ./venv/bin/python emp/src/score_casts.py
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
from bench_canon_spread import build_segs, m9c_caught, hit, forms    # noqa: E402
from bench_cast import cast_index, planted_forms                     # noqa: E402
from fuzzy_canon import flags_for                                    # noqa: E402

CODE_ED, MIN_RATIO = 1, 0.55
SID = "20260211-210718"
CASTDIR = ROOT / "emp" / "results" / "canon-spread" / "casts"


def fuzzy_flags(cards, singles, cast):
    return flags_for(cards, singles, *cast_index(cast), CODE_ED, MIN_RATIO)


def main():
    sources = {p.stem: json.loads(p.read_text())["casts"] for p in sorted(CASTDIR.glob("*.json"))}
    if not sources:
        print("no cached casts yet — run gen_casts.py first")
        return
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    worlds = list(dict.fromkeys(["Mahabharata"] + [s["world"] for s in spread]))

    # union of every model's cast, per world
    union = {}
    for w in worlds:
        seen, names = set(), []
        for casts in sources.values():
            for n in casts.get(w, []):
                if n.lower() not in seen:
                    seen.add(n.lower())
                    names.append(n)
        union[w] = names
    sources["UNION"] = union

    # precompute cards once
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    mcards = story_name_cards(rich["segments"], recover=True)
    msingles = proper_name_candidates(rich["segments"])
    items = load_items(SID) or {}
    spread_cards = [(st, story_name_cards(build_segs(st["lines"]), recover=True),
                     proper_name_candidates(build_segs(st["lines"]))) for st in spread]

    print(f"{'cast source':22} {'maha recall':>11} {'maha prec':>10} {'spread':>9} {'false':>6}  maha-names/Pandavas")
    print("-" * 86)
    for tag, casts in sources.items():
        mr = score_canon(items, fuzzy_flags(mcards, msingles, casts.get("Mahabharata", [])))
        real, n = mr["matrix"]["M9c"]["flagged"], mr["flagged"]
        tot = caught = fp = 0
        for st, cards, singles in spread_cards:
            flags = fuzzy_flags(cards, singles, casts.get(st["world"], []))
            cg = m9c_caught(flags)
            caught += sum(1 for pl in st["planted"] if hit(pl["heard"], cg))
            pf = planted_forms(st["planted"])
            fp += sum(1 for f in flags if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pf))
            tot += len(st["planted"])
        mcast = casts.get("Mahabharata", [])
        has_p = "pandavas" in {x.lower() for x in mcast}
        print(f"{tag:22} {str(mr['caught']) + '/' + str(mr['gold_m9c']):>11} "
              f"{str(real) + '/' + str(n):>10} {str(caught) + '/' + str(tot):>9} {fp:>6}  "
              f"{len(mcast)} names{' [Pandavas]' if has_p else ''}")
    print("\n  reference: Gemma cast + exact 0.50; Qwen cast + fuzzy 0.62; clear-error ceiling ~4-5/5")


if __name__ == "__main__":
    main()
