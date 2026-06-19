#!/usr/bin/env python3
"""The best-working combination: Qwen3-4B-generated cast list (clean, knows Karna) + FUZZY
sound-matching (catches degraded names exact matching misses). Stacks the two levers that
actually worked. Scored on the real Mahabharata (by-ear key) and the 7-world spread.

    ./venv/bin/python emp/src/bench_cast_fuzzy.py [model_id]
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
from bench_cast_qwen import make_qwen_reader, generate_cast, MODEL   # noqa: E402  (tolerant parse)
from fuzzy_canon import flags_for                                    # noqa: E402  (fuzzy matcher)

CODE_ED, MIN_RATIO = 1, 0.55      # the sweet spot from the fuzzy sweep


def fuzzy_flags(cards, singles, cast):
    canon_forms, code_to_name = cast_index(cast)
    return flags_for(cards, singles, canon_forms, code_to_name, CODE_ED, MIN_RATIO)


def main():
    gen = make_qwen_reader(MODEL)
    tag = MODEL.split("/")[-1]

    print(f"\n=== PART A — real Mahabharata (by-ear), {tag} cast + FUZZY match ===")
    SID = "20260211-210718"
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    segs = rich["segments"]
    cards = story_name_cards(segs, recover=True)
    singles = proper_name_candidates(segs)
    cast = generate_cast(gen, "Mahabharata")
    r = score_canon(load_items(SID) or {}, fuzzy_flags(cards, singles, cast))
    real, n = r["matrix"]["M9c"]["flagged"], r["flagged"]
    print(f"  recall {r['caught']}/{r['gold_m9c']}   precision {real}/{n}   caught {r['hits']}")
    print(f"  (exact-match + Qwen cast was 4/11; best clear-error recall so far 4/5)", flush=True)

    print(f"\n=== PART B — spread, {tag} cast + FUZZY match ===")
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    tot = caught = fp = 0
    for st in spread:
        segs = build_segs(st["lines"])
        cards = story_name_cards(segs, recover=True)
        singles = proper_name_candidates(segs)
        cast = generate_cast(gen, st["world"])
        flags = fuzzy_flags(cards, singles, cast)
        cg = m9c_caught(flags)
        rc = sum(1 for p in st["planted"] if hit(p["heard"], cg))
        pf = planted_forms(st["planted"])
        fpc = sum(1 for f in flags if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pf))
        tot += len(st["planted"]); caught += rc; fp += fpc
        print(f"  {st['world'][:34]:36} {rc}/{len(st['planted'])}   (false {fpc})", flush=True)
    print(f"  SPREAD recall {caught}/{tot} = {caught / tot:.2f}   false flags {fp}")
    print(f"  (exact-match + Qwen cast: 21/40 = 0.53; Gemma cast: 0.50)")


if __name__ == "__main__":
    main()
