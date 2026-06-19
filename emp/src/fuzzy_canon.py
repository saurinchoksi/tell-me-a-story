#!/usr/bin/env python3
"""Does FUZZY sound-matching catch the degraded names that exact Double-Metaphone misses?

Exact-code matching caught bishma/garn/urjun but missed Bandos->Pandavas and Urzi->Arjuna
(their sound-codes differ). This relaxes the match: a token matches a canon name if their DM
codes are within a small edit distance AND the spellings are similar enough (a string-ratio
guard to keep precision). Swept over a few thresholds, scored on the real Mahabharata (recall +
precision vs the by-ear key) AND the made-up controls (any flag = a false alarm).

NO MODEL — pure code, safe to run while a model job holds the GPU. Uses data/mahabharata.json as
a stand-in name list; the matcher change is independent of where the list comes from, so the
result carries over to the model-generated cast.

    ./venv/bin/python emp/src/fuzzy_canon.py
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from detectors.story_names._audit import story_name_cards, gate_canon_flags  # noqa: E402
from detectors.story_names._names import proper_name_candidates              # noqa: E402
from detectors.phonetics import codes, clean                                 # noqa: E402
from score_canon_heldout import score_canon, load_items                     # noqa: E402
from phonetic_canon import canon_index, MIN_LEN                             # noqa: E402

SID = "20260211-210718"


def lev(a, b):
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match(tok, canon_forms, code_to_name, code_ed, min_ratio):
    """Return the canon name `tok` resolves to, or None. Exact DM-code first; then fuzzy:
    a canon code within `code_ed` edits AND spelling-similarity >= `min_ratio`."""
    if tok in canon_forms:
        return None                                          # already a correct spelling
    tcodes = codes(tok)
    for c in tcodes:
        if c in code_to_name:
            return code_to_name[c]                           # exact sound match
    if code_ed <= 0:
        return None
    best, best_r = None, 0.0
    for c in tcodes:
        for cc, name in code_to_name.items():
            if lev(c, cc) <= code_ed:
                r = SequenceMatcher(None, tok, clean(name)).ratio()
                if r >= min_ratio and r > best_r:
                    best, best_r = name, r
    return best


def flags_for(cards, singles, canon_forms, code_to_name, code_ed, min_ratio):
    flags = []
    for card in cards:
        for cl in card["clean"]:
            for tok in cl.split(" "):
                if len(tok) < MIN_LEN:
                    continue
                m = match(tok, canon_forms, code_to_name, code_ed, min_ratio)
                if m:
                    flags.append({"case": "M9c", "canonical": m,
                                  "wrong_surface": [tok], "wrong_cleaned": [tok],
                                  "all_spellings": card["surface"], "card_id": card["id"],
                                  "evidence": card["examples"][0] if card["examples"] else ""})
    return gate_canon_flags(flags, singles)


def control_false(canon_forms, code_to_name, code_ed, min_ratio):
    """Total M9c flags on the made-up control stories — every one is a false alarm."""
    from bench_canon_spread import build_segs
    controls = json.loads((ROOT / "emp" / "results" / "canon-spread" / "controls.json").read_text())["stories"]
    total = 0
    for st in controls:
        segs = build_segs(st["lines"])
        cards = story_name_cards(segs, recover=True)
        singles = proper_name_candidates(segs)
        fl = flags_for(cards, singles, canon_forms, code_to_name, code_ed, min_ratio)
        total += sum(1 for f in fl if f.get("case") == "M9c")
    return total


def main():
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    segs = rich["segments"]
    cards = story_name_cards(segs, recover=True)
    singles = proper_name_candidates(segs)
    canon_forms, code_to_name = canon_index(ROOT / "data" / "mahabharata.json")
    items = load_items(SID) or {}

    print(f"{'setting':28} {'maha recall':>12} {'precision':>11} {'control FP':>11}")
    print("-" * 64)
    configs = [
        ("exact only", 0, 1.0),
        ("code<=1, ratio>=0.70", 1, 0.70),
        ("code<=1, ratio>=0.55", 1, 0.55),
        ("code<=1, ratio>=0.45", 1, 0.45),
        ("code<=2, ratio>=0.55", 2, 0.55),
    ]
    for name, ce, mr in configs:
        fl = flags_for(cards, singles, canon_forms, code_to_name, ce, mr)
        r = score_canon(items, fl)
        real, n = r["matrix"]["M9c"]["flagged"], r["flagged"]
        cfp = control_false(canon_forms, code_to_name, ce, mr)
        prec = f"{real}/{n}" + (f"={real / n:.2f}" if n else "")
        print(f"{name:28} {str(r['caught']) + '/' + str(r['gold_m9c']):>12} {prec:>11} {cfp:>11}")
        if ce >= 1:
            print(f"     caught: {r['hits']}")


if __name__ == "__main__":
    main()
