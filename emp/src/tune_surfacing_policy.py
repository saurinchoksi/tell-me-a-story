#!/usr/bin/env python3
"""Sweep the SURFACING POLICY for the canon detector (M9c), holding the voting fixed.

The shipped detector computes a combined catch set (order-robust judge UNION sound-match) and
then surfaces ONLY the "confident" ones — those whose suggested spelling shares a Double-Metaphone
code with the heard token. That drops genuine-but-badly-garbled catches (Dhrashtra->Dhritarashtra
5/7, Yudhisthir->Yudhishthira 3/7 on a real Mahabharata night). This asks: if we ALSO surfaced a
lower-confidence "best guess" tier, how much real recall do we gain and how much precision do we pay?

It reuses the PRODUCTION judge layer (the same functions tune_judge_vote.py drives) so the numbers
transfer. Voting is fixed at the shipped K=7 / cutoff=3; the only variable is which combined catches
we show. Scored on the held-out Mahabharata by-ear key AND the synthetic 7-world planted spread.

    ./venv/bin/python emp/src/tune_surfacing_policy.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

import detectors.story_names._qwen35 as q                              # noqa: E402
from detectors.story_names._audit import story_name_cards, gate_canon_flags  # noqa: E402
from detectors.story_names._names import proper_name_candidates        # noqa: E402
from detectors.phonetics import clean, codes                          # noqa: E402
from qwen35 import make_reader                                         # noqa: E402
from score_canon_heldout import score_canon, load_items               # noqa: E402
from bench_canon_spread import build_segs, hit, forms                 # noqa: E402

HELDOUT = "20260211-210718"   # the never-tuned Mahabharata held-out (single story)
K, CUTOFF = 7, 3              # the SHIPPED voting config — fixed; we vary only surfacing

# Surfacing policies: keep a combined catch iff predicate(confident, votes) is true.
# A phonetic (sound-matched) catch is always confident; a judge-only catch may not be.
POLICIES = {
    "confident-only (today)": lambda conf, v: conf,
    "+best-guess vote>=5":    lambda conf, v: conf or v >= 5,
    "+best-guess vote>=4":    lambda conf, v: conf or v >= 4,
    "+best-guess (all)":      lambda conf, v: True,
}

gen = make_reader()


def runs_for(world, names, k=K, seed_base=0):
    """k UNGATED production judge passes over deterministic shuffles."""
    out = []
    for r in range(k):
        sh = list(names)
        random.Random(seed_base + r).shuffle(sh)
        out.append(q._judge_raw(gen, world, sh))
    return out


def voted_with_counts(runs, k=K, cutoff=CUTOFF):
    """The shipped vote, but each surviving flag keeps its vote_count so a surfacing policy
    can gate on it. Mirrors q.judge_names_voted; returns UNGATED flags (caller gates once)."""
    cnt, canons = Counter(), {}
    for run in runs[:k]:
        seen = set()
        for f in run:
            key = tuple(sorted(f["wrong_cleaned"]))
            if key in seen:
                continue
            seen.add(key)
            cnt[key] += 1
            canons.setdefault(key, Counter())[f["canonical"]] += 1
    flags = []
    for key, n in cnt.items():
        if n >= cutoff:
            can = sorted(canons[key].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            flags.append({"case": "M9c", "canonical": can, "wrong_surface": list(key),
                          "wrong_cleaned": list(key), "all_spellings": list(key),
                          "card_id": -1, "evidence": "", "methods": ["judge"], "vote_count": n})
    return flags


def is_confident(flag):
    """Does the suggested spelling SOUND like the heard token (shared DM code)? — the same test
    expand_combined uses for suggestion_confident."""
    return any(bool(codes(wc) & codes(clean(flag["canonical"]))) for wc in flag["wrong_cleaned"])


def surface(comb, keep):
    """Apply a surfacing predicate to combined catches. A catch carries a vote_count only if a
    judge round produced it; a phonetic-only catch has none but is confident, so votes default 0."""
    out = []
    for f in comb:
        conf = is_confident(f)
        votes = f.get("vote_count", 0)
        if keep(conf, votes):
            out.append(f)
    return out


def build_world(world, names, cards, singles):
    """One model-bound build per world: K judge shuffles + cast + phonetic, then the shipped vote.
    Returns the combined catch set (each judge catch tagged with its vote_count)."""
    runs = runs_for(world, names)
    cast = q.generate_cast(gen, world)
    phon = q.phonetic_flags(cards, singles, cast)            # production phonetic (gated, confident)
    judge = gate_canon_flags(voted_with_counts(runs), singles)
    return q.combine(judge, phon)


def main():
    # --- held-out Mahabharata (one region == whole transcript) ---
    rich = json.loads((ROOT / "sessions" / HELDOUT / "transcript-rich.json").read_text())
    hcards = story_name_cards(rich["segments"], recover=True)
    hsingles = proper_name_candidates(rich["segments"])
    hnames = sorted({s for c in hcards for s in c["surface"]})
    hitems = load_items(HELDOUT) or {}
    h_comb = build_world("Mahabharata", hnames, hcards, hsingles)

    # --- synthetic 7-world planted spread ---
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    syn = []
    for st in spread:
        segs = build_segs(st["lines"])
        cards = story_name_cards(segs, recover=True)
        singles = proper_name_candidates(segs)
        names = sorted({s for c in cards for s in c["surface"]})
        comb = build_world(st["world"], names, cards, singles)
        syn.append((st, comb))

    def synth_score(keep):
        caught = tot = fp = 0
        for st, comb in syn:
            shown = surface(comb, keep)
            cg = forms([c for f in shown if f.get("case") == "M9c" for c in f["wrong_cleaned"]])
            caught += sum(1 for pl in st["planted"] if hit(pl["heard"], cg))
            pforms = ({clean(t) for pl in st["planted"] for t in pl["heard"].split()}
                      | {clean(pl["heard"]) for pl in st["planted"]})
            fp += sum(1 for f in shown if f.get("case") == "M9c"
                      and not (forms(f["wrong_cleaned"]) & pforms))
            tot += len(st["planted"])
        return caught, tot, fp

    print(f"\n{'policy':24} | held-out maha recall  prec | synthetic recall  false")
    print("-" * 78)
    for label, keep in POLICIES.items():
        h_shown = surface(h_comb, keep)
        r = score_canon(hitems, h_shown)
        real_flagged = r["matrix"]["M9c"]["flagged"]
        c, tot, fp = synth_score(keep)
        prec = f"{real_flagged}/{r['flagged']}" if r["flagged"] else "—"
        print(f"{label:24} | {r['caught']}/{r['gold_m9c']}={r['caught']/max(r['gold_m9c'],1):.2f}  "
              f"{prec:>6} | {c}/{tot}={c/max(tot,1):.2f}  {fp}", flush=True)

    print("\nRead: recall should rise with looser policies; watch the 'false'/precision cost.")
    print("The concrete Mahabharata-2 (Dhrashtra/Yudhisthir) and Thomas (invented copies) sessions "
          "are scored separately from their captures.")


if __name__ == "__main__":
    main()
