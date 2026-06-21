#!/usr/bin/env python3
"""Tune the ORDER-ROBUST judge against what actually SHIPS. The single judge call is
order-sensitive: a borderline name (Bandos->Pandavas on the held-out) is caught in ~5/8 random
orderings of the name list. So run the judge over K deterministic shuffles and keep a wrong-spelling
caught in >= cutoff of them. This sweeps K x cutoff against the held-out by-ear key AND the synthetic
7-world spread.

FAITHFUL to production: it drives the real `_qwen35` functions (the shipped JUDGE_PROMPT via
_judge_raw, generate_cast, phonetic_flags, combine) through the production reader (make_reader), so
the numbers transfer to the live detector — unlike a near-copy of the EMP experiment prompt, which a
4B model reads differently. Efficiency: MAXK shuffles per world run ONCE; every (K, cutoff) is a
cheap re-count over the first K runs (no extra model calls).

Held-out absolute is offset by the gold's phrase-form keys ("like urjun" vs the filler-stripped
"urjun"), constant across configs — so rank by the explicit Bandos?/Hastinapur? columns (single-token
gold keys that DO join). The hard bar is the synthetic judge+phonetic spread (must hold >= 34/40).

    ./venv/bin/python emp/src/tune_judge_vote.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

import detectors.story_names._qwen35 as q                          # noqa: E402  (production judge layer)
from detectors.story_names._audit import story_name_cards, gate_canon_flags  # noqa: E402
from detectors.story_names._names import proper_name_candidates      # noqa: E402
from detectors.phonetics import clean                                # noqa: E402
from qwen35 import make_reader                                       # noqa: E402  (production reader)
from score_canon_heldout import score_canon, load_items             # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit, forms    # noqa: E402

SID = "20260211-210718"
MAXK = 7
CUTOFFS = {5: [2, 3, 4], 7: [3, 4, 5]}   # integer cutoffs (fractions collapse: ceil(.5*5)==ceil(.6*5))

gen = make_reader()


def runs_for(world, names, k=MAXK, seed_base=0):
    """k UNGATED production judge passes over deterministic shuffles (seeds 0..k-1)."""
    out = []
    for r in range(k):
        sh = list(names)
        random.Random(seed_base + r).shuffle(sh)
        out.append(q._judge_raw(gen, world, sh))
    return out


def vote(runs, k, cutoff):
    """From the first k runs, keep wrong-spellings caught in >= cutoff rounds, majority canonical.
    Mirrors q.judge_names_voted's voting; returns UNGATED flags (caller gates once)."""
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
                          "card_id": -1, "evidence": "", "methods": ["judge"]})
    return flags


def caught_keys(flags):
    return {wc for f in flags for wc in f.get("wrong_cleaned", [])}


def main():
    # --- held-out (1 region == whole transcript) ---
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    mcards = story_name_cards(rich["segments"], recover=True)
    msingles = proper_name_candidates(rich["segments"])
    mnames = sorted({s for c in mcards for s in c["surface"]})
    items = load_items(SID) or {}
    m_cast = q.generate_cast(gen, "Mahabharata")
    mp = q.phonetic_flags(mcards, msingles, m_cast)                 # production phonetic (gated)
    m_runs = runs_for("Mahabharata", mnames)
    m_sorted = q._judge_raw(gen, "Mahabharata", sorted(mnames))     # pre-voting baseline pass

    # --- synthetic 7-world spread ---
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    syn = []
    for st in spread:
        segs = build_segs(st["lines"])
        cds = story_name_cards(segs, recover=True)
        sg = proper_name_candidates(segs)
        nm = sorted({s for c in cds for s in c["surface"]})
        cast = q.generate_cast(gen, st["world"])
        pf = q.phonetic_flags(cds, sg, cast)
        syn.append((st, runs_for(st["world"], nm), sg, pf,
                    q._judge_raw(gen, st["world"], sorted(nm))))

    def synth(picker):
        caught = tot = fp = 0
        for st, runs, sg, pf, srt in syn:
            comb = q.combine(gate_canon_flags(picker(runs, srt), sg), pf)
            cg = m9c_caught(comb)
            caught += sum(1 for pl in st["planted"] if hit(pl["heard"], cg))
            pforms = ({clean(t) for pl in st["planted"] for t in pl["heard"].split()}
                      | {clean(pl["heard"]) for pl in st["planted"]})
            fp += sum(1 for f in comb if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pforms))
            tot += len(st["planted"])
        return caught, tot, fp

    print(f"\n{'config':12} | maha j+p prec Bandos Hastin | spread j+p  false")
    print("-" * 62)

    def row(label, m_judge_flags, picker):
        comb = q.combine(gate_canon_flags(m_judge_flags, msingles), mp)
        r = score_canon(items, comb)
        keys = caught_keys(comb)
        bn = "YES" if "bandos" in keys else "no "
        hp = "YES" if "hastinapur" in keys else "no "
        c, tot, fp = synth(picker)
        print(f"{label:12} | {r['caught']}/{r['gold_m9c']}  {r['matrix']['M9c']['flagged']}/{r['flagged']}  {bn}  {hp}  |  "
              f"{c}/{tot}={c / tot:.2f}  {fp}", flush=True)

    # baseline: single judge pass over the SORTED list (production's pre-voting behavior)
    row("sorted x1", m_sorted, lambda runs, srt: srt)
    for k in (5, 7):
        for cut in CUTOFFS[k]:
            row(f"K{k} cut{cut}", vote(m_runs, k, cut),
                lambda runs, srt, k=k, cut=cut: vote(runs, k, cut))

    print("\nPick: max maha recall with Bandos=YES, synthetic j+p >= 34/40 (0.85), false <= baseline.")


if __name__ == "__main__":
    main()
