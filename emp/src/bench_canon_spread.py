#!/usr/bin/env python3
"""Measure the canon name-detector's recall across a fame/difficulty spread of worlds.

The held-out by-ear test scored ~1-in-12, but on the two HARDEST worlds (a non-Western
epic and a 2025 film). This asks the honest question: is that a Mahabharata problem or a
general one? It runs the REAL production path (the same world-namer + gated name-audit the
m9c-canon detector uses) on fabricated stories spanning Star Wars -> Steven Universe ->
Mahabharata, each seeded with known name misspellings.

It separates the TWO walls per world:
  - recognition: did the world-namer place the world at all? (if not, the check is off)
  - name recall (given the world): of the planted misspellings, how many does the audit
    catch when handed the correct world? This isolates the model's name knowledge from the
    capitalization gate via a `reachable` count (was the name even surfaced as a candidate).

Synthetic (no family content) and read-only. One local-model load for the whole spread.

    ./venv/bin/python emp/src/bench_canon_spread.py            # run + score the spread
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from story_segment import make_reader, pass2_name                  # noqa: E402
from detectors.story_names._audit import story_name_cards, run_v2  # noqa: E402
from detectors.phonetics import clean                              # noqa: E402

BENCH = ROOT / "emp" / "results" / "canon-spread" / "stories.json"
OUT = ROOT / "emp" / "results" / "canon-spread" / "results.json"


def build_segs(lines):
    """Fabricated transcript -> the segment/word shape the detector reads. One segment per
    line; each word gets a fake timestamp. Names that the fabricator capitalized mid-sentence
    become proper-name candidates exactly as in a real transcript."""
    segs, t = [], 0.0
    for i, line in enumerate(lines):
        words, start = [], t
        for w in line.split():
            words.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2)})
            t += 0.3
        segs.append({"id": i, "start": round(start, 2), "end": round(t, 2),
                     "text": line.strip(), "words": words})
        t += 0.2
    return segs


def forms(cleaned_list):
    """Both the per-token and the space-stripped joined form of each cleaned spelling, so a
    single name and a multi-word phrase card both join to a space-free clean(heard)."""
    out = set()
    for cl in cleaned_list:
        out.add(cl.replace(" ", ""))
        for tok in cl.split(" "):
            if tok:
                out.add(tok)
    return out


def m9c_caught(flags):
    return forms([c for f in flags if f.get("case") == "M9c" for c in f.get("wrong_cleaned", [])])


def hit(heard, found):
    """A planted name counts as present in `found` if its space-stripped form OR any of its
    component tokens is there — so a multi-word name ('Dark Vader', 'Obi Wan Canopy') matches
    however the detector tokenized the misspelled part."""
    toks = {clean(t) for t in heard.split() if clean(t)} | {clean(heard)}
    return bool(toks & found)


def run():
    bench = json.loads(BENCH.read_text())
    stories = bench["stories"]
    gen = make_reader()
    results = []
    for st in stories:
        world, lines, planted = st["world"], st["lines"], st["planted"]
        segs = build_segs(lines)
        lines_text = "\n".join(f'[{s["id"]}] "{s["text"]}"' for s in segs if s["text"])
        _, pred_world = pass2_name(gen, lines_text, [])

        cards = story_name_cards(segs, recover=True)
        reachable = forms([cl for c in cards for cl in c["clean"]])

        given_flags, _ = run_v2(gen, world, segs, cards, [])          # correct world handed in
        e2e_flags, _ = run_v2(gen, pred_world, segs, cards, [])       # the world it recognized
        caught_given, caught_e2e = m9c_caught(given_flags), m9c_caught(e2e_flags)

        marks = []
        for p in planted:
            marks.append({"heard": p["heard"], "correct": p["correct"],
                          "reachable": hit(p["heard"], reachable),
                          "caught_given": hit(p["heard"], caught_given),
                          "caught_e2e": hit(p["heard"], caught_e2e)})
        n = len(marks)
        results.append({
            "world": world, "recognized_as": pred_world,
            "n_planted": n,
            "reachable": sum(m["reachable"] for m in marks),
            "recall_given": sum(m["caught_given"] for m in marks),
            "recall_e2e": sum(m["caught_e2e"] for m in marks),
            "marks": marks,
        })
        print(f"  {world:34} world->{pred_world!r:24} "
              f"given {sum(m['caught_given'] for m in marks)}/{n}  "
              f"e2e {sum(m['caught_e2e'] for m in marks)}/{n}  "
              f"(reachable {sum(m['reachable'] for m in marks)}/{n})", flush=True)

    OUT.write_text(json.dumps({"results": results}, indent=2))
    # ---- summary ----
    tot_n = sum(r["n_planted"] for r in results)
    tot_given = sum(r["recall_given"] for r in results)
    tot_e2e = sum(r["recall_e2e"] for r in results)
    tot_reach = sum(r["reachable"] for r in results)
    print("\n" + "=" * 78)
    print("CANON NAME-RECALL ACROSS WORLDS  (production path: world-namer + gated audit)")
    print("=" * 78)
    print(f"  {'world':34} {'recognized?':>11} {'given':>7} {'e2e':>6} {'reach':>7}")
    for r in results:
        ok = "yes" if r["recognized_as"] else "NO (made-up)"
        print(f"  {r['world']:34} {ok:>11} "
              f"{r['recall_given']}/{r['n_planted']:<5} {r['recall_e2e']}/{r['n_planted']:<4} "
              f"{r['reachable']}/{r['n_planted']}")
    print("-" * 78)
    print(f"  {'TOTAL':34} {'':>11} {tot_given}/{tot_n:<5} {tot_e2e}/{tot_n:<4} {tot_reach}/{tot_n}")
    print(f"\n  name-recall given the correct world : {tot_given}/{tot_n} = {tot_given / tot_n:.2f}")
    print(f"  end-to-end (recognition + audit)    : {tot_e2e}/{tot_n} = {tot_e2e / tot_n:.2f}")
    print(f"  of the reachable names only         : {tot_given}/{tot_reach} = "
          f"{tot_given / tot_reach:.2f}" if tot_reach else "  (none reachable)")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
