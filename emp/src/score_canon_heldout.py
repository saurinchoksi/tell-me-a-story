#!/usr/bin/env python3
"""Score the PRODUCTION canon name-reader (m9c-canon) against a by-ear key — held-out.

This grades the SHIPPED detector exactly as it runs: it reads `sessions/<id>/detections.json`
(dictionary gate and all), NOT the sealed eval auditor (`audit_names.py`, which has no gate).
It is the recall trust-gate for the two held-out sessions — once the by-ear key is marked,
one command gives the M9c precision/recall.

FIREWALL: this is the ONLY side that reads name-truth.json (the human gold). The detector
never does. READ-ONLY — nothing here writes a session or a key.

Name-grain (per distinct spelling), session-scope: a name is canon-gold if the ear marked it
"sourced canon" with a true spelling that differs from the delivered one. Per-STORY scope (a
name canon in one story, ordinary in another) needs the saved story regions the held-out
sessions don't have yet — a noted limitation, fine for the single-story Mahabharata held-out.

    ./venv/bin/python emp/src/score_canon_heldout.py                  # both held-out sessions
    ./venv/bin/python emp/src/score_canon_heldout.py 20260211-210718  # one
    ./venv/bin/python emp/src/score_canon_heldout.py --selftest       # verify the logic, no files
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from detectors.phonetics import clean          # noqa: E402
from score_names import gold_class             # noqa: E402  (the shared gold-derivation rule)
from name_truth import sidecar_path            # noqa: E402

HELDOUT = ["20260211-210718", "20260414-213156"]   # the two never-tuned sessions

# Gold buckets the canon reader is graded on. M9c is the positive; everything else that is a
# real name (correctly-spelled canon, improvised, family) is a negative it must NOT call canon.
NEGATIVE = {"no-error"}
OTHER_CASE = {"M9b", "M9d", "M9a"}             # a real error, but not THIS detector's case


def _norm(cleaned):
    """Join key: spaces removed, so a phrase card the ear keys 'like urjun' lines up with
    the production flag whose cleaned token is the space-stripped 'likeurjun'."""
    return cleaned.replace(" ", "")


def canon_gold(items):
    """{join-key -> (display spelling, gold class)} for every reviewable name in the key,
    with the phrase-member fold from score_names.build_gold: a single-word item whose token
    is part of a multi-word phrase key (e.g. 'urjun' inside 'like urjun') is dropped — the
    phrase card carries the verdict, so counting both would double the same name. Items the
    ear left uncategorized (gold_class -> 'unjudged') are skipped, so an unmarked key scores
    nothing rather than scoring everything as a negative."""
    phrase_members = {tok for k in items if " " in k for tok in k.split(" ")}
    out = {}
    for cleaned, meta in items.items():
        if " " not in cleaned and cleaned in phrase_members:
            continue                                    # folded into its phrase card
        g = gold_class(meta.get("category"), cleaned, meta.get("true_spelling", ""))
        if g != "unjudged":
            out[_norm(cleaned)] = (cleaned, g)
    return out


def flagged_canon(flags):
    """The set of join-keys the production reader flagged as M9c — the union of each flag's
    own token and its card's wrong-spelling set, space-normalized so it joins to the gold
    however the per-occurrence explosion landed on it."""
    out = set()
    for f in flags:
        if f.get("case") and f["case"] != "M9c":
            continue
        if f.get("cleaned"):
            out.add(_norm(f["cleaned"]))
        out.update(_norm(c) for c in f.get("wrong_cleaned", []))
    return out


def score_canon(items, flags):
    """Confusion of (human gold) x (reader flagged M9c?), name-grain. Pure — no file I/O —
    so it is `--selftest`-able. Returns counts + the off-diagonal lists to read the failures."""
    gold = canon_gold(items)
    flagged = flagged_canon(flags)
    rows = ["M9c", "no-error", "M9b", "M9d", "M9a", "phantom"]
    matrix = {r: Counter() for r in rows}
    misses, false_pos, caught = [], [], []

    for key, (display, g) in gold.items():
        hit = key in flagged
        col = "flagged" if hit else "none"
        row = g if g in matrix else "no-error"
        matrix[row][col] += 1
        if g == "M9c":
            (caught if hit else misses).append(display)
        elif hit:                                   # flagged a name that is NOT a canon error
            false_pos.append((display, g))

    gold_keys = set(gold)
    for key in flagged:                             # reader flagged a token the key doesn't cover
        if key not in gold_keys:
            matrix["phantom"]["flagged"] += 1
            false_pos.append((key, "phantom"))

    gold_m9c = sum(matrix["M9c"].values())
    n_caught = matrix["M9c"]["flagged"]
    n_flagged = sum(matrix[r]["flagged"] for r in rows)
    return {"matrix": matrix, "gold_m9c": gold_m9c, "caught": n_caught, "flagged": n_flagged,
            "misses": sorted(misses), "false_pos": sorted(false_pos), "hits": sorted(caught)}


def _safe(n, d):
    return f"{n}/{d} = {n / d:.2f}" if d else f"{n}/{d} = — (nothing to score)"


def load_flags(sid):
    p = ROOT / "sessions" / sid / "detections.json"
    if not p.exists():
        return None
    sec = json.loads(p.read_text()).get("detectors", {}).get("m9c-canon")
    return sec.get("flags", []) if sec else []


def load_items(sid):
    p = sidecar_path(sid)
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("items", {})


def report(sid):
    print(f"\n{'=' * 72}\n{sid}\n{'=' * 72}")
    flags = load_flags(sid)
    items = load_items(sid)
    if flags is None:
        print("  no detections.json — run: python src/detect.py --detector m9c-canon", sid)
        return
    n_marked = sum(1 for m in (items or {}).values() if m.get("category"))
    if not items or not n_marked:
        flagged = sorted(flagged_canon(flags))
        print(f"  by-ear key not marked yet ({0 if not items else len(items)} names swept, "
              f"{n_marked} categorized).")
        print(f"  the reader currently flags {len(flagged)} name(s) as canon: {flagged}")
        print(f"  -> mark the key first:  python emp/src/name_truth.py {sid} --serve")
        return

    r = score_canon(items, flags)
    print(f"  by-ear key: {n_marked} names categorized   |   reader flagged {r['flagged']} as canon")
    print(f"  M9c recall    {_safe(r['caught'], r['gold_m9c'])}   (real canon misspellings caught)")
    real_flagged = r["matrix"]["M9c"]["flagged"]
    print(f"  M9c precision {_safe(real_flagged, r['flagged'])}   (flags that land on a real canon error)")
    print("  confusion (row = ear gold, col = reader):   flagged   none")
    for row, c in r["matrix"].items():
        if sum(c.values()):
            print(f"    {row:>9} |   {c['flagged']:>6}  {c['none']:>5}")
    for n in r["misses"]:
        print(f"    MISS  {n!r}  — ear says canon-misspelled, reader stayed silent")
    for n, g in r["false_pos"]:
        why = "flagged an ordinary word / correct name" if g in ("no-error", "phantom") \
            else f"flagged a {g} name as canon"
        print(f"    FP    {n!r}  ({why})")


def selftest():
    """Prove the scoring logic on a synthetic key + flags — no real files touched."""
    items = {
        "urjun": {"category": "tv-canon", "true_spelling": "Arjuna"},   # M9c — should be caught
        "bhishma": {"category": "tv-canon", "true_spelling": "Bhishma"},  # canon spelled right -> no-error
        "arrows": {"category": "not-a-name", "true_spelling": ""},        # ordinary word -> no-error
        "jammus": {"category": "improvised", "true_spelling": "Jammus"},  # improvised name -> not canon
        "drona": {"category": "tv-canon", "true_spelling": "Drona"},      # canon spelled right -> no-error
    }
    # reader: catches urjun (good), wrongly flags arrows (FP), misses nothing else
    flags = [{"case": "M9c", "cleaned": "urjun", "wrong_cleaned": ["urjun"]},
             {"case": "M9c", "cleaned": "arrows", "wrong_cleaned": ["arrows"]}]
    r = score_canon(items, flags)
    assert r["gold_m9c"] == 1 and r["caught"] == 1, r          # recall 1/1
    assert r["matrix"]["no-error"]["flagged"] == 1, r          # arrows is a false positive
    assert ("arrows", "no-error") in r["false_pos"], r
    assert r["misses"] == [], r
    # and the ungated reader (extra arrows/wars flags) would show worse precision:
    noisy = flags + [{"case": "M9c", "cleaned": "wars", "wrong_cleaned": ["wars"]}]
    rn = score_canon(items, noisy)
    assert rn["flagged"] == 3 and rn["matrix"]["M9c"]["flagged"] == 1, rn   # precision 1/3

    # phrase fold + space join: the ear keys the SAME name twice — phrase 'like urjun' (with
    # space) and single 'urjun' — while the reader flags the space-stripped 'likeurjun'. The
    # single must fold into the phrase, and the spaced phrase must still join to the flag, so
    # this counts as ONE gold M9c, caught — not a miss plus a phantom.
    phrase_items = {
        "like urjun": {"category": "tv-canon", "true_spelling": "Arjuna"},
        "urjun": {"category": "tv-canon", "true_spelling": "Arjuna"},
    }
    rp = score_canon(phrase_items, [{"case": "M9c", "cleaned": "likeurjun",
                                     "wrong_cleaned": ["likeurjun"]}])
    assert rp["gold_m9c"] == 1, rp          # the single folded into the phrase — one gold name
    assert rp["caught"] == 1, rp            # spaced phrase joined to the stripped flag
    assert rp["misses"] == [] and rp["matrix"]["phantom"]["flagged"] == 0, rp
    print("selftest OK — recall 1/1, gated precision 1/2, ungated precision 1/3, "
          "phrase fold+join clean, no false misses")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sessions", nargs="*", help=f"session ids (default: {HELDOUT})")
    ap.add_argument("--selftest", action="store_true", help="verify the scoring logic, no files")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    for sid in (args.sessions or HELDOUT):
        report(sid)


if __name__ == "__main__":
    main()
