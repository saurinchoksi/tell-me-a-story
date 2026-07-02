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
sys.path.insert(0, str(ROOT))                  # so `api.helpers` (the shipped tier policy) imports
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from detectors.phonetics import clean          # noqa: E402
from score_names import gold_class             # noqa: E402  (the shared gold-derivation rule)
from name_truth import sidecar_path            # noqa: E402
# Reuse the PRODUCTION surfacing policy as the single source of truth, so this eval grades exactly
# what the Monitor shows. If Choksi retunes BEST_GUESS_VOTE_MIN, the score follows for free — the
# eval can never drift from the product (the "bent ruler" bug the EMP got burned by once).
from api.helpers import canon_tier, BEST_GUESS_VOTE_MIN   # noqa: E402

HELDOUT = ["20260211-210718", "20260414-213156"]   # the two never-tuned sessions

# The view layer (api.helpers.annotate_canon_tiers) shows confident + best_guess by default and
# hides `low` behind "Show all". Scoring all tiers would understate precision by counting low-tier
# false alarms the user never sees — so grade per policy and headline the shipped default.
TIER_POLICIES = [
    ("confident-only",          {"confident"}),
    ("+best_guess (default)",   {"confident", "best_guess"}),
    ("all-tiers (Show all)",    {"confident", "best_guess", "low"}),
]


def by_tier(flags, allowed):
    """The flags whose view-time tier is in `allowed` — the shipped surfacing policy applied to
    a raw detections.json (which stores per-flag suggestion_confident + vote_count, not `tier`)."""
    return [f for f in flags if canon_tier(f) in allowed]

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


def _stale_key_warning(sid):
    """Settled 2026-07-02: a key is stamped with the transcript it was marked against;
    grading a different transcript state is the bent-ruler mistake (hit 3x). Warn loudly."""
    import json as _json
    kp = ROOT / "emp" / "results" / "visuals" / sid / "name-truth.json"
    key_data = _json.loads(kp.read_text()) if kp.exists() else {}
    stamp = key_data.get("_transcript_fingerprint")
    import hashlib, json as _json
    rich = _json.loads((ROOT / "sessions" / sid / "transcript-rich.json").read_text())
    parts = [w.get("word", "") for seg in rich.get("segments", []) for w in seg.get("words", [])]
    now = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    if stamp is None:
        print(f"  !! KEY UNSTAMPED — marked before fingerprinting existed; treat scores as "
              f"indicative only (re-save in name_truth --serve to stamp)")
    elif stamp != now:
        print(f"  !! STALE KEY — marked against transcript {stamp}, current is {now}. "
              f"Scores below grade a DIFFERENT transcript state; re-mark before trusting.")


def report(sid):
    print(f"\n{'=' * 72}\n{sid}\n{'=' * 72}")
    _stale_key_warning(sid)
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

    # Grade per surfacing policy: the recall denominator (gold_m9c) is policy-independent; only what
    # the reader surfaces changes. The shipped default (+best_guess) is the honest "what ships" line.
    graded = {label: score_canon(items, by_tier(flags, allowed)) for label, allowed in TIER_POLICIES}
    gold_m9c = next(iter(graded.values()))["gold_m9c"]
    print(f"  by-ear key: {n_marked} names categorized   |   gold canon errors: {gold_m9c}")
    print(f"  {'tier policy':<24} {'recall':<16} {'precision':<16} flags")
    for label, _ in TIER_POLICIES:
        r = graded[label]
        real_flagged = r["matrix"]["M9c"]["flagged"]
        print(f"  {label:<24} {_safe(r['caught'], r['gold_m9c']):<16} "
              f"{_safe(real_flagged, r['flagged']):<16} {r['flagged']}")

    # Read both off-diagonals AT THE SHIPPED DEFAULT — that's what the user actually sees.
    rd = graded["+best_guess (default)"]
    print("  off-diagonals at the shipped default (+best_guess):")
    recovered = set(graded["all-tiers (Show all)"]["hits"]) - set(rd["hits"])  # caught only via Show all
    for n in rd["misses"]:
        tail = "  [recoverable via Show all]" if n in recovered else ""
        print(f"    MISS  {n!r}  — ear says canon-misspelled, reader stayed silent{tail}")
    for n, g in rd["false_pos"]:
        why = "flagged an ordinary word / correct name" if g in ("no-error", "phantom") \
            else f"flagged a {g} name as canon"
        print(f"    FP    {n!r}  ({why})")
    if not rd["misses"] and not rd["false_pos"]:
        print("    (clean — no misses, no false positives at the shipped default)")


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

    # tier policy: a low-confidence catch is hidden from the shipped default but reachable via
    # "Show all"; a confident (sound-alike) or sufficiently-voted best-guess catch shows by default.
    confident = {"case": "M9c", "cleaned": "bishma", "wrong_cleaned": ["bishma"],
                 "suggestion_confident": True}
    best_guess = {"case": "M9c", "cleaned": "dhrashtra", "wrong_cleaned": ["dhrashtra"],
                  "suggestion_confident": False, "vote_count": BEST_GUESS_VOTE_MIN}
    low = {"case": "M9c", "cleaned": "urzi", "wrong_cleaned": ["urzi"],
           "suggestion_confident": False, "vote_count": BEST_GUESS_VOTE_MIN - 1}
    assert (canon_tier(confident), canon_tier(best_guess), canon_tier(low)) \
        == ("confident", "best_guess", "low")
    default_flags = by_tier([confident, best_guess, low], {"confident", "best_guess"})
    assert confident in default_flags and best_guess in default_flags and low not in default_flags
    assert len(by_tier([confident, best_guess, low], {"confident", "best_guess", "low"})) == 3
    print("selftest OK — recall 1/1, gated precision 1/2, ungated precision 1/3, "
          "phrase fold+join clean, no false misses, low-tier hidden from default")


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
