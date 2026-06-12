#!/usr/bin/env python3
"""Score the M9b name-consistency detector against the human M9 coding.  READ-ONLY.

The detector (src/detectors/name_consistency.py) flags inconsistently-spelled
names by code-only phonetic clustering — no roster, no model. The human coding is
per-SEGMENT (axial-labels.json, codes incl. "M9"); we reconcile at the segment
level, the same grain as score_m9a.py.

Ground truth: a segment is an **M9b segment** if the human marked it M9 *and* it
contains a token in the M9b variant list (emp/results/pivot-notes.json), using the
same variant→case logic as count.py. So:

  M9b precision -- of the detector's flagged segments, how many are M9b-truth?
  M9b recall    -- of M9b-truth segments, how many did the detector flag?

Because the detector is roster-agnostic it *also* catches inconsistently-spelled
family names (M9a) and sourced-canon names (M9c). A flagged segment is tagged
M9a / M9b / M9c / none by which variant-list tokens it carries, so a "false
positive" that is really an inconsistent *family* name is reported distinctly
from a genuine misfire (two unrelated real words that rhyme). A second,
broader **name precision** counts a flag correct if it lands on any real name
error (M9a/b/c).

PRIVACY: per-flag output can echo family-name variants, so it is written to the
gitignored emp/results/visuals/<id>/m9b-flags.json; only aggregates print.

Usage: python emp/src/score_m9b.py            # all five coded sessions
       python emp/src/score_m9b.py 20260129-204404
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from detectors.name_consistency import NameConsistencyDetector  # noqa: E402

# Session id -> story name (the five EMP-coded sessions).
SESSIONS = {
    "20251207-195607": "Moon",
    "20251207-202105": "Cruel Baby",
    "20251210-203654": "Rubber Ducky",
    "20260117-202237": "Pandavas",
    "20260129-204404": "Portal",
}
M9B_BEARING = {"20251207-202105", "20260129-204404"}  # the rest are negative controls

# count.py's token normalization (keeps hyphens/apostrophes) — used for ground
# truth so it matches how the variant lists were built.
_WORD_RE = re.compile(r"\W*([\w\-'’]+)\W*")


def clean_word(w):
    m = _WORD_RE.fullmatch((w or "").strip())
    return (m.group(1) if m else (w or "").strip()).lower().replace("’", "'")


def variant_to_case():
    """token -> 'M9a' | 'M9b' | 'M9c', merging the public cases with the gitignored
    M9a family-name list (counting only; never rendered)."""
    cases = json.loads((ROOT / "emp" / "results" / "pivot-notes.json").read_text())["_m9_cases"]
    vmap = {}
    for c in ("M9a", "M9b", "M9c"):
        for v in cases.get(c, {}).get("variants", []):
            vmap[clean_word(v)] = c
    priv = ROOT / "emp" / "results" / "pivot-notes.private.json"
    if priv.exists():
        for c, vs in json.loads(priv.read_text()).get("m9_variants", {}).items():
            for v in vs:
                vmap[clean_word(v)] = c
    return vmap


def segment_cases(seg, vmap):
    """The set of name-cases ({'M9a','M9b',...}) present as tokens in a segment."""
    found = set()
    for w in seg.get("words", []):
        c = vmap.get(clean_word(w.get("word", "")))
        if c:
            found.add(c)
    return found


def score_session(sid, vmap, detector):
    base = ROOT / "sessions" / sid
    human = {str(l["segmentId"]): l["codes"]
             for l in json.loads((base / "axial-labels.json").read_text())["labels"]
             if l.get("codes")}
    rich = json.loads((base / "transcript-rich.json").read_text())
    seg_by_id = {str(s["id"]): s for s in rich["segments"]}

    human_m9 = {s for s, codes in human.items() if any(c.startswith("M9") for c in codes)}
    # M9b-truth: human M9 segments that carry an M9b variant token.
    m9b_truth = {s for s in human_m9
                 if "M9b" in segment_cases(seg_by_id.get(s, {}), vmap)}

    flags = detector.run(base)["flags"]
    flag_segs = {}
    for f in flags:
        flag_segs.setdefault(str(f["segment_id"]), []).append(f)

    # Per flagged segment: which name-cases does it actually contain?
    seg_tag = {s: segment_cases(seg_by_id.get(s, {}), vmap) for s in flag_segs}

    tp = len(set(flag_segs) & m9b_truth)
    fp = len(set(flag_segs) - m9b_truth)
    recall_caught = len(m9b_truth & set(flag_segs))
    # Broader "name precision": a flag is a real name error if its segment has any case.
    name_hits = sum(1 for s in flag_segs if seg_tag[s])

    # gitignored per-flag dump (echoes name variants)
    out = ROOT / "emp" / "results" / "visuals" / sid
    out.mkdir(parents=True, exist_ok=True)
    (out / "m9b-flags.json").write_text(json.dumps({
        "session": sid, "n_flags": len(flags),
        "n_flagged_segments": len(flag_segs), "flags": flags,
    }, indent=2))

    return {
        "name": SESSIONS[sid], "n_flags": len(flags),
        "flag_segs": flag_segs, "seg_tag": seg_tag,
        "human_m9": human_m9, "m9b_truth": m9b_truth,
        "tp": tp, "fp": fp, "name_hits": name_hits, "recall_caught": recall_caught,
        "text": {s: (seg_by_id[s].get("text") or "").strip() for s in flag_segs},
    }


def report(r):
    sid_name = r["name"]
    nseg = len(r["flag_segs"])
    print(f"\n=== {sid_name} ===")
    print(f"  human M9 segments: {len(r['human_m9'])}  | of those M9b: {len(r['m9b_truth'])}")
    print(f"  detector: {r['n_flags']} flags across {nseg} segments")

    # case mix of flagged segments
    mix = {"M9a": 0, "M9b": 0, "M9c": 0, "none": 0}
    for s, tags in r["seg_tag"].items():
        if not tags:
            mix["none"] += 1
        else:
            for t in tags:
                mix[t] += 1
    print(f"  flagged-segment name mix: M9a={mix['M9a']} M9b={mix['M9b']} "
          f"M9c={mix['M9c']} no-listed-name={mix['none']}")

    if r["m9b_truth"]:
        rec = r["recall_caught"] / len(r["m9b_truth"])
        print(f"  M9b recall = {r['recall_caught']}/{len(r['m9b_truth'])} = {rec:.3f}")
    if nseg:
        m9b_prec = r["tp"] / nseg
        name_prec = r["name_hits"] / nseg
        print(f"  M9b precision  = {r['tp']}/{nseg} = {m9b_prec:.3f}")
        print(f"  name precision = {r['name_hits']}/{nseg} = {name_prec:.3f}  "
              f"(flag lands on any real name error)")

    # the unexplained flags — neither M9a/b/c: new finds or misfires, eyeball these
    unexplained = [s for s, tags in r["seg_tag"].items() if not tags]
    for s in sorted(unexplained, key=lambda x: int(x) if x.isdigit() else 1e9)[:12]:
        toks = sorted({f["token"] for f in r["flag_segs"][s]})
        spell = r["flag_segs"][s][0]["cluster_spellings"]
        print(f"    seg {s:<5} no-listed-name  {toks} (cluster {spell}) "
              f"| \"{r['text'][s][:55]}\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", help="session ids (default: all five coded)")
    args = ap.parse_args()
    sids = args.sessions or list(SESSIONS)

    vmap = variant_to_case()
    detector = NameConsistencyDetector()
    print(f"M9b name-consistency detector (v{detector.version}) — segment-level scoring")
    print(f"variant map: {sum(1 for c in vmap.values() if c=='M9b')} M9b tokens, "
          f"{sum(1 for c in vmap.values() if c=='M9a')} M9a, "
          f"{sum(1 for c in vmap.values() if c=='M9c')} M9c")

    results = {sid: score_session(sid, vmap, detector) for sid in sids}
    for sid in sids:
        report(results[sid])

    # pooled over the M9b-bearing sessions
    pool = [results[s] for s in sids if s in M9B_BEARING]
    if pool:
        tp = sum(r["tp"] for r in pool)
        nseg = sum(len(r["flag_segs"]) for r in pool)
        name_hits = sum(r["name_hits"] for r in pool)
        truth = sum(len(r["m9b_truth"]) for r in pool)
        caught = sum(r["recall_caught"] for r in pool)
        print("\n=== POOLED (M9b-bearing sessions) ===")
        if truth:
            print(f"  M9b recall    = {caught}/{truth} = {caught/truth:.3f}")
        if nseg:
            print(f"  M9b precision = {tp}/{nseg} = {tp/nseg:.3f}")
            print(f"  name precision = {name_hits}/{nseg} = {name_hits/nseg:.3f}")


if __name__ == "__main__":
    main()
