#!/usr/bin/env python3
"""
Score L1 (the deterministic M9a detector) against a human ear pass.  READ-ONLY.

The held-out, out-of-sample validation. The human coding is per-SEGMENT
(axial-labels.json, codes incl. "M9"); L1's sealed flags are per-TOKEN (with a
segment_id). We reconcile at the segment level:

  precision -- of L1's flags, how many land on a segment the human marked as a
               name (M9) error (true positive) vs not (false positive).
  recall    -- of the human's M9 (name-error) segments, how many did L1 flag?
               The ones L1 missed are shown with their text, to be classified by
               eye as a CHILD-name dissolved case (a real M9a miss, audio-only)
               or a character-name error (M9b -- not L1's job). Never auto-decided.

Usage: python emp/src/score_m9a.py 20260414-213156
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(sid):
    base = ROOT / "sessions" / sid
    ax = json.loads((base / "axial-labels.json").read_text())
    human = {str(l["segmentId"]): l["codes"] for l in ax["labels"] if l.get("codes")}
    rich = json.loads((base / "transcript-rich.json").read_text())
    text = {str(s["id"]): (s.get("text") or "").strip() for s in rich["segments"]}
    flags = json.loads((ROOT / "emp" / "results" / "visuals" / sid
                        / "m9a-l1-flags.json").read_text())["flags"]
    return human, text, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    sid = ap.parse_args().session
    human, text, flags = load(sid)

    human_m9 = {s for s, codes in human.items() if any(c.startswith("M9") for c in codes)}
    flag_segs = {}
    for f in flags:
        flag_segs.setdefault(str(f["segment_id"]), []).append(f["token"])

    print(f"=== {sid} ===")
    print(f"human-coded segments: {len(human)}  "
          f"(M9 name-errors: {len(human_m9)} -> {sorted(human_m9, key=int)})")
    print(f"L1 flags: {len(flags)} tokens across {len(flag_segs)} segments\n")

    # ---- PRECISION: each L1 flag vs the human coding ----
    print("PRECISION  (each L1 flag -- did the human mark that segment a name error?)")
    tp = fp = 0
    for seg in sorted(flag_segs, key=int):
        agree = seg in human_m9
        tp += agree
        fp += not agree
        for tok in flag_segs[seg]:
            verdict = "TP  agree" if agree else "FP  human did NOT mark"
            print(f"   seg {seg:<4} {tok:<10} -> {verdict:<22} | \"{text.get(seg,'')[:60]}\"")
    prec = tp / (tp + fp) if (tp + fp) else 0
    print(f"   precision = {tp}/{tp+fp} = {prec:.3f}\n")

    # ---- RECALL: each human M9 segment vs L1 ----
    print("RECALL  (each human name-error -- did L1 flag it?)")
    caught = sorted(human_m9 & set(flag_segs), key=int)
    missed = sorted(human_m9 - set(flag_segs), key=int)
    for seg in caught:
        print(f"   seg {seg:<4} CAUGHT  {flag_segs[seg]}")
    for seg in missed:
        print(f"   seg {seg:<4} *** L1 MISS -> classify by eye: "
              f"\"{text.get(seg,'')[:80]}\"")
    rec = len(caught) / len(human_m9) if human_m9 else 0
    print(f"   recall (vs all human M9) = {len(caught)}/{len(human_m9)} = {rec:.3f}")
    if missed:
        print("   ^ recall is provisional until each MISS is classified "
              "M9a-dissolved (real miss) vs M9b (character, not L1's job)")

    # ---- the non-M9 tags the human added, for completeness ----
    other = {s: c for s, c in human.items() if s not in human_m9}
    if other:
        print("\nother human tags (ignored for the L1 score):")
        for s in sorted(other, key=int):
            print(f"   seg {s:<4} {other[s]} | \"{text.get(s,'')[:60]}\"")


if __name__ == "__main__":
    main()
