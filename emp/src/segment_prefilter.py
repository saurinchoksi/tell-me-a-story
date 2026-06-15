#!/usr/bin/env python3
"""Recall harness for the Stage-0 pre-filter (read-only, no model).

The cue+gap pre-filter itself lives in `segment.py` (single source of truth, so
this harness tests what the real segmenter runs). This file just *scores its
recall*: does it propose a candidate within tolerance of every true boundary in
the locked human truth? Design target is recall — the reader confirms/refines a
candidate but never invents one near nothing, so a missed boundary is fatal and
a false candidate is cheap.

    python emp/src/segment_prefilter.py            # eyeball candidates vs truth, all sessions
    python emp/src/segment_prefilter.py --gap 4.0  # try a different time-gap threshold
    python emp/src/segment_prefilter.py 20260129-204404   # one session, verbose
"""
import argparse

from segment import SESSIONS, load_segments, load_truth, propose, cluster


def truth_boundary_positions(stories, order):
    """The boundary points to detect: every story's start and end position."""
    pos = {x: i for i, x in enumerate(order)}
    out = []
    for k, st in enumerate(stories):
        out.append((pos[st["start"]], "start", k))
        out.append((pos[st["end"]], "end", k))
    return out


def nearest_candidate(target, cand_positions):
    if not cand_positions:
        return None, None
    best = min(cand_positions, key=lambda p: abs(p - target))
    return best, abs(best - target)


def eyeball(sid, gap_threshold=3.0, tol=2, verbose=False):
    name = SESSIONS[sid]
    segs = load_segments(sid)
    order = [s["id"] for s in segs]
    stories = load_truth()[sid]
    cands = propose(segs, gap_threshold)
    cand_pos = [c["pos"] for c in cands]
    zones = cluster(cands)
    boundaries = truth_boundary_positions(stories, order)

    print(f"\n========== {name} ({sid}) — {len(segs)} segs, {len(stories)} GT stor{'y' if len(stories)==1 else 'ies'} ==========")
    print(f"  pre-filter: {len(cands)} candidate points -> {len(zones)} clustered zones (gap>={gap_threshold}s, tol=+/-{tol})")

    hit = 0
    for tpos, kind, k in boundaries:
        bp, dist = nearest_candidate(tpos, cand_pos)
        ok = dist is not None and dist <= tol
        hit += ok
        seg = segs[tpos]
        print(f"    [{'OK ' if ok else 'MISS'}] story{k+1} {kind:5s} @ pos {tpos:3d} (id {seg['id']}) "
              f"nearest cand pos {bp} (d={dist})  \"{seg['text'][:55]}\"")
    print(f"  --> boundary recall: {hit}/{len(boundaries)}")

    if verbose:
        print("  candidate zones:")
        for z in zones:
            lo, hi = z["positions"][0], z["positions"][-1]
            near = any(abs(tpos - p) <= tol for p in z["positions"] for tpos, _, _ in boundaries)
            tag = "  (near a GT boundary)" if near else "  (false -> reader rejects)"
            print(f"     pos {lo:3d}-{hi:3d} [{','.join(sorted(z['signals']))}]{tag}  \"{segs[lo]['text'][:60]}\"")
    return hit, len(boundaries), len(cands), len(zones)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?")
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--tol", type=int, default=2)
    args = ap.parse_args()
    if args.session:
        eyeball(args.session, args.gap, args.tol, verbose=True)
    else:
        tot_hit = tot_b = tot_c = 0
        for sid in SESSIONS:
            h, b, c, _ = eyeball(sid, args.gap, args.tol, verbose=True)
            tot_hit += h; tot_b += b; tot_c += c
        print(f"\n=========================================================")
        print(f"  POOLED boundary recall: {tot_hit}/{tot_b} = {tot_hit/tot_b:.2f}")
        print(f"  total candidate points across 5 sessions: {tot_c}")


if __name__ == "__main__":
    main()
