#!/usr/bin/env python3
"""Score the Stage-0 segmenter against the locked human truth (read-only).

The validation set is tiny (5 sessions, exactly ONE multi-story positive — see
emp.md), so this reports a SPREAD of honest numbers with their denominators
rather than one collapsed score:

  - story COUNT per session (target 1/1/1/1/3), with over-split (a single-story
    control predicted as >1 — the failure to avoid) and under-split flagged;
  - region overlap (IoU) and boundary DISTANCE (positions and seconds) for each
    greedily-matched story — a start/end is "correct" within +/-tol positions
    because the start is a fuzzy zone, but the raw distance is always shown;
  - WORLD/TITLE side-by-side vs truth for the human to eyeball (not auto-scored:
    "original" vs a free-text title can't be string-matched fairly).

    python emp/src/score_segmentation.py            # all sessions
    python emp/src/score_segmentation.py --tol 3    # looser boundary tolerance
"""
import argparse
import json

from segment import SESSIONS, load_segments, load_truth, PRED_OUT


def to_pos(stories, order, start_key, end_key):
    pos = {x: i for i, x in enumerate(order)}
    out = []
    for st in stories:
        s, e = pos.get(st[start_key]), pos.get(st[end_key])
        if s is None or e is None:
            continue
        out.append({"start_pos": min(s, e), "end_pos": max(s, e),
                    "title": st.get("title", ""), "world": st.get("world", "")})
    return out


def overlap(a, b):
    return max(0, min(a["end_pos"], b["end_pos"]) - max(a["start_pos"], b["start_pos"]) + 1)


def iou(a, b):
    inter = overlap(a, b)
    union = (a["end_pos"] - a["start_pos"] + 1) + (b["end_pos"] - b["start_pos"] + 1) - inter
    return inter / union if union else 0.0


def greedy_match(gt, pred):
    """One-to-one match by max position-overlap (GT longest first). Returns list of
    (gt_idx, pred_idx) and the leftover indices."""
    pairs, used = [], set()
    for gi in sorted(range(len(gt)), key=lambda i: gt[i]["start_pos"] - gt[i]["end_pos"]):
        best, bj = 0, None
        for pj in range(len(pred)):
            if pj in used:
                continue
            o = overlap(gt[gi], pred[pj])
            if o > best:
                best, bj = o, pj
        if bj is not None:
            pairs.append((gi, bj)); used.add(bj)
    matched_gt = {gi for gi, _ in pairs}
    return (pairs,
            [i for i in range(len(gt)) if i not in matched_gt],
            [j for j in range(len(pred)) if j not in used])


def fmt_t(segs, pos, which):
    t = segs[pos][which]
    return "—" if t is None else f"{t/60:.0f}:{t%60:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=2, help="boundary-correct tolerance (+/- positions)")
    args = ap.parse_args()
    tol = args.tol
    truth = load_truth()
    pred_all = json.loads(PRED_OUT.read_text())["sessions"]

    P = {"count_exact": 0, "over_split": 0, "under_split": 0,
         "ious": [], "start_ok": 0, "end_ok": 0, "matched": 0,
         "start_dpos": [], "end_dpos": [], "start_dsec": [], "end_dsec": [],
         "false": 0, "missed": 0}
    n_controls = sum(1 for sid in SESSIONS if len(truth[sid]) == 1)

    for sid, name in SESSIONS.items():
        segs = load_segments(sid)
        order = [s["id"] for s in segs]
        gt = to_pos(truth[sid], order, "start", "end")
        pr = to_pos(pred_all.get(sid, {}).get("stories", []), order, "start_id", "end_id")
        n_gt, n_pred = len(gt), len(pr)
        exact = n_pred == n_gt
        over = len(truth[sid]) == 1 and n_pred > 1
        under = n_pred < n_gt
        P["count_exact"] += exact; P["over_split"] += over; P["under_split"] += under

        print(f"\n===== {name} ({sid}) =====")
        flag = "OK" if exact else ("OVER-SPLIT" if over else ("UNDER-SPLIT" if under else "count off"))
        print(f"  stories: predicted {n_pred} vs truth {n_gt}   [{flag}]")

        pairs, miss_gt, false_pr = greedy_match(gt, pr)
        P["false"] += len(false_pr); P["missed"] += len(miss_gt)
        for gi, pj in pairs:
            g, p = gt[gi], pr[pj]
            this_iou = iou(g, p)
            sdp, edp = abs(p["start_pos"] - g["start_pos"]), abs(p["end_pos"] - g["end_pos"])
            sds = abs((segs[p["start_pos"]]["start"] or 0) - (segs[g["start_pos"]]["start"] or 0))
            eds = abs((segs[p["end_pos"]]["end"] or 0) - (segs[g["end_pos"]]["end"] or 0))
            s_ok, e_ok = sdp <= tol, edp <= tol
            P["ious"].append(this_iou); P["matched"] += 1
            P["start_ok"] += s_ok; P["end_ok"] += e_ok
            P["start_dpos"].append(sdp); P["end_dpos"].append(edp)
            P["start_dsec"].append(sds); P["end_dsec"].append(eds)
            print(f"    story{gi+1}: IoU {this_iou:.2f}  | start d={sdp}pos/{sds:.1f}s [{'ok' if s_ok else 'X'}] "
                  f"(truth {g['start_pos']}@{fmt_t(segs,g['start_pos'],'start')} vs pred {p['start_pos']}@{fmt_t(segs,p['start_pos'],'start')})")
            print(f"             | end   d={edp}pos/{eds:.1f}s [{'ok' if e_ok else 'X'}] "
                  f"(truth {g['end_pos']}@{fmt_t(segs,g['end_pos'],'end')} vs pred {p['end_pos']}@{fmt_t(segs,p['end_pos'],'end')})")
            print(f"             | world: truth \"{g['world']}\"  vs pred \"{p['world']}\"   (title: \"{p['title']}\")")
        for gi in miss_gt:
            print(f"    MISSED truth story{gi+1}: pos {gt[gi]['start_pos']}-{gt[gi]['end_pos']} \"{gt[gi]['world']}\"")
        for pj in false_pr:
            print(f"    FALSE pred story: pos {pr[pj]['start_pos']}-{pr[pj]['end_pos']} \"{pr[pj]['world']}\" (\"{pr[pj]['title']}\")")

    n = len(SESSIONS)
    m = P["matched"] or 1
    print("\n" + "=" * 60)
    print("  POOLED (every number with its denominator)")
    print(f"  story-count exact:     {P['count_exact']}/{n} sessions")
    print(f"  over-split controls:   {P['over_split']}/{n_controls}   (single-story sessions wrongly split — the key failure)")
    print(f"  under-split:           {P['under_split']}/{n} sessions")
    print(f"  matched stories:       {P['matched']}   (false predicted: {P['false']}, missed truth: {P['missed']})")
    print(f"  mean region IoU:       {sum(P['ious'])/m:.2f}  over {P['matched']} matched")
    print(f"  start correct @+/-{tol}:  {P['start_ok']}/{P['matched']}   mean dist {sum(P['start_dpos'])/m:.1f} pos / {sum(P['start_dsec'])/m:.1f} s")
    print(f"  end   correct @+/-{tol}:  {P['end_ok']}/{P['matched']}   mean dist {sum(P['end_dpos'])/m:.1f} pos / {sum(P['end_dsec'])/m:.1f} s")


if __name__ == "__main__":
    main()
