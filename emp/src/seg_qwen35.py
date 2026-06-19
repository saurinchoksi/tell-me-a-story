#!/usr/bin/env python3
"""Step-1 experiment: can Qwen3.5-4B segment a bedtime recording into its stories as
well as / better than Gemma-4 E4B? If yes, the canon path unifies on one model.

This swaps ONLY the reader and the prompts; every deterministic piece (the pre-filter,
the windows, the stateful walk's region assembly, the merge's consecutive-run logic,
and the IoU/count scorer) is imported verbatim from the sealed eval so the comparison
isolates exactly one variable: the model. Gemma's baseline is 5/5 story counts and
mean region IoU 0.86 (emp.md, score_segmentation.py).

PLAIN TEXT in and out — no JSON (memory plain-text-not-json). PASS1 emits
`START <id>` / `END <id>` / `NONE` lines; the merge emits one line of letters per
story. Qwen3.5 is a VLM -> mlx_vlm; a reasoning model -> enable_thinking=False.

We TUNE like the plan asks: 4 plain-text PASS1 variants and 2 merge variants, scored
against the locked truth, keep the best. World/title naming (pass 2) is skipped here —
the count+IoU score only uses boundaries; world recognition is its own validated task.

    ./venv/bin/python emp/src/seg_qwen35.py smoke           # 1 variant, 1 session, dump raw
    ./venv/bin/python emp/src/seg_qwen35.py smoke terse 20260117-202237
    ./venv/bin/python emp/src/seg_qwen35.py all             # full PASS1 + merge sweep vs Gemma
    ./venv/bin/python emp/src/seg_qwen35.py pass1           # PASS1 sweep only (merge=faithful)
"""
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from mlx_vlm import load, generate                                       # noqa: E402
# Deterministic scaffolding — imported, never reimplemented (sealed eval of record).
from segment import (SESSIONS, load_segments, load_truth, propose, cluster,          # noqa: E402
                     build_window, render_window, _mkregion, _state_line, coerce_id,
                     _render_regions_for_merge, _winddown_fraction)
from score_segmentation import to_pos, iou, greedy_match                  # noqa: E402

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

# ============================ reader (Qwen3.5, plain text) ====================
_model = _processor = _tok = None


def make_reader():
    """Load Qwen3.5-4B once; return gen(prompt, max_tokens) -> raw string. VLM build
    (mlx_vlm), reasoning disabled (enable_thinking=False), greedy, plain text."""
    global _model, _processor, _tok
    if _model is None:
        _model, _processor = load(MODEL)
        _tok = getattr(_processor, "tokenizer", _processor)

    def gen(prompt, max_tokens=200):
        fmt = _tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        o = generate(_model, _processor, fmt, max_tokens=max_tokens, temperature=0.0, verbose=False)
        return (getattr(o, "text", o) or "").strip()
    return gen


# ============================ prompts (plain text) ============================
FMT_P1 = ("\n\nOutput — for THIS window only, write one line per boundary you find:\n"
          "START <id>   (a new story begins at that line)\n"
          "END <id>     (the currently-open story ends at that line)\n"
          "Use only the [id] numbers shown above. If nothing in this window begins or "
          "ends a story, write exactly:\nNONE\nWrite nothing else — no explanations, no JSON.\n\n"
          "Window:\n{window}")

P1 = {
    "faithful": (
        "You are reading a transcript of a parent telling a young child bedtime stories, in "
        "order, one window at a time. A recording usually holds just ONE story; occasionally a "
        "parent tells two or three, with negotiation and milk breaks between them. Each line is "
        "tagged with its segment id in [brackets]; some lines were auto-flagged as possible "
        "boundaries.\n\nCURRENT STATE: {state}\n\n"
        "Decide what — if anything — happens in THIS window. Be conservative: MOST windows "
        "contain no boundary.\n"
        "- If NO story is open and a new story clearly BEGINS here, mark START on the launch-cue "
        'line ("start the story", "let\'s tell a different story", "you help me tell it", "once '
        'upon a time"); if there is no cue, the first real sentence. (If this window is the very '
        "START of the recording and the parent is already mid-story, mark that first narration line.)\n"
        "- If a story IS open, you are almost always STILL INSIDE it — stories are long and full "
        "of pauses, milk breaks, repeated words, and characters talking. An interruption (a spill, "
        "getting milk) is STILL the same story if the telling resumes. Mark END only when the open "
        'story clearly concludes: it says "the end", or gives way for good to settling down, a sung '
        "wind-down, or negotiating the next story.\n"
        "- Otherwise, nothing happens here." + FMT_P1),
    "terse": (
        "A parent is telling a child bedtime stories. Usually the whole recording is ONE story; "
        "sometimes two or three, separated by breaks. You are reading ONE window at a time.\n\n"
        "CURRENT STATE: {state}\n\n"
        "Most windows have NO boundary. Mark START only where a new story clearly begins; mark END "
        'only where the open story clearly finishes ("the end", or settling down / choosing the '
        "next story). A pause or interruption is the SAME story." + FMT_P1),
    "conservative": (
        "You are splitting a bedtime recording into its stories, reading one window at a time. The "
        "DEFAULT answer is NONE — a recording is usually a single long story, and most windows are "
        "deep inside it.\n\nCURRENT STATE: {state}\n\n"
        "Only break the default when you are SURE:\n"
        "- START: a genuinely new story opens here (a clear launch cue, or — at the very start of "
        "the recording — the first line of narration).\n"
        '- END: the open story has clearly finished ("the end", a sung wind-down, or the parent and '
        "child start choosing a different story).\n"
        "Pauses, milk breaks, spills, repeated lines, and characters talking are NOT boundaries — "
        "they are the same story. When unsure, answer NONE." + FMT_P1),
    "cue_first": (
        "You are finding story boundaries in a parent's bedtime recording, one window at a time. "
        "Each line shows its [id] and any auto-flagged signals.\n\nCURRENT STATE: {state}\n\n"
        "Look for these concrete moments:\n"
        'START cues: "once upon a time", "let\'s start", "say the words", "you help me tell it", '
        '"let\'s tell a different story". At the recording\'s very start with no cue, the first '
        "narration line is the START.\n"
        'END cues: "the end", "that\'s the end", a sung wind-down ("I see trees of green"), or the '
        "pair negotiating which story to tell next.\n"
        "Everything else — pauses, milk, spills, repeated words, dialogue inside the story — is NOT "
        "a boundary. A recording is usually one story; be conservative." + FMT_P1),
}

FMT_MERGE = ("\n\nOutput: one line per distinct story, listing that story's segment letters "
             "separated by spaces. Every letter from A to {last} must appear on exactly one line. "
             "Write only letters — no other words, no JSON. Example:\nA B\nC")

MERGE = {
    "faithful": (
        "A first pass split ONE bedtime recording into the candidate story-segments below, labelled "
        "by LETTER, in order. Group the segments that belong to the SAME story; keep different "
        "stories separate. Every segment is part of some story — your only job is grouping.\n"
        "- SAME story: same characters and plot; a later segment just CONTINUES after an "
        "interruption (milk, a spill, a tangent). A back-and-forth where parent and child build ONE "
        "story together is still one story.\n"
        "- DIFFERENT stories: the characters or world clearly change, or someone announces a switch "
        '("let\'s tell a different story"). When unsure, keep them separate.\n\nSegments:\n{segments}'
        + FMT_MERGE),
    "terse": (
        "Below are candidate story-segments from ONE bedtime recording, labelled by letter, in "
        "order. Some are halves of the SAME story split by an interruption (milk, a spill). Group "
        "segments that share characters and plot into one story; keep clearly different stories "
        "(different characters or world, or an announced switch) separate. When unsure, keep "
        "separate.\n\nSegments:\n{segments}" + FMT_MERGE),
}


# ============================ plain-text parsers =============================
def parse_pass1(raw, valid_ids, by_id):
    """Parse START/END lines -> walk events. Robust to trailing text on the line."""
    events = []
    for line in (raw or "").splitlines():
        mt = re.match(r"\s*(START|END)\b", line, re.I)
        if not mt:
            continue
        mid = re.search(r"(gap_[\d.]+|\d+)", line[mt.end():])
        if not mid:
            continue
        sid = coerce_id(mid.group(1), valid_ids)
        if sid is None:
            continue
        typ = "story_start" if mt.group(1).upper() == "START" else "story_end"
        events.append({"type": typ, "segment_id": sid, "quote": by_id.get(sid, "")})
    return events


def parse_groups(raw, n):
    """Parse merge groups: each non-empty line is one story's letters. A token counts
    only if it is exactly a single in-range letter (drops stray prose like 'Group')."""
    last = chr(64 + n)
    groups = []
    for line in (raw or "").splitlines():
        letts = [t.upper() for t in re.split(r"[\s,]+", line.strip())
                 if len(t) == 1 and "A" <= t.upper() <= last]
        if letts:
            groups.append([ord(c) - 65 for c in letts])
    return groups


# ============================ the walk (ported, plain-text passes) ===========
def build_regions(sid, gen, p1_variant, raws=None, verbose=False):
    """Stateful left-to-right walk -> PRE-MERGE regions. Mirrors segment.segment_session's
    assembly exactly; only the pass-1 call is plain-text Qwen3.5. Returns (segs, order, regions)."""
    segs = load_segments(sid)
    order = [s["id"] for s in segs]
    pos_of = {x: i for i, x in enumerate(order)}
    by_id = {s["id"]: s["text"] for s in segs}
    valid_ids = set(order)
    zones = cluster(propose(segs))

    regions, open_story = [], None
    for zi, z in enumerate(zones):
        positions, elide = build_window(segs, z)
        window_text = render_window(segs, positions, z["sig_at"], elide)
        prompt = P1[p1_variant].format(state=_state_line(open_story), window=window_text)
        t0 = time.monotonic()
        try:
            raw = gen(prompt, max_tokens=200)
        except Exception as ex:
            raw = f"ERROR {ex!r}"
        if verbose:
            print(f"    zone {zi + 1}/{len(zones)} ({len(window_text)} chars) "
                  f"-> {time.monotonic() - t0:5.1f}s  {raw[:40]!r}", flush=True)
        if raws is not None:
            raws.append({"state": _state_line(open_story)[:40], "raw": raw})
        evs = parse_pass1(raw, valid_ids, by_id)
        for e in sorted(evs, key=lambda e: (pos_of[e["segment_id"]], 0 if e["type"] == "story_end" else 1)):
            ep = pos_of[e["segment_id"]]
            if e["type"] == "story_start":
                if open_story is None:
                    open_story = e
                elif ep - 1 >= pos_of[open_story["segment_id"]]:
                    regions.append(_mkregion(open_story, order[ep - 1], "<next story starts>", pos_of))
                    open_story = e
                # else: spurious backward start — keep the open story
            else:  # story_end
                if open_story is None:  # recording opened mid-story
                    sp = min((regions[-1]["end_pos"] + 1) if regions else 0, ep)
                    open_story = {"segment_id": order[sp], "quote": "<opened mid-story>"}
                regions.append(_mkregion(open_story, e["segment_id"], e.get("quote", ""), pos_of))
                open_story = None
    if open_story is not None:
        regions.append(_mkregion(open_story, order[-1], "<runs to end>", pos_of))
    return segs, order, regions


def consolidate(gen, segs, regions, merge_variant, winddown_thresh=0.34):
    """Global merge pass — ported from segment.global_consolidate, plain-text groups.
    Deterministic wind-down drop, then MODEL merge-only over consecutive same-label runs."""
    kept = [r for r in regions if _winddown_fraction(segs, r) < winddown_thresh] or regions
    n = len(kept)
    if n <= 1:
        return kept
    prompt = MERGE[merge_variant].format(segments=_render_regions_for_merge(segs, kept), last=chr(64 + n))
    raw = gen(prompt, max_tokens=200)
    label = {}
    for gid, g in enumerate(parse_groups(raw, n)):
        for i in g:
            if 0 <= i < n:
                label.setdefault(i, ("g", gid))
    for i in range(n):
        label.setdefault(i, ("solo", i))  # never silently lose a region
    runs, cur = [], None
    for i in range(n):
        lab = label[i]
        if cur is not None and cur["lab"] == lab:
            cur["last"] = i
        else:
            cur = {"lab": lab, "first": i, "last": i}
            runs.append(cur)
    return [{"start_id": kept[c["first"]]["start_id"], "end_id": kept[c["last"]]["end_id"],
             "start_pos": kept[c["first"]]["start_pos"], "end_pos": kept[c["last"]]["end_pos"]} for c in runs]


def regions_to_stories(regions):
    return [{"start_id": r["start_id"], "end_id": r["end_id"], "title": "", "world": ""} for r in regions]


# ============================ scoring ========================================
def score_session(sid, order, truth_stories, pred_stories):
    gt = to_pos(truth_stories, order, "start", "end")
    pr = to_pos(pred_stories, order, "start_id", "end_id")
    pairs, miss, false = greedy_match(gt, pr)
    return {"n_gt": len(gt), "n_pred": len(pr), "exact": len(pr) == len(gt),
            "over": len(truth_stories) == 1 and len(pr) > 1,
            "ious": [iou(gt[gi], pr[pj]) for gi, pj in pairs],
            "false": len(false), "missed": len(miss)}


def aggregate(per_session):
    ce = sum(s["exact"] for s in per_session)
    ov = sum(s["over"] for s in per_session)
    ious = [x for s in per_session for x in s["ious"]]
    miou = sum(ious) / len(ious) if ious else 0.0
    return {"count_exact": ce, "over": ov, "miou": miou,
            "false": sum(s["false"] for s in per_session), "missed": sum(s["missed"] for s in per_session)}


# ============================ modes ==========================================
def run_combo(gen, p1, merge_variant, truth, cache=None, progress=False):
    """Full walk over all 5 sessions for one (p1, merge) combo -> aggregate score.
    Caches pre-merge regions per (p1, sid) so a merge sweep can reuse them."""
    per = []
    for sid in SESSIONS:
        t0 = time.monotonic()
        if cache is not None and (p1, sid) in cache:
            segs, order, regions = cache[(p1, sid)]
            built = "cached"
        else:
            segs, order, regions = build_regions(sid, gen, p1)
            built = f"{time.monotonic() - t0:.0f}s"
            if cache is not None:
                cache[(p1, sid)] = (segs, order, regions)
        merged = consolidate(gen, segs, regions, merge_variant)
        sc = score_session(sid, order, truth[sid], regions_to_stories(merged))
        per.append(sc)
        if progress:
            print(f"      {SESSIONS[sid]:12} [{built:>6}] pred {sc['n_pred']} vs truth {sc['n_gt']} "
                  f"{'OK' if sc['exact'] else 'OFF'}", flush=True)
    return aggregate(per), per


def smoke(variant="faithful", sid="20260117-202237"):
    gen = make_reader()
    truth = load_truth()
    print(f"SMOKE — PASS1='{variant}' on {SESSIONS.get(sid, sid)} ({sid})\n", flush=True)
    raws = []
    segs, order, regions = build_regions(sid, gen, variant, raws=raws)
    print(f"raw PASS1 outputs (first 8 of {len(raws)} zones):")
    for r in raws[:8]:
        print(f"  [{r['state']}...] -> {r['raw']!r}")
    print(f"\npre-merge regions: {[(r['start_id'], r['end_id']) for r in regions]}")
    merged = consolidate(gen, segs, regions, "faithful")
    print(f"post-merge regions: {[(r['start_id'], r['end_id']) for r in merged]}")
    sc = score_session(sid, order, truth[sid], regions_to_stories(merged))
    print(f"\ntruth stories: {[(t['start'], t['end']) for t in truth[sid]]}")
    print(f"score: pred {sc['n_pred']} vs truth {sc['n_gt']}  exact={sc['exact']}  "
          f"IoUs={[round(x, 2) for x in sc['ious']]}  false={sc['false']} missed={sc['missed']}")


def sweep_pass1(gen, truth, cache):
    print(f"\n{'PASS1 variant':14} | count  over | mean IoU | false missed", flush=True)
    print("-" * 56)
    results = {}
    for v in P1:
        print(f"  --- PASS1 variant '{v}' (merge=faithful) ---", flush=True)
        agg, _ = run_combo(gen, v, "faithful", truth, cache=cache, progress=True)
        results[v] = agg
        print(f"{v:14} |  {agg['count_exact']}/5    {agg['over']}  |   {agg['miou']:.2f}   |   "
              f"{agg['false']}     {agg['missed']}", flush=True)
    best = max(results, key=lambda v: (results[v]["count_exact"], results[v]["miou"], -results[v]["over"]))
    print(f"\n  best PASS1 (merge=faithful): {best}  "
          f"({results[best]['count_exact']}/5 counts, IoU {results[best]['miou']:.2f})")
    print("  Gemma baseline: 5/5 counts, mean IoU 0.86")
    return best


def sweep_merge(gen, best_p1, truth, cache):
    print(f"\n{'merge variant':14} | count  over | mean IoU | false missed   (PASS1=" + best_p1 + ")", flush=True)
    print("-" * 56)
    results = {}
    for mv in MERGE:
        agg, _ = run_combo(gen, best_p1, mv, truth, cache=cache)  # pre-merge regions cached
        results[mv] = agg
        print(f"{mv:14} |  {agg['count_exact']}/5    {agg['over']}  |   {agg['miou']:.2f}   |   "
              f"{agg['false']}     {agg['missed']}", flush=True)
    best = max(results, key=lambda v: (results[v]["count_exact"], results[v]["miou"], -results[v]["over"]))
    print(f"\n  best merge: {best}")
    return best


def probe(variant="terse", sid="20260117-202237"):
    """Instrumented single-session run — times the model load and EVERY pass-1 call, live.
    Answers 'hung or just slow?': watch per-zone latency stream in real time."""
    print(f"PROBE — loading {MODEL} ...", flush=True)
    t0 = time.monotonic()
    gen = make_reader()
    print(f"  model loaded in {time.monotonic() - t0:.1f}s", flush=True)
    print(f"  warming up (1 tiny call) ...", flush=True)
    tw = time.monotonic()
    _ = gen("Reply with the single word: ok", max_tokens=5)
    print(f"  warmup call: {time.monotonic() - tw:.1f}s", flush=True)
    print(f"\nPASS1='{variant}' on {SESSIONS.get(sid, sid)} ({sid}) — per-zone timing:", flush=True)
    tb = time.monotonic()
    segs, order, regions = build_regions(sid, gen, variant, verbose=True)
    print(f"\n  build_regions total: {time.monotonic() - tb:.1f}s  "
          f"-> pre-merge regions {[(r['start_id'], r['end_id']) for r in regions]}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke(*sys.argv[2:4])
        return
    if mode == "probe":
        probe(*sys.argv[2:4])
        return
    gen = make_reader()
    truth = load_truth()
    cache = {}
    if mode == "pass1":
        sweep_pass1(gen, truth, cache)
    elif mode == "all":
        best_p1 = sweep_pass1(gen, truth, cache)
        best_merge = sweep_merge(gen, best_p1, truth, cache)
        print(f"\n=== Qwen3.5 segmentation: PASS1='{best_p1}', merge='{best_merge}' ===")
        agg, per = run_combo(gen, best_p1, best_merge, truth, cache=cache)
        for sid, s in zip(SESSIONS, per):
            print(f"  {SESSIONS[sid]:12} pred {s['n_pred']} vs truth {s['n_gt']}  "
                  f"{'OK' if s['exact'] else 'OFF'}  IoU {[round(x, 2) for x in s['ious']]}")
        print(f"\n  FINAL: {agg['count_exact']}/5 counts, mean IoU {agg['miou']:.2f}, "
              f"over-splits {agg['over']}, false {agg['false']}, missed {agg['missed']}")
        print("  Gemma:  5/5 counts, mean IoU 0.86, over-splits 0")
    else:
        print(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
