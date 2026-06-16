#!/usr/bin/env python3
"""Offline worker for the per-story name auditor — runs under venv-mlx-vlm.

Invoked by detector.py (the main-venv caller) as a subprocess:

    venv-mlx-vlm/bin/python  src/detectors/story_names/_worker.py  <session_dir>

In ONE model load it: (1) segments the session into stories (live Stage 0),
(2) per story builds recall-recovered name cards and runs the v2 audit (+ canon
shield), (3) EXPANDS each per-spelling verdict to the standard per-occurrence flag
schema (segment_id/word_index/...), and prints {n_word_tokens, flags} as JSON on
stdout. ALL diagnostics go to stderr so stdout stays clean JSON for the caller.

This module mirrors the design of name_consistency_judge.py's worker, but where the
M9b judge crosses only a tiny judge() callable into the venv, here the WHOLE
detection lives in the worker because segmentation + audit + shield all need the
model. The flag-expansion (the one production-specific adaptation) and the
live-segmentation region adapter live here; the validated logic is imported,
verbatim, from the ported _segment/_audit/_names modules.
"""
import json
import sys
from pathlib import Path

# This file lives at src/detectors/story_names/_worker.py — put src/ on the path so the
# `detectors` package imports resolve (the venv's sys.path[0] is this dir, not src/).
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detectors.story_names._segment import segment_session, make_reader, load_segments
from detectors.story_names._audit import story_name_cards, run_v2
from detectors.story_names._names import story_segments
from detectors.phonetics import clean, codes


def build_regions(stories, pos_of):
    """Live equivalent of the EMP's load_regions: map each Stage-0 story's start/end
    ids to transcript positions, in order. The story SOURCE is the live segmenter
    output (not a pre-baked file); the id->position bridge is unchanged."""
    out = []
    for i, st in enumerate(stories):
        sp = pos_of.get(st["start_id"])
        ep = pos_of.get(st["end_id"])
        if sp is None or ep is None:
            print(f"  WARNING: story ids {st.get('start_id')}-{st.get('end_id')} "
                  f"absent from transcript — skipped", file=sys.stderr)
            continue
        if ep < sp:
            sp, ep = ep, sp
        out.append({"idx": i, "start_id": st["start_id"], "end_id": st["end_id"],
                    "start_pos": sp, "end_pos": ep,
                    "world": st.get("world", ""), "title": st.get("title", "")})
    return out


def expand_flag(flag, card, seg_by_id, story_idx, world):
    """Expand one per-(story, spelling) v2 flag into per-occurrence Monitor flags.

    Walk the card's occurrence positions and emit one flag per occurrence whose token
    (single word, or the reconstructed bigram for a phrase card) cleans into the flag's
    wrong_cleaned set. Because run_v2's canon shield rebuilds wrong_cleaned to EXCLUDE
    shielded spellings, a protected real name (James) yields zero occurrences here with
    no special-casing."""
    wrong = set(flag["wrong_cleaned"])
    is_phrase = card.get("is_phrase", False)
    out = []
    for o in card["occ"]:
        seg = seg_by_id.get(o["seg_id"])
        if not seg:
            continue
        words = seg.get("words") or []
        wi = o["wi"]
        if wi >= len(words):
            continue
        w0 = words[wi]
        if is_phrase:
            if wi + 1 >= len(words):
                continue
            w1 = words[wi + 1]
            raw = (w0["word"].strip() + " " + w1["word"].strip()).strip()
            start, end = w0.get("start"), w1.get("end")
        else:
            raw = w0["word"].strip()
            start, end = w0.get("start"), w0.get("end")
        c = clean(raw)
        if c not in wrong:
            continue
        rec = {
            "segment_id": o["seg_id"],
            "word_index": wi,
            "start": start,
            "end": end,
            "token": raw,
            "cleaned": c,
            "dm_codes": sorted(codes(c)),
            "case": flag["case"],            # "M9b" or "M9c"
            "canonical": flag["canonical"],  # spelling to standardize on (NOT applied — detection only)
            "card_id": flag["card_id"],
            "all_spellings": flag["all_spellings"],
            "wrong_cleaned": flag["wrong_cleaned"],
            "evidence": flag.get("evidence", ""),
            "story_id": story_idx,
            "story_world": world,
        }
        if flag.get("shielded"):  # M9b only — the canon spellings the shield protected
            rec["shielded"] = flag["shielded"]
        out.append(rec)
    return out


def count_word_tokens(rich):
    """Non-empty cleaned word tokens across all segments — matches the M9b detector's
    n_word_tokens so the Monitor's 'flags / tokens' line is comparable."""
    n = 0
    for s in rich["segments"]:
        for w in s.get("words", []):
            if clean(w["word"].strip()):
                n += 1
    return n


def run(session_dir):
    session_dir = Path(session_dir)
    rich = json.loads((session_dir / "transcript-rich.json").read_text())
    seg_by_id = {s["id"]: s for s in rich["segments"]}

    gen = make_reader()                                   # one model load
    seg_result, _ = segment_session(session_dir, gen)    # live Stage 0 (returns a tuple)
    pos_of = {s["id"]: s["pos"] for s in load_segments(session_dir)}
    regions = build_regions(seg_result["stories"], pos_of)
    print(f"  auditing names across {len(regions)} story region(s)", file=sys.stderr)

    flags = []
    for r in regions:
        segs = story_segments(rich, r, pos_of)
        cards = story_name_cards(segs, recover=True)      # recall recovery
        card_by_id = {c["id"]: c for c in cards}
        v2_flags, _ = run_v2(gen, r["world"], segs, cards, raw_log=[])  # + canon shield
        for f in v2_flags:
            if f["case"] == "M9d-suspect":                # dropped for v1 (low-confidence substitution)
                continue
            card = card_by_id.get(f["card_id"])
            if card is None:                              # v2 always sets a real card_id; defensive
                continue
            flags.extend(expand_flag(f, card, seg_by_id, r["idx"], r["world"]))

    print(json.dumps({"n_word_tokens": count_word_tokens(rich), "flags": flags}))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: _worker.py <session_dir>", file=sys.stderr)
        sys.exit(2)
    run(sys.argv[1])
