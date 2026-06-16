#!/usr/bin/env python3
"""Shared helpers for the Stage-1 per-story name auditor (EMP).

Both the auditor (audit_names.py) and the scorer (score_names.py) import this so
they agree on (a) what a "story region" is, (b) how a word occurrence maps to a
story, and (c) how the DELIVERED transcript is read.

Two non-negotiables encoded here (see emp.md "Stage 1, step 1"):
  - Read the DELIVERED word tokens, never the segment `text`. LLM-normalization
    rewrites the words but not `text` (a known pipeline bug), and the answer keys
    reflect the delivered words. `seg_word_text()` joins words, like name_truth.py.
  - A story is a [start_pos, end_pos] region from the Stage-0 segmenter; anything
    no region covers is non-story (preamble, milk breaks, wind-down) and is scored
    as out-of-story, not silently dropped.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from segment import SESSIONS, PRED_OUT, load_segments  # noqa: E402
from detectors.phonetics import clean  # noqa: E402

__all__ = ["SESSIONS", "ROOT", "clean", "load_rich", "load_regions",
           "story_of_pos", "seg_word_text", "story_segments"]


def load_rich(sid):
    return json.loads((ROOT / "sessions" / sid / "transcript-rich.json").read_text())


def seg_word_text(seg):
    """The segment's sentence built from the (corrected) WORD tokens — NOT seg['text'],
    which the normalizer leaves stale. Empty string if a gap segment with no words."""
    ws = seg.get("words") or []
    return " ".join(w["word"].strip() for w in ws).strip() if ws else (seg.get("text") or "").strip()


def load_regions(sid):
    """Return (regions, pos_of) for a session.

    regions: [{idx, start_id, end_id, start_pos, end_pos, world, title}] in order.
    pos_of:  {segment_id -> position index in transcript-rich} (incl. gap_* string ids).

    Positions come from load_segments (segment.py), so they share the segmenter's
    id<->position space exactly. A story whose ids are absent from the current
    transcript is skipped with a warning (stale pred vs re-enriched transcript)."""
    segs = load_segments(sid)
    pos_of = {s["id"]: s["pos"] for s in segs}
    pred = json.loads(PRED_OUT.read_text())["sessions"]
    stories = pred.get(sid, {}).get("stories", [])
    out = []
    for i, st in enumerate(stories):
        sp, ep = pos_of.get(st["start_id"]), pos_of.get(st["end_id"])
        if sp is None or ep is None:
            print(f"  WARNING: {sid} story ids {st.get('start_id')}-{st.get('end_id')} "
                  f"absent from current transcript — skipped")
            continue
        if ep < sp:
            sp, ep = ep, sp
        out.append({"idx": i, "start_id": st["start_id"], "end_id": st["end_id"],
                    "start_pos": sp, "end_pos": ep,
                    "world": st.get("world", ""), "title": st.get("title", "")})
    return out, pos_of


def story_of_pos(pos, regions):
    """Index of the region covering this position, or None (non-story)."""
    if pos is None:
        return None
    for r in regions:
        if r["start_pos"] <= pos <= r["end_pos"]:
            return r["idx"]
    return None


def story_segments(rich, region, pos_of):
    """The rich segment dicts that fall inside one story region, in order."""
    lo, hi = region["start_pos"], region["end_pos"]
    return [s for s in rich["segments"]
            if lo <= pos_of.get(s["id"], -1) <= hi]
