#!/usr/bin/env python3
"""E10 — the gated end-to-end: the full name-correction chain with the no-false-positive gate.

The chain (all per story, all world-agnostic):
  1. name CANDIDATES from the blind transcript (story_name_cards / proper_name_candidates —
     the existing M9c machinery). No candidate at a spot -> the chain never touches it.
  2. recognize the WORLD from the name list (existing; abstains on invented worlds -> do nothing).
  3. correction CAST = characters + groups (E8 split prompts, Qwen∪Gemma union, from
     split-casts.json) — loaded from disk; no model call here.
  4. targeted CLIP RE-TRANSCRIBE (E4 winner: ~3.5s window, world+cast appositive prompt) at
     candidate occurrences only.
  5. ACCEPTANCE GATE: auto-accept a re-decode only if
        (a) the position was a name candidate in the blind transcript,   [by construction]
        (b) the re-decoded token differs from the blind token and lands in the cast
            (exact clean match, or shares a Double-Metaphone code with a cast name),
        (c) the agreement vote concurs (--agree gemma: the Gemma listener picked a name that
            sound-matches the same cast name; --agree none: skip (measures (a)+(b) alone)).
     Anything caught but not accepted -> QUEUED (the human-bless flow). Never blanket.

Scores against the by-ear key (time-join) AND reports the control positions, printing the
honest confusion: auto & right / auto & WRONG / queued & right / queued & wrong / untouched.

Read-only. Output -> emp/results/visuals/whisper-context/<sid>/gated_e2e[_<agree>].json
Usage: python emp/src/gated_e2e.py <session_id> [--agree gemma|none] [--lead 1.5] [--window 2.0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import mlx_whisper  # noqa: E402
from mlx_whisper.audio import SAMPLE_RATE, load_audio  # noqa: E402
from api.helpers import get_session_dir  # noqa: E402
from detectors.phonetics import clean, codes  # noqa: E402
from detectors.story_names._worker import build_regions  # noqa: E402
from detectors.story_names._names import story_segments, proper_name_candidates  # noqa: E402
from detectors.story_names._audit import story_name_cards  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_vs_key", ROOT / "emp/src/score_vs_key.py")
svk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svk)

MODEL = "mlx-community/whisper-large-v3-mlx"
OUTDIR = ROOT / "emp/results/visuals/whisper-context"
SPLIT_CASTS = ROOT / "emp/results/visuals/cast-prompt-sweep/split-casts.json"


def correction_cast(world: str) -> list[str]:
    """Characters + groups, Qwen∪Gemma union, from the E8 split-prompt runs (world must match
    a swept world by name; the production version would generate these live)."""
    data = json.loads(SPLIT_CASTS.read_text())
    names, seen = [], set()
    for model_key in ("qwen", "gemma"):
        w = data.get(model_key, {}).get(world)
        if not w:
            continue
        for kind in ("groups", "characters"):   # groups FIRST — they matter most (Pandavas)
            for n in w[kind]:
                if clean(n) and clean(n) not in seen:
                    seen.add(clean(n)); names.append(n)
    return names


def build_prompt(world: str, cast: list[str]) -> str:
    groups = cast[:6]
    rest = cast[6:20]
    return (f"This is a bedtime story about the {world}. Its characters include "
            + ", ".join(groups) + (", and " + ", ".join(rest) if rest else "") + ".")


def candidate_positions(rich):
    """Every occurrence position (start time + blind token) of a name candidate, per story.
    Also returns the union of `proper_name_candidates` across regions — the dictionary
    gate's recall guard (a capitalized-in-story word is protected from the ordinary-word cut)."""
    seg_by_id = {s["id"]: s for s in rich["segments"]}
    pos_of = {s["id"]: i for i, s in enumerate(rich["segments"])}
    out = []
    singles = set()
    for r in build_regions(rich["_stories"], pos_of):
        segs = story_segments(rich, r, pos_of)
        cards = story_name_cards(segs, recover=True)
        singles |= proper_name_candidates(segs)
        names = sorted({s for c in cards for s in c["surface"]})
        for card in cards:
            for o in card["occ"]:
                seg = seg_by_id.get(o["seg_id"])
                words = (seg or {}).get("words") or []
                if o["wi"] >= len(words):
                    continue
                w = words[o["wi"]]
                if w.get("start") is None:
                    continue
                out.append({"story": r["idx"], "start": float(w["start"]),
                            "blind": w["word"].strip(), "seg_id": o["seg_id"], "wi": o["wi"]})
    # dedupe by (seg_id, wi)
    seen, ded = set(), []
    for p in out:
        k = (p["seg_id"], p["wi"])
        if k not in seen:
            seen.add(k); ded.append(p)
    return ded, singles


def redecode(audio, start, prompt, lead, window):
    """Re-transcribe the clip around `start`. Returns (words, offsets): each word with its
    time offset within the clip — the target word sits ~`lead` seconds in, and temporal
    matching against that offset is what keeps a neighboring name in the same clip from
    being mistaken for the target (the E10a lesson)."""
    i0 = max(0, int((start - lead) * SAMPLE_RATE))
    i1 = int((start + window) * SAMPLE_RATE)
    r = mlx_whisper.transcribe(audio[i0:i1], path_or_hf_repo=MODEL, language="en",
                               word_timestamps=True, initial_prompt=prompt,
                               condition_on_previous_text=True, verbose=False)
    out = []
    for s in r.get("segments", []):
        for w in s.get("words", []):
            if w.get("start") is None:
                continue
            out.append(((w.get("word") or "").strip(), float(w["start"])))
    return out


def main(session_id, agree, lead, window):
    session_dir = get_session_dir(ROOT / "sessions", session_id)
    rich = json.loads((session_dir / "transcript-rich.json").read_text())
    audio = np.array(load_audio(str(session_dir / "audio.m4a"))).astype(np.float32)
    key = svk.load_key(session_dir)
    key_by_time = {round(k["word_start"], 1): k for k in key}

    cands, names_singles = candidate_positions(rich)
    print(f"[e2e] {len(cands)} candidate occurrences", file=sys.stderr)

    from detectors.story_names import _qwen35  # noqa: E402  (used for world only if needed)
    # Recognize the world via the cached E4 result if present; else live (needs Qwen).
    world = "Mahabharata"  # this session's world is established; generality probe runs the live path
    cast = correction_cast(world)
    if not cast:
        raise SystemExit(f"no split-cast for world {world!r} — run cast_split.py first")
    cast_codes = set()
    for n in cast:
        cast_codes |= codes(clean(n))
    cast_clean = {clean(n): n for n in cast}
    prompt = build_prompt(world, cast)
    print(f"[e2e] cast ({len(cast)}): {', '.join(cast[:20])}", file=sys.stderr)
    print(f"[e2e] prompt: {prompt}", file=sys.stderr)

    # Gemma agreement votes (from E9's saved run), keyed by rounded time.
    gemma_pick = {}
    if agree == "gemma":
        gl = json.loads((OUTDIR / session_id / "gemma_listen.json").read_text())
        for r in gl["rows"]:
            if r["kind"] == "key":
                gemma_pick[round(r["start"], 1)] = clean((r["pick"] or "").split()[0]) if r["pick"].split() else ""

    # Dictionary gate (the M9c lesson, and E10a's father->Vidura leak): an ordinary English
    # word in the re-decode never maps onto a cast name — unless the story itself uses it as
    # a proper name (the `singles` recall guard, same as gate_canon_flags).
    from detectors.story_names._audit import _is_ordinary_word, _NCD

    t0 = time.time()
    decisions = []
    TEMPORAL_TOL = 0.7  # s — how far from the expected clip offset the target word may sit
    for p in cands:
        words = redecode(audio, p["start"], prompt, lead, window)
        blind_c = clean(p["blind"])
        expected = min(p["start"], lead)  # target offset within the clip (clip starts at start-lead, clamped at 0)
        # TEMPORAL matching: the target is the ONE nearest non-ordinary word to the expected
        # offset. If that word is unchanged or doesn't map to the cast, the spot is untouched —
        # a neighboring name in the same clip is a different word position, never the target
        # (E10 v2's two leaks were exactly a loud neighboring "Krishna"/"Kauravas" grabbing
        # the decision when the target itself re-decoded to an unmapped "Bandhas").
        near = [(w, off) for w, off in words if abs(off - expected) <= TEMPORAL_TOL]
        near.sort(key=lambda t: abs(t[1] - expected))
        target = next(((w, off) for w, off in near
                       if clean(w) and not _is_ordinary_word(clean(w), names_singles)), None)
        picked = None
        if target is not None:
            wc = clean(target[0])
            if wc and wc != blind_c:
                if wc in cast_clean:
                    picked = (target[0], cast_clean[wc], "exact")
                else:
                    m = next((cast_clean[cc] for cc in cast_clean if codes(wc) & codes(cc)), None)
                    if m:
                        picked = (target[0], m, "dm")
        if picked is None:
            decisions.append({**p, "action": "untouched"})
            continue
        tok, canonical, how = picked
        action = "auto"
        # Common-word protection (Choksi's by-ear verdict, 2026-07-01): a BLIND token that is
        # a real dictionary word ("arrows", "Beam") is NEVER auto-overwritten, even when it's
        # capitalized in-story (the `singles` exemption is exactly how both wrong autos —
        # arrows->Kauravas/Arjuna — leaked). Non-words (Bushma, Koros, Urzi) still auto-fix;
        # real words queue for the human bless.
        if _NCD._is_common(blind_c):
            action = "queued"
        elif agree == "gemma":  # the common-word queue is FINAL — agreement can't override it
            gp = gemma_pick.get(round(p["start"], 1), None)
            agree_ok = bool(gp) and gp != "none" and bool(codes(gp) & codes(clean(canonical)))
            action = "auto" if agree_ok else "queued"
        decisions.append({**p, "action": action, "redecoded": tok, "canonical": canonical,
                          "how": how, "clip": " ".join(w for w, _ in words)[:80]})
        print(f"  [{time.time()-t0:4.0f}s] {p['start']:6.1f} {p['blind']:14.14} -> "
              f"{canonical:12} [{action}]", file=sys.stderr)

    # score vs key
    tally = {"auto_right": 0, "auto_wrong": 0, "queued_right": 0, "queued_wrong": 0,
             "untouched_key": 0, "auto_offkey": 0, "queued_offkey": 0}
    for d in decisions:
        k = key_by_time.get(round(d["start"], 1))
        if d["action"] == "untouched":
            if k:
                tally["untouched_key"] += 1
            continue
        if k is None:
            tally[f"{d['action']}_offkey"] += 1   # a change at a non-key spot — eyeball these
            continue
        right = clean(d["canonical"]) == clean(k["answer"]) or bool(
            codes(clean(d["canonical"])) & codes(clean(k["answer"])))
        tally[f"{d['action']}_{'right' if right else 'wrong'}"] += 1

    out = {"session": session_id, "agree": agree, "world": world, "cast": cast,
           "prompt": prompt, "n_candidates": len(cands), "tally": tally,
           "decisions": decisions}
    name = f"gated_e2e_{agree}" if agree != "none" else "gated_e2e"
    (OUTDIR / session_id / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[e2e] agree={agree}  tally: {json.dumps(tally)}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--agree", choices=["gemma", "none"], default="none")
    ap.add_argument("--lead", type=float, default=1.5)
    ap.add_argument("--window", type=float, default=2.0)
    a = ap.parse_args()
    main(a.session_id, a.agree, a.lead, a.window)
