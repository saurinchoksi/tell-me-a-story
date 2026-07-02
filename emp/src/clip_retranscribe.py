#!/usr/bin/env python3
"""Clip-level re-transcription: re-decode a SHORT window around each key name with context.

E1 showed a single upfront `initial_prompt` recovers "Pandavas" only near the start of the
13-min file — the bias decays. The fix this tests: cut a short clip around EACH key name
occurrence and re-transcribe that clip with the world+cast prompt, so the bias is fresh for
every occurrence. If the answer name appears anywhere in the clip's transcription, that
occurrence was recovered.

Two honest controls to separate "context recovered the real name" from "the prompt FORCED a
name that wasn't said":
  - --prompt-mode cast   : world + cast (the treatment)
  - --prompt-mode world  : world sentence only (context, no name list)
  - --prompt-mode none   : no prompt (pure clip re-decode — isolates the windowing effect)

Read-only on session data. One model load, loops the clips. Output ->
emp/results/visuals/whisper-context/<session>/<out>.clips.json + a printed per-name table.

Usage:
    python emp/src/clip_retranscribe.py <session_id> --out E4_clip_cast --prompt-mode cast \
        [--lead 1.5] [--window 2.0]
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
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_vs_key", ROOT / "emp/src/score_vs_key.py")
svk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svk)

MODEL = "mlx-community/whisper-large-v3-mlx"
OUTDIR = ROOT / "emp/results/visuals/whisper-context"

WORLD = "This is a bedtime story about the Mahabharata, the ancient Indian epic."
CAST = (WORLD + " Its characters include Dhritarashtra, Gandhari, Kunti, Draupadi, Drona, "
        "Krishna, the Kauravas, Duryodhana, Dushasana, Karna, Bhishma, and the five Pandavas: "
        "Yudhishthira, Nakula, Sahadeva, Bhima, and Arjuna.")
# COMBINED: the Qwen∪Gemma model casts (which individually MISS the groups) WITH the groups
# "Pandavas"/"Kauravas" restored — the finding that the correction cast must carry groups.
COMBINED = (WORLD + " Its characters include the Pandavas, the Kauravas, Yudhishthira, Bhima, "
            "Arjuna, Nakula, Sahadeva, Draupadi, Karna, Krishna, Gandhari, Duryodhana, Bhishma, "
            "Vidura, Drona, Ashwatthama, Abhimanyu, Dhritarashtra, Kunti, Dushasana.")
# LEAN: minimal prose (the prose "ancient Indian epic" is what leaked verbatim in the precision check).
LEAN = ("Mahabharata story. Names: Pandavas, Kauravas, Bhishma, Bhima, Arjuna, Yudhishthira, "
        "Nakula, Sahadeva, Karna, Duryodhana, Krishna, Drona, Draupadi.")
PROMPTS = {"cast": CAST, "world": WORLD, "none": None, "combined": COMBINED, "lean": LEAN}


def clip_words(audio, ws, lead, window, prompt):
    """Re-transcribe the window [ws-lead, ws+window] and return its list of word strings."""
    i0 = max(0, int((ws - lead) * SAMPLE_RATE))
    i1 = int((ws + window) * SAMPLE_RATE)
    r = mlx_whisper.transcribe(audio[i0:i1], path_or_hf_repo=MODEL, language="en",
                               word_timestamps=True, initial_prompt=prompt,
                               condition_on_previous_text=True, verbose=False)
    return [(w.get("word") or "").strip() for s in r.get("segments", []) for w in s.get("words", [])]


def run(session_id, out_name, prompt_mode, lead=1.5, window=2.0):
    session_dir = get_session_dir(ROOT / "sessions", session_id)
    audio = np.array(load_audio(str(session_dir / "audio.m4a"))).astype(np.float32)
    key = svk.load_key(session_dir)
    prompt = PROMPTS[prompt_mode]
    print(f"[clip] {session_id} out={out_name} mode={prompt_mode} lead={lead} window={window}", file=sys.stderr)

    rows, hits, scoreable = [], 0, 0
    t0 = time.time()
    for i, k in enumerate(key):
        ans, ws = k["answer"], k["word_start"]
        if ans in svk.UNKNOWABLE:
            rows.append({"answer": ans, "word_start": round(ws, 2), "clip_words": "", "match": "unknowable"})
            continue
        scoreable += 1
        words = clip_words(audio, ws, lead, window, prompt)
        match, matched = "miss", ""
        for w in words:
            m = svk._matches(w, ans)
            if m == "exact":
                match, matched = "exact", w; break
            if m == "dm" and match == "miss":
                match, matched = "dm", w
        if match in ("exact", "dm"):
            hits += 1
        rows.append({"answer": ans, "word_start": round(ws, 2), "clip_words": " ".join(words),
                     "matched_word": matched, "match": match})
        print(f"  [{time.time()-t0:4.0f}s] {ws:6.1f} {ans:14.14} -> {match:6} "
              f"({matched or '—'})  clip=\"{' '.join(words)[:70]}\"", file=sys.stderr)

    recall = round(hits / scoreable, 3) if scoreable else 0.0
    outdir = OUTDIR / session_id
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{out_name}.clips.json").write_text(json.dumps(
        {"label": out_name, "prompt_mode": prompt_mode, "prompt": prompt, "lead": lead,
         "window": window, "recall": recall, "hits": hits, "scoreable": scoreable,
         "rows": rows}, ensure_ascii=False, indent=2))
    exact = sum(1 for r in rows if r["match"] == "exact")
    dm = sum(1 for r in rows if r["match"] == "dm")
    print(f"\n[clip] {out_name}: HITS {hits}/{scoreable} (recall {recall}) | exact {exact} sound-alike {dm}\n")
    return recall


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-mode", choices=list(PROMPTS), default="cast")
    ap.add_argument("--lead", type=float, default=1.5)
    ap.add_argument("--window", type=float, default=2.0)
    a = ap.parse_args()
    run(a.session_id, a.out, a.prompt_mode, lead=a.lead, window=a.window)
