#!/usr/bin/env python3
"""Precision / forcing check: does the clip+cast prompt HALLUCINATE names where none were said?

The clip+cast win (E4) is only safe to ship if the cast prompt recovers real names without
FORCING cast names into ordinary speech. This samples control positions — ordinary words far
from any key name — cuts a clip around each, and re-transcribes it BOTH with the cast prompt
and with no prompt. A "force" = a cast name (Double-Metaphone match) appears in a clip whose
honest transcript had no such name. Comparing cast vs none isolates the prompt's forcing cost.

Read-only. Output -> emp/results/visuals/whisper-context/<session>/precision.json
Usage: python emp/src/precision_check.py <session_id> [--n 25] [--lead 1.5] [--window 2.0]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import mlx_whisper  # noqa: E402
from mlx_whisper.audio import SAMPLE_RATE, load_audio  # noqa: E402
from api.helpers import get_session_dir  # noqa: E402
from detectors.phonetics import clean, codes  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_vs_key", ROOT / "emp/src/score_vs_key.py")
svk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svk)

_clip_spec = importlib.util.spec_from_file_location("clip_retranscribe", ROOT / "emp/src/clip_retranscribe.py")
clipmod = importlib.util.module_from_spec(_clip_spec)
_clip_spec.loader.exec_module(clipmod)

MODEL = "mlx-community/whisper-large-v3-mlx"
OUTDIR = ROOT / "emp/results/visuals/whisper-context"

CAST_NAMES = ["Pandavas", "Kauravas", "Bhishma", "Bhima", "Arjuna", "Karna", "Duryodhana",
              "Krishna", "Drona", "Draupadi", "Yudhishthira", "Nakula", "Sahadeva",
              "Dushasana", "Kunti", "Gandhari", "Dhritarashtra", "Mahabharata"]
CAST_CODES = set()
for n in CAST_NAMES:
    CAST_CODES |= codes(clean(n))


def is_cast_name(tok: str) -> bool:
    c = clean(tok)
    return len(c) >= 4 and bool(codes(c) & CAST_CODES)


def pick_controls(transcript, key, n, min_gap=4.0):
    """Ordinary words (len>=4, alphabetic, NOT sounding like a cast name) at least min_gap
    seconds from any key position — spread evenly across the audio."""
    key_times = [k["word_start"] for k in key]
    cands = []
    for s in transcript["segments"]:
        for w in s.get("words", []):
            if w.get("start") is None:
                continue
            t, tok = float(w["start"]), (w.get("word") or "").strip()
            c = clean(tok)
            if len(c) < 4 or is_cast_name(tok):
                continue
            if all(abs(t - kt) >= min_gap for kt in key_times):
                cands.append({"start": t, "word": tok})
    if len(cands) <= n:
        return cands
    step = len(cands) / n
    return [cands[int(i * step)] for i in range(n)]


def run(session_id, n=25, lead=1.5, window=2.0):
    session_dir = get_session_dir(ROOT / "sessions", session_id)
    audio = np.array(load_audio(str(session_dir / "audio.m4a"))).astype(np.float32)
    transcript = json.loads((session_dir / "transcript-rich.json").read_text())
    key = svk.load_key(session_dir)
    controls = pick_controls(transcript, key, n)
    print(f"[precision] {len(controls)} control positions (ordinary words, >=4s from any key name)", file=sys.stderr)

    rows = {"cast": [], "none": []}
    forced = {"cast": 0, "none": 0}
    for mode, prompt in (("cast", clipmod.CAST), ("none", None)):
        for c in controls:
            words = clipmod.clip_words(audio, c["start"], lead, window, prompt)
            names_here = sorted({w.strip() for w in words if is_cast_name(w)})
            if names_here:
                forced[mode] += 1
            rows[mode].append({"start": round(c["start"], 1), "orig_word": c["word"],
                               "forced_names": names_here, "clip": " ".join(words)[:80]})
            if names_here:
                print(f"  [{mode}] {c['start']:6.1f} orig={c['word']!r} FORCED {names_here} | {' '.join(words)[:60]}",
                      file=sys.stderr)

    total = len(controls)
    out = {"n_controls": total, "lead": lead, "window": window,
           "forced_cast": forced["cast"], "forced_none": forced["none"],
           "force_rate_cast": round(forced["cast"] / total, 3) if total else 0,
           "force_rate_none": round(forced["none"] / total, 3) if total else 0,
           "rows": rows}
    (OUTDIR / session_id / "precision.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[precision] control positions: {total}", file=sys.stderr)
    print(f"[precision] FORCED a cast name — cast prompt: {forced['cast']}/{total} "
          f"({out['force_rate_cast']}) | no prompt: {forced['none']}/{total} ({out['force_rate_none']})",
          file=sys.stderr)
    print(f"[precision] net forcing attributable to the cast prompt: "
          f"{forced['cast'] - forced['none']}/{total}", file=sys.stderr)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--lead", type=float, default=1.5)
    ap.add_argument("--window", type=float, default=2.0)
    a = ap.parse_args()
    run(a.session_id, n=a.n, lead=a.lead, window=a.window)
