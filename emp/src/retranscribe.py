#!/usr/bin/env python3
"""Re-transcribe a session's audio with an optional context prompt, then score vs the by-ear key.

The core of the "Whisper-with-context" investigation: Whisper transcribed this audio BLIND.
If we hand it the story's world (and cast) via `initial_prompt`, does it decode the real
names from the AUDIO — turning "Fondos" back into "Pandavas" — instead of us guessing the
spelling downstream? mlx-whisper 0.4.3 supports `initial_prompt` (documented for exactly this:
"custom vocabularies or proper nouns"). The prompt is capped at ~223 tokens (large-v3), and
the tail is what's kept — so the caller should put the most important names LAST.

Runs full-file (one model load, then the process exits and frees Metal — that's why each
experiment is its own subprocess). Saves the raw result and prints the score vs the key.

Read-only on session data. Output -> emp/results/visuals/whisper-context/<session>/<out>.json
Usage:
    python emp/src/retranscribe.py <session_id> --out <name> [--prompt "..."] \
        [--lang en] [--no-condition] [--tol 1.0]
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
from mlx_whisper.audio import load_audio  # noqa: E402
from api.helpers import get_session_dir  # noqa: E402
import importlib.util  # noqa: E402

# import the sibling scorer by path (emp/src isn't a package)
_spec = importlib.util.spec_from_file_location("score_vs_key", ROOT / "emp/src/score_vs_key.py")
svk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svk)

MODEL = "mlx-community/whisper-large-v3-mlx"
OUTDIR = ROOT / "emp/results/visuals/whisper-context"


def retranscribe(session_id: str, out_name: str, prompt: str | None, lang: str = "en",
                 condition: bool = True, tol: float = 1.0) -> dict:
    session_dir = get_session_dir(ROOT / "sessions", session_id)
    audio = np.array(load_audio(str(session_dir / "audio.m4a"))).astype(np.float32)

    print(f"[retranscribe] {session_id}  out={out_name}  lang={lang}  condition={condition}", file=sys.stderr)
    print(f"[retranscribe] initial_prompt = {prompt!r}", file=sys.stderr)
    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL, language=lang, word_timestamps=True,
        initial_prompt=prompt or None, condition_on_previous_text=condition, verbose=False)
    print(f"[retranscribe] done in {time.time()-t0:.0f}s", file=sys.stderr)

    outdir = OUTDIR / session_id
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"_prompt": prompt, "_lang": lang, "_condition": condition,
               "text": result.get("text", ""), "language": result.get("language"),
               "segments": result.get("segments", [])}
    (outdir / f"{out_name}.json").write_text(json.dumps(payload, ensure_ascii=False))

    key = svk.load_key(session_dir)
    res = svk.score(key, svk.flatten_words(payload), tol=tol)
    svk.print_report(res, f"{out_name}  (prompt={prompt!r})")
    (outdir / f"{out_name}.score.json").write_text(json.dumps(
        {"label": out_name, "prompt": prompt, "lang": lang, "condition": condition,
         "recall": res["recall"], "hits": res["hits"], "scoreable": res["scoreable"],
         "tally": res["tally"], "rows": res["rows"]}, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--no-condition", action="store_true", help="condition_on_previous_text=False")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    retranscribe(a.session_id, a.out, a.prompt, lang=a.lang,
                 condition=not a.no_condition, tol=a.tol)
