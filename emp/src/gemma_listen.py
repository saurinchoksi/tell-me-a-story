#!/usr/bin/env python3
"""E9 — Gemma as a second listener: play it each clip, ask WHICH cast name is spoken.

Gemma-4 E4B has a built-in audio encoder (config `audio_config`, gemma4_audio), and the
installed mlx_vlm 0.5.0 drives it (`generate(..., audio=[path])`). The question is
CONSTRAINED — pick one name from the cast, or 'none' — which is easier than open
transcription, and Gemma's failure mode is independent of Whisper's. Its job in the target
architecture is the VETO vote in the acceptance gate, so the number that matters most is the
false-pick rate on control clips (ordinary speech, no cast name), not raw recall.

Cuts wav clips (ffmpeg, LEAD before / TAIL after each position) for the key positions and
the precision-check control positions, then one Gemma load loops them all.

Also probes: an OPEN "transcribe this clip" variant on the key clips (Gemma's raw hearing),
and reports whether Qwen3.5's config carries an audio tower (a possible third listener).

Read-only on session data. Output -> emp/results/visuals/whisper-context/<sid>/gemma_listen.json
Usage: python emp/src/gemma_listen.py <session_id> [--lead 1.5] [--tail 2.0] [--open-too]
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
from detectors.phonetics import clean, codes  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_vs_key", ROOT / "emp/src/score_vs_key.py")
svk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svk)

OUTDIR = ROOT / "emp/results/visuals/whisper-context"
GEMMA = "mlx-community/gemma-4-e4b-it-4bit"

# The E4-winning cast (groups included), offered as the constrained choice list.
CAST = ["Pandavas", "Kauravas", "Bhishma", "Bhima", "Arjuna", "Yudhishthira", "Nakula",
        "Sahadeva", "Karna", "Duryodhana", "Krishna", "Drona", "Draupadi", "Dushasana",
        "Kunti", "Gandhari", "Dhritarashtra", "Mahabharata", "Arti"]

PICK_PROMPT = ("Listen to the audio clip of a parent and child telling a bedtime story. "
               "Which ONE of these names is spoken in the clip: " + ", ".join(CAST) +
               "? If none of them is spoken, answer exactly 'none'. Answer with one word only.")
OPEN_PROMPT = "Transcribe exactly what is said in this audio clip."


def cut_wav_clips(session_dir, positions, outdir, lead, tail):
    """Cut a mono 16k wav per position (start time -> [start-lead, start+tail])."""
    ff = shutil.which("ffmpeg") or "ffmpeg"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, p in enumerate(positions):
        frm = max(0.0, p["start"] - lead)
        out = outdir / f"{p['kind']}-{i}.wav"
        cmd = [ff, "-y", "-ss", f"{frm:.3f}", "-i", str(session_dir / "audio.m4a"),
               "-t", f"{lead + tail:.3f}", "-ac", "1", "-ar", "16000", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg failed for {out}: {r.stderr[-200:]}")
        paths.append(out)
    return paths


def qwen_audio_check():
    """Does the cached Qwen3.5 carry an audio tower? (a possible third listener)"""
    try:
        p = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/*/config.json"))[0]
        c = json.loads(open(p).read())
        return {"model_type": c.get("model_type"),
                "audio_keys": [k for k in c if "audio" in k.lower()]}
    except Exception as e:
        return {"error": repr(e)[:120]}


def main(session_id, lead, tail, open_too):
    session_dir = get_session_dir(ROOT / "sessions", session_id)
    key = svk.load_key(session_dir)
    key_pos = [{"kind": "key", "start": k["word_start"], "answer": k["answer"]}
               for k in key if k["answer"] not in svk.UNKNOWABLE]

    prec_path = OUTDIR / session_id / "precision.json"
    ctrl_pos = []
    if prec_path.exists():
        prec = json.loads(prec_path.read_text())
        ctrl_pos = [{"kind": "ctrl", "start": r["start"], "answer": None}
                    for r in prec["rows"]["none"]]

    clipdir = OUTDIR / session_id / "listen-clips"
    positions = key_pos + ctrl_pos
    print(f"[listen] cutting {len(positions)} clips ({len(key_pos)} key + {len(ctrl_pos)} control)",
          file=sys.stderr)
    paths = cut_wav_clips(session_dir, positions, clipdir, lead, tail)

    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    t0 = time.time()
    model, processor = load(GEMMA)
    print(f"[listen] gemma loaded in {time.time()-t0:.0f}s", file=sys.stderr)

    def ask(prompt_text, wav):
        fmt = apply_chat_template(processor, model.config, prompt_text, num_audios=1)
        res = generate(model, processor, fmt, audio=[str(wav)], max_tokens=60,
                       temperature=0.0, verbose=False)
        return (getattr(res, "text", res) or "").strip()

    rows, hits, false_picks = [], 0, 0
    for p, wav in zip(positions, paths):
        pick = ask(PICK_PROMPT, wav)
        pick_clean = clean(pick.split()[0]) if pick.split() else ""
        row = {"kind": p["kind"], "start": round(p["start"], 1), "answer": p["answer"],
               "pick": pick[:60]}
        if p["kind"] == "key":
            ok = bool(pick_clean) and pick_clean != "none" and p["answer"] and (
                pick_clean == clean(p["answer"]) or bool(codes(pick_clean) & codes(clean(p["answer"]))))
            row["match"] = "hit" if ok else "miss"
            hits += ok
        else:
            picked_name = bool(pick_clean) and pick_clean != "none"
            row["match"] = "false_pick" if picked_name else "correct_none"
            false_picks += picked_name
        if open_too and p["kind"] == "key":
            row["open"] = ask(OPEN_PROMPT, wav)[:120]
        rows.append(row)
        print(f"  [{time.time()-t0:4.0f}s] {p['kind']} {p['start']:6.1f} "
              f"ans={str(p['answer']):14.14} pick={pick[:24]:24} {row['match']}", file=sys.stderr)

    out = {"session": session_id, "lead": lead, "tail": tail, "cast": CAST,
           "pick_prompt": PICK_PROMPT,
           "key_hits": hits, "key_total": len(key_pos),
           "key_recall": round(hits / len(key_pos), 3) if key_pos else 0,
           "ctrl_false_picks": false_picks, "ctrl_total": len(ctrl_pos),
           "ctrl_false_rate": round(false_picks / len(ctrl_pos), 3) if ctrl_pos else None,
           "qwen_audio_tower": qwen_audio_check(), "rows": rows}
    (OUTDIR / session_id / "gemma_listen.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[listen] KEY {hits}/{len(key_pos)} ({out['key_recall']})  "
          f"CTRL false-picks {false_picks}/{len(ctrl_pos)} ({out['ctrl_false_rate']})", file=sys.stderr)
    print(f"[listen] qwen audio tower: {out['qwen_audio_tower']}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--lead", type=float, default=1.5)
    ap.add_argument("--tail", type=float, default=2.0)
    ap.add_argument("--open-too", action="store_true")
    a = ap.parse_args()
    main(a.session_id, a.lead, a.tail, a.open_too)
