#!/usr/bin/env python3
"""
LLM precision-filter test: can Gemma 4 E2B (~2B, the smallest deployable Gemma 4)
clean up the M9a code detector's false positives?  EXPERIMENT (branch only).

L1 (the deterministic detector) flags any token that *sounds like* a family name;
it can't tell a real name from a homophone (e.g. "where'd" -> code ART == "Arti").
This feeds each of L1's flags, WITH its surrounding sentence, to E2B and asks a
single yes/no in context: is the flagged word a person's NAME, or an ordinary word
that merely sounds like one?  It then scores those keep/drop calls against the same
human ear-pass ground truth (axial-labels M9 segments) that L1 was scored on.

Roster-free by design: L1 already did the roster match; the model only adds context
judgment, so no real names go in the prompt.  Read-only re: data; the only output is
a gitignored JSON of verdicts.  MUST run in the mlx-vlm venv:

    ./venv-mlx-vlm/bin/python emp/src/filter_e2b.py

Model note: mlx-vlm supports the *multimodal* gemma4 build, not the text-only
`gemma4_text` checkpoint (and mlx-lm can't load these int4 builds) -- so the only
working path on this Mac is the multimodal build driven with a text-only prompt.
"""
import json
from collections import Counter
from pathlib import Path

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

ROOT = Path(__file__).resolve().parents[2]
MODEL = "mlx-community/gemma-4-e2b-it-4bit"
SESSIONS = ["20260414-213156", "20260211-210718"]
RUNS = 3


def load_session(sid):
    base = ROOT / "sessions" / sid
    rich = json.loads((base / "transcript-rich.json").read_text())
    text = {str(s["id"]): (s.get("text") or "").strip() for s in rich["segments"]}
    ax = json.loads((base / "axial-labels.json").read_text())
    m9 = {str(l["segmentId"]) for l in ax["labels"]
          if any(c.startswith("M9") for c in l.get("codes", []))}
    flags = json.loads((ROOT / "emp" / "results" / "visuals" / sid
                        / "m9a-l1-flags.json").read_text())["flags"]
    return text, m9, flags


def build_prompt(token, sentence):
    return (
        "You are checking a speech-to-text transcript. A word was automatically "
        "flagged because it sounds like a name. Decide, from the context of the "
        "sentence, whether the flagged word is actually being used as a person's "
        "NAME, or is just an ordinary word or contraction that happens to sound "
        "like a name.\n\n"
        f'Sentence: "{sentence}"\n'
        f'Flagged word: "{token}"\n\n'
        "Answer with exactly one word: NAME or WORD."
    )


def classify(model, processor, token, sentence):
    prompt = apply_chat_template(processor, model.config, build_prompt(token, sentence))
    try:
        res = generate(model, processor, prompt, max_tokens=8, temperature=0.0, verbose=False)
    except Exception as e:               # surface a crash as its own verdict
        return "CRASH", repr(e)[:160]
    raw = (getattr(res, "text", res) or "").strip()
    up = raw.upper()
    verdict = "NAME" if "NAME" in up else ("WORD" if "WORD" in up else "?")
    return verdict, raw


def main():
    print(f"loading {MODEL} ...")
    model, processor = load(MODEL)
    print("loaded.\n")

    agg = Counter()          # pooled tallies
    for sid in SESSIONS:
        text, m9, flags = load_session(sid)
        out = []
        print(f"=== {sid} ({len(flags)} flags) ===")
        print(f"{'seg':>5} {'token':<10} {'truth':<11} {'3 runs':<22} {'verdict':<7} result")
        for f in flags:
            seg, token = str(f["segment_id"]), f["token"]
            sentence = text.get(seg, "")
            runs = [classify(model, processor, token, sentence) for _ in range(RUNS)]
            verdicts = [v for v, _ in runs]
            majority = Counter(verdicts).most_common(1)[0][0]
            stable = len(set(verdicts)) == 1
            is_name = seg in m9                       # human truth: real M9a name?
            ideal = "NAME" if is_name else "WORD"
            correct = majority == ideal
            agg["names" if is_name else "fps"] += 1
            if is_name and majority == "NAME":   agg["names_kept"] += 1
            if is_name and majority == "WORD":   agg["names_dropped"] += 1   # recall damage
            if not is_name and majority == "WORD": agg["fps_dropped"] += 1   # precision gain
            if not is_name and majority == "NAME": agg["fps_kept"] += 1
            agg["stable"] += stable
            if "CRASH" in verdicts: agg["crashes"] += 1
            out.append({"segment_id": f["segment_id"], "token": token, "sentence": sentence,
                        "human_truth": ideal, "verdicts": verdicts, "majority": majority,
                        "stable": stable, "correct": correct, "raw": [r for _, r in runs]})
            mark = "ok" if correct else "**MISS**"
            print(f"{seg:>5} {token:<10} {('NAME' if is_name else 'word-FP'):<11} "
                  f"{str(verdicts):<22} {majority:<7} {mark}")
        outdir = ROOT / "emp" / "results" / "visuals" / sid
        (outdir / "m9a-e2b-filter.json").write_text(json.dumps(
            {"_about": "Gemma 4 E2B precision-filter verdicts over L1 flags (3 runs each); "
                       "NAME=keep, WORD=drop. Gitignored (tokens).",
             "model": MODEL, "session": sid, "runs": RUNS, "flags": out}, indent=2))
        print(f"   -> wrote {outdir.relative_to(ROOT)}/m9a-e2b-filter.json\n")

    # ---- pooled scorecard ----
    names, fps = agg["names"], agg["fps"]
    kept = agg["names_kept"] + agg["fps_kept"]
    prec = agg["names_kept"] / kept if kept else 0
    rec = agg["names_kept"] / names if names else 0
    print("=" * 60)
    print("POOLED FILTER SCORECARD (L1 flags -> E2B keep/drop)")
    print(f"  real names kept (NAME):      {agg['names_kept']}/{names}"
          f"   {'<-- recall damage: '+str(agg['names_dropped']) if agg['names_dropped'] else ''}")
    print(f"  false positives dropped:     {agg['fps_dropped']}/{fps}"
          f"   {'(fp kept: '+str(agg['fps_kept'])+')' if agg['fps_kept'] else ''}")
    print(f"  stable across {RUNS} runs:        {agg['stable']}/{names+fps}    crashes: {agg['crashes']}")
    print(f"  --> L1+E2B precision = {prec:.3f}   recall = {rec:.3f}")
    print(f"      (L1 alone: precision 0.909, recall 1.000; capitalization gate also drops the FP)")


if __name__ == "__main__":
    main()
