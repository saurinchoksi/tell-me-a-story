#!/usr/bin/env python3
"""(a) Does LOWERING the 4-letter candidate floor surface the short names (Ray/Fin/Po) to the
judge, and what does it cost in false flags? Judge-only, spread, floors 4/3/2. Plain text.

    ./venv/bin/python emp/src/test_floor.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from mlx_vlm import load, generate                                  # noqa: E402
from detectors.phonetics import clean, is_capitalized               # noqa: E402
from detectors.story_names._audit import gate_canon_flags           # noqa: E402
from detectors.story_names._names import proper_name_candidates      # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit, forms    # noqa: E402

model, processor = load("mlx-community/Qwen3.5-4B-MLX-4bit")
tok = getattr(processor, "tokenizer", processor)
SENT_END = (".", "!", "?", "…", '."', '?"', '!"')
JUDGE = ('This bedtime story is set in the world of "{world}". Below is a list of names exactly '
         'as a transcriber wrote them by ear — some may be misspelled.\n\nFor each name that is '
         'really a character or place from {world} (or other very famous canon) but spelled '
         'WRONG, write ONE line:\n<name as written> -> <correct spelling>\nSkip any name already '
         'spelled correctly, made up, or not a real name. Output only the correction lines.\n\n'
         'Names:\n{names}')


def surface_candidates(segs, min_len):
    out = set()
    for s in segs:
        words = s.get("words", [])
        for i, w in enumerate(words):
            raw = w["word"].strip()
            c = clean(raw)
            prev = words[i - 1]["word"].strip() if i > 0 else ""
            si = i == 0 or prev.endswith(SENT_END)
            if c and len(c) >= min_len and is_capitalized(raw) and not si:
                out.add(raw.strip('.,?!"\''))
    return sorted(out)


def judge(world, names):
    fmt = tok.apply_chat_template([{"role": "user", "content": JUDGE.format(world=world, names="\n".join(names))}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    raw = getattr(generate(model, processor, fmt, max_tokens=500, temperature=0.0, verbose=False), "text", "") or ""
    flags, seen = [], set()
    for m in re.finditer(r"^\s*([A-Za-z][\w '.-]{1,30}?)\s*-+>\s*([A-Za-z][\w '.-]{1,30})\s*$", raw, re.M):
        wc = clean(m.group(1))
        if wc and clean(m.group(2)) != wc and wc not in seen:
            seen.add(wc)
            flags.append({"case": "M9c", "canonical": m.group(2).strip(), "wrong_surface": [m.group(1).strip()],
                          "wrong_cleaned": [wc], "all_spellings": [m.group(1).strip()], "card_id": len(flags), "evidence": ""})
    return flags


def main():
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    print(f"{'floor':6} {'spread recall':>14} {'false':>6}   (short-name worlds: Star Wars)")
    for ml in (4, 3, 2):
        tot = caught = fp = 0
        for st in spread:
            segs = build_segs(st["lines"])
            names = surface_candidates(segs, ml)
            singles = proper_name_candidates(segs, min_len=ml)
            fl = gate_canon_flags(judge(st["world"], names), singles)
            cg = m9c_caught(fl)
            caught += sum(1 for pl in st["planted"] if hit(pl["heard"], cg))
            pf = {clean(t) for pl in st["planted"] for t in pl["heard"].split()} | {clean(pl["heard"]) for pl in st["planted"]}
            fp += sum(1 for f in fl if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pf))
            tot += len(st["planted"])
        print(f"{ml:>6} {str(caught) + '/' + str(tot) + ' = ' + format(caught / tot, '.2f'):>14} {fp:>6}", flush=True)
    print("\n  reference: floor=4 judge-only was 0.72 spread, 4 false")


if __name__ == "__main__":
    main()
