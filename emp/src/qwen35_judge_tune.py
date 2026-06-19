#!/usr/bin/env python3
"""Prompt-tune the Qwen3.5 name judge. Four plain-text variants (no JSON):
  base      - the current judge.
  precision - adds a confidence guard to cut over-corrections (the 4 spread false flags).
  fewshot   - adds catch/skip examples to calibrate.
  cast      - "other methods PLUS": hand the judge the world's known cast as a reference list.
Scored on real Mahabharata (by-ear) + the spread.  Plain text in and out.

    ./venv/bin/python emp/src/qwen35_judge_tune.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from mlx_vlm import load, generate                                  # noqa: E402
from detectors.story_names._audit import story_name_cards, gate_canon_flags  # noqa: E402
from detectors.story_names._names import proper_name_candidates      # noqa: E402
from detectors.phonetics import clean                                # noqa: E402
from score_canon_heldout import score_canon, load_items             # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit, forms    # noqa: E402

SID = "20260211-210718"
model, processor = load("mlx-community/Qwen3.5-4B-MLX-4bit")
tok = getattr(processor, "tokenizer", processor)
CASTS = {p.stem: json.loads(p.read_text())["casts"]
         for p in (ROOT / "emp" / "results" / "canon-spread" / "casts").glob("qwen35.json")}

HEAD = ('This bedtime story is set in the world of "{world}". Below is a list of names exactly '
        'as a transcriber wrote them by ear — some may be misspelled.\n')
TASK = ('\nFor each name that is really a character or place from {world} (or other very famous '
        'canon) but spelled WRONG, write ONE line:\n<name as written> -> <correct spelling>\n'
        'Skip any name already spelled correctly, made up, or not a real name. Output only the '
        'correction lines, nothing else.\n\nNames:\n{names}')
GUARD = ('\nOnly output a correction when you are CONFIDENT the name is a real, well-known '
         'character or place from {world} that has been misspelled. If a name could be made up, '
         'or you are unsure, do NOT output a line for it.')
SHOTS = ('\nFor example, in a Mahabharata story you would output "Bishma -> Bhishma" and '
         '"Drauna -> Drona", but skip "arrows" (an ordinary word) and "Jammus" (a made-up name).')


def build(variant, world, names):
    n = "\n".join(names)
    if variant == "base":
        return HEAD.format(world=world) + TASK.format(world=world, names=n)
    if variant == "precision":
        return HEAD.format(world=world) + GUARD.format(world=world) + TASK.format(world=world, names=n)
    if variant == "fewshot":
        return HEAD.format(world=world) + SHOTS + TASK.format(world=world, names=n)
    if variant == "cast":
        cast = CASTS.get("qwen35", {}).get(world, [])
        ref = f'\nWell-known {world} names include: {", ".join(cast[:30])}.' if cast else ""
        return HEAD.format(world=world) + ref + TASK.format(world=world, names=n)


def judge(variant, world, names):
    prompt = build(variant, world, names)
    fmt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    o = generate(model, processor, fmt, max_tokens=500, temperature=0.0, verbose=False)
    raw = getattr(o, "text", o) or ""
    flags, seen = [], set()
    for m in re.finditer(r"^\s*([A-Za-z][\w '.-]{1,30}?)\s*-+>\s*([A-Za-z][\w '.-]{1,30})\s*$", raw, re.M):
        wrong, correct = m.group(1).strip(), m.group(2).strip()
        wc = clean(wrong)
        if not wc or clean(correct) == wc or wc in seen:
            continue
        seen.add(wc)
        flags.append({"case": "M9c", "canonical": correct, "wrong_surface": [wrong],
                      "wrong_cleaned": [wc], "all_spellings": [wrong], "card_id": len(flags), "evidence": ""})
    return flags


def main():
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    mcards = story_name_cards(rich["segments"], recover=True)
    msingles = proper_name_candidates(rich["segments"])
    mnames = sorted({s for c in mcards for s in c["surface"]})
    items = load_items(SID) or {}
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    spread_pre = [(st, story_name_cards(build_segs(st["lines"]), recover=True),
                   proper_name_candidates(build_segs(st["lines"]))) for st in spread]

    print(f"{'variant':10} | maha recall  prec | spread recall  false")
    print("-" * 56)
    for v in ("base", "precision", "fewshot", "cast"):
        r = score_canon(items, gate_canon_flags(judge(v, "Mahabharata", mnames), msingles))
        tot = caught = fp = 0
        for st, cds, sg in spread_pre:
            nm = sorted({s for c in cds for s in c["surface"]})
            fl = gate_canon_flags(judge(v, st["world"], nm), sg)
            cg = m9c_caught(fl)
            caught += sum(1 for pl in st["planted"] if hit(pl["heard"], cg))
            pf = {clean(t) for pl in st["planted"] for t in pl["heard"].split()} | {clean(pl["heard"]) for pl in st["planted"]}
            fp += sum(1 for f in fl if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pf))
            tot += len(st["planted"])
        print(f"{v:10} |   {r['caught']}/{r['gold_m9c']}      {r['matrix']['M9c']['flagged']}/{r['flagged']} |   "
              f"{caught}/{tot} = {caught / tot:.2f}    {fp}", flush=True)
    print("\n  baseline judge: 8/11 real (8/9 prec), 0.68 spread, 4 false")


if __name__ == "__main__":
    main()
