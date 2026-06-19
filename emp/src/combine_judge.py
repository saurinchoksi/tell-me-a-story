#!/usr/bin/env python3
"""Stack the plain Qwen3.5 judge with sound-matching by UNIONING their catches (independent
methods that fail differently). Tests judge / judge+phonetic / judge+fuzzy / judge+both on the
real Mahabharata + spread. The matcher uses Qwen3.5's own cast (no other models). Judge runs
once per story; the four configs just combine the precomputed flag sets.

    ./venv/bin/python emp/src/combine_judge.py
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
from bench_cast import cast_index                                    # noqa: E402
from fuzzy_canon import flags_for                                    # noqa: E402

SID = "20260211-210718"
model, processor = load("mlx-community/Qwen3.5-4B-MLX-4bit")
tok = getattr(processor, "tokenizer", processor)
QCAST = json.loads((ROOT / "emp" / "results" / "canon-spread" / "casts" / "qwen35.json").read_text())["casts"]

JUDGE = ('This bedtime story is set in the world of "{world}". Below is a list of names exactly '
         'as a transcriber wrote them by ear — some may be misspelled.\n\nFor each name that is '
         'really a character or place from {world} (or other very famous canon) but spelled '
         'WRONG, write ONE line:\n<name as written> -> <correct spelling>\nSkip any name already '
         'spelled correctly, made up, or not a real name. Output only the correction lines.\n\n'
         'Names:\n{names}')


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


def match(cards, singles, cast, code_ed):
    return flags_for(cards, singles, *cast_index(cast), code_ed, 0.55)


def combine(*lists):
    out, seen = [], set()
    for fl in lists:
        for f in fl:
            key = tuple(sorted(f.get("wrong_cleaned", [])))
            if key and key not in seen:
                seen.add(key)
                out.append(f)
    return out


CONFIGS = {"judge": (1, 0, 0), "judge+phonetic": (1, 1, 0), "judge+fuzzy": (1, 0, 1), "judge+both": (1, 1, 1)}


def pick(j, p, f, flags_on):
    parts = [x for x, on in zip((j, p, f), flags_on) if on]
    return combine(*parts)


def main():
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    mcards = story_name_cards(rich["segments"], recover=True)
    msingles = proper_name_candidates(rich["segments"])
    mnames = sorted({s for c in mcards for s in c["surface"]})
    items = load_items(SID) or {}
    mj = gate_canon_flags(judge("Mahabharata", mnames), msingles)
    mp = match(mcards, msingles, QCAST.get("Mahabharata", []), 0)
    mf = match(mcards, msingles, QCAST.get("Mahabharata", []), 1)

    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    sp = []
    for st in spread:
        segs = build_segs(st["lines"])
        cds, sg = story_name_cards(segs, recover=True), proper_name_candidates(build_segs(st["lines"]))
        nm = sorted({s for c in cds for s in c["surface"]})
        jf = gate_canon_flags(judge(st["world"], nm), sg)
        cast = QCAST.get(st["world"], [])
        sp.append((st, jf, match(cds, sg, cast, 0), match(cds, sg, cast, 1)))

    print(f"{'config':16} | maha recall  prec | spread       false")
    print("-" * 56)
    for name, on in CONFIGS.items():
        r = score_canon(items, pick(mj, mp, mf, on))
        tot = caught = fp = 0
        for st, jf, pf, ff in sp:
            cg = m9c_caught(pick(jf, pf, ff, on))
            caught += sum(1 for pl in st["planted"] if hit(pl["heard"], cg))
            pforms = {clean(t) for pl in st["planted"] for t in pl["heard"].split()} | {clean(pl["heard"]) for pl in st["planted"]}
            fp += sum(1 for f in pick(jf, pf, ff, on) if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pforms))
            tot += len(st["planted"])
        print(f"{name:16} |   {r['caught']}/{r['gold_m9c']}     {r['matrix']['M9c']['flagged']}/{r['flagged']} |   "
              f"{caught}/{tot}={caught / tot:.2f}    {fp}", flush=True)


if __name__ == "__main__":
    main()
