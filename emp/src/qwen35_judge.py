#!/usr/bin/env python3
"""Qwen3.5 as a NAME JUDGE — skip the cast + fuzzy entirely. Feed Qwen3.5 the actual transcript
names (plain text, no JSON) and let it reason about which are misspelled canon names. Tested
with thinking on and off, scored on the real Mahabharata. This is the LLM-judge approach that
FAILED on Gemma — the question is whether Qwen3.5's reasoning makes it work.

    ./venv/bin/python emp/src/qwen35_judge.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

import json                                                          # noqa: E402
from mlx_vlm import load, generate                                  # noqa: E402
from detectors.story_names._audit import story_name_cards, gate_canon_flags  # noqa: E402
from detectors.story_names._names import proper_name_candidates      # noqa: E402
from detectors.phonetics import clean                                # noqa: E402
from score_canon_heldout import score_canon, load_items             # noqa: E402

SID = "20260211-210718"
MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

# PLAIN TEXT in and out — no JSON anywhere (we format on our side).
JUDGE = """This bedtime story is set in the world of "{world}". Below is a list of names exactly as a transcriber wrote them down by ear — some may be misspelled.

For each name that is really a character or place from {world} (or other very famous canon) but spelled WRONG, write ONE line:
<name as written> -> <correct spelling>

Skip any name that is already spelled correctly, is made up, or is not a real name. Output only the correction lines, nothing else.

Names:
{names}"""

model, processor = load(MODEL)
tok = getattr(processor, "tokenizer", processor)


def judge(world, names, think):
    fmt = tok.apply_chat_template([{"role": "user", "content": JUDGE.format(world=world, names="\n".join(names))}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=think)
    o = generate(model, processor, fmt, max_tokens=2500 if think else 500, temperature=0.0, verbose=False)
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
    segs = rich["segments"]
    cards = story_name_cards(segs, recover=True)
    singles = proper_name_candidates(segs)
    names = sorted({s for c in cards for s in c["surface"]})        # the distinct spellings to judge
    items = load_items(SID) or {}
    print(f"REAL MAHABHARATA — {len(names)} names judged (plain text, no thinking)")
    flags = gate_canon_flags(judge("Mahabharata", names, False), singles)
    r = score_canon(items, flags)
    print(f"  recall {r['caught']}/{r['gold_m9c']}  precision {r['matrix']['M9c']['flagged']}/{r['flagged']}  caught {r['hits']}")

    # ---- spread (the generalization check) ----
    from bench_canon_spread import build_segs, m9c_caught, hit, forms
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    print("\nSPREAD — judge per world (no thinking):")
    tot = caught = fp = 0
    for st in spread:
        s2 = build_segs(st["lines"])
        cds = story_name_cards(s2, recover=True)
        sg = proper_name_candidates(s2)
        nm = sorted({s for c in cds for s in c["surface"]})
        fl = gate_canon_flags(judge(st["world"], nm, False), sg)
        cg = m9c_caught(fl)
        rc = sum(1 for pl in st["planted"] if hit(pl["heard"], cg))
        pf = {clean(t) for pl in st["planted"] for t in pl["heard"].split()} | {clean(pl["heard"]) for pl in st["planted"]}
        fpc = sum(1 for f in fl if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pf))
        tot += len(st["planted"]); caught += rc; fp += fpc
        print(f"  {st['world'][:32]:34} {rc}/{len(st['planted'])}  (false {fpc})", flush=True)
    print(f"  SPREAD recall {caught}/{tot} = {caught / tot:.2f}   false flags {fp}")
    print("\n  reference: union cast+fuzzy = 8/11 real, 0.68 spread; Gemma judge (old) 0-1/11")


if __name__ == "__main__":
    main()
