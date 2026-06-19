#!/usr/bin/env python3
"""Why did spell-it produce ZERO flags on the real Mahabharata? Re-run the spell pass on that
session's real cards with the model's RAW output shown, to tell apart (a) the model genuinely
returning no corrections from (b) a parse failure silently dropping flags.

    ./venv/bin/python emp/src/diag_spell_real.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from story_segment import make_reader, segment_session, extract_json   # noqa: E402
from detectors.story_names._audit import (story_name_cards, render_cards,  # noqa: E402
                                          CARD_CHUNK)
from detectors.story_names._names import story_segments                # noqa: E402
from audit_spell import SPELL_PROMPT                                   # noqa: E402

SID = "20260211-210718"
gen = make_reader()
rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
pos_of = {s["id"]: i for i, s in enumerate(rich["segments"])}
seg_result, _ = segment_session(ROOT / "sessions" / SID, gen)
st = seg_result["stories"][0]
sp, ep = pos_of[st["start_id"]], pos_of[st["end_id"]]
if ep < sp:
    sp, ep = ep, sp
segs = story_segments(rich, {"start_pos": sp, "end_pos": ep}, pos_of)
cards = story_name_cards(segs, recover=True)
world = st.get("world", "")

print(f"world={world!r}   n_cards={len(cards)}   CARD_CHUNK={CARD_CHUNK}   "
      f"-> {(len(cards) + CARD_CHUNK - 1) // CARD_CHUNK} chunk(s)")
print("cards that should be canon misspellings:")
for c in cards:
    j = " ".join(c["clean"])
    if any(x in j for x in ["urjun", "bishma", "garn", "bandos", "duryod", "beem", "drop"]):
        print(f"   card[{c['id']}] {c['surface']}")

for i in range(0, len(cards), CARD_CHUNK):
    chunk = cards[i:i + CARD_CHUNK]
    raw = gen(SPELL_PROMPT.format(world=world, cards=render_cards(chunk)), max_tokens=900)
    obj = extract_json(raw)
    print(f"\n{'=' * 60}\nCHUNK {i // CARD_CHUNK}: {len(chunk)} cards   parsed={obj is not None}")
    if isinstance(obj, dict):
        names = obj.get("names", [])
        nonempty = [(n.get("id"), n.get("correct")) for n in names
                    if isinstance(n, dict) and str(n.get("correct", "")).strip()]
        print(f"  model returned {len(names)} entries; non-empty corrections: {nonempty}")
    print(f"  RAW (first 700 chars):\n{raw[:700]}")
