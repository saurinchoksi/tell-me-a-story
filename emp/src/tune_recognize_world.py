#!/usr/bin/env python3
"""Recognize the world from the NAME LIST, not the rambling transcript. Production found
that pass-2 world naming over a long whole-recording region abstains to empty (both
held-outs: world=''), so the canon check never starts — even though the judge clearly
knows the names (it corrects Bishma->Bhishma). This tunes a focused recognize_world prompt
fed only the story's distinct name spellings.

Target: the Mahabharata held-out -> a Mahabharata world; the KPop held-out -> KPop Demon
Hunters (or at least non-empty). Plain text in/out.

    ./venv/bin/python emp/src/tune_recognize_world.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from detectors.story_names._audit import story_name_cards                 # noqa: E402
from story_segment_qwen35 import make_reader, parse_name                  # noqa: E402

HELDOUT = {"20260211-210718": "Mahabharata (expected)", "20260414-213156": "KPop Demon Hunters (expected)"}

HEAD = ("These names were heard in ONE children's bedtime story, written down by ear — some may "
        "be misspelled:\n\n{names}\n\n")
TAIL = '\n\nWrite one line, nothing else:\nWorld: <the world, or "original" if made up>'

VARIANTS = {
    "basic": (
        "What well-known fictional WORLD (a book, film, show, myth, or franchise) do these "
        "characters and places come from? If they clearly come from a real, widely-known world, "
        'name it. If they are made-up with no recognizable world, write "original".'),
    "soft": (
        "Do these characters and places come from a REAL, widely-known story world (a book, film, "
        "show, myth, or franchise) — even one the story never names aloud? If you can place them, "
        'name that world. Write "original" ONLY when you cannot place the names in any known world.'),
    "byear": (
        "Think about which well-known characters these garbled, by-ear names SOUND like, then name "
        "the single world they all come from. Many famous worlds are never named aloud — recognize "
        'them from the characters. Write "original" only if the names point to no known world.'),
}


def names_for(sid):
    rich = json.loads((ROOT / "sessions" / sid / "transcript-rich.json").read_text())
    cards = story_name_cards(rich["segments"], recover=True)
    return sorted({s for c in cards for s in c["surface"]})


def main():
    gen = make_reader()
    data = {sid: names_for(sid) for sid in HELDOUT}
    for sid, exp in HELDOUT.items():
        print(f"\n{sid} — {exp}   ({len(data[sid])} names: {data[sid][:18]}...)", flush=True)
        for v, body in VARIANTS.items():
            prompt = HEAD.format(names="\n".join(data[sid])) + body + TAIL
            _, world = parse_name(gen(prompt, max_tokens=120))
            print(f"    {v:7} -> {world!r}", flush=True)


if __name__ == "__main__":
    main()
