#!/usr/bin/env python3
"""Tune the pass-2 WORLD prompt for Qwen3.5. Validation found it abstains on the Pandavas
Mahabharata story (returns "original") even though it names the characters — too eager to
abstain. Goal: recognize a world the characters clearly point to (Pandavas->Mahabharata,
Portal-3->Thomas) WITHOUT making the 5 genuinely made-up stories hallucinate one.

Scored on the 7 hand-marked truth spans. The win condition: the 2 canon spans get a
non-empty, correct-sounding world; the 5 'original' spans stay empty. Plain text in/out.

    ./venv/bin/python emp/src/tune_pass2_worlds.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from segment import SESSIONS, load_truth                                  # noqa: E402
from story_segment import load_segments, full_region_lines               # noqa: E402
from story_segment_qwen35 import make_reader, parse_name                 # noqa: E402

HEAD = ("You are reading the full transcript of ONE bedtime story a parent told a young child, "
        "out loud — so it rambles and proper names may be mis-transcribed.\n\n")
FMT = ("\n\nWrite exactly two lines, nothing else:\n"
       "Title: <short title>\nWorld: <the world, or \"original\" if made up>\n\nStory lines:\n{lines}")

# Variant bodies — the instruction between HEAD and FMT.
VARIANTS = {
    "current": (
        "Give it a short descriptive title, and decide what fictional WORLD it is set in. Think "
        "in two quick steps:\n"
        "1) Note the distinctive character names, place names, and signature words. Names may be "
        "garbled — judge by what they sound like.\n"
        "2) Do they match a REAL, widely-known story world (a book, film, show, myth, or "
        'franchise)? If yes, name that world. If it is an original, made-up story from no known '
        'world, write "original". When unsure, write "original".'),
    "recognize": (
        "Give it a short descriptive title, and decide what fictional WORLD it is set in. Think "
        "in two quick steps:\n"
        "1) Note the distinctive character names, place names, and signature words. Names may be "
        "garbled — judge by what they sound like.\n"
        "2) Use your knowledge of stories. Many well-known worlds are NEVER named aloud — you "
        "recognize them from their characters and places. If the names point to a real book, "
        "film, show, myth, or franchise, NAME that world. Only write \"original\" if the story is "
        "genuinely made-up with no recognizable world."),
    "example": (
        "Give it a short descriptive title, and decide what fictional WORLD it is set in.\n"
        "Use your knowledge of stories: recognize a world from its CHARACTERS and PLACES, even if "
        "the story never says the world's name. For instance, a story about a boy wizard at a "
        "school of magic is \"Harry Potter\" even if those words are never spoken; a story about "
        "two warring families of cousins, with a great archer and a charioteer prince, is the "
        "\"Mahabharata\". If the names point to a real book, film, show, myth, or franchise, name "
        "it. Only write \"original\" if the story is genuinely made-up with no recognizable world."),
    "twostep_soft": (
        "Give it a short descriptive title, and decide what fictional WORLD it is set in. Think "
        "in two quick steps:\n"
        "1) Note the distinctive character names, place names, and signature words. Names may be "
        "garbled — judge by what they sound like.\n"
        "2) Do these characters and places come from a REAL, widely-known story world (a book, "
        "film, show, myth, or franchise) — even one never named aloud in the story? If you can "
        "place them, name that world. Write \"original\" ONLY when you cannot place the characters "
        "in any known world."),
}

CANON = {"Pandavas", "Portal-3"}  # the two spans that SHOULD get a non-empty world


def main():
    truth = load_truth()
    gen = make_reader()
    # Build the 7 (label, lines, truth_world) spans once.
    spans = []
    for sid, name in SESSIONS.items():
        segs = load_segments(str(ROOT / "sessions" / sid))
        pos_of = {s["id"]: i for i, s in enumerate(segs)}
        for k, st in enumerate(truth[sid]):
            sp, ep = pos_of.get(st["start"]), pos_of.get(st["end"])
            if sp is None or ep is None:
                continue
            label = f"{name}-{k+1}" if len(truth[sid]) > 1 else name
            region = {"start_pos": min(sp, ep), "end_pos": max(sp, ep)}
            spans.append((label, full_region_lines(segs, region), st.get("world", "")))

    for v, body in VARIANTS.items():
        prompt = HEAD + body + FMT
        print(f"\n=== variant '{v}' ===", flush=True)
        n_canon_ok = n_made_up_clean = 0
        for label, lines, tw in spans:
            raw = gen(prompt.format(lines=lines), max_tokens=160)
            _, world = parse_name(raw)
            is_canon = label in CANON
            ok = (bool(world) if is_canon else not world)
            n_canon_ok += (is_canon and bool(world))
            n_made_up_clean += (not is_canon and not world)
            mark = "ok " if ok else "XX "
            print(f"  {mark}{label:12} truth={tw[:20] or '(original)':20} -> {world!r}", flush=True)
        print(f"  canon recognized {n_canon_ok}/2   made-up kept empty {n_made_up_clean}/5", flush=True)


if __name__ == "__main__":
    main()
