#!/usr/bin/env python3
"""E8 — split cast prompts: CHARACTERS and GROUPS asked separately, per model.

The cast-prompt sweep found that asking for "characters" makes both Qwen and Gemma drop the
GROUP names (Pandavas, Kauravas — 14 of the 36 by-ear key items). Instead of one prompt that
does both badly, ask twice: the settled characters prompt, plus a dedicated families/groups/
factions prompt, and union the results into the correction cast.

One model per invocation (16 GB: one local model at a time). Output merges into
emp/results/visuals/cast-prompt-sweep/split-casts.json  ({model: {world: {characters, groups}}}).

Usage: python emp/src/cast_split.py qwen|gemma
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

WORLDS = ["Mahabharata", "Star Wars", "Harry Potter", "Thomas & Friends",
          "Steven Universe", "Teenage Mutant Ninja Turtles", "Super Mario"]

CHARACTERS_PROMPT = (
    'List the main characters of the fictional world "{world}" - real individual people only, '
    'never places, groups, items, species, or titles. Give the 12 to 18 most central characters, '
    'most important first, and be sure to include the main protagonist(s), the main antagonist or '
    'villain, and the most important elder or mentor. Use the correct canonical English spelling of '
    'each name. Output the names only, one name per line - no numbers, no descriptions, no '
    'commentary, and no duplicate or alternate spellings. List each character exactly once and stop '
    'once the central cast is covered.')

GROUPS_PROMPT = (
    'List the important named GROUPS of the fictional world "{world}" - the families, clans, teams, '
    'houses, factions, peoples, and collective names the story uses (for example, the name for the '
    'heroes as a group, and the name for their enemies as a group). Only real collective names from '
    'that world, not individual people and not places. Give up to 8, most important first. Use the '
    'correct canonical English spelling. Output the names only, one per line - no numbers, no '
    'descriptions, no commentary. Stop when the important groups are covered.')

OUT = ROOT / "emp/results/visuals/cast-prompt-sweep/split-casts.json"


def parse(raw):
    out, seen = [], set()
    for line in raw.splitlines():
        s = re.sub(r"^[\s\-\*.\d)]+", "", line).strip().strip('",')
        s = re.sub(r"^[Tt]he\s+", "", s)  # normalize "The Kauravas" -> "Kauravas"
        if not s or len(s) > 40 or s.lower() in seen:
            continue
        if re.search(r"[^\x00-\x7f]", s) or re.search(r"[:{}\[\]]", s):
            continue
        seen.add(s.lower()); out.append(s)
    return out


def main(model_key):
    if model_key == "qwen":
        from qwen35 import make_reader
    elif model_key == "gemma":
        from story_segment import make_reader
    else:
        raise SystemExit("usage: cast_split.py qwen|gemma")
    t0 = time.time()
    gen = make_reader()
    print(f"{model_key} loaded in {time.time()-t0:.0f}s", file=sys.stderr)

    res = {}
    for w in WORLDS:
        chars = parse(gen(CHARACTERS_PROMPT.format(world=w), max_tokens=500))
        groups = parse(gen(GROUPS_PROMPT.format(world=w), max_tokens=200))
        res[w] = {"characters": chars, "groups": groups}
        print(f"[{time.time()-t0:4.0f}s] {w:30} chars={len(chars)} groups={len(groups)} "
              f"| groups: {', '.join(groups)}", file=sys.stderr)

    data = json.loads(OUT.read_text()) if OUT.exists() else {"prompts": {
        "characters": CHARACTERS_PROMPT, "groups": GROUPS_PROMPT}}
    data[model_key] = res
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print("WROTE", OUT, file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
