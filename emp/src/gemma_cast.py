#!/usr/bin/env python3
"""Generate casts with GEMMA (gemma-4-e4b-it-4bit) using the improved prompt — to compare with
Qwen and to build a combined (union) cast. Answers Choksi's "what does Gemma give us / can we
combine cast lists?" One Gemma load (process exits → frees Metal). Read-only.

Output -> emp/results/visuals/cast-prompt-sweep/gemma-casts.json
Usage: python emp/src/gemma_cast.py
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from story_segment import make_reader  # noqa: E402  (the Gemma reader)

WORLDS = ["Mahabharata", "Star Wars", "Harry Potter", "Thomas & Friends",
          "Steven Universe", "Teenage Mutant Ninja Turtles", "Super Mario"]

PROMPT = ('List the main characters of the fictional world "{world}" - real individual people only, '
          'never places, groups, items, species, or titles. Give the 12 to 18 most central characters, '
          'most important first, and be sure to include the main protagonist(s), the main antagonist or '
          'villain, and the most important elder or mentor. Use the correct canonical English spelling of '
          'each name. Output the names only, one name per line - no numbers, no descriptions, no '
          'commentary, and no duplicate or alternate spellings. List each character exactly once and stop '
          'once the central cast is covered.')


def parse(raw):
    out, seen = [], set()
    for line in raw.splitlines():
        s = re.sub(r"^[\s\-\*.\d)]+", "", line).strip().strip('",')
        if not s or len(s) > 40 or s.lower() in seen:
            continue
        if re.search(r"[^\x00-\x7f]", s) or re.search(r"[:{}\[\]]", s):
            continue
        seen.add(s.lower()); out.append(s)
    return out


if __name__ == "__main__":
    t0 = time.time()
    gen = make_reader()
    print(f"gemma loaded in {time.time()-t0:.0f}s", file=sys.stderr)
    res = {}
    for w in WORLDS:
        raw = gen(PROMPT.format(world=w), max_tokens=500)
        res[w] = parse(raw)
        print(f"[{time.time()-t0:4.0f}s] {w:32} -> {len(res[w])} names: {', '.join(res[w])}", file=sys.stderr)
    out = ROOT / "emp/results/visuals/cast-prompt-sweep/gemma-casts.json"
    out.write_text(json.dumps({"prompt": PROMPT, "casts": res}, ensure_ascii=False, indent=2))
    print("WROTE", out, file=sys.stderr)
