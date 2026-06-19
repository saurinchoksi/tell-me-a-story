#!/usr/bin/env python3
"""Generate + cache a cast list per world from one model, with a completeness-oriented prompt
and a robust parser (strips non-English script and prompt-echo fragments — the junk that broke
Qwen3.5). Cached to emp/results/canon-spread/casts/<tag>.json so scoring/union can be done
later with no GPU.

    ./venv/bin/python emp/src/gen_casts.py <model_id> <gemma|qwen> <tag>
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

MODEL, LOADER, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
# 4th arg "nothink" prepends Qwen's /no_think directive — thinking models (Qwen3-8B, Qwen3.5)
# otherwise spend the whole budget on a hidden reasoning block before the JSON.
NOTHINK = len(sys.argv) > 4 and sys.argv[4] == "nothink"

# Completeness prompt: explicitly ask for GROUPS/collectives and PLACES, not just heroes —
# the "Pandavas" the earlier prompt dropped.
PROMPT = """List the named characters, FAMILIES or GROUPS (for example "the Pandavas", "the Kauravas", "the X-Men", "the Crystal Gems"), and important PLACES of the fictional world "{world}". Use ONLY the common ENGLISH spelling of each.

Rules:
- one plain name per entry, exactly as written in English;
- INCLUDE collectives/groups and place names, not just individual heroes;
- NO non-English script, NO parentheses, NO notes or descriptions;
- list only names you are sure are real — do not invent names to pad.

Return JSON only, no other text:
{{"names": ["Name", "Name", ...]}}
"""

STOP = re.compile(r'\b(names?|characters?|places?|important|world|list|spelling|english|entry|group|family|families|collective)\b', re.I)


def parse_cast(raw):
    """Tolerant: pull names from the (possibly looping/unterminated) `names` array; drop
    non-ASCII (Devanagari) and prompt-echo fragments; dedupe order-preserving."""
    m = re.search(r'"names"\s*:\s*\[', raw)
    body = raw[m.end():] if m else raw
    names, seen = [], set()
    for s in re.findall(r'"([^"]{2,40})"', body):
        s = s.strip().lstrip("the ").strip()
        if not s or s.lower() in seen:
            continue
        if re.search(r'[^\x00-\x7f]', s) or STOP.search(s):
            continue
        seen.add(s.lower())
        names.append(s)
    return names


def main():
    if LOADER == "gemma":
        from story_segment import make_reader
        gen = make_reader()
    else:
        from bench_cast_qwen import make_qwen_reader
        gen = make_qwen_reader(MODEL)

    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    worlds = list(dict.fromkeys(["Mahabharata"] + [s["world"] for s in spread]))
    casts = {}
    for w in worlds:
        prompt = ("/no_think\n" if NOTHINK else "") + PROMPT.format(world=w)
        raw = gen(prompt, max_tokens=1000)
        casts[w] = parse_cast(raw)
        has_p = "pandavas" in {n.lower() for n in casts[w]}
        print(f"  {w[:34]:36} {len(casts[w]):>2} names" + ("   [has Pandavas]" if has_p else ""), flush=True)

    out = ROOT / "emp" / "results" / "canon-spread" / "casts" / f"{TAG}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "casts": casts}, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
