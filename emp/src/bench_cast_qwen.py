#!/usr/bin/env python3
"""Same cast+phonetic experiment as bench_cast.py, but generate the cast list with a DIFFERENT
small model — Qwen3-4B-Instruct-2507 (4-bit, ~2.3GB, runs <5GB via mlx_lm) instead of Gemma-4
E4B. Tests whether a newer, more knowledgeable small model produces better cast lists (does it
remember Karna? does it know more of each world?) — everything else (prompt, phonetic match,
gate, scoring) is identical, so any change is the model.

    ./venv/bin/python emp/src/bench_cast_qwen.py [model_id]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

import re                                                             # noqa: E402
import mlx_lm                                                          # noqa: E402
from detectors.story_names._audit import story_name_cards            # noqa: E402
from detectors.story_names._names import proper_name_candidates      # noqa: E402
from phonetic_canon import phonetic_flags                            # noqa: E402
from score_canon_heldout import score_canon, load_items             # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit, forms    # noqa: E402
from bench_cast import CAST_PROMPT, cast_index, planted_forms        # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def generate_cast(gen, world):
    """Tolerant cast parse: small models can loop and never close the JSON array (Qwen did at
    temp 0). Pull every quoted name out of the `names` list and dedupe, order-preserving — so a
    repetition loop or an unterminated array still yields the full, clean cast."""
    if not world:
        return []
    raw = gen(CAST_PROMPT.format(world=world), max_tokens=1100)
    m = re.search(r'"names"\s*:\s*\[', raw)
    body = raw[m.end():] if m else raw
    names, seen = [], set()
    for s in re.findall(r'"([^"]{2,40})"', body):
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            names.append(s)
    return names


def make_qwen_reader(model_id):
    print(f"loading {model_id} via mlx_lm (downloads on first run)...", flush=True)
    model_obj, tokenizer = mlx_lm.load(model_id)

    def gen(prompt_text, max_tokens=900):
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return mlx_lm.generate(model_obj, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
    return gen


def main():
    gen = make_qwen_reader(MODEL)
    print(f"\n=== PART A — real Mahabharata (vs by-ear key), cast by {MODEL.split('/')[-1]} ===")
    SID = "20260211-210718"
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    segs = rich["segments"]
    cards = story_name_cards(segs, recover=True)
    singles = proper_name_candidates(segs)
    cast = generate_cast(gen, "Mahabharata")
    print(f"  cast ({len(cast)}): {cast[:30]}")
    print(f"  Karna in cast? {'karna' in {c.lower() for c in cast}}")
    r = score_canon(load_items(SID) or {}, phonetic_flags(cards, singles, *cast_index(cast)))
    real, n = r["matrix"]["M9c"]["flagged"], r["flagged"]
    print(f"  recall {r['caught']}/{r['gold_m9c']}   precision {real}/{n}   caught {r['hits']}")
    print(f"  (Gemma cast got 4/11; LLM worksheet 1; curated list 3)", flush=True)

    print(f"\n=== PART B — synthetic spread (generalization), cast by {MODEL.split('/')[-1]} ===")
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    tot = caught = fp = 0
    for st in spread:
        segs = build_segs(st["lines"])
        cards = story_name_cards(segs, recover=True)
        singles = proper_name_candidates(segs)
        cast = generate_cast(gen, st["world"])
        flags = phonetic_flags(cards, singles, *cast_index(cast))
        cg = m9c_caught(flags)
        rc = sum(1 for p in st["planted"] if hit(p["heard"], cg))
        pf = planted_forms(st["planted"])
        fpc = sum(1 for f in flags if f.get("case") == "M9c" and not (forms(f.get("wrong_cleaned", [])) & pf))
        tot += len(st["planted"]); caught += rc; fp += fpc
        print(f"  {st['world'][:34]:36} {rc}/{len(st['planted'])}   (cast {len(cast)}, false {fpc})", flush=True)
    print(f"  SPREAD recall {caught}/{tot} = {caught / tot:.2f}   false flags {fp}")
    print(f"  (Gemma cast spread: 20/40 = 0.50)")


if __name__ == "__main__":
    main()
