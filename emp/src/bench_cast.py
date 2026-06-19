#!/usr/bin/env python3
"""Last experiment: have the MODEL generate the cast list for the recognized world, then
phonetic-match the transcript's names against THAT generated list (no curated roster). Uses the
model for recall (listing a famous world's characters — its strength) and code for the catch
(Double-Metaphone sound-matching — reliable), routing around the name-JUDGING step that failed.

  Part A: real Mahabharata, scored vs the by-ear key — does a generated list match the curated
          mahabharata.json result (3 clear catches)?
  Part B: the 7-world synthetic spread, scored vs planted keys — does it GENERALIZE to Star Wars,
          Steven Universe, etc., where we have NO hand-made list?

One model load (one cast-list call per world; the match is code). Read-only, experimental.

    ./venv/bin/python emp/src/bench_cast.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from story_segment import make_reader, extract_json                  # noqa: E402
from detectors.story_names._audit import story_name_cards            # noqa: E402
from detectors.story_names._names import proper_name_candidates      # noqa: E402
from detectors.phonetics import codes, clean                         # noqa: E402
from phonetic_canon import phonetic_flags                            # noqa: E402
from score_canon_heldout import score_canon, load_items             # noqa: E402
from bench_canon_spread import build_segs, m9c_caught, hit, forms    # noqa: E402

CAST_PROMPT = """List the characters and important places of the fictional world "{world}", using ONLY the common ENGLISH spelling of each name. Give 20 to 25 names — the main characters plus the secondary ones you are confident about.

Strict format rules:
- one plain name per entry, exactly as it is normally written in English;
- NO non-English script, NO translations, NO parentheses, NO notes, titles, or descriptions;
- do not invent names to pad the list — only names you are sure are real.

Return JSON only, no other text:
{{"names": ["Name", "Name", ...]}}
"""


def generate_cast(gen, world):
    if not world:
        return []
    obj = extract_json(gen(CAST_PROMPT.format(world=world), max_tokens=900))
    names = obj.get("names") if isinstance(obj, dict) else None
    return [str(n).strip() for n in names if str(n).strip()] if isinstance(names, list) else []


def cast_index(names):
    forms_set = {clean(n) for n in names if clean(n)}
    code_to_name = {}
    for n in names:
        for cd in codes(n):
            code_to_name.setdefault(cd, n)
    return forms_set, code_to_name


def planted_forms(planted):
    s = set()
    for p in planted:
        s |= {clean(t) for t in p["heard"].split() if clean(t)}
        s.add(clean(p["heard"]))
    return s


def main():
    gen = make_reader()

    print("=" * 74)
    print("PART A — real Mahabharata (vs by-ear key), model-generated cast")
    print("=" * 74)
    SID = "20260211-210718"
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    segs = rich["segments"]
    cards = story_name_cards(segs, recover=True)
    singles = proper_name_candidates(segs)
    cast = generate_cast(gen, "Mahabharata")
    print(f"  model cast ({len(cast)}): {cast[:28]}")
    fset, c2n = cast_index(cast)
    r = score_canon(load_items(SID) or {}, phonetic_flags(cards, singles, fset, c2n))
    real, n = r["matrix"]["M9c"]["flagged"], r["flagged"]
    print(f"  recall {r['caught']}/{r['gold_m9c']}   precision {real}/{n}   caught {r['hits']}")
    print(f"  reference points: curated-list 3, LLM worksheet 1, LLM spell-it 0", flush=True)

    print("\n" + "=" * 74)
    print("PART B — synthetic spread (generalization to worlds with no curated list)")
    print("=" * 74)
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
    print("-" * 74)
    print(f"  SPREAD recall {caught}/{tot} = {caught / tot:.2f}   false flags {fp}")
    print(f"  reference points: spell-it 23/40 = 0.57, worksheet 8/40 = 0.20")


if __name__ == "__main__":
    main()
