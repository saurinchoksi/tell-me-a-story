#!/usr/bin/env python3
"""Phonetic canon matching — the no-model alternative to the LLM name-judge.

The why-check proved the model returns "no correction" for real garbled canon names (it sees
the correct spelling also present in the card and calls the name fine). So drop the model from
the CATCH entirely: take the names the transcript contains and match them, by SOUND, against a
canon name LIST — exactly how the family-name detector (M9a) hits ~0.99 recall. The reference
list here is data/mahabharata.json (a world the family actually recurs on; we already have it).

Scored against the by-ear key with the real scorer. World recognition is assumed (this is the
Mahabharata session); the approach only applies to worlds we have a list for — its honest limit.

    ./venv/bin/python emp/src/phonetic_canon.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "emp" / "src"))

from detectors.story_names._audit import story_name_cards, gate_canon_flags  # noqa: E402
from detectors.story_names._names import proper_name_candidates              # noqa: E402
from detectors.phonetics import codes, clean                                 # noqa: E402
from score_canon_heldout import score_canon, load_items                     # noqa: E402

SID = "20260211-210718"
MIN_LEN = 4


def canon_index(ref_path):
    """From the reference: the set of correctly-spelled canon forms, and a map from each canon
    Double-Metaphone code to a canonical spelling (so a garbled token that SOUNDS like a canon
    name resolves to which name it is)."""
    ref = json.loads(Path(ref_path).read_text())
    forms, code_to_name = set(), {}
    for e in ref["entries"]:
        spellings = [e["canonical"]] + e.get("variants", []) + e.get("aliases", [])
        for f in spellings:
            cf = clean(f)
            if cf:
                forms.add(cf)
        for f in [e["canonical"]] + e.get("variants", []):   # codes from real spellings only
            for cd in codes(f):
                code_to_name.setdefault(cd, e["canonical"])
    return forms, code_to_name


def phonetic_flags(cards, singles, canon_forms, code_to_name):
    """Flag a card's spelling when it SOUNDS like a canon name but is NOT itself a correct canon
    form — then run the same production dictionary gate so an ordinary word can't be flagged."""
    flags = []
    for card in cards:
        for cl in card["clean"]:
            for tok in cl.split(" "):
                if len(tok) < MIN_LEN or tok in canon_forms:
                    continue                              # too short, or already a correct canon spelling
                matched = next((code_to_name[c] for c in codes(tok) if c in code_to_name), None)
                if not matched:
                    continue                              # doesn't sound like any canon name
                flags.append({
                    "case": "M9c", "canonical": matched,
                    "wrong_surface": [s for s in card["surface"] if clean(s) == tok] or [tok],
                    "wrong_cleaned": [tok],
                    "all_spellings": card["surface"], "card_id": card["id"],
                    "evidence": card["examples"][0] if card["examples"] else "",
                })
    return gate_canon_flags(flags, singles)


def main():
    rich = json.loads((ROOT / "sessions" / SID / "transcript-rich.json").read_text())
    segs = rich["segments"]
    cards = story_name_cards(segs, recover=True)
    singles = proper_name_candidates(segs)
    canon_forms, code_to_name = canon_index(ROOT / "data" / "mahabharata.json")
    flags = phonetic_flags(cards, singles, canon_forms, code_to_name)

    items = load_items(SID) or {}
    r = score_canon(items, flags)
    print(f"{SID}  — phonetic match vs data/mahabharata.json  (no model in the catch)")
    print(f"  recall    {r['caught']}/{r['gold_m9c']}")
    real, n = r["matrix"]["M9c"]["flagged"], r["flagged"]
    print(f"  precision {real}/{n}" + (f" = {real / n:.2f}" if n else " (no flags)"))
    if r["hits"]:
        print(f"  caught   : {r['hits']}")
    if r["misses"]:
        print(f"  missed   : {r['misses']}")
    for name, g in r["false_pos"]:
        print(f"  FALSE FLAG: {name!r} (gold {g})")
    print("\n  for reference on this session: LLM worksheet 1/11, LLM spell-it 0/11")


if __name__ == "__main__":
    main()
