#!/usr/bin/env python3
"""Experimental canon name-audit: the "spell-it" approach.

The diagnostic (diag_canon_spread) found the worksheet reader often KNOWS the right spelling
but routes the name to the wrong verdict bucket ("ok"/"inconsistent") instead of "canon_wrong"
— so the detector throws the knowledge away (Peech->"ok" canon="Peach"). This drops the
verdict scheme entirely: ask the model only for each name's correct canon spelling, and let
US flag the mismatch. Then the SAME production dictionary gate runs, so an ordinary word can't
be flagged.

EXPERIMENT ONLY — not wired into production. Compared against run_v2 by bench_spell.py.
"""
from detectors.story_names._audit import (CARD_CHUNK, render_cards, gate_canon_flags)
from detectors.story_names._names import proper_name_candidates
from story_segment import extract_json
from detectors.phonetics import clean

SPELL_PROMPT = """You are checking the NAMES in one bedtime story set in the world of: {world}.

For EACH numbered name below, decide ONE thing: is it a real, well-known character or place from {world} (or other famous canon — Star Wars, Harry Potter, the Mahabharata, Thomas & Friends, etc.) whose spelling here is WRONG — a mis-hearing of the real name? Judge by SOUND, because the transcriber may have mangled it ("Choobaka" for Chewbacca, "Bishma" for Bhishma, "Peech" for Peach).

- If yes, give the CORRECT canonical spelling.
- If the name is already spelled correctly, OR is a made-up / original name with no real-world source, OR is an ordinary word, give an empty string.

Names:
{cards}

Return JSON only, no other text:
{{"names": [{{"id": <number>, "correct": "<the correct canon spelling, or empty>"}}]}}
"""


def run_spell(gen, world, segs, cards, raw_log=None):
    """Ask for each name's correct canon spelling; flag any card whose spelling(s) differ from
    it; then apply the production dictionary gate. Returns M9c flags in the run_v2 flag shape."""
    raw_log = raw_log if raw_log is not None else []
    flags, by_id = [], {c["id"]: c for c in cards}
    for i in range(0, len(cards), CARD_CHUNK):
        chunk = cards[i:i + CARD_CHUNK]
        raw = gen(SPELL_PROMPT.format(world=world, cards=render_cards(chunk)), max_tokens=900)
        raw_log.append({"pass": "spell", "raw": raw})
        obj = extract_json(raw)
        arr = obj.get("names") if isinstance(obj, dict) else None
        if not isinstance(arr, list):
            continue
        for v in arr:
            if not isinstance(v, dict):
                continue
            try:
                cid = int(str(v.get("id")).strip().strip("[]"))
            except (ValueError, TypeError):
                continue
            card = by_id.get(cid)
            if not card:
                continue
            correct = str(v.get("correct", "")).strip()
            cc = clean(correct)
            if not cc:
                continue                                   # model says: fine / made-up / not a name
            wrong_surface = [s for s in card["surface"] if clean(s) != cc]
            if not wrong_surface:
                continue                                   # already spelled correctly
            flags.append({
                "case": "M9c", "canonical": correct,
                "wrong_surface": sorted(set(wrong_surface)),
                "wrong_cleaned": sorted({clean(s) for s in wrong_surface if clean(s)}),
                "all_spellings": card["surface"], "card_id": card["id"],
                "evidence": card["examples"][0] if card["examples"] else "",
            })
    return gate_canon_flags(flags, proper_name_candidates(segs))
