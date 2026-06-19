#!/usr/bin/env python3
"""Qwen3.5 canon-name layer (M9c) — the production port of the validated EMP design.

The winning recipe (emp.md, 2026-06-18): with the story's world recognized, run TWO
independent catches and UNION their outputs —
  - judge_names: feed Qwen3.5 the transcript's distinct name spellings in PLAIN TEXT and
    let it reason about which are misspelled canon and the correct spelling. This is the
    step that scored 8/11 on the held-out real Mahabharata (effectively 8/8 genuine
    errors), catching even the degraded Urzi->Arjuna — once Qwen3.5 drives it (it failed
    on Gemma at ~1/11).
  - phonetic_flags: sound-match the transcript names (Double Metaphone) against a
    Qwen3.5-GENERATED cast of the world's characters (no curated roster).
The two fail differently, so `combine` (union by wrong-spelling) lifts the synthetic
spread 0.72->0.85 with zero extra false flags. The WRONG way — feeding the cast into the
judge's prompt as context — over-eagered it (precision halved); union the OUTPUTS instead.

Every flag passes the recall-safe dictionary gate (`_audit.gate_canon_flags`) so an
ordinary English word can't be flagged as a misspelled canon name.

PLAIN TEXT across the model boundary, never JSON (memory plain-text-not-json). The judge
and cast prompts here byte-match the validated emp/src experiments (`qwen35_judge.py`,
`gen_cast_qwen35.py`) — small wording changes move a 4B model's behaviour a lot, so do
not "lightly adapt" them without re-scoring. Occurrence expansion lives in `_worker`.
"""
import re

from detectors.phonetics import clean, codes
from detectors.story_names._audit import gate_canon_flags

MIN_LEN = 4  # candidate floor (shared with the EMP phonetic matcher)

# --- cast generation (plain lines, no JSON) — byte-matches gen_cast_qwen35.cast_for ----
CAST_PROMPT = ('List the 25 best-known characters, families/groups, and important places of the '
               'fictional world "{world}". Output names only, one per line, English spelling, '
               'nothing else — no numbers, no JSON, no commentary.')

# --- name judge (plain "wrong -> correct" lines) — byte-matches qwen35_judge.JUDGE -----
JUDGE_PROMPT = """This bedtime story is set in the world of "{world}". Below is a list of names exactly as a transcriber wrote them down by ear — some may be misspelled.

For each name that is really a character or place from {world} (or other very famous canon) but spelled WRONG, write ONE line:
<name as written> -> <correct spelling>

Skip any name that is already spelled correctly, is made up, or is not a real name. Output only the correction lines, nothing else.

Names:
{names}"""

# wrong -> correct, tolerant of arrow length and surrounding text on the line
_JUDGE_LINE = re.compile(r"^\s*([A-Za-z][\w '.-]{1,30}?)\s*-+>\s*([A-Za-z][\w '.-]{1,30})\s*$", re.M)

# --- world recognition from the NAME LIST (not the rambling transcript) ----------------
# Production found pass-2 world naming over a long whole-recording region abstains to empty,
# so the canon check never starts (held-out Mahabharata: world=''). The names ARE the signal
# — the judge clearly knows them — so recognize the world from the distinct name list. The
# `soft` wording (emp/src/tune_recognize_world.py) recognized Mahabharata 3/3 AND correctly
# abstained on the KPop session's garbled/generic names (vs a variant that hallucinated Oz).
RECOGNIZE_PROMPT = (
    "These names were heard in ONE children's bedtime story, written down by ear — some may be "
    "misspelled:\n\n{names}\n\n"
    "Do these characters and places come from a REAL, widely-known story world (a book, film, "
    "show, myth, or franchise) — even one the story never names aloud? If you can place them, "
    'name that world. Write "original" ONLY when you cannot place the names in any known world.'
    '\n\nWrite one line, nothing else:\nWorld: <the world, or "original" if made up>')

_NON_WORLDS = {"", "original", "made up", "made-up", "none", "unknown", "n/a"}


def recognize_world(gen, names):
    """The world the story's names come from, or "" when not placeable (canon check stays off).
    Reads the distinct name list, NOT the transcript — the reliable signal on long recordings."""
    if not names:
        return ""
    raw = gen(RECOGNIZE_PROMPT.format(names="\n".join(names)), max_tokens=120)
    world = ""
    for line in (raw or "").splitlines():
        m = re.match(r"\s*world\s*[:\-]\s*(.+)", line, re.I)
        if m:
            world = m.group(1).strip().strip('".')
    return "" if world.lower() in _NON_WORLDS else world


def generate_cast(gen, world):
    """Plain-line cast for `world` (empty world -> []). Dedupe handles the temp-0 list loop;
    the filters drop numbering, non-ASCII, and JSON-ish lines."""
    if not world:
        return []
    raw = gen(CAST_PROMPT.format(world=world), max_tokens=500)
    out, seen = [], set()
    for line in raw.splitlines():
        s = re.sub(r"^[\s\-\*•\d.)]+", "", line).strip().strip('",')
        if not s or len(s) > 40 or s.lower() in seen:
            continue
        if re.search(r"[^\x00-\x7f]", s) or re.search(r"[:{}\[\]]", s):
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def judge_names(gen, world, names, singles):
    """Per-(story) M9c flags from the plain-text judge, dictionary-gated. `names` is the
    story's distinct surface spellings. Empty world or no names -> []."""
    if not world or not names:
        return []
    raw = gen(JUDGE_PROMPT.format(world=world, names="\n".join(names)), max_tokens=500)
    flags, seen = [], set()
    for m in _JUDGE_LINE.finditer(raw or ""):
        wrong, correct = m.group(1).strip(), m.group(2).strip()
        wc = clean(wrong)
        if not wc or clean(correct) == wc or wc in seen:
            continue
        seen.add(wc)
        flags.append({"case": "M9c", "canonical": correct, "wrong_surface": [wrong],
                      "wrong_cleaned": [wc], "all_spellings": [wrong], "card_id": -1, "evidence": ""})
    return gate_canon_flags(flags, singles)


def cast_index(names):
    """From a cast list: the set of correctly-spelled forms, and a map from each cast
    Double-Metaphone code to a canonical spelling (port of bench_cast.cast_index)."""
    forms = {clean(n) for n in names if clean(n)}
    code_to_name = {}
    for n in names:
        for cd in codes(n):
            code_to_name.setdefault(cd, n)
    return forms, code_to_name


def phonetic_flags(cards, singles, cast):
    """Flag a card spelling that SOUNDS like a cast name (exact Double-Metaphone) but is not
    itself a correct cast form, then dictionary-gate (port of phonetic_canon.phonetic_flags,
    code_ed=0). Empty cast -> []."""
    if not cast:
        return []
    canon_forms, code_to_name = cast_index(cast)
    flags = []
    for card in cards:
        # Phrase (multi-word) cards are handled by the judge. Splitting one here would flag a
        # single word whose per-occurrence flag expand_flag can't reconstruct (it rebuilds the
        # whole bigram and cleans it, so a single-word wrong_cleaned never matches).
        if card.get("is_phrase"):
            continue
        for cl in card["clean"]:
            for tok in cl.split(" "):
                if len(tok) < MIN_LEN or tok in canon_forms:
                    continue
                matched = next((code_to_name[c] for c in codes(tok) if c in code_to_name), None)
                if not matched:
                    continue
                flags.append({
                    "case": "M9c", "canonical": matched,
                    "wrong_surface": [s for s in card["surface"] if clean(s) == tok] or [tok],
                    "wrong_cleaned": [tok], "all_spellings": card["surface"],
                    "card_id": card["id"], "evidence": card["examples"][0] if card["examples"] else "",
                })
    return gate_canon_flags(flags, singles)


def combine(*flag_lists):
    """Union independent catches by their wrong-spelling key (port of combine_judge.combine).
    The judge and phonetic methods fail differently; unioning their OUTPUTS covers more."""
    out, seen = [], set()
    for fl in flag_lists:
        for f in fl:
            key = tuple(sorted(f["wrong_cleaned"]))  # fail loud: every flag carries wrong_cleaned
            if key and key not in seen:
                seen.add(key)
                out.append(f)
    return out
