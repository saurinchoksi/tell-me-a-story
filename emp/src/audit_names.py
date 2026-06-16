#!/usr/bin/env python3
"""Stage-1 per-story name auditor — M9b (improvised inconsistency) + M9c (sourced
canon recognition) in ONE local-LLM pass per story.  EMP.

Three architectures, swept (the choice is made by the data, like the M9a encoder
ladder and the M9b judge prompt ladder — see emp.md):

  worksheet  — per story, every candidate name (clustered by sound) with its
               spellings + example lines + the segmenter's inferred world; the model
               classifies each. Bounded; reuses the M9b-judge cluster+examples shape.
  fulltext   — the story's DELIVERED words in windows; the model surfaces name errors
               as it reads (segmenter pass-1 shape). Sees lowercase/short tokens the
               cap-gated candidate set drops.
  hybrid     — the worksheet card set PLUS a sampled story-context block per call.

All three emit the SAME flag schema, so score_names.py is architecture-agnostic.

FIREWALL: this module reads the transcript + the Stage-0 regions ONLY. It NEVER reads
name-truth.json (that is the answer key; only the scorer reads it).

LOCAL-ONLY: Gemma-4 E4B via MLX under venv-mlx-vlm (the M9b-judge + segmenter model);
family content never leaves the machine. Output carries name variants -> gitignored.

    ./venv-mlx-vlm/bin/python emp/src/audit_names.py                  # all 3 archs, 5 sessions
    ./venv-mlx-vlm/bin/python emp/src/audit_names.py 20260117-202237  # one session
    ./venv-mlx-vlm/bin/python emp/src/audit_names.py --arch worksheet # one architecture
    ./venv/bin/python         emp/src/audit_names.py --show-cards     # no model: inspect worksheets
"""
import argparse
import json
from collections import defaultdict

from audit_common import (SESSIONS, ROOT, clean, load_rich, load_regions,
                          seg_word_text, story_segments)
from name_truth import proper_name_candidates, detect_phrases
from segment import make_reader, extract_json, MODEL_ID
from detectors.phonetics import codes

ARCHS = ("worksheet", "fulltext", "hybrid")
CARD_CHUNK = 16        # worksheet cards per model call (keeps each generation bounded)
FULLTEXT_WIN = 24      # non-empty story lines per fulltext window


# ============================ candidate worksheet =============================
def story_name_cards(segs):
    """Per-story name cards (transcript-only). A card is a phonetic cluster of the
    capitalized name candidates (Double Metaphone union-find) OR a multi-word phrase
    card, each with its distinct spellings, occurrence count, up to 3 example lines,
    and occurrence positions. Phrase members are folded out of the single-word cards
    (the 'names are spans' rule)."""
    singles = proper_name_candidates(segs)
    phrases = detect_phrases(segs, singles)
    phrase_members = {tok for ph in phrases for tok in ph.split(" ")}
    cluster_singles = {c for c in singles if c not in phrase_members}
    seg_sent = {s["id"]: seg_word_text(s) for s in segs}

    # union-find on shared DM codes
    parent = {c: c for c in cluster_singles}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    code_to_forms = defaultdict(list)
    for c in cluster_singles:
        for code in codes(c):
            code_to_forms[code].append(c)
    for forms in code_to_forms.values():
        for o in forms[1:]:
            union(forms[0], o)

    surf, cln, cnt, occ = defaultdict(set), defaultdict(set), defaultdict(int), defaultdict(list)
    ex = defaultdict(list)
    for s in segs:
        sent = seg_sent[s["id"]]
        for wi, w in enumerate(s.get("words", [])):
            raw = w["word"].strip()
            c = clean(raw)
            if c in cluster_singles:
                r = find(c)
                surf[r].add(raw); cln[r].add(c); cnt[r] += 1
                occ[r].append({"seg_id": s["id"], "wi": wi})
                if sent and sent not in ex[r] and len(ex[r]) < 3:
                    ex[r].append(sent)

    cards = []
    for r in cln:
        cards.append({"clean": sorted(cln[r]), "surface": sorted(surf[r]), "n": cnt[r],
                      "examples": ex[r], "is_phrase": False, "occ": occ[r]})
    for ph, occs in phrases.items():
        sents, seen = [], set()
        for o in occs:
            t = seg_sent.get(o["seg_id"], "")
            if t and t not in seen:
                seen.add(t); sents.append(t)
        cards.append({"clean": [ph], "surface": sorted({o["surface"] for o in occs}),
                      "n": len(occs), "examples": sents[:3], "is_phrase": True,
                      "occ": [{"seg_id": o["seg_id"], "wi": o["wi"]} for o in occs]})

    cards.sort(key=lambda c: -c["n"])
    for i, c in enumerate(cards):
        c["id"] = i + 1
    return cards


# ================================== prompts ===================================
WORKSHEET_PROMPT = """You are checking the NAMES in one bedtime story a parent told a young child.
This story's world is: {world}.
{context}
Below are the names found in the story — each with the spelling(s) the transcriber used and example lines. For EACH numbered entry, choose ONE verdict:
- "canon_wrong": a real name from this world (or other well-known canon, e.g. Thomas & Friends, Mahabharata) that is SPELLED WRONG. Give the correct spelling in "canonical".
- "inconsistent": ONE made-up name the transcriber spelled SEVERAL different ways. Give the spelling to standardize on in "canonical".
- "substitution": it may be a made-up name written as a DIFFERENT real word/name (you cannot be sure from text alone).
- "ok": a name spelled correctly — a canon name spelled right, OR a made-up name spelled the same way every time.
- "not_name": an ordinary word, not a name.

Names:
{cards}

Return JSON only, no other text:
{{"verdicts": [{{"id": <number>, "verdict": "canon_wrong|inconsistent|substitution|ok|not_name", "canonical": "<correct spelling or empty>", "wrong_spellings": ["<the misspelled form(s)>"]}}]}}
"""

FULLTEXT_PROMPT = """You are reading part of ONE bedtime story (world: {world}). List any NAME (a person, character, or place) the transcriber SPELLED WRONG:
- a known canon name from this world spelled wrong (case "canon"), or
- a made-up name spelled INCONSISTENTLY across the story (case "madeup").
Ignore ordinary words and correctly-spelled names. If unsure, leave it out.

Lines:
{window}

Return JSON only, no other text:
{{"errors": [{{"wrong": "<spelling as written>", "correct": "<the right spelling>", "case": "canon|madeup"}}]}}
If there are none, return {{"errors": []}}.
"""

_VERDICT_CASE = {"canon_wrong": "M9c", "inconsistent": "M9b", "substitution": "M9d-suspect"}
_FULLTEXT_CASE = {"canon": "M9c", "madeup": "M9b"}


# ================================ renderers ===================================
def render_cards(cards):
    out = []
    for c in cards:
        forms = ", ".join(f'"{s}"' for s in c["surface"])
        kind = " [multi-word]" if c["is_phrase"] else ""
        out.append(f'[{c["id"]}] spellings: {forms}  ({c["n"]} uses){kind}')
        for e in c["examples"][:3]:
            out.append(f'    - "{e[:130]}"')
    return "\n".join(out)


def sample_context(segs, head=6, mid=4, tail=4):
    lines = [(s["id"], seg_word_text(s)) for s in segs if seg_word_text(s)]
    if len(lines) <= head + mid + tail:
        picks = lines
    else:
        mids = [lines[len(lines) * (k + 1) // (mid + 1)] for k in range(mid)]
        picks = lines[:head] + mids + lines[-tail:]
    seen, out = set(), []
    for sid, t in picks:
        if sid in seen:
            continue
        seen.add(sid); out.append(f'[{sid}] "{t[:140]}"')
    return "\n".join(out)


# ============================ per-architecture run ============================
def _flag_from_verdict(card, verdict, canonical, wrong):
    case = _VERDICT_CASE.get(verdict)
    if not case:
        return None
    wrong_surface = [w for w in wrong if w] or [s for s in card["surface"] if clean(s) != clean(canonical)]
    if not wrong_surface:
        wrong_surface = card["surface"]
    return {"case": case, "canonical": canonical,
            "wrong_surface": sorted(set(wrong_surface)),
            "wrong_cleaned": sorted({clean(s) for s in wrong_surface if clean(s)}),
            "all_spellings": card["surface"], "card_id": card["id"],
            "evidence": card["examples"][0] if card["examples"] else ""}


def _run_card_pass(gen, world, cards, raw_log, context=""):
    """Worksheet/hybrid: classify the cards (chunked). Returns (flags, verdicts)."""
    flags, verdicts = [], []
    by_id = {c["id"]: c for c in cards}
    for i in range(0, len(cards), CARD_CHUNK):
        chunk = cards[i:i + CARD_CHUNK]
        prompt = WORKSHEET_PROMPT.format(world=world, context=context,
                                         cards=render_cards(chunk))
        raw = gen(prompt, max_tokens=1100)
        raw_log.append({"pass": "cards", "chunk": [c["id"] for c in chunk], "raw": raw})
        obj = extract_json(raw)
        arr = obj.get("verdicts") if isinstance(obj, dict) else None
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
            verdict = str(v.get("verdict", "")).strip().lower()
            canonical = str(v.get("canonical", "")).strip()
            wrong = [str(x).strip() for x in (v.get("wrong_spellings") or []) if str(x).strip()]
            verdicts.append({"id": cid, "spellings": card["surface"],
                             "cleaned": card["clean"], "verdict": verdict,
                             "canonical": canonical})
            f = _flag_from_verdict(card, verdict, canonical, wrong)
            if f:
                flags.append(f)
    return flags, verdicts


def run_worksheet(gen, world, segs, cards, raw_log):
    return _run_card_pass(gen, world, cards, raw_log, context="")


def run_hybrid(gen, world, segs, cards, raw_log):
    ctx = sample_context(segs)
    block = f"\nFor context, here are sampled lines of the story (in order):\n{ctx}\n"
    return _run_card_pass(gen, world, cards, raw_log, context=block)


def run_fulltext(gen, world, segs, raw_log):
    lines = [(s["id"], seg_word_text(s)) for s in segs if seg_word_text(s)]
    flags = {}
    for i in range(0, len(lines), FULLTEXT_WIN):
        chunk = lines[i:i + FULLTEXT_WIN]
        wtext = "\n".join(f'[{sid}] "{t[:160]}"' for sid, t in chunk)
        raw = gen(FULLTEXT_PROMPT.format(world=world, window=wtext), max_tokens=400)
        raw_log.append({"pass": "fulltext", "window": [c[0] for c in chunk], "raw": raw})
        obj = extract_json(raw)
        errs = obj.get("errors") if isinstance(obj, dict) else None
        if not isinstance(errs, list):
            continue
        for e in errs:
            if not isinstance(e, dict):
                continue
            wrong = str(e.get("wrong", "")).strip()
            cw = clean(wrong)
            if not cw:
                continue
            correct = str(e.get("correct", "")).strip()
            case = _FULLTEXT_CASE.get(str(e.get("case", "")).strip().lower(), "M9c")
            if cw not in flags:
                flags[cw] = {"case": case, "canonical": correct, "wrong_surface": [wrong],
                             "wrong_cleaned": [cw], "all_spellings": [wrong],
                             "card_id": None, "evidence": ""}
    return list(flags.values()), []


_RUNNERS = {"worksheet": run_worksheet, "fulltext": None, "hybrid": run_hybrid}


# ============================== orchestration =================================
def audit_session(sid, gen, archs, show_cards=False):
    rich = load_rich(sid)
    regions, pos_of = load_regions(sid)
    out = {a: {"session": sid, "architecture": a, "model": MODEL_ID, "stories": []} for a in archs}
    raws = {a: [] for a in archs}

    for r in regions:
        segs = story_segments(rich, r, pos_of)
        cards = story_name_cards(segs)
        story_meta = {"idx": r["idx"], "start_id": r["start_id"], "end_id": r["end_id"],
                      "world": r["world"], "title": r["title"]}
        if show_cards:
            print(f"\n--- {SESSIONS[sid]} story {r['idx']} (segs {r['start_id']}-{r['end_id']}) "
                  f"world={r['world']!r} — {len(cards)} cards ---")
            print(render_cards(cards))
            continue
        for a in archs:
            if a == "fulltext":
                flags, verdicts = run_fulltext(gen, r["world"], segs, raws[a])
            else:
                flags, verdicts = _RUNNERS[a](gen, r["world"], segs, cards, raws[a])
            out[a]["stories"].append({"story": story_meta, "n_cards": len(cards),
                                      "flags": flags, "verdicts": verdicts})
            print(f"  {SESSIONS[sid]:11s} story {r['idx']} [{a:9s}]: "
                  f"{len(cards)} cards -> {len(flags)} flag(s) "
                  + ", ".join(f'{f["case"]}:{"/".join(f["wrong_surface"])}' for f in flags))
    return out, raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", help="one session id; default = all five")
    ap.add_argument("--arch", choices=ARCHS, action="append", help="architecture(s); default = all 3")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--show-cards", action="store_true", help="print worksheets, no model")
    args = ap.parse_args()
    sids = [args.session] if args.session else list(SESSIONS)
    archs = args.arch or list(ARCHS)

    if args.show_cards:
        for sid in sids:
            audit_session(sid, gen=None, archs=archs, show_cards=True)
        return

    print(f"loading {args.model} ...  (architectures: {', '.join(archs)})")
    gen = make_reader(args.model)
    print("loaded.\n")
    for sid in sids:
        out, raws = audit_session(sid, gen, archs)
        vis = ROOT / "emp" / "results" / "visuals" / sid
        vis.mkdir(parents=True, exist_ok=True)
        for a in archs:
            (vis / f"name-audit-{a}.json").write_text(json.dumps(out[a], indent=2, ensure_ascii=False))
            (vis / f"name-audit-{a}.raw.json").write_text(json.dumps(raws[a], indent=2, ensure_ascii=False))
    print("\nwrote name-audit-<arch>.json per session (gitignored under emp/results/visuals/<id>/)")


if __name__ == "__main__":
    main()
