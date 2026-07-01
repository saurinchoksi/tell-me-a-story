#!/usr/bin/env python3
"""World-grounded name normalizer — correct mis-heard names, but only within the story's world.

The old normalizer (``normalize.py``, removed 2026-07-01) was WORLD-BLIND: it handed the whole
transcript to an LLM with a generic "fix proper nouns" prompt, so the model free-associated a
"correct" spelling from its training priors and confidently rewrote the child's word for the
Pandavas (heard as ``Fondos``/``Bondos``) into ``Bhishma`` — a character on the enemy's side.

This rebuild reuses the validated M9c detection brain (``detectors/story_names/_qwen35.py`` +
``_worker.expand_combined``) but points it at CORRECTING: per story region it recognizes the
world FROM THE NAME LIST (the segmenter often abstains on a long recording, so we re-recognize),
generates the world's cast, runs the order-robust judge and the phonetic sound-matcher, unions
them, and expands to per-occurrence catches — then CLASSIFIES each into a hybrid safety tier:

    classify precedence (see ``classify``):
      1. dictionary hit    -> AUTO  (a human blessed this variant before; deterministic, wins)
      2. sound-alike        -> AUTO  (heard form shares a Double-Metaphone code with the suggestion)
      3. otherwise          -> QUEUE (judge's best guess, not sound-alike -> a human blesses it)
    unrecognized world      -> nothing (never guess into the transcript)

Only sound-alikes and blessed variants touch the words; everything else stays honest and waits
for a human. This is deliberately conservative: on the held-out Mahabharata the child's
``Fondos`` -> Pandavas is NOT a sound-alike, so it is queued (kept honest), not auto-substituted.

Single-model on Qwen3.5-4B via ``qwen35.make_reader`` (see memory eight-gb-single-model-constraint).
``run_worldnorm`` is pure with an injectable ``gen`` (unit-testable with a fake reader); the
subprocess wrapper ``world_normalize`` loads the model in a fresh process (frees Metal on exit).
"""
import re
from collections import Counter

from detectors.phonetics import clean
from detectors.story_names import _qwen35
from detectors.story_names._audit import story_name_cards
from detectors.story_names._names import proper_name_candidates, story_segments
from detectors.story_names._worker import build_regions, expand_combined

_EDGE_PUNCT = re.compile(r"^\W+|\W+$")


def _display_heard(occurrences) -> str:
    """A readable "as heard" spelling for a group: the most common surface token with edge
    punctuation trimmed (keeps internal apostrophes). Ties broken by first-seen order."""
    counts = Counter(_EDGE_PUNCT.sub("", o["token"].strip()) for o in occurrences)
    return counts.most_common(1)[0][0] if counts else ""


def _group_catches(expanded):
    """Fold per-occurrence M9c catches into per-(story, heard-spelling) groups. Each group is a
    correction decision unit — one heard spelling in one story maps to one suggested canonical."""
    groups: dict[tuple, dict] = {}
    for rec in expanded:
        key = (rec["story_id"], rec["cleaned"])
        g = groups.get(key)
        if g is None:
            g = {
                "world": rec["story_world"],
                "story_id": rec["story_id"],
                "heard_cleaned": rec["cleaned"],
                "suggestion": rec["canonical"],
                "suggestion_confident": rec["suggestion_confident"],
                "methods": rec["methods"],
                "vote_count": rec.get("vote_count"),
                "vote_rounds": rec.get("vote_rounds"),
                "evidence": rec.get("evidence", ""),
                "occurrences": [],
            }
            groups[key] = g
        g["occurrences"].append({
            "segment_id": rec["segment_id"], "word_index": rec["word_index"],
            "start": rec["start"], "end": rec["end"], "token": rec["token"],
        })
    for g in groups.values():
        g["heard"] = _display_heard(g["occurrences"])
    return list(groups.values())


def classify(group: dict, variant_map: dict) -> dict:
    """Decide a group's fate under the hybrid safety policy. ``variant_map`` maps a CLEANED heard
    form -> blessed canonical (already cleaned by ``run_worldnorm``). Returns the group augmented
    with ``action`` ("auto"|"pending"), ``canonical`` (the resolved spelling), and ``method``.

    Dictionary precedence is the point of the bless loop: once a human confirms "Bondos means
    Pandavas", that entry auto-applies and OVERRIDES a phonetic collision (Bushma->Bhishma)."""
    heard = group["heard_cleaned"]
    if heard in variant_map:
        decided = {"action": "auto", "canonical": variant_map[heard], "method": "dictionary"}
    elif group["suggestion_confident"]:
        decided = {"action": "auto", "canonical": group["suggestion"], "method": "sound-alike"}
    else:
        decided = {"action": "pending", "canonical": group["suggestion"], "method": "judge"}
    return {**group, **decided}


def _variant_map_for(world: str, world_dicts) -> dict:
    """Cleaned-key variant map for a world. ``world_dicts`` None -> load from disk (production);
    a dict -> use it as the authoritative source (tests/measurement inject the cold empty case).
    Keys are cleaned so they share ``heard_cleaned``'s space; last write wins on a clean collision."""
    if world_dicts is None:
        import worlddict  # local import: keeps run_worldnorm's core pure/disk-free when injected
        raw = worlddict.load_variant_map(world)
    else:
        raw = world_dicts.get(world, {})
    return {clean(k): v for k, v in raw.items() if clean(k)}


def run_worldnorm(gen, transcript: dict, world_dicts=None) -> dict:
    """Per-story world-grounded name correction. Returns ``{auto, pending, worlds}``:

      - ``auto``    — groups to auto-apply (feed ``auto_to_corrections`` -> ``apply_corrections``)
      - ``pending`` — groups queued for human bless (best guesses; kept out of the transcript)
      - ``worlds``  — per-region recognition record (saved vs recognized world) for measurement

    ``gen`` is a ``qwen35.make_reader`` closure (injectable). ``world_dicts`` maps world -> its
    variant map; None loads per-recognized-world from disk. Requires ``_stories`` on the
    transcript (the pipeline segments before this pass — see the Phase-2 reorder)."""
    segments = transcript["segments"]
    seg_by_id = {s["id"]: s for s in segments}
    pos_of = {s["id"]: i for i, s in enumerate(segments)}
    regions = build_regions(transcript["_stories"], pos_of)

    auto, pending, worlds = [], [], []
    for r in regions:
        segs = story_segments(transcript, r, pos_of)
        cards = story_name_cards(segs, recover=True)
        singles = proper_name_candidates(segs)
        names = sorted({s for c in cards for s in c["surface"]})

        # Re-recognize the world from the NAME LIST, not the saved _stories world: pass-2
        # segmentation over a long whole-recording region abstains (this session: world='').
        world = _qwen35.recognize_world(gen, names)
        worlds.append({"story_id": r["idx"], "title": r.get("title", ""),
                       "saved_world": r.get("world", ""), "recognized_world": world,
                       "n_names": len(names)})
        if not world:
            continue  # unrecognized world -> no correction (never guess into the transcript)

        cast = _qwen35.generate_cast(gen, world)
        judge_flags = _qwen35.judge_names_voted(gen, world, names, singles)
        phon_flags = _qwen35.phonetic_flags(cards, singles, cast)
        combined = _qwen35.combine(judge_flags, phon_flags)
        expanded = expand_combined(combined, cards, seg_by_id, r["idx"], world)

        variant_map = _variant_map_for(world, world_dicts)
        for group in _group_catches(expanded):
            decided = classify(group, variant_map)
            (auto if decided["action"] == "auto" else pending).append(decided)

    return {"auto": auto, "pending": pending, "worlds": worlds}


def auto_to_corrections(auto) -> list[dict]:
    """Turn auto-apply groups into ``apply_corrections`` input ({transcribed, correct}), keyed by
    the cleaned heard form (which ``apply_corrections`` matches case-insensitively, punctuation and
    possessive tolerant). Deduped by heard form — first canonical wins on any collision.

    NOTE (Phase 2): a blessed variant that the judge/phonetic pass does not re-flag this run won't
    appear here — auto only covers THIS run's catches. Making the dictionary apply to every
    occurrence regardless of the judge needs a standalone dictionary sweep over all candidates;
    it's a no-op in Phase 1 (dicts start empty) and is added when the bless loop fills them."""
    seen: dict[str, dict] = {}
    for g in auto:
        t = g["heard_cleaned"]
        if t and t not in seen:
            seen[t] = {"transcribed": t, "correct": g["canonical"]}
    return list(seen.values())


# --------------------------- subprocess entry (production) ---------------------------
def _worldnorm_in_subprocess(transcript, world_dicts=None):
    """Module-level (picklable) worker: load Qwen3.5 once, run the normalizer, return the result.
    Runs in a fresh process via run_model so the Metal allocation is freed on exit."""
    from qwen35 import make_reader
    gen = make_reader()
    return run_worldnorm(gen, transcript, world_dicts)


def world_normalize(transcript: dict, world_dicts=None, timeout: int = 1800) -> dict:
    """Run ``run_worldnorm`` in a fresh subprocess (loads the model, frees Metal on exit).
    Returns ``{auto, pending, worlds}``. Raises TimeoutError past ``timeout`` seconds."""
    from model_runner import run_model
    return run_model(_worldnorm_in_subprocess, transcript, world_dicts, timeout=timeout)
