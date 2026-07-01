"""Unit tests for the world-grounded normalizer (worldnorm.py) + per-world dicts (worlddict.py).

Pure/fast: a FAKE `gen` returns canned Qwen output dispatched by prompt, so no model loads.
Covers the hybrid safety policy (dictionary > sound-alike > queue), the auto/pending split
end-to-end, auto_to_corrections round-tripping through the real applier, and the bless loop.
"""
import pytest

import worldnorm
import worlddict
from corrections import apply_corrections
from api.helpers import canon_tier


# --------------------------- fake Qwen reader ---------------------------
def make_fake_gen(world="Mahabharata",
                  cast=("Bhishma", "Pandavas", "Arjuna", "Krishna"),
                  judge_lines=("Bishma -> Bhishma", "Bondos -> Pandavas")):
    """A gen(prompt, max_tokens) closure that answers the three prompt kinds by substring.
    Judge is deterministic across all 7 vote rounds, so every catch gets vote_count == rounds."""
    def gen(prompt, max_tokens=256):
        if "REAL, widely-known" in prompt:            # recognize_world
            return "original" if not world else f"World: {world}"
        if "List the 25 best-known" in prompt:        # generate_cast
            return "\n".join(cast) + "\n"
        if "spelled WRONG" in prompt:                 # judge
            return "\n".join(judge_lines) + "\n"
        return ""
    return gen


def make_transcript():
    """Tiny one-story transcript. Bishma/Bushma cluster (both DM code PXM) sound like the cast's
    Bhishma; Bondos (PNTS) does not sound like Pandavas (PNTFS) -> judge-only, must queue."""
    def w(word, start):
        return {"word": word, "start": start, "end": start + 0.3}
    segments = [
        {"id": 0, "words": [w("Tell", 0.0), w("me", 0.4), w("about", 0.8),
                            w("Bishma", 1.2), w("today.", 1.6)]},
        {"id": 1, "words": [w("The", 2.0), w("Bondos", 2.4), w("were", 2.8), w("good.", 3.2)]},
        {"id": 2, "words": [w("And", 4.0), w("Bushma", 4.4), w("fought.", 4.8)]},
    ]
    return {"segments": segments,
            "_stories": [{"start_id": 0, "end_id": 2, "world": "", "title": "Test Story"}]}


# ------------------------------ classify ------------------------------
def _group(heard_cleaned, suggestion, confident, vote_count=7):
    return {"world": "Mahabharata", "story_id": 0, "heard": heard_cleaned,
            "heard_cleaned": heard_cleaned, "suggestion": suggestion,
            "suggestion_confident": confident, "methods": ["judge"],
            "vote_count": vote_count, "vote_rounds": 7, "evidence": "", "occurrences": []}


def test_classify_sound_alike_auto_applies():
    d = worldnorm.classify(_group("bishma", "Bhishma", True), {})
    assert d["action"] == "auto" and d["method"] == "sound-alike" and d["canonical"] == "Bhishma"


def test_classify_non_sound_alike_queues():
    d = worldnorm.classify(_group("bondos", "Pandavas", False), {})
    assert d["action"] == "pending" and d["method"] == "judge" and d["canonical"] == "Pandavas"


def test_classify_dictionary_hit_auto_applies():
    d = worldnorm.classify(_group("bondos", "Pandavas", False), {"bondos": "Pandavas"})
    assert d["action"] == "auto" and d["method"] == "dictionary" and d["canonical"] == "Pandavas"


def test_classify_dictionary_overrides_sound_alike_collision():
    # The Bushma->Bhishma trap: sound alone would auto-apply Bhishma, but a human blessed
    # bushma->Pandavas. Dictionary precedence must win.
    d = worldnorm.classify(_group("bushma", "Bhishma", True), {"bushma": "Pandavas"})
    assert d["action"] == "auto" and d["method"] == "dictionary" and d["canonical"] == "Pandavas"


def test_classify_agrees_with_canon_tier():
    # auto (sound-alike) <=> canon_tier 'confident'; pending (judge, >=3 votes) <=> 'best_guess'.
    sound = _group("bishma", "Bhishma", True)
    guess = _group("bondos", "Pandavas", False, vote_count=7)
    assert worldnorm.classify(sound, {})["action"] == "auto"
    assert canon_tier(sound) == "confident"
    assert worldnorm.classify(guess, {})["action"] == "pending"
    assert canon_tier(guess) == "best_guess"


# --------------------------- run_worldnorm (end-to-end, fake gen) ---------------------------
def test_run_worldnorm_splits_auto_and_pending():
    res = worldnorm.run_worldnorm(make_fake_gen(), make_transcript(), world_dicts={})
    autos = {g["heard_cleaned"]: g for g in res["auto"]}
    pends = {g["heard_cleaned"]: g for g in res["pending"]}

    # Recognized the world from the name list even though _stories saved world=''.
    assert res["worlds"][0]["recognized_world"] == "Mahabharata"
    assert res["worlds"][0]["saved_world"] == ""

    # Sound-alikes auto-apply; the non-sound-alike hero word queues (kept honest).
    assert autos["bishma"]["canonical"] == "Bhishma" and autos["bishma"]["method"] == "sound-alike"
    assert autos["bushma"]["canonical"] == "Bhishma"      # empty dict -> sound-alike wins
    assert "bondos" in pends and pends["bondos"]["canonical"] == "Pandavas"
    assert "bondos" not in autos                            # Bondos is NOT auto-substituted


def test_run_worldnorm_dictionary_overrides_at_runtime():
    res = worldnorm.run_worldnorm(make_fake_gen(), make_transcript(),
                                  world_dicts={"Mahabharata": {"Bushma": "Pandavas"}})
    autos = {g["heard_cleaned"]: g for g in res["auto"]}
    assert autos["bushma"]["canonical"] == "Pandavas" and autos["bushma"]["method"] == "dictionary"


def test_run_worldnorm_unrecognized_world_does_nothing():
    res = worldnorm.run_worldnorm(make_fake_gen(world=""), make_transcript(), world_dicts={})
    assert res["auto"] == [] and res["pending"] == []
    assert res["worlds"][0]["recognized_world"] == ""


def test_occurrences_carry_positions():
    res = worldnorm.run_worldnorm(make_fake_gen(), make_transcript(), world_dicts={})
    bishma = next(g for g in res["auto"] if g["heard_cleaned"] == "bishma")
    occ = bishma["occurrences"][0]
    assert occ["segment_id"] == 0 and occ["word_index"] == 3 and occ["token"].strip() == "Bishma"


# --------------------------- auto_to_corrections + applier round-trip ---------------------------
def test_auto_to_corrections_applies_to_transcript():
    res = worldnorm.run_worldnorm(make_fake_gen(), make_transcript(), world_dicts={})
    corrections = worldnorm.auto_to_corrections(res["auto"])
    fixed, n = apply_corrections(make_transcript(), corrections, "worldnorm")

    # The sound-alikes are corrected in the words; the queued Bondos is untouched.
    words = {(s["id"], wi): w["word"].strip()
             for s in fixed["segments"] for wi, w in enumerate(s["words"])}
    assert words[(0, 3)] == "Bhishma" and words[(2, 1)] == "Bhishma"
    assert words[(1, 1)] == "Bondos"          # queued, not applied
    assert n == 2 and words[(0, 3)] != "Bishma"


def test_auto_to_corrections_dedupes_by_heard():
    auto = [_group("bishma", "Bhishma", True) | {"canonical": "Bhishma"},
            _group("bishma", "Bhishma", True) | {"canonical": "Bhishma"}]
    corr = worldnorm.auto_to_corrections(auto)
    assert corr == [{"transcribed": "bishma", "correct": "Bhishma"}]


# ------------------------------ worlddict ------------------------------
def test_slug():
    assert worlddict.slug("Thomas & Friends") == "thomas-friends"
    assert worlddict.slug("Mahabharata") == "mahabharata"
    assert worlddict.slug("Steven Universe!") == "steven-universe"


def test_load_world_dict_missing_is_empty(tmp_path):
    assert worlddict.load_world_dict("Nowhere", base=tmp_path) == {}
    assert worlddict.load_variant_map("Nowhere", base=tmp_path) == {}


def test_bless_round_trip(tmp_path):
    worlddict.bless("Mahabharata", "Bondos", "Pandavas", base=tmp_path, now="2026-07-01T00:00:00Z")
    vmap = worlddict.load_variant_map("Mahabharata", base=tmp_path)
    assert vmap.get("bondos") == "Pandavas"
    data = worlddict.load_world_dict("Mahabharata", base=tmp_path)
    assert data["_blessings"][0]["heard"] == "Bondos"


def test_bless_is_idempotent_on_variant(tmp_path):
    worlddict.bless("Mahabharata", "Bondos", "Pandavas", base=tmp_path, now="t1")
    worlddict.bless("Mahabharata", "Bondos", "Pandavas", base=tmp_path, now="t2")
    data = worlddict.load_world_dict("Mahabharata", base=tmp_path)
    entry = next(e for e in data["entries"] if e["canonical"] == "Pandavas")
    assert entry["variants"] == ["Bondos"]            # variant added once
    assert len(data["_blessings"]) == 2                # but each bless logged


def test_bless_then_dictionary_precedence_end_to_end(tmp_path):
    # Bless bushma->Pandavas, then run with that world dict loaded: the collision is overridden.
    worlddict.bless("Mahabharata", "Bushma", "Pandavas", base=tmp_path, now="t1")
    vmap = worlddict.load_variant_map("Mahabharata", base=tmp_path)
    res = worldnorm.run_worldnorm(make_fake_gen(), make_transcript(),
                                  world_dicts={"Mahabharata": vmap})
    autos = {g["heard_cleaned"]: g for g in res["auto"]}
    assert autos["bushma"]["canonical"] == "Pandavas" and autos["bushma"]["method"] == "dictionary"


def test_bless_requires_world():
    with pytest.raises(ValueError):
        worlddict.bless("", "Bondos", "Pandavas")
