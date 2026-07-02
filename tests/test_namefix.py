"""Unit tests for the namefix stage (namefix.py) + the split-prompt cast (worldcast.py).

Fast/pure: fake gen + fake redecode, no models. Each gate rule here was bought by a measured
failure in the 2026-07-01 validation (emp/emp.md) — these tests pin them:
  - temporal nearest-non-ordinary-word matching (the neighboring-Krishna leak)
  - ordinary re-decode never maps to cast (father->Vidura)
  - common blind token never auto-overwritten (arrows->Kauravas, Choksi's ear-check)
  - blessed dictionary entries win outright
  - unrecognized world -> the stage does nothing (generality 6/6)
"""
import json

import pytest

import namefix
import worldcast
from corrections import apply_corrections


# ------------------------------ worldcast ------------------------------
def fake_gen_factory(world_answer="Mahabharata"):
    def gen(prompt, max_tokens=256):
        if "named GROUPS" in prompt:
            return "The Pandavas\nKauravas\n"
        if "main characters" in prompt:
            return "Arjuna\nBhima\nBhishma\nKrishna\nDuryodhana\n"
        if "REAL, widely-known" in prompt:
            return "original" if not world_answer else f"World: {world_answer}"
        return ""
    return gen


def test_generate_cast_split_parses_both():
    split = worldcast.generate_cast_split(fake_gen_factory(), "Mahabharata")
    assert split["groups"] == ["Pandavas", "Kauravas"]   # "The " stripped
    assert "Bhishma" in split["characters"]


def test_correction_cast_groups_first_dedup():
    split = {"groups": ["Pandavas", "Kauravas"], "characters": ["Arjuna", "pandavas", "Bhima"]}
    cast = worldcast.correction_cast(split)
    assert cast[:2] == ["Pandavas", "Kauravas"]
    assert "pandavas" not in cast[2:]          # deduped by cleaned form
    assert cast == ["Pandavas", "Kauravas", "Arjuna", "Bhima"]


def test_cached_cast_split_round_trip(tmp_path):
    calls = []
    def counting_gen(prompt, max_tokens=256):
        calls.append(1)
        return fake_gen_factory()(prompt, max_tokens)
    s1 = worldcast.cached_cast_split(counting_gen, "Mahabharata", cache_dir=tmp_path)
    n_after_first = len(calls)
    s2 = worldcast.cached_cast_split(counting_gen, "Mahabharata", cache_dir=tmp_path)
    assert s1 == s2 and len(calls) == n_after_first   # second hit served from cache
    assert (tmp_path / "mahabharata.json").exists()


# ------------------------------ the gate (pure) ------------------------------
CAST_CLEAN = {"pandavas": "Pandavas", "kauravas": "Kauravas", "bhishma": "Bhishma",
              "bhima": "Bhima", "arjuna": "Arjuna", "vidura": "Vidura"}


def test_gate_sound_match_autofixes_nonword():
    d = namefix.gate_decision([("Bhishma", 1.5)], "Bushma", CAST_CLEAN, set(), {})
    assert d["action"] == "auto" and d["canonical"] == "Bhishma"


def test_gate_common_blind_token_queues():
    # Choksi's ear-check rule: "arrows" is a real word -> never auto-overwritten.
    d = namefix.gate_decision([("Kauravas", 1.5)], "arrows", CAST_CLEAN, {"arrows"}, {})
    assert d["action"] == "queued" and d["canonical"] == "Kauravas"


def test_gate_ordinary_redecode_never_maps():
    # father (FTR) sound-matches Vidura (FTR) but is an ordinary word -> skipped -> untouched.
    d = namefix.gate_decision([("father", 1.5)], "Fondos", CAST_CLEAN, set(), {})
    assert d is None


def test_gate_neighboring_name_is_not_the_target():
    # nearest non-ordinary word is the unmapped "Bandhas" -> untouched; the loud "Kauravas"
    # sitting further away must NOT grab the decision (the v2 leak).
    near = [("Bandhas", 1.5), ("Kauravas", 2.1)]
    d = namefix.gate_decision(near, "Bandos", CAST_CLEAN, set(), {})
    assert d is None


def test_gate_unchanged_target_untouched():
    d = namefix.gate_decision([("Bushma", 1.5)], "Bushma", CAST_CLEAN, set(), {})
    assert d is None


def test_gate_blessed_dictionary_wins():
    d = namefix.gate_decision([("anything", 1.5)], "Bondos", CAST_CLEAN, set(),
                              {"bondos": "Pandavas"})
    assert d["action"] == "auto" and d["how"] == "dictionary" and d["canonical"] == "Pandavas"


# ------------------------------ run_namefix (fakes end-to-end) ------------------------------
def make_transcript():
    def w(word, start):
        return {"word": word, "start": start, "end": start + 0.3, "probability": 0.9}
    segments = [
        {"id": 0, "text": " Tell me about Bushma today.",
         "words": [w("Tell", 0.0), w("me", 0.4), w("about", 0.8), w("Bushma", 1.2), w("today.", 1.6)]},
        {"id": 1, "text": " The Bondos were good.",
         "words": [w("The", 2.0), w("Bondos", 2.4), w("were", 2.8), w("good.", 3.2)]},
    ]
    return {"text": "", "segments": segments,
            "_stories": [{"start_id": 0, "end_id": 1, "world": "", "title": "T"}]}


def fake_wac(world="Mahabharata"):
    return [{"story": 0, "world": world,
             "split": {"characters": ["Bhishma", "Arjuna"], "groups": ["Pandavas", "Kauravas"]}}]


def fake_redecode_factory(mapping):
    """mapping: blind cleaned -> re-decoded word placed at the expected offset."""
    def fake(audio_path, jobs, singles, variant_maps):
        from detectors.phonetics import clean as _c
        out = []
        for j in jobs:
            p = j["cand"]
            re_word = mapping.get(_c(p["blind"]))
            near = [(re_word, min(p["start"], namefix.LEAD))] if re_word else []
            d = namefix.gate_decision(near, p["blind"], j["cast_clean"], set(singles),
                                      variant_maps.get(j["world"], {}))
            if d:
                out.append({**p, **d, "world": j["world"]})
        return out
    return fake


def test_run_namefix_auto_and_untouched():
    res = namefix.run_namefix(make_transcript(), "unused.m4a",
                              worlds_and_casts=fake_wac(),
                              redecode=fake_redecode_factory({"bushma": "Bhishma",
                                                              "bondos": "Bandhas"}))
    autos = {g["heard_cleaned"]: g for g in res["auto"]}
    assert autos["bushma"]["canonical"] == "Bhishma"
    assert "bondos" not in autos and res["pending"] == []   # Bandhas unmapped -> untouched
    assert res["worlds"][0]["recognized_world"] == "Mahabharata"


def test_run_namefix_unrecognized_world_does_nothing():
    res = namefix.run_namefix(make_transcript(), "unused.m4a",
                              worlds_and_casts=[{"story": 0, "world": "",
                                                 "split": {"characters": [], "groups": []}}],
                              redecode=fake_redecode_factory({"bushma": "Bhishma"}))
    assert res["auto"] == [] and res["pending"] == []
    assert res["worlds"][0]["recognized_world"] == ""


def test_auto_corrections_apply_id_safe():
    t = make_transcript()
    res = namefix.run_namefix(t, "unused.m4a", worlds_and_casts=fake_wac(),
                              redecode=fake_redecode_factory({"bushma": "Bhishma"}))
    fixed, n = apply_corrections(t, namefix.auto_to_corrections(res["auto"]), "namefix")
    assert n == 1
    # ids + word counts unchanged (the axial-label safety property)
    assert [s["id"] for s in fixed["segments"]] == [s["id"] for s in t["segments"]]
    assert [len(s["words"]) for s in fixed["segments"]] == [len(s["words"]) for s in t["segments"]]
    assert fixed["segments"][0]["words"][3]["word"].strip() == "Bhishma"
    assert fixed["segments"][0]["words"][3]["_original"] == "Bushma"
    assert "Bhishma" in fixed["segments"][0]["text"]        # text healed from words


def test_write_pending_shape(tmp_path):
    t = make_transcript()
    res = {"auto": [], "worlds": [], "pending": [
        {"world": "Mahabharata", "story_id": 0, "heard": "arrows", "heard_cleaned": "arrows",
         "suggestion": "Kauravas", "canonical": "Kauravas", "method": "exact",
         "action": "queued", "occurrences": [{"segment_id": 0, "word_index": 1,
                                              "start": 2.4, "token": "arrows"}]}]}
    path = namefix.write_pending(tmp_path, res, t)
    data = json.loads(path.read_text())
    assert data["pending"][0]["canonical"] == "Kauravas"
    assert data["config_fingerprint"] and data["transcript_fingerprint"]
    assert data["_rejected"] == []


# ------------------------------ the frequency wordlist ------------------------------
def test_commonwords_kills_the_websters_ghosts():
    """The three Webster's-1934 failures, pinned against the new frequency list:
    real words are common; archaic ghosts and canon names are not."""
    from commonwords import is_common
    for real in ("arrows", "beam", "father", "wars"):
        assert is_common(real), f"{real} should be protected"
    for ghost in ("bando", "jami", "kauravas", "bushma", "bandos", "bheem"):
        assert not is_common(ghost), f"{ghost} must not be falsely protected"


def test_gate_blind_bandos_now_autofixes_without_dictionary():
    # Under Webster's, "Bandos" queued (bando ghost). Under the frequency list it
    # auto-fixes on a sound/cast match — no blessing needed for the non-word garble.
    d = namefix.gate_decision([("Pandavas", 1.5)], "Bandos", CAST_CLEAN, set(), {})
    assert d["action"] == "auto" and d["canonical"] == "Pandavas"
