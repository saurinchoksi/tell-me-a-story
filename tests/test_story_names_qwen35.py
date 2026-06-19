"""Unit tests for the Qwen3.5 canon layer (_qwen35) + the combined occurrence expander
(_worker.expand_combined). The model is mocked (a canned `gen`); the dictionary gate is
isolated to identity where we test parsing/matching (the gate has its own tests in
test_canon_dictionary_gate.py). Fast — no model load."""
import detectors.story_names._qwen35 as q
from detectors.story_names._worker import expand_combined


# ----------------------------- the name judge -------------------------------
def test_judge_parses_wrong_correct_lines(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)  # isolate the parser
    raw = ("Bishma -> Bhishma\n"
           "Garn -> Karna\n"
           "this is not a correction line\n"
           "Krishna -> Krishna\n")  # already-correct: dropped (clean equal)
    flags = q.judge_names(lambda p, max_tokens=500: raw, "Mahabharata", ["Bishma", "Garn", "Krishna"], set())
    assert {f["canonical"] for f in flags} == {"Bhishma", "Karna"}
    assert {f["wrong_cleaned"][0] for f in flags} == {"bishma", "garn"}
    assert all(f["case"] == "M9c" for f in flags)


def test_judge_empty_world_or_names_returns_empty():
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not be called"))
    assert q.judge_names(boom, "", ["Bishma"], set()) == []
    assert q.judge_names(boom, "Mahabharata", [], set()) == []


def test_judge_dedupes_repeated_wrong_spelling(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    raw = "Bishma -> Bhishma\nBishma -> Bhishma\n"
    flags = q.judge_names(lambda p, max_tokens=500: raw, "Mahabharata", ["Bishma"], set())
    assert len(flags) == 1


# ----------------------------- the cast generator ---------------------------
def test_generate_cast_parses_plain_lines_and_filters():
    raw = "Bhishma\nKarna\n1. Arjuna\n- Drona\n{\"names\": x}\nBhishma\n"  # numbering, bullet, JSON, dup
    cast = q.generate_cast(lambda p, max_tokens=500: raw, "Mahabharata")
    assert cast == ["Bhishma", "Karna", "Arjuna", "Drona"]  # JSON-ish dropped, dup deduped, order kept


def test_generate_cast_empty_world():
    assert q.generate_cast(lambda *a, **k: "x", "") == []


# ----------------------------- world recognition ----------------------------
def test_recognize_world_parses_world_line():
    assert q.recognize_world(lambda p, max_tokens=120: "World: Mahabharata", ["Bishma", "Arjun"]) == "Mahabharata"


def test_recognize_world_abstains_on_original():
    assert q.recognize_world(lambda p, max_tokens=120: "World: original", ["Blarg", "Florp"]) == ""


def test_recognize_world_empty_names_no_model_call():
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not be called"))
    assert q.recognize_world(boom, []) == ""


# ----------------------------- phonetic matching ----------------------------
def _card(cid, clean_forms, surface, occ=()):
    return {"id": cid, "clean": list(clean_forms), "surface": list(surface),
            "n": 1, "examples": [""], "is_phrase": False, "occ": list(occ)}


def test_phonetic_flags_matches_by_sound(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    cast = ["Bhishma", "Karna", "Arjuna"]
    flags = q.phonetic_flags([_card(0, ["bishma"], ["Bishma"])], set(), cast)
    assert len(flags) == 1 and flags[0]["canonical"] == "Bhishma" and flags[0]["wrong_cleaned"] == ["bishma"]


def test_phonetic_flags_skips_correct_and_short(monkeypatch):
    monkeypatch.setattr(q, "gate_canon_flags", lambda flags, singles: flags)
    cast = ["Bhishma", "Karna"]
    # a correctly-spelled cast form (karna) and a too-short token (po) must not flag
    flags = q.phonetic_flags([_card(0, ["karna"], ["Karna"]), _card(1, ["po"], ["Po"])], set(), cast)
    assert flags == []


def test_phonetic_flags_empty_cast():
    assert q.phonetic_flags([_card(0, ["bishma"], ["Bishma"])], set(), []) == []


# ----------------------------- the union ------------------------------------
def test_combine_unions_by_wrong_spelling():
    a = [{"wrong_cleaned": ["bishma"], "canonical": "Bhishma"}]
    b = [{"wrong_cleaned": ["bishma"], "canonical": "Bhishma"},   # dup of a
         {"wrong_cleaned": ["garn"], "canonical": "Karna"}]       # new
    out = q.combine(a, b)
    assert len(out) == 2
    assert {tuple(f["wrong_cleaned"]) for f in out} == {("bishma",), ("garn",)}


# ----------------------------- occurrence expansion -------------------------
def test_expand_combined_maps_flag_to_occurrences():
    card = _card(0, ["bishma"], ["Bishma"], occ=[{"seg_id": 5, "wi": 2}])
    seg_by_id = {5: {"id": 5, "words": [{"word": "and"}, {"word": "then"},
                                        {"word": "Bishma", "start": 1.0, "end": 1.5}]}}
    flag = {"case": "M9c", "canonical": "Bhishma", "wrong_cleaned": ["bishma"], "all_spellings": ["Bishma"]}
    out = expand_combined([flag], [card], seg_by_id, story_idx=0, world="Mahabharata")
    assert len(out) == 1
    rec = out[0]
    assert rec["segment_id"] == 5 and rec["word_index"] == 2
    assert rec["canonical"] == "Bhishma" and rec["token"] == "Bishma" and rec["case"] == "M9c"
    assert rec["story_world"] == "Mahabharata"


def test_expand_combined_dedupes_across_cards():
    # the same occurrence reachable via two cards must land once
    card_a = _card(0, ["bishma"], ["Bishma"], occ=[{"seg_id": 5, "wi": 0}])
    card_b = _card(1, ["bishma"], ["Bishma"], occ=[{"seg_id": 5, "wi": 0}])
    seg_by_id = {5: {"id": 5, "words": [{"word": "Bishma", "start": 1.0, "end": 1.5}]}}
    flag = {"case": "M9c", "canonical": "Bhishma", "wrong_cleaned": ["bishma"], "all_spellings": ["Bishma"]}
    out = expand_combined([flag], [card_a, card_b], seg_by_id, story_idx=0, world="Mahabharata")
    assert len(out) == 1
