"""Unit tests for the world-classifier benchmark: the scorer's matching logic and
the runner's input rendering. Model-free and fast — they exercise the pure
functions only, never loading Gemma.

The eval lives under emp/src/ (not on the default pytest pythonpath), so we add it
explicitly here — the only deviation from the repo's bare-import convention.
"""
import sys
from pathlib import Path

import pytest

EMP_SRC = Path(__file__).resolve().parents[1] / "emp" / "src"
sys.path.insert(0, str(EMP_SRC))

import score_worlds as sw  # noqa: E402
import bench_worlds as bw  # noqa: E402


# --- helpers to build gold stories --------------------------------------------
def canon(world, aliases, difficulty="medium"):
    return {"bucket": "canon", "difficulty": difficulty,
            "world_gold": world, "world_aliases": aliases}


def made_up(difficulty="subtle"):
    return {"bucket": "made_up", "difficulty": difficulty,
            "world_gold": "", "world_aliases": []}


# --- normalize / claims_none --------------------------------------------------
def test_normalize_strips_punctuation_case_and_leading_the():
    assert sw.normalize("The Star-Wars Universe!") == "star wars universe"
    assert sw.normalize("  Mahābhārata  ") == "mah bh rata"  # non-ascii drops out
    assert sw.normalize("") == ""


@pytest.mark.parametrize("pred", ["", "made up", "Made-up", "original / made-up",
                                  "Original story", "an invented world", "unknown",
                                  "it's fiction / imaginary"])
def test_claims_none_true_for_made_up_answers(pred):
    assert sw.claims_none(pred) is True


@pytest.mark.parametrize("pred", ["Star Wars", "He-Man", "Harry Potter", "Mahabharata"])
def test_claims_none_false_for_real_worlds(pred):
    assert sw.claims_none(pred) is False


# --- identifies ---------------------------------------------------------------
def test_identifies_matches_world_gold_and_aliases():
    story = canon("Star Wars", ["jedi", "the force", "skywalker"])
    assert sw.identifies("Star Wars", story) == "Star Wars"
    assert sw.identifies("a Jedi adventure", story) == "jedi"          # alias substring
    assert sw.identifies("the Star Wars universe", story) == "Star Wars"  # two-way substring


def test_identifies_ignores_short_aliases_and_non_matches():
    story = canon("Star Wars", ["ww"])           # 'ww' is too short to match on
    assert sw.identifies("ww", story) is None
    assert sw.identifies("Harry Potter", story) is None
    assert sw.identifies("", story) is None


# --- classify (the four outcomes) ---------------------------------------------
def test_classify_world_buckets():
    story = canon("Star Wars", ["jedi", "the force"])
    assert sw.classify(story, "Star Wars") == "correct"
    assert sw.classify(story, "a Jedi story") == "correct"
    assert sw.classify(story, "made up") == "missed_canon"   # the costly Feb-11 error
    assert sw.classify(story, "") == "missed_canon"
    assert sw.classify(story, "Harry Potter") == "wrong_world"


def test_classify_made_up_bucket():
    story = made_up()
    assert sw.classify(story, "") == "correct"
    assert sw.classify(story, "an original made-up story") == "correct"
    assert sw.classify(story, "Star Wars") == "false_world"  # named a world that isn't there


# --- runner rendering ---------------------------------------------------------
def test_render_full_numbers_every_line():
    lines = ["once there was a boy", "and then what?", "he flew away"]
    out = bw.render_full(lines)
    assert out.splitlines() == ['[0] "once there was a boy"',
                                '[1] "and then what?"',
                                '[2] "he flew away"']


def test_render_sampled_keeps_all_when_short():
    lines = [f"line {i}" for i in range(10)]   # 10 <= head+mid+tail (12)
    assert bw.render_sampled(lines).count("\n") + 1 == 10


def test_render_sampled_thins_long_stories_like_production():
    # 20 lines -> head[0..5] + mids[5,10,15] + tail[17,18,19], deduped = 11 picks.
    lines = [f"line {i}" for i in range(20)]
    out = bw.render_sampled(lines)
    picked = [ln.split("]")[0].lstrip("[") for ln in out.splitlines()]
    assert picked == ["0", "1", "2", "3", "4", "5", "10", "15", "17", "18", "19"]
    assert '[6] "line 6"' not in out          # a dropped middle line
    assert '[19] "line 19"' in out            # the true last line is kept
