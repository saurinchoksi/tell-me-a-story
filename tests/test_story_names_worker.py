"""The story-names worker reads the pipeline's saved stories, falling back to live
segmentation only for sessions processed before story segmentation shipped.

All model-touching functions are mocked so no Gemma is ever loaded."""
import json
from unittest.mock import patch

from detectors.story_names import CanonNameDetector, _worker


def _write_rich(tmp_path, segments, stories=None):
    rich = {"segments": segments}
    if stories is not None:
        rich["_stories"] = stories
    (tmp_path / "transcript-rich.json").write_text(json.dumps(rich))
    return tmp_path


_SEGS = [{"id": 0, "text": "a", "words": []}, {"id": 1, "text": "b", "words": []}]


def test_worker_uses_saved_stories_without_resegmenting(tmp_path):
    stories = [{"index": 0, "start_id": 0, "end_id": 1, "title": "A", "world": "W"}]
    sd = _write_rich(tmp_path, _SEGS, stories=stories)
    with patch("detectors.story_names._worker.make_reader", return_value=lambda *a, **k: ""), \
         patch("detectors.story_names._worker.story_segments", return_value=[]), \
         patch("detectors.story_names._worker.story_name_cards", return_value=[]), \
         patch("detectors.story_names._worker.run_v2", return_value=([], [])), \
         patch("detectors.story_names._worker.segment_session") as mock_seg:
        result = _worker.run(str(sd))
    mock_seg.assert_not_called()  # saved stories were used — no live re-split
    assert result == {"n_word_tokens": 0, "flags": []}


def test_worker_falls_back_to_live_segmentation(tmp_path):
    sd = _write_rich(tmp_path, _SEGS, stories=None)  # no _stories present
    seg_result = ({"stories": [{"start_id": 0, "end_id": 1, "title": "A", "world": "W"}]}, [])
    with patch("detectors.story_names._worker.make_reader", return_value=lambda *a, **k: ""), \
         patch("detectors.story_names._worker.story_segments", return_value=[]), \
         patch("detectors.story_names._worker.story_name_cards", return_value=[]), \
         patch("detectors.story_names._worker.run_v2", return_value=([], [])), \
         patch("detectors.story_names._worker.segment_session", return_value=seg_result) as mock_seg:
        result = _worker.run(str(sd))
    mock_seg.assert_called_once()  # no saved stories → fell back to live segmentation
    assert result == {"n_word_tokens": 0, "flags": []}


def test_worker_uses_empty_saved_stories_without_resegmenting(tmp_path):
    """A session the pipeline segmented to ZERO stories (_stories == []) uses the saved
    empty list — it must NOT fall back to a wasteful live re-split (the falsy-[] bug)."""
    sd = _write_rich(tmp_path, _SEGS, stories=[])  # segmented, found none
    with patch("detectors.story_names._worker.make_reader", return_value=lambda *a, **k: ""), \
         patch("detectors.story_names._worker.segment_session") as mock_seg:
        result = _worker.run(str(sd))
    mock_seg.assert_not_called()  # empty saved list is still "segmented" — no re-split
    assert result == {"n_word_tokens": 0, "flags": []}


def test_canon_detector_emits_only_m9c_flags(tmp_path):
    """CanonNameDetector surfaces only the M9c slice of the converged worker output —
    M9b is the separate m9b-name-consistency detector's job. The worker (the expensive
    converged pass) runs exactly once; filtering its result is free."""
    mixed = {
        "n_word_tokens": 100,
        "flags": [
            {"case": "M9c", "token": "Pondavas", "canonical": "Pandavas"},
            {"case": "M9b", "token": "Jameis", "canonical": "Jammus"},
            {"case": "M9c", "token": "Dhrashtra", "canonical": "Dhritarashtra"},
        ],
    }
    with patch("model_runner.run_model", return_value=mixed) as mock_run:
        result = CanonNameDetector().run(tmp_path)
    mock_run.assert_called_once()                                  # converged engine runs once
    assert result["n_word_tokens"] == 100                          # denominator unchanged
    assert [f["case"] for f in result["flags"]] == ["M9c", "M9c"]  # M9b filtered out
