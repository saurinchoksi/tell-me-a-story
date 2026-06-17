"""Tests for enrich_transcript function (in pipeline module)."""

import copy
from unittest.mock import patch, MagicMock

from pipeline import enrich_transcript


# --- Test fixtures ---

_CLEAN_TRANSCRIPT = {
    "text": "hello world",
    "segments": [{
        "text": "hello world",
        "words": [
            {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " world", "start": 0.5, "end": 1.0, "probability": 0.8},
        ],
    }],
}

_FAKE_DIARIZATION = {
    "segments": [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ],
}


# --- enrich_transcript tests ---

def test_enrich_all_stages_succeed():
    """All 4 enrichment stages succeed, producing 4 processing entries."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    llm_entry = {"stage": "llm_normalization", "model": "qwen3:8b",
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    diar_entry = {"stage": "diarization_enrichment",
                  "model": "pyannote/speaker-diarization-community-1",
                  "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0,
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)) as mock_llm, \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", side_effect=[
             (copy.deepcopy(transcript), 2),
             (copy.deepcopy(transcript), 1),
         ]), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path="fake/lib.json", verbose=False
        )

    assert len(processing) == 4
    assert processing[0]["stage"] == "diarization_enrichment"
    assert processing[0]["status"] == "success"
    assert "timestamp" in processing[0]
    assert processing[1]["stage"] == "gap_detection"
    assert processing[1]["status"] == "success"
    assert processing[2]["stage"] == "llm_normalization"
    assert processing[2]["status"] == "success"
    assert "timestamp" in processing[2]
    assert processing[2]["corrections_applied"] == 2
    assert processing[3]["stage"] == "dictionary_normalization"
    assert processing[3]["status"] == "success"
    assert "timestamp" in processing[3]
    assert counts["llm_count"] == 2
    assert counts["dict_count"] == 1


def test_enrich_llm_fails_others_continue():
    """LLM failure is recorded but dict + diarization + gap detection still run."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    diar_entry = {"stage": "diarization_enrichment",
                  "model": "pyannote/speaker-diarization-community-1",
                  "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0,
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}

    with patch("pipeline.llm_normalize", side_effect=RuntimeError("Ollama down")), \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(
             copy.deepcopy(transcript), 3,
         )), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path="fake/lib.json", verbose=False
        )

    assert len(processing) == 4
    assert processing[0]["stage"] == "diarization_enrichment"
    assert processing[0]["status"] == "success"
    assert "timestamp" in processing[0]
    assert processing[1]["stage"] == "gap_detection"
    assert processing[1]["status"] == "success"
    assert processing[2]["stage"] == "llm_normalization"
    assert processing[2]["status"] == "error"
    assert "Ollama down" in processing[2]["error"]
    assert "timestamp" in processing[2]
    assert processing[3]["stage"] == "dictionary_normalization"
    assert processing[3]["status"] == "success"
    assert "timestamp" in processing[3]
    assert counts["llm_count"] == 0
    assert counts["dict_count"] == 3


def test_enrich_does_not_set_processing_on_transcript():
    """enrich_transcript does not set _processing on the returned transcript."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    llm_entry = {"stage": "llm_normalization", "model": "qwen3:8b",
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    diar_entry = {"stage": "diarization_enrichment",
                  "model": "pyannote/speaker-diarization-community-1",
                  "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0,
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)), \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", side_effect=[
             (copy.deepcopy(transcript), 0),
             (copy.deepcopy(transcript), 0),
         ]), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path="fake/lib.json", verbose=False
        )

    assert "_processing" not in result


def test_enrich_skips_dictionary_when_no_library():
    """Dictionary normalization is skipped when no library_path is provided."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    llm_entry = {"stage": "llm_normalization", "model": "qwen3:8b",
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    diar_entry = {"stage": "diarization_enrichment",
                  "model": "pyannote/speaker-diarization-community-1",
                  "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0,
                 "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)), \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(
             copy.deepcopy(transcript), 0,
         )), \
         patch("pipeline.load_library") as mock_load_lib, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    # Should have 4 entries: diar, gap, llm, dict(skipped)
    assert len(processing) == 4
    assert processing[3]["stage"] == "dictionary_normalization"
    assert processing[3]["status"] == "skipped"
    assert processing[3]["reason"] == "no_library_path"
    assert counts["dict_count"] == 0
    # load_library should never be called
    mock_load_lib.assert_not_called()


def test_enrich_passes_cache_dir_to_llm_normalize(tmp_path):
    """enrich_transcript threads cache_dir through to llm_normalize, so a re-enrich can
    reuse the cached corrections instead of reloading the model."""
    transcript = {"segments": []}
    diarization = {"segments": []}
    diar_entry = {"stage": "diarization_enrichment"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0}
    llm_entry = {"stage": "llm_normalization", "model": "m", "status": "success",
                 "from_cache": False, "timestamp": "t"}

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)) as mock_llm, \
         patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(transcript, 0)), \
         patch("pipeline.segment_transcript", return_value=[]), \
         patch("pipeline.enrich_with_stories", side_effect=lambda t, s: t), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):
        enrich_transcript(transcript, diarization, cache_dir=tmp_path, verbose=False)

    assert mock_llm.call_args.kwargs.get("cache_dir") == tmp_path


def test_enrich_pass5_segmentation_success(tmp_path):
    """With a cache_dir, Pass 5 runs and records a story_segmentation processing entry."""
    transcript = {"segments": [{"id": 0, "text": " a", "words": [{"word": " a"}]}]}
    diarization = {"segments": []}
    diar_entry = {"stage": "diarization_enrichment"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0}
    llm_entry = {"stage": "llm_normalization", "model": "m", "status": "success",
                 "from_cache": False, "timestamp": "t"}
    stories = [{"start_id": 0, "end_id": 0, "title": "A", "world": "W"}]

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)), \
         patch("pipeline.extract_text", return_value="a"), \
         patch("pipeline.apply_corrections", return_value=(transcript, 0)), \
         patch("pipeline.segment_transcript", return_value=stories) as mock_seg, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):
        result, processing, _ = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, verbose=False)

    seg_entries = [p for p in processing if p["stage"] == "story_segmentation"]
    assert len(seg_entries) == 1
    assert seg_entries[0]["status"] == "success"
    assert seg_entries[0]["stories_found"] == 1
    assert result["_stories"][0]["title"] == "A"  # enrich_with_stories ran for real
    mock_seg.assert_called_once()


def test_enrich_pass5_degrades_safely(tmp_path):
    """A segmentation failure records an error entry but does not sink the pass."""
    transcript = {"segments": [{"id": 0, "text": " a", "words": [{"word": " a"}]}]}
    diarization = {"segments": []}
    diar_entry = {"stage": "diarization_enrichment"}
    gap_entry = {"stage": "gap_detection", "gaps_found": 0}
    llm_entry = {"stage": "llm_normalization", "model": "m", "status": "success",
                 "from_cache": False, "timestamp": "t"}

    with patch("pipeline.llm_normalize", return_value=([], llm_entry)), \
         patch("pipeline.extract_text", return_value="a"), \
         patch("pipeline.apply_corrections", return_value=(transcript, 0)), \
         patch("pipeline.segment_transcript", side_effect=RuntimeError("model down")), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, diar_entry)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, gap_entry)):
        result, processing, _ = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, verbose=False)

    seg = [p for p in processing if p["stage"] == "story_segmentation"][0]
    assert seg["status"] == "error" and "model down" in seg["error"]
    # the other passes still ran
    assert any(p["stage"] == "llm_normalization" for p in processing)
    assert "_stories" not in result  # segmentation failed, nothing tagged


def test_enrich_reuses_cache_across_runs(tmp_path):
    """Two enrich runs with the same cache_dir hit the cache: the expensive model computes
    (normalize + segmentation) each run ONCE, the second run is from_cache. The headline
    're-enrich doesn't reload the model' behavior, wired through the in-memory transcript
    fingerprints — mocks only the model COMPUTES, so the real cache/fingerprints run."""
    base = {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": " hello",
         "words": [{"word": " hello", "start": 0.0, "end": 1.0}]},
        {"id": 1, "start": 1.0, "end": 2.0, "text": " world",
         "words": [{"word": " world", "start": 1.0, "end": 2.0}]},
    ]}
    diarization = {"segments": []}
    stories = [{"start_id": 0, "end_id": 1, "title": "T", "world": "W"}]

    with patch("normalize._call_mlx", return_value='{"corrections": []}') as mock_mlx, \
         patch("pipeline.segment_transcript", return_value=stories) as mock_seg, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, {"stage": "diarization_enrichment"})), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, {"stage": "gap_detection", "gaps_found": 0})):
        _, p1, _ = enrich_transcript(copy.deepcopy(base), diarization, cache_dir=tmp_path, verbose=False)
        r2, p2, _ = enrich_transcript(copy.deepcopy(base), diarization, cache_dir=tmp_path, verbose=False)

    assert mock_mlx.call_count == 1   # normalization computed once, reused on the 2nd run
    assert mock_seg.call_count == 1   # segmentation computed once, reused on the 2nd run
    assert [e for e in p1 if e["stage"] == "llm_normalization"][0]["from_cache"] is False
    assert [e for e in p2 if e["stage"] == "llm_normalization"][0]["from_cache"] is True
    assert [e for e in p2 if e["stage"] == "story_segmentation"][0]["from_cache"] is True
    assert r2["_stories"][0]["title"] == "T"  # stories still enriched in on the cached run
