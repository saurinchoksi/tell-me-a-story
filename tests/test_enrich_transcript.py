"""Tests for enrich_transcript function (in pipeline module).

The world-blind LLM name-normalizer (old Pass 3) was removed 2026-07-01 — it
confidently substituted wrong names. Enrichment now runs: diarization (Pass 1),
gap detection (Pass 2), story segmentation (Pass 3, only with a cache_dir),
namefix (Pass 4 — the audio-grounded replacement; needs cache_dir + audio_path
+ _stories, else records a "skipped" entry), and dictionary normalization
(Pass 5, off by default).
"""

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

_DIAR_ENTRY = {"stage": "diarization_enrichment",
               "model": "pyannote/speaker-diarization-community-1",
               "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}
_GAP_ENTRY = {"stage": "gap_detection", "gaps_found": 0,
              "status": "success", "timestamp": "2026-01-01T00:00:00+00:00"}


# --- enrich_transcript tests ---

def test_enrich_all_stages_succeed():
    """With a library (and no cache_dir/audio), the stages run: diar, gap,
    namefix(skipped), dictionary."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(copy.deepcopy(transcript), 1)), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path="fake/lib.json", verbose=False
        )

    assert len(processing) == 4
    assert processing[0]["stage"] == "diarization_enrichment"
    assert processing[0]["status"] == "success"
    assert processing[1]["stage"] == "gap_detection"
    assert processing[1]["status"] == "success"
    assert processing[2]["stage"] == "namefix"
    assert processing[2]["status"] == "skipped"
    assert processing[3]["stage"] == "dictionary_normalization"
    assert processing[3]["status"] == "success"
    assert counts["dict_count"] == 1
    assert counts["namefix_auto"] == 0
    assert "llm_count" not in counts


def test_enrich_does_not_set_processing_on_transcript():
    """enrich_transcript does not set _processing on the returned transcript."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("pipeline.extract_text", return_value="hello world"), \
         patch("pipeline.apply_corrections", return_value=(copy.deepcopy(transcript), 0)), \
         patch("pipeline.load_library", return_value={"entries": []}), \
         patch("pipeline.build_variant_map", return_value={}), \
         patch("pipeline.normalize_variants", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, library_path="fake/lib.json", verbose=False
        )

    assert "_processing" not in result


def test_enrich_skips_dictionary_when_no_library():
    """Dictionary normalization is skipped when no library_path is provided."""
    transcript = copy.deepcopy(_CLEAN_TRANSCRIPT)
    diarization = copy.deepcopy(_FAKE_DIARIZATION)

    with patch("pipeline.load_library") as mock_load_lib, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):

        result, processing, counts = enrich_transcript(
            transcript, diarization, verbose=False
        )

    # diar, gap, namefix(skipped), dict(skipped) = 4
    assert len(processing) == 4
    assert processing[2]["stage"] == "namefix"
    assert processing[2]["status"] == "skipped"
    assert processing[3]["stage"] == "dictionary_normalization"
    assert processing[3]["status"] == "skipped"
    assert processing[3]["reason"] == "no_library_path"
    assert counts["dict_count"] == 0
    mock_load_lib.assert_not_called()


def test_enrich_namefix_runs_with_audio_and_stories(tmp_path):
    """With cache_dir + audio_path + stories, Pass 4 runs namefix: autos applied to the
    words, the pending queue written to the session dir, counts reported."""
    transcript = {"segments": [{
        "id": 0, "start": 0.0, "end": 2.0, "text": " about Bushma today",
        "words": [
            {"word": " about", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " Bushma", "start": 0.5, "end": 1.0, "probability": 0.9},
            {"word": " today", "start": 1.0, "end": 1.5, "probability": 0.9},
        ],
    }]}
    diarization = {"segments": []}
    stories = [{"start_id": 0, "end_id": 0, "title": "T", "world": ""}]
    fake_result = {
        "auto": [{"world": "Mahabharata", "story_id": 0, "heard": "Bushma",
                  "heard_cleaned": "bushma", "suggestion": "Bhishma", "canonical": "Bhishma",
                  "method": "exact", "action": "auto",
                  "occurrences": [{"segment_id": 0, "word_index": 1, "start": 0.5,
                                   "token": "Bushma"}]}],
        "pending": [{"world": "Mahabharata", "story_id": 0, "heard": "arrows",
                     "heard_cleaned": "arrows", "suggestion": "Kauravas",
                     "canonical": "Kauravas", "method": "exact", "action": "queued",
                     "occurrences": []}],
        "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}],
    }
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    with patch("pipeline.segment_transcript", return_value=stories), \
         patch("namefix.run_namefix", return_value=fake_result) as mock_nf, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, counts = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, audio_path=str(audio), verbose=False)

    mock_nf.assert_called_once()
    nf = [p for p in processing if p["stage"] == "namefix"][0]
    assert nf["status"] == "success" and nf["auto_applied"] == 1 and nf["pending"] == 1
    assert result["segments"][0]["words"][1]["word"].strip() == "Bhishma"
    assert result["segments"][0]["words"][1]["_original"] == "Bushma"
    assert counts["namefix_auto"] == 1 and counts["namefix_pending"] == 1
    assert (tmp_path / "pending-name-corrections.json").exists()


def test_enrich_does_not_realign(tmp_path):
    """Realignment left the enrichment loop: enrich_transcript emits no word_realignment
    stage and never re-times words — so --re-enrich reuses transcript-raw's corrected
    timings, even when a session audio file sits in cache_dir."""
    transcript = {"segments": [{
        "id": 0, "text": " hello world",
        "words": [
            {"word": " hello", "start": 0.11, "end": 0.52, "probability": 0.9, "_align_conf": 0.88},
            {"word": " world", "start": 0.52, "end": 1.03, "probability": 0.8, "_align_conf": 0.91},
        ],
    }]}
    before = copy.deepcopy(transcript["segments"][0]["words"])
    diarization = {"segments": []}
    (tmp_path / "audio.m4a").write_bytes(b"not real audio")  # present but must be ignored

    with patch("pipeline.segment_transcript", return_value=[]), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, _ = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, verbose=False)

    assert not any(p["stage"] == "word_realignment" for p in processing)
    assert result["segments"][0]["words"] == before  # timings untouched by enrichment


def test_enrich_pass5_segmentation_success(tmp_path):
    """With a cache_dir, Pass 5 runs and records a story_segmentation processing entry."""
    transcript = {"segments": [{"id": 0, "text": " a", "words": [{"word": " a"}]}]}
    diarization = {"segments": []}
    stories = [{"start_id": 0, "end_id": 0, "title": "A", "world": "W"}]

    with patch("pipeline.segment_transcript", return_value=stories) as mock_seg, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
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

    with patch("pipeline.segment_transcript", side_effect=RuntimeError("model down")), \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, _DIAR_ENTRY)), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, _GAP_ENTRY)):
        result, processing, _ = enrich_transcript(
            transcript, diarization, cache_dir=tmp_path, verbose=False)

    seg = [p for p in processing if p["stage"] == "story_segmentation"][0]
    assert seg["status"] == "error" and "model down" in seg["error"]
    # the other passes still ran
    assert any(p["stage"] == "gap_detection" for p in processing)
    assert "_stories" not in result  # segmentation failed, nothing tagged


def test_enrich_reuses_segmentation_cache_across_runs(tmp_path):
    """Two enrich runs with the same cache_dir hit the segmentation cache: the expensive
    segmenter computes once, the second run is from_cache — the 're-enrich doesn't reload
    the model' behavior, wired through the in-memory transcript fingerprints (mocks only
    the model COMPUTE, so the real cache/fingerprints run)."""
    base = {"segments": [
        {"id": 0, "start": 0.0, "end": 1.0, "text": " hello",
         "words": [{"word": " hello", "start": 0.0, "end": 1.0}]},
        {"id": 1, "start": 1.0, "end": 2.0, "text": " world",
         "words": [{"word": " world", "start": 1.0, "end": 2.0}]},
    ]}
    diarization = {"segments": []}
    stories = [{"start_id": 0, "end_id": 1, "title": "T", "world": "W"}]

    with patch("pipeline.segment_transcript", return_value=stories) as mock_seg, \
         patch("pipeline.enrich_with_diarization", side_effect=lambda t, d: (t, {"stage": "diarization_enrichment"})), \
         patch("pipeline.detect_unintelligible_gaps", side_effect=lambda t, d: (t, {"stage": "gap_detection", "gaps_found": 0})):
        _, p1, _ = enrich_transcript(copy.deepcopy(base), diarization, cache_dir=tmp_path, verbose=False)
        r2, p2, _ = enrich_transcript(copy.deepcopy(base), diarization, cache_dir=tmp_path, verbose=False)

    assert mock_seg.call_count == 1   # segmentation computed once, reused on the 2nd run
    assert [e for e in p1 if e["stage"] == "story_segmentation"][0]["from_cache"] is False
    assert [e for e in p2 if e["stage"] == "story_segmentation"][0]["from_cache"] is True
    assert r2["_stories"][0]["title"] == "T"  # stories still enriched in on the cached run
