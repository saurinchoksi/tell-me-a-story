"""Speaker diarization using pyannote.audio.

Also contains enrichment functions that apply diarization results to transcripts,
adding _speaker metadata to each word based on temporal overlap with speaker segments.
"""

import copy
import os
import subprocess
import tempfile
from datetime import datetime, timezone
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# PyTorch 2.6+ changed weights_only default to True for security.
# pyannote.audio models need these classes allowlisted to load.
# This is safe because we're loading official pyannote models from Hugging Face.
from pyannote.audio.core.task import Specifications, Problem, Resolution, Scope
torch.serialization.add_safe_globals([Specifications, Problem, Resolution, Scope])

_SCHEMA_VERSION = "1.0.0"
_GENERATOR_VERSION = "pyannote-speaker-diarization-community-1"
MODEL = "pyannote/speaker-diarization-community-1"


def load_diarization_model() -> Pipeline:
    """Load the speaker diarization model.
    
    Requires HF_TOKEN environment variable to be set.
    First run will download the model (~1GB).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    print("Loading diarization model (this may take a minute)...")
    model = Pipeline.from_pretrained(
        MODEL,
        token=token
    )
    return model


def prepare_audio_for_diarization(audio_path: str) -> str:
    """Convert audio file to 16kHz mono WAV for pyannote compatibility.
    
    Returns path to temporary WAV file. Caller is responsible for cleanup.
    """
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # mono
        "-loglevel", "error",  # suppress ffmpeg output
        temp_wav.name
    ], check=True)
    
    return temp_wav.name


def diarize(audio_path: str, model: Pipeline = None, num_speakers: int = None) -> dict:
    """Run speaker diarization on an audio file.

    Args:
        audio_path: Path to audio file
        model: Optional pre-loaded diarization model (loads one if not provided)
        num_speakers: Optional hint for exact number of speakers (improves accuracy)

    Returns:
        Dict with '_schema_version', '_generator_version', and 'segments' keys.
        Segments is a list of dicts with 'start', 'end', 'speaker' keys.
    """
    if model is None:
        model = load_diarization_model()
    
    # Convert to 16kHz mono WAV for compatibility with pyannote
    wav_path = prepare_audio_for_diarization(audio_path)
    
    try:
        # Run diarization with progress feedback
        with ProgressHook() as hook:
            output = model(wav_path, hook=hook, num_speakers=num_speakers)
        
        # Extract segments using exclusive mode (one speaker at a time)
        # This simplifies alignment with transcription timestamps
        segments = []
        for turn, speaker in output.exclusive_speaker_diarization:
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        return {
            "_schema_version": _SCHEMA_VERSION,
            "_generator_version": _GENERATOR_VERSION,
            "segments": segments
        }
    finally:
        # Clean up temp file
        os.unlink(wav_path)


# ---------------------------------------------------------------------------
# Diarization enrichment — apply speaker labels to transcript words
# ---------------------------------------------------------------------------

_ENRICHED_SCHEMA_VERSION = "1.2.0"


def _compute_speaker_coverage(word_start, word_end, diar_segments):
    """Compute which speaker best covers a word's time range.

    For each diarization segment, calculates the temporal overlap with the
    word. The speaker with the greatest total overlap wins.

    Args:
        word_start: Word start time in seconds.
        word_end: Word end time in seconds.
        diar_segments: List of diarization segment dicts with start, end,
            and speaker keys.

    Returns:
        Dict with 'label' (speaker string or None) and 'coverage' (float
        0.0-1.0 indicating what fraction of the word duration is covered
        by the best-matching speaker).
    """
    word_duration = word_end - word_start
    if word_duration <= 0:
        return {"label": None, "coverage": 0.0}

    overlap_by_speaker = {}
    for seg in diar_segments:
        overlap = max(0, min(word_end, seg["end"]) - max(word_start, seg["start"]))
        if overlap > 0:
            speaker = seg["speaker"]
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0) + overlap

    if not overlap_by_speaker:
        return {"label": None, "coverage": 0.0}

    best_speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
    coverage = min(overlap_by_speaker[best_speaker] / word_duration, 1.0)

    return {"label": best_speaker, "coverage": coverage}


def enrich_with_diarization(transcript, diarization):
    """Add speaker labels to each word in a transcript using diarization data.

    Deep-copies the transcript, then assigns a _speaker dict to every word
    based on temporal overlap with diarization segments. Does not touch
    _processing or _schema_version -- the pipeline handles those.

    Args:
        transcript: Whisper transcript dict with segments containing words.
        diarization: Diarization result dict with a 'segments' list of
            {start, end, speaker} dicts.

    Returns:
        Tuple of (enriched_transcript, processing_entry).
        enriched_transcript: Deep-copied transcript with _speaker metadata.
        processing_entry: Dict with stage metadata.
    """
    result = copy.deepcopy(transcript)
    diar_segments = diarization.get("segments", [])

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            speaker_info = _compute_speaker_coverage(
                word["start"], word["end"], diar_segments
            )
            word["_speaker"] = speaker_info

    entry = {
        "stage": "diarization_enrichment",
        "model": MODEL,
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result, entry
