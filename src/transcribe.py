"""Transcribe an audio file using MLX Whisper."""

import mlx_whisper

_SCHEMA_VERSION = "1.0.0"
_GENERATOR_VERSION = "mlx-whisper-1.0"


def transcribe(audio_path: str, model: str = None) -> dict:
    """Transcribe audio file and return result dict.

    Args:
        audio_path: Path to audio file
        model: Optional model path/repo

    Returns:
        Dict with 'text', 'language', 'segments' keys.
        Each segment has a 'words' list with word-level timestamps.
    """
    kwargs = {"word_timestamps": True}

    if model:
        kwargs["path_or_hf_repo"] = model

    result = mlx_whisper.transcribe(audio_path, **kwargs)
    return result


def clean_transcript(transcript: dict) -> dict:
    """Remove garbage from transcript.

    Removes:
    - Zero-duration words (end == start) - fabrications that never existed
    - Empty segments (no text, no words, or zero duration)

    This is garbage removal, not quality filtering. Quality filtering
    (probability thresholds) happens at query time.

    Args:
        transcript: Raw transcript dict from transcribe()

    Returns:
        New transcript dict with garbage removed.
    """
    cleaned_segments = []
    for seg in transcript.get("segments", []):
        # Skip empty/garbage segments
        text = seg.get("text", "").strip()
        has_duration = seg.get("end", 0) > seg.get("start", 0)
        if not text or not has_duration:
            continue

        words = seg.get("words", [])
        # Keep only words with positive duration
        cleaned_words = [w for w in words if w.get("end", 0) > w.get("start", 0)]

        cleaned_seg = seg.copy()
        cleaned_seg["words"] = cleaned_words
        cleaned_segments.append(cleaned_seg)

    result = transcript.copy()
    result["segments"] = cleaned_segments
    result["_schema_version"] = _SCHEMA_VERSION
    result["_generator_version"] = _GENERATOR_VERSION
    return result
