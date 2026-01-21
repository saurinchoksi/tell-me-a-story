"""Speaker diarization using pyannote.audio."""

import os
import subprocess
import tempfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# PyTorch 2.6+ changed weights_only default to True for security.
# pyannote.audio models need these classes allowlisted to load.
# This is safe because we're loading official pyannote models from Hugging Face.
from pyannote.audio.core.task import Specifications, Problem, Resolution, Scope
torch.serialization.add_safe_globals([Specifications, Problem, Resolution, Scope])


def load_pipeline() -> Pipeline:
    """Load the speaker diarization pipeline.
    
    Requires HF_TOKEN environment variable to be set.
    First run will download the model (~1GB).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token
    )
    return pipeline


def convert_to_wav_16k(audio_path: str) -> str:
    """Convert audio file to 16kHz mono WAV using ffmpeg.
    
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


def diarize(audio_path: str, pipeline: Pipeline = None) -> list[dict]:
    """Run speaker diarization on an audio file.
    
    Args:
        audio_path: Path to audio file
        pipeline: Optional pre-loaded pipeline (loads one if not provided)
    
    Returns:
        List of dicts with 'start', 'end', 'speaker' keys
    """
    if pipeline is None:
        pipeline = load_pipeline()
    
    # Convert to 16kHz mono WAV for compatibility with pyannote
    wav_path = convert_to_wav_16k(audio_path)
    
    try:
        # Run diarization with progress feedback
        with ProgressHook() as hook:
            output = pipeline(wav_path, hook=hook)
        
        # Extract segments using exclusive mode (one speaker at a time)
        # This simplifies alignment with transcription timestamps
        segments = []
        for turn, speaker in output.exclusive_speaker_diarization:
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        return segments
    finally:
        # Clean up temp file
        os.unlink(wav_path)


def print_diarization(segments: list[dict]) -> None:
    """Print diarization results in a readable format."""
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]
        print(f"[{start:6.1f} - {end:6.1f}] {speaker}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/diarize.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    print(f"Diarizing: {audio_file}")
    print("First run will download the model (~1GB)...")
    print()
    
    segments = diarize(audio_file)
    print_diarization(segments)
    print(f"\nFound {len(segments)} segments")
