"""Run the full transcription and alignment pipeline."""

import sys
from transcribe import transcribe
from diarize import diarize
from align import align, format_transcript, group_words_by_speaker


def extract_words(transcript: dict) -> list[dict]:
    """Extract all words from transcript segments.
    
    Args:
        transcript: Result from transcribe() with word_timestamps=True
    
    Returns:
        Flat list of word dicts with 'start', 'end', 'word' keys.
    """
    all_words = []
    for seg in transcript["segments"]:
        if seg.get("words"):
            all_words.extend(seg["words"])
    return all_words


def run_pipeline(audio_path: str, verbose: bool = True) -> dict:
    """Run full pipeline on an audio file.
    
    Args:
        audio_path: Path to audio file
        verbose: Print progress messages
    
    Returns:
        Dict with 'transcript', 'diarization', 'words', 'utterances' keys.
    """
    if verbose:
        print(f"Transcribing: {audio_path}")
        print("Using large model for better child speech recognition...")
    
    transcript = transcribe(
        audio_path,
        word_timestamps=True,
        model="mlx-community/whisper-large-v3-turbo"
    )
    
    if verbose:
        print("\nDiarizing (this takes a few minutes)...")
    
    segments = diarize(audio_path)
    
    # Extract words from transcript
    words = extract_words(transcript)
    
    if verbose:
        print(f"\nAligning {len(words)} words to {len(segments)} speaker segments...")
    
    # Run alignment pipeline
    utterances = align(words, segments)
    
    return {
        "transcript": transcript,
        "diarization": segments,
        "words": words,
        "utterances": utterances
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/pipeline.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    result = run_pipeline(audio_path)
    
    print("\n--- Speaker-labeled transcript ---\n")
    print(format_transcript(result["utterances"]))
