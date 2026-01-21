"""Transcribe an audio file using MLX Wisper."""

import mlx_whisper


def transcribe(audio_path: str, word_timestamps: bool = False, model: str = None) -> dict:
    """Transcribe audio file and return result dict.
    
    Args:
        audio_path: Path to audio file
        word_timestamps: If True, include word-level timestamps in segments
    
    Returns:
        Dict with 'text', 'language', 'segments' keys.
        If word_timestamps=True, each segment also has 'words' list.
    """
    
    kwargs = {"word_timestamps": word_timestamps}

    if model:
        kwargs["path_or_hf_repo"] = model

    result = mlx_whisper.transcribe(audio_path, **kwargs)
    return result

def save_transcript(result: dict, output_path: str) -> None:
    """Save transcript to a text file."""
    with open(output_path, "w") as f:
        f.write("--- TRANSCRIPT ---\n")
        f.write(result["text"].strip())
        f.write("\n\n--- SEGMENTS ---\n")
        for seg in result["segments"]:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]
            f.write(f"[{start:6.1f} - {end:6.1f}] {text}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print ("Usage: python src/transcribe.py <audio_file>")
        sys.exit

    audio_file = sys.argv[1]
    print(f"Transcribing: {audio_file}")
    print("This may take a moment on first run (downloading model)...")

    result = transcribe(audio_file)

    print("\n--- TRANSCRIPT ---")
    print(result["text"])

    print("\n--- SEGMENTS ---")
    for seg in result["segments"]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        print(f"[{start:6.1f} - {end:6.1f}] {text}")

    output_file = audio_file.rsplit(".", 1)[0] + ".txt"
    save_transcript(result, output_file)
    print(f"\nSaved to: {output_file}")
