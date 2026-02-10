"""
Inspect audio file and print its properties.
First script for Tell Me A Story - just proving we can work with audio.
"""

from pathlib import Path
from mutagen import File


def get_audio_info(filepath: str) -> dict | None:
    """Return basic properties of an audio file, or None if unreadable."""
    path = Path(filepath)

    if not path.exists():
        return None

    audio = File(path)

    if audio is None:
        return None

    return {
        "filename": path.name,
        "format": type(audio).__name__,
        "duration_seconds": audio.info.length,
        "sample_rate": audio.info.sample_rate,
        "channels": audio.info.channels,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/inspect_audio.py <audio_file>")
        sys.exit(1)

    info = get_audio_info(sys.argv[1])
    if info is None:
        print(f"Could not read audio file: {sys.argv[1]}")
        sys.exit(1)

    for key, value in info.items():
        print(f"{key}: {value}")
