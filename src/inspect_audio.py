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


def print_audio_info(filepath: str) -> None:
    """Print basic properties of an audio file."""

    if info is None:
        print(f"Could not read audio file: {filepath}")
        return

    print(f"File: {path.name}")
    print(f"Format: {type(audio).__name__}")
    print(f"Duration: {audio.info.length:.1f} seconds")
    print(f"Sample rate: {audio.info.sample_rate} Hz")
    print(f"Channels: {audio.info.channels}")


if __name__ == "__main__":
    inspect_audio("data/New Recording 63.m4a")
