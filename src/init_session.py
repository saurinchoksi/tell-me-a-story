"""Initialize session folders from inbox audio files."""

import shutil
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()  # src/
PROJECT_DIR = SCRIPT_DIR.parent               # tell-me-a-story/
SESSIONS_DIR = PROJECT_DIR / "sessions"
INBOX_DIR = SESSIONS_DIR / "inbox"
SUPPORTED_FORMATS = {".m4a", ".mp3", ".wav"}


def get_creation_time(filepath: Path) -> datetime:
    """Get file creation time.

    Uses st_birthtime on macOS, falls back to st_mtime (modification time)
    on Linux and other platforms where birth time is not available.
    """
    stat = filepath.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    return datetime.fromtimestamp(timestamp)


def generate_session_id(dt: datetime) -> str:
    """Format datetime as YYYYMMDD-HHMMSS."""
    return dt.strftime("%Y%m%d-%H%M%S")


def init_session(audio_path: Path) -> tuple[str, str] | None:
    """Move audio file to new session folder.

    Returns (session_id, audio_filename) on success, None if skipped.
    """
    creation_time = get_creation_time(audio_path)
    session_id = generate_session_id(creation_time)
    session_dir = SESSIONS_DIR / session_id

    if session_dir.exists():
        print(f"  {audio_path.name} → SKIPPED (session {session_id} already exists)")
        return None

    session_dir.mkdir(parents=True)
    dest_path = session_dir / f"audio{audio_path.suffix.lower()}"
    shutil.move(audio_path, dest_path)
    print(f"  {audio_path.name} → {session_id}/{dest_path.name}")
    return session_id, dest_path.name


def main():
    """Process all audio files in inbox."""
    if not INBOX_DIR.exists():
        INBOX_DIR.mkdir(parents=True)
        print(f"Created inbox directory: {INBOX_DIR}")
        print("Drop audio files here and run again.")
        return

    audio_files = [
        f for f in INBOX_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not audio_files:
        print(f"No audio files found in {INBOX_DIR}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return

    print("Processing inbox...")
    initialized = []
    skipped = 0

    for audio_path in sorted(audio_files):
        result = init_session(audio_path)
        if result:
            initialized.append(result)
        else:
            skipped += 1

    print()
    print(f"Initialized {len(initialized)} sessions. Skipped {skipped}.")

    if initialized:
        print("Ready for pipeline:")
        for session_id, audio_filename in initialized:
            print(f"  python src/pipeline.py sessions/{session_id}/{audio_filename}")


if __name__ == "__main__":
    main()
