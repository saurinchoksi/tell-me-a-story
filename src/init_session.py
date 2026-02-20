"""Initialize session folders from inbox audio files."""

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()  # src/
PROJECT_DIR = SCRIPT_DIR.parent               # tell-me-a-story/
SESSIONS_DIR = PROJECT_DIR / "sessions"
INBOX_DIR = PROJECT_DIR / "inbox"
DUPLICATES_DIR = INBOX_DIR / "duplicates"
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


def file_hash(filepath: Path) -> str:
    """Compute MD5 hash of file contents."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def init_session(audio_path: Path) -> tuple[str, str] | None:
    """Move audio file to new session folder.

    Returns (session_id, audio_filename) on success, None if skipped.
    """
    creation_time = get_creation_time(audio_path)
    session_id = generate_session_id(creation_time)
    session_dir = SESSIONS_DIR / session_id

    if session_dir.exists():
        # Find existing audio file in session
        existing_audio = None
        for ext in SUPPORTED_FORMATS:
            candidate = session_dir / f"audio{ext}"
            if candidate.exists():
                existing_audio = candidate
                break

        if existing_audio and file_hash(audio_path) == file_hash(existing_audio):
            # True duplicate - move to duplicates folder
            DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)
            dup_dest = DUPLICATES_DIR / f"{session_id}_{audio_path.name}"
            shutil.move(audio_path, dup_dest)
            print(f"  {audio_path.name} → DUPLICATE of session {session_id} (moved to duplicates/)")
            return None
        else:
            # Collision - different files with same timestamp
            print(f"  {audio_path.name} → COLLISION: session {session_id} exists with different content")
            print(f"              Manual resolution required - file left in inbox")
            return None

    session_dir.mkdir(parents=True)
    dest_path = session_dir / f"audio{audio_path.suffix.lower()}"
    try:
        shutil.move(audio_path, dest_path)
    except Exception:
        session_dir.rmdir()  # Clean up the empty folder
        raise  # Re-raise so caller knows it failed
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
