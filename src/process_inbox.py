#!/usr/bin/env python3
"""Process all audio files in the inbox: init session → run pipeline → save artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from init_session import INBOX_DIR, SESSIONS_DIR, SUPPORTED_FORMATS, init_session
from pipeline import run_pipeline, save_computed


def process_inbox():
    audio_files = [f for f in INBOX_DIR.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]

    if not audio_files:
        print("Inbox is empty — nothing to process.")
        return

    print(f"Found {len(audio_files)} file(s) in inbox.\n")

    initialized = []
    skipped = []
    failed = []

    # Phase 1: init all sessions
    for audio_path in sorted(audio_files):
        print(f"Initializing: {audio_path.name}")
        try:
            result = init_session(audio_path)
            if result is None:
                skipped.append(audio_path.name)
            else:
                initialized.append(result)
        except Exception as e:
            print(f"  ✗ Init failed: {e}")
            failed.append((audio_path.name, str(e)))

    if not initialized:
        print("\nNo new sessions to process.")
        _print_summary([], skipped, failed)
        return

    # Phase 2: run pipeline on each initialized session
    created = []
    for session_id, audio_filename in initialized:
        audio_path = SESSIONS_DIR / session_id / audio_filename
        print(f"\nRunning pipeline: {session_id}")
        try:
            result = run_pipeline(str(audio_path))
            save_computed(
                str(SESSIONS_DIR / session_id),
                result["transcript_raw"],
                result["transcript"],
                result["diarization"],
            )
            created.append(session_id)
            print(f"  ✓ Done — {session_id}")
        except Exception as e:
            print(f"  ✗ Pipeline failed: {e}")
            failed.append((session_id, str(e)))

    _print_summary(created, skipped, failed)


def _print_summary(created, skipped, failed):
    print("\n" + "─" * 50)
    if created:
        print(f"\nCreated ({len(created)}):")
        for s in created:
            print(f"  ✓ {s}")
    if skipped:
        print(f"\nSkipped — duplicates ({len(skipped)}):")
        for name in skipped:
            print(f"  ~ {name}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  ✗ {name}: {err}")
    if not created and not skipped and not failed:
        print("\n  (nothing processed)")


if __name__ == "__main__":
    process_inbox()
