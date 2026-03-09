#!/usr/bin/env python3
"""Process all audio files in the inbox: init session → run pipeline → save artifacts."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from init_session import INBOX_DIR, SESSIONS_DIR, SUPPORTED_FORMATS, init_session
from pipeline import run_pipeline, save_computed
from embeddings import save_embeddings


def process_inbox(target_file: str | None = None):
    if target_file:
        path = INBOX_DIR / target_file
        if not path.exists():
            raise FileNotFoundError(f"File not found in inbox: {target_file}")
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        audio_files = [path]
    else:
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
            if result.get("embeddings") is not None:
                save_embeddings(
                    result["embeddings"],
                    str(SESSIONS_DIR / session_id / "embeddings.json"),
                )

                # Auto-identify against existing profiles
                from profiles import load_profiles
                from identify import identify_speakers, save_identifications
                profiles = load_profiles()
                if profiles.get("profiles"):
                    identifications = identify_speakers(
                        str(SESSIONS_DIR / session_id / "embeddings.json")
                    )
                    save_identifications(
                        identifications,
                        str(SESSIONS_DIR / session_id / "identifications.json"),
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
    parser = argparse.ArgumentParser(description="Process inbox audio files.")
    parser.add_argument(
        "--file",
        help="Process only this filename from the inbox (e.g. 'New Recording 37.m4a')",
    )
    args = parser.parse_args()
    process_inbox(target_file=args.file)
