"""Step-through normalization validation.

Run both normalization passes on an existing transcript,
pausing between each step to inspect the results.

Usage:
    cd src && python validate_normalization.py
"""

import json
import sys
from pathlib import Path

from corrections import extract_text, apply_corrections
from normalize import normalize as llm_normalize
from dictionary import load_library, build_variant_map, normalize_variants

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sessions" / "00000000-000000" / "transcript.json"
LIBRARY_PATH = Path(__file__).parent.parent / "data" / "mahabharata.json"
LLM_MODEL = "qwen3:8b"


def pause(msg="Press Enter to continue..."):
    input(f"\n{msg}")
    print()


def print_corrections(corrections: list[dict], label: str):
    """Print a corrections list in a readable format."""
    if not corrections:
        print(f"  (no corrections from {label})")
        return
    for c in corrections:
        print(f'  "{c["transcribed"]}" → "{c["correct"]}"')


def print_changed_words(original: dict, corrected: dict):
    """Compare two transcripts and print words that changed."""
    changes = []
    for seg_o, seg_c in zip(original["segments"], corrected["segments"]):
        for word_o, word_c in zip(seg_o["words"], seg_c["words"]):
            if word_o["word"] != word_c["word"]:
                changes.append({
                    "original": word_c.get("_original", word_o["word"].strip()),
                    "corrected": word_c["word"].strip(),
                    "corrections": word_c.get("_corrections", []),
                    "start": word_c.get("start", "?"),
                })
    if not changes:
        print("  (no words changed)")
        return
    for ch in changes:
        chain = " → ".join(
            [ch["original"]] + [c["to"] for c in ch["corrections"]]
        )
        print(f"  [{ch['start']:.1f}s] {chain}")
    print(f"\n  Total: {len(changes)} words corrected")


def main():
    # Load transcript
    print("=" * 60)
    print("STEP 1: Load existing transcript")
    print("=" * 60)

    with open(TRANSCRIPT_PATH) as f:
        transcript = json.load(f)

    text = extract_text(transcript)
    print(f"\nTranscript loaded: {len(transcript['segments'])} segments")
    print(f"\n--- Raw text ---\n")
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... ({len(text)} chars total)")

    pause("Ready to run LLM normalization? Press Enter...")

    # LLM pass
    print("=" * 60)
    print(f"STEP 2: LLM normalization ({LLM_MODEL})")
    print("=" * 60)
    print(f"\nCalling Ollama with {LLM_MODEL}... (may take a minute)")

    llm_corrections = llm_normalize(text, model=LLM_MODEL)

    print(f"\nLLM returned {len(llm_corrections)} corrections:\n")
    print_corrections(llm_corrections, "LLM")

    pause("Review the corrections above. Ready to apply them? Press Enter...")

    # Apply LLM corrections
    print("=" * 60)
    print("STEP 3: Apply LLM corrections")
    print("=" * 60)

    transcript_after_llm, llm_count = apply_corrections(transcript, llm_corrections, "llm")
    print(f"\n{llm_count} words changed in transcript:\n")
    print_changed_words(transcript, transcript_after_llm)

    pause("Ready to run dictionary normalization? Press Enter...")

    # Dictionary pass
    print("=" * 60)
    print("STEP 4: Dictionary normalization")
    print("=" * 60)

    library = load_library(str(LIBRARY_PATH))
    variant_map = build_variant_map(library)
    print(f"\nLoaded {len(variant_map)} variants from library")

    text_after_llm = extract_text(transcript_after_llm)
    dict_corrections = normalize_variants(text_after_llm, variant_map)

    print(f"\nDictionary found {len(dict_corrections)} corrections:\n")
    print_corrections(dict_corrections, "dictionary")

    pause("Review the corrections above. Ready to apply them? Press Enter...")

    # Apply dictionary corrections
    print("=" * 60)
    print("STEP 5: Apply dictionary corrections"
    )
    print("=" * 60)

    transcript_final, dict_count = apply_corrections(transcript_after_llm, dict_corrections, "dictionary")
    print(f"\n{dict_count} words changed in transcript:\n")
    print_changed_words(transcript_after_llm, transcript_final)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY: Full correction chains")
    print("=" * 60)
    print()

    # Find words with multi-stage corrections
    multi_stage = []
    single_stage = []
    for seg in transcript_final["segments"]:
        for word in seg["words"]:
            if "_corrections" in word:
                if len(word["_corrections"]) > 1:
                    multi_stage.append(word)
                else:
                    single_stage.append(word)

    if multi_stage:
        print("Words corrected by BOTH passes:")
        for word in multi_stage:
            chain = " → ".join(
                [word["_original"]] + [c["to"] for c in word["_corrections"]]
            )
            print(f"  {chain}")

    if single_stage:
        print(f"\nWords corrected by ONE pass only:")
        for word in single_stage:
            c = word["_corrections"][0]
            print(f"  [{c['stage']}] {word['_original']} → {c['to']}")

    print(f"\n--- Final corrected text ---\n")
    final_text = extract_text(transcript_final)
    print(final_text[:2000])
    if len(final_text) > 2000:
        print(f"\n... ({len(final_text)} chars total)")

    # Save for inspection
    output_path = Path(__file__).parent.parent / "sessions" / "00000000-000000" / "transcript_normalized.json"
    with open(output_path, "w") as f:
        json.dump(transcript_final, f, indent=2)
    print(f"\nSaved normalized transcript to: {output_path}")


if __name__ == "__main__":
    main()
