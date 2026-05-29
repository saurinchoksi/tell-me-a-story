"""Add Mode 9 axial labels for a session's known name variants.

Walks transcript-rich.json, finds words matching the given variant list, and
adds an M9 axial label to each parent segment. Always merges additively with
any existing axial-labels.json — never destructive.

Per-segment merge rules:
  - no entry yet                     → create new entry with codes=["M9"]
  - already has M9                   → no change
  - codes == ["NotA"] (NotA-only)    → replace with codes=["M9"]
                                       (NotA was wrong; segment is not clean)
  - NotA + other codes (defensive)   → drop NotA, append M9
  - failure codes only, no M9        → append M9 (multi-tag)

Backs up the existing axial-labels.json to a timestamped copy before any write.
Sanity-checks invariants (no M1–M8 count change, no segmentId dropped, M9 and
entry counts strictly non-decreasing) and aborts if anything looks off.

Examples:
    python emp/src/populate_mode9.py 20260117-202237 \\
        Duryodhan Yudhisthir Pondavas Dhrashtra

    python emp/src/populate_mode9.py 20251210-203654 \\
        Artie "Artie's" Arthie "Arthie's"
"""

import argparse
import collections
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.helpers import get_session_dir  # noqa: E402

WORD_RE = re.compile(r"\W*([\w\-'’]+)\W*")


def clean(word: str) -> str:
    m = WORD_RE.fullmatch(word.strip())
    return m.group(1) if m else word.strip()


def code_freq(labels: list[dict]) -> collections.Counter:
    f: collections.Counter = collections.Counter()
    for L in labels:
        for c in L.get("codes", []):
            f[c] += 1
    return f


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Mode 9 axial labels for a session's name variants (merge-only, never destructive).",
    )
    parser.add_argument("session_id", help="Session ID (YYYYMMDD-HHMMSS)")
    parser.add_argument(
        "variants",
        nargs="+",
        help="Name variants to tag as M9 / #2 instances",
    )
    args = parser.parse_args()

    variants = set(args.variants)
    session_dir = get_session_dir(ROOT / "sessions", args.session_id)
    transcript_path = session_dir / "transcript-rich.json"
    labels_path = session_dir / "axial-labels.json"

    # Load existing labels (or start empty if file doesn't exist).
    existing_labels: list[dict] = []
    if labels_path.exists():
        with open(labels_path) as f:
            existing_labels = json.load(f).get("labels", [])

    pre_freq = code_freq(existing_labels)
    pre_entry_count = len(existing_labels)
    pre_segment_ids = {L["segmentId"] for L in existing_labels}

    print(f"Session:               {args.session_id}")
    print(f"Variants targeted:     {sorted(variants)}")
    print(f"Pre-merge entries:     {pre_entry_count}")
    print(f"Pre-merge code freq:   {dict(sorted(pre_freq.items()))}")

    # Scan transcript for segments containing any variant token.
    with open(transcript_path) as f:
        tx = json.load(f)

    seg_to_variants: dict[int | str, list[str]] = {}
    for seg in tx["segments"]:
        if seg.get("_source") == "diarization_gap":
            continue
        sid = seg["id"]
        for w in seg.get("words", []):
            cleaned = clean(w["word"])
            if cleaned in variants:
                seg_to_variants.setdefault(sid, []).append(cleaned)

    # Build merged labels in-memory.
    by_id: dict = {L["segmentId"]: dict(L) for L in existing_labels}  # shallow copies
    now = datetime.now(timezone.utc).isoformat()
    stats = {"new": 0, "nota_replaced": 0, "multitag_added": 0, "already_m9": 0}

    for sid in seg_to_variants:
        entry = by_id.get(sid)
        if entry is None:
            by_id[sid] = {
                "segmentId": sid,
                "codes": ["M9"],
                "createdAt": now,
                "updatedAt": now,
            }
            stats["new"] += 1
            continue

        codes = list(entry.get("codes", []))
        if "M9" in codes:
            stats["already_m9"] += 1
            continue

        if codes == ["NotA"]:
            entry["codes"] = ["M9"]
            entry["updatedAt"] = now
            stats["nota_replaced"] += 1
            continue

        # Defensive: NotA co-existing with other codes — drop NotA.
        if "NotA" in codes:
            codes = [c for c in codes if c != "NotA"]
        codes.append("M9")
        entry["codes"] = codes
        entry["updatedAt"] = now
        stats["multitag_added"] += 1

    merged_labels = list(by_id.values())
    post_freq = code_freq(merged_labels)
    post_segment_ids = {L["segmentId"] for L in merged_labels}

    print(f"Post-merge entries:    {len(merged_labels)}")
    print(f"Post-merge code freq:  {dict(sorted(post_freq.items()))}")
    print("Categories:")
    print(f"  new (no prior label):       {stats['new']}")
    print(f"  NotA replaced with M9:      {stats['nota_replaced']}")
    print(f"  M9 added as multi-tag:      {stats['multitag_added']}")
    print(f"  already had M9 (no change): {stats['already_m9']}")

    # Sanity-check invariants before any write.
    failures = []
    for code in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"):
        if post_freq.get(code, 0) != pre_freq.get(code, 0):
            failures.append(
                f"{code} count changed: {pre_freq.get(code, 0)} -> {post_freq.get(code, 0)}"
            )
    expected_nota = pre_freq.get("NotA", 0) - stats["nota_replaced"]
    if post_freq.get("NotA", 0) != expected_nota:
        failures.append(
            f"NotA count off: expected {expected_nota}, got {post_freq.get('NotA', 0)}"
        )
    if post_freq.get("M9", 0) < pre_freq.get("M9", 0):
        failures.append(
            f"M9 count decreased: {pre_freq.get('M9', 0)} -> {post_freq.get('M9', 0)}"
        )
    if len(merged_labels) < pre_entry_count:
        failures.append(
            f"Entry count decreased: {pre_entry_count} -> {len(merged_labels)}"
        )
    missing = pre_segment_ids - post_segment_ids
    if missing:
        sample = sorted(missing, key=lambda x: str(x))[:5]
        failures.append(f"Pre-merge segmentIds dropped: {sample}...")

    if failures:
        print("\nINVARIANT CHECK FAILED — aborting write:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        sys.exit(2)

    # Backup before write (only if a labels file exists).
    if labels_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = labels_path.parent / f"axial-labels.json.pre-m9-merge-{stamp}"
        shutil.copy2(labels_path, backup)
        if backup.stat().st_size != labels_path.stat().st_size:
            print(
                f"Backup size mismatch — aborting write. Backup at {backup}",
                file=sys.stderr,
            )
            sys.exit(3)
        print(f"\nBacked up to: {backup}")

    # Write merged file.
    with open(labels_path, "w") as f:
        json.dump({"labels": merged_labels}, f, indent=2)

    total_instances = sum(len(v) for v in seg_to_variants.values())
    observed = sorted({v for vs in seg_to_variants.values() for v in vs})
    print(f"\nVariants observed:       {observed}")
    print(f"Total variant instances: {total_instances}")
    print(f"Wrote: {labels_path}")


if __name__ == "__main__":
    main()
