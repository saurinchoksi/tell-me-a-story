#!/usr/bin/env python3
"""Per-world name dictionaries — the human-blessed memory the world-grounded normalizer grows.

Each recognized story world (Mahabharata, Thomas & Friends, Steven Universe, ...) gets its
own file at ``data/worlds/<slug>.json``. A world starts EMPTY (missing file -> ``{}``) and
grows one entry at a time, only from a human confirming a correction (the bless loop). This
is the deliberate "first contact" stance decided with Choksi: no pre-seeded rosters, so the
system generalises to brand-new audio with brand-new worlds. ``data/mahabharata.json`` is
retired precisely because it pre-seeds — a world dict here holds only what a person blessed.

Schema mirrors the old reference library so ``dictionary.build_variant_map`` reads it verbatim::

    {"_world": "Mahabharata", "_version": 1, "entries": [
        {"canonical": "Pandavas", "variants": ["Bondos", "Fondos"], "aliases": []}, ...],
     "_blessings": [{"heard": "Bondos", "canonical": "Pandavas", "provenance": "...", "at": "..."}]}

Only ``variants`` is load-bearing for correction (``build_variant_map`` maps lowered variant ->
canonical, and a blessed variant then takes precedence over the phonetic guess forever after).
``_blessings`` is an append-only audit trail. Missing/empty file -> an empty variant map, so
Phase 1 (nothing blessed yet) auto-applies only genuine sound-alikes.
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dictionary import build_variant_map

# Repo-root/data/worlds — src/ is one level under the root.
WORLDS_DIR = Path(__file__).resolve().parents[1] / "data" / "worlds"


def slug(world: str) -> str:
    """Filesystem-safe slug for a world name. Lowercase, non-alphanumeric runs -> single
    hyphen, trimmed. "Thomas & Friends" -> "thomas-friends"; "" -> "" (caller guards)."""
    s = re.sub(r"[^a-z0-9]+", "-", world.strip().lower()).strip("-")
    return s


def world_dict_path(world: str, base: Path = WORLDS_DIR) -> Path:
    """Path to a world's dictionary file (may not exist)."""
    return Path(base) / f"{slug(world)}.json"


def load_world_dict(world: str, base: Path = WORLDS_DIR) -> dict:
    """The raw per-world dict, or ``{}`` when the world has no file yet (first contact).
    Fail-loud on a corrupt file — a malformed dict is a real bug, not an empty world."""
    path = world_dict_path(world, base)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_variant_map(world: str, base: Path = WORLDS_DIR) -> dict[str, str]:
    """The lowered-variant -> canonical map for one world (empty when nothing blessed)."""
    return build_variant_map(load_world_dict(world, base))


def load_world_dicts(worlds, base: Path = WORLDS_DIR) -> dict[str, dict[str, str]]:
    """{world -> its variant map} for a set of recognized worlds. The shape
    ``run_worldnorm`` consumes: it looks up each region's world to get its blessed variants."""
    return {w: load_variant_map(w, base) for w in worlds if w}


def _atomic_write(path: Path, obj: dict) -> None:
    """Write JSON to ``path`` atomically (temp file + os.replace), creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def bless(world: str, heard: str, canonical: str, provenance: str = "human",
          base: Path = WORLDS_DIR, now: str | None = None) -> dict:
    """Record a human-confirmed correction: ``heard`` is a variant of ``canonical`` in ``world``.

    Read-merge-write, atomic. Adds ``heard`` to the canonical's ``variants`` (deduped,
    case-insensitively) and appends to the ``_blessings`` audit trail. Idempotent: blessing the
    same pair twice adds the variant once but logs each bless. Returns the updated dict.

    A blessed variant becomes a deterministic dictionary entry that takes precedence over the
    phonetic guess (see ``worldnorm.classify``) — so a human's "Bondos means Pandavas" applies to
    every occurrence, and overrides a sound-alike collision (the Bushma->Bhishma case) forever.
    """
    if not world:
        raise ValueError("bless: world is required (an unrecognized world cannot be blessed)")
    if not heard.strip() or not canonical.strip():
        raise ValueError("bless: heard and canonical must be non-empty")
    data = load_world_dict(world, base) or {}
    data.setdefault("_world", world)
    data.setdefault("_version", 1)
    entries = data.setdefault("entries", [])

    entry = next((e for e in entries if e.get("canonical", "").lower() == canonical.lower()), None)
    if entry is None:
        entry = {"canonical": canonical, "variants": [], "aliases": []}
        entries.append(entry)
    variants = entry.setdefault("variants", [])
    if heard.lower() not in {v.lower() for v in variants} and heard.lower() != canonical.lower():
        variants.append(heard)

    data.setdefault("_blessings", []).append({
        "heard": heard, "canonical": canonical, "provenance": provenance,
        "at": now or datetime.now(timezone.utc).isoformat(),
    })
    _atomic_write(world_dict_path(world, base), data)
    return data
