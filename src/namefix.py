#!/usr/bin/env python3
"""namefix — the audio-grounded, gated name-correction stage (the normalizer's replacement).

The removed world-blind normalizer guessed "correct" spellings from garbled text and rewrote
the child's word for the heroes as an enemy's name. This stage grounds correction in the
AUDIO instead, validated end-to-end on 2026-07-01 (emp/emp.md; held-out Mahabharata: 32
auto-corrections / 0 wrong / 4 queued vs the old ~43 with 13 catastrophic):

  1. NAME CANDIDATES from the blind transcript (the M9c card machinery). Not a candidate ->
     never touched.
  2. Per story, recognize the WORLD from the name list (abstains on invented/unknown worlds ->
     the stage does nothing there; an invented name has no canon to correct toward).
  3. Generate the world's correction CAST — characters AND groups, split prompts (worldcast).
  4. Re-transcribe a short clip around each candidate with the world+cast in Whisper's ear
     (a single upfront prompt decays over a long file; per-occurrence re-prompting doesn't).
  5. The v4 ACCEPTANCE GATE, each rule bought by a measured failure:
       - judge ONLY the nearest non-ordinary re-decoded word at the target's time offset
         (a neighboring name in the same clip is a different word position);
       - an ordinary-word re-decode never maps to a cast name (father->Vidura leak);
       - a BLIND token that is a real dictionary word is NEVER auto-overwritten (Choksi's
         ear-check: both wrong autos were the real word "arrows") — it queues;
       - otherwise: exact/sound match into the cast -> AUTO; no match -> untouched.
     Auto-fixes apply via corrections.apply_corrections (in-place token rewrites — segment ids
     and word counts preserved by construction, so axial labels stay bound).
  6. Everything caught-but-not-bulletproof -> sessions/<id>/pending-name-corrections.json,
     the human bless queue (worlddict.bless makes a blessing deterministic forever after).

Two SEQUENTIAL model subprocesses (model_runner.run_model; one model at a time): worker A
loads Qwen3.5 (world + cast per story), worker B loads Whisper (clip re-decodes + gate).
Blessed per-world dictionary entries (worlddict) auto-apply ahead of the sound-match.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from detectors.phonetics import clean, codes
from detectors.story_names._audit import story_name_cards, _is_ordinary_word, _NCD
from detectors.story_names._names import story_segments, proper_name_candidates
from detectors.story_names._worker import build_regions

MODEL = "mlx-community/whisper-large-v3-mlx"
LEAD, WINDOW = 1.5, 2.0     # the E4/E5-tuned clip: [start-LEAD, start+WINDOW]
TEMPORAL_TOL = 0.7          # max offset drift between blind word time and its re-decode
VERSION = "1.0.0"


def config_fingerprint() -> str:
    """Ties a pending file / processing entry to the exact chain config that produced it."""
    import worldcast
    blob = (worldcast.CHARACTERS_PROMPT + worldcast.GROUPS_PROMPT +
            f"{MODEL}|{LEAD}|{WINDOW}|{TEMPORAL_TOL}|{VERSION}")
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def transcript_fingerprint(transcript: dict) -> str:
    """Word-level fingerprint (mirrors the detectors' convention): a transcript edit stales
    the pending file so the UI can prompt a re-run instead of showing drifted positions."""
    parts = []
    for s in transcript.get("segments", []):
        for w in s.get("words", []):
            parts.append(w.get("word", ""))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


# --------------------------- stage 1: candidates (pure) ---------------------------
def candidate_positions(transcript: dict):
    """Per-story candidate occurrences + per-story name surfaces + the capitalized-in-story
    guard set. Returns (regions_out, singles) where regions_out[i] = {region, names, cands}."""
    seg_by_id = {s["id"]: s for s in transcript["segments"]}
    pos_of = {s["id"]: i for i, s in enumerate(transcript["segments"])}
    singles = set()
    out = []
    for r in build_regions(transcript.get("_stories") or [], pos_of):
        segs = story_segments(transcript, r, pos_of)
        cards = story_name_cards(segs, recover=True)
        singles |= proper_name_candidates(segs)
        names = sorted({s for c in cards for s in c["surface"]})
        cands, seen = [], set()
        for card in cards:
            for o in card["occ"]:
                seg = seg_by_id.get(o["seg_id"])
                words = (seg or {}).get("words") or []
                if o["wi"] >= len(words) or words[o["wi"]].get("start") is None:
                    continue
                k = (o["seg_id"], o["wi"])
                if k in seen:
                    continue
                seen.add(k)
                w = words[o["wi"]]
                cands.append({"story": r["idx"], "start": float(w["start"]),
                              "blind": w["word"].strip(), "seg_id": o["seg_id"], "wi": o["wi"]})
        out.append({"region": r, "names": names, "cands": cands})
    return out, singles


# --------------------------- stage 2+3: worlds + casts (Qwen worker) ---------------------------
def _worlds_and_casts_worker(region_names: list[dict]) -> list[dict]:
    """Subprocess worker A: ONE Qwen3.5 load; per story recognize the world from its name
    list and generate the split cast. Picklable in/out."""
    from qwen35 import make_reader
    from detectors.story_names import _qwen35
    import worldcast
    gen = make_reader()
    out = []
    for rn in region_names:
        world = _qwen35.recognize_world(gen, rn["names"])
        split = worldcast.cached_cast_split(gen, world) if world else {"characters": [], "groups": []}
        out.append({"story": rn["story"], "world": world, "split": split})
    return out


def build_prompt(world: str, cast: list[str]) -> str:
    """The validated appositive prompt shape: groups lead, then the characters."""
    groups, rest = cast[:6], cast[6:20]
    return (f"This is a bedtime story about the {world}. Its characters include "
            + ", ".join(groups) + (", and " + ", ".join(rest) if rest else "") + ".")


# --------------------------- stage 4+5: re-decode + gate (Whisper worker) ---------------------------
def gate_decision(near_words, blind: str, cast_clean: dict, singles: set,
                  variant_map: dict) -> dict | None:
    """The v4 gate on one candidate. `near_words` = [(word, offset)] within TEMPORAL_TOL of
    the expected clip offset, sorted nearest-first. Returns None (untouched) or a decision
    {canonical, redecoded, how, action}. Pure — unit-testable without any model."""
    blind_c = clean(blind)
    # blessed dictionary entry wins outright (deterministic human knowledge)
    if blind_c in variant_map:
        return {"canonical": variant_map[blind_c], "redecoded": blind,
                "how": "dictionary", "action": "auto"}
    # Target = nearest re-decoded word that is either name-shaped (not an ordinary word) or
    # an EXACT cast spelling. The cast exemption matters because Webster's-1934 contains some
    # canon names verbatim ("Kauravas" is a real entry!) — an exact cast member is the very
    # vocabulary we primed Whisper with, never "ordinary" here. Sound-alikes get no exemption,
    # which keeps the father(FTR)->Vidura(FTR) leak closed.
    target = next(((w, off) for w, off in near_words
                   if clean(w) and (clean(w) in cast_clean
                                    or not _is_ordinary_word(clean(w), singles))), None)
    if target is None:
        return None
    wc = clean(target[0])
    if not wc or wc == blind_c:
        return None
    if wc in cast_clean:
        canonical, how = cast_clean[wc], "exact"
    else:
        canonical = next((cast_clean[cc] for cc in cast_clean if codes(wc) & codes(cc)), None)
        how = "dm"
        if canonical is None:
            return None
    # Choksi's rule: a real dictionary word in the transcript is never auto-overwritten.
    action = "queued" if _NCD._is_common(blind_c) else "auto"
    return {"canonical": canonical, "redecoded": target[0], "how": how, "action": action}


def _redecode_worker(audio_path: str, jobs: list[dict], singles: list[str],
                     variant_maps: dict) -> list[dict]:
    """Subprocess worker B: ONE Whisper load; per candidate cut the in-memory clip, re-decode
    with the story's world+cast prompt, run the gate. Picklable in/out."""
    import numpy as np
    import mlx_whisper
    from mlx_whisper.audio import SAMPLE_RATE, load_audio
    audio = np.array(load_audio(audio_path)).astype(np.float32)
    singles = set(singles)
    decisions = []
    for j in jobs:
        p = j["cand"]
        i0 = max(0, int((p["start"] - LEAD) * SAMPLE_RATE))
        i1 = int((p["start"] + WINDOW) * SAMPLE_RATE)
        r = mlx_whisper.transcribe(audio[i0:i1], path_or_hf_repo=MODEL, language="en",
                                   word_timestamps=True, initial_prompt=j["prompt"],
                                   condition_on_previous_text=True, verbose=False)
        words = [((w.get("word") or "").strip(), float(w["start"]))
                 for s in r.get("segments", []) for w in s.get("words", [])
                 if w.get("start") is not None]
        expected = min(p["start"], LEAD)
        near = sorted(((w, off) for w, off in words if abs(off - expected) <= TEMPORAL_TOL),
                      key=lambda t: abs(t[1] - expected))
        d = gate_decision(near, p["blind"], j["cast_clean"], singles,
                          variant_maps.get(j["world"], {}))
        if d:
            decisions.append({**p, **d, "world": j["world"]})
    return decisions


# --------------------------- orchestration ---------------------------
def group_decisions(decisions: list[dict]) -> dict:
    """Fold per-occurrence decisions into per-(story, heard) groups — the worldnorm Phase-1
    shape the bless queue and worlddict already speak."""
    groups = {}
    for d in decisions:
        key = (d["story"], clean(d["blind"]), d["action"])
        g = groups.get(key)
        if g is None:
            g = {"world": d["world"], "story_id": d["story"], "heard": d["blind"].strip(" .,?!"),
                 "heard_cleaned": clean(d["blind"]), "suggestion": d["canonical"],
                 "canonical": d["canonical"], "method": d["how"], "action": d["action"],
                 "occurrences": []}
            groups[key] = g
        g["occurrences"].append({"segment_id": d["seg_id"], "word_index": d["wi"],
                                 "start": d["start"], "token": d["blind"]})
    auto = [g for g in groups.values() if g["action"] == "auto"]
    pending = [g for g in groups.values() if g["action"] == "queued"]
    return {"auto": auto, "pending": pending}


def auto_to_corrections(auto: list[dict]) -> list[dict]:
    """apply_corrections input from the auto groups (first canonical wins a collision)."""
    seen = {}
    for g in auto:
        t = g["heard_cleaned"]
        if t and t not in seen:
            seen[t] = {"transcribed": t, "correct": g["canonical"]}
    return list(seen.values())


def run_namefix(transcript: dict, audio_path: str, worlds_and_casts=None,
                redecode=None, timeout: int = 3600) -> dict:
    """The full stage. Returns {auto, pending, worlds}. `worlds_and_casts` and `redecode`
    are injectable for tests (fakes); production runs the two model subprocesses."""
    import worldcast
    import worlddict
    from model_runner import run_model

    regions, singles = candidate_positions(transcript)
    region_names = [{"story": r["region"]["idx"], "names": r["names"]} for r in regions]

    if worlds_and_casts is None:
        worlds_and_casts = run_model(_worlds_and_casts_worker, region_names, timeout=timeout)
    wac_by_story = {w["story"]: w for w in worlds_and_casts}

    jobs, worlds_out, variant_maps = [], [], {}
    for r in regions:
        sid = r["region"]["idx"]
        wac = wac_by_story.get(sid, {"world": "", "split": {"characters": [], "groups": []}})
        world = wac["world"]
        worlds_out.append({"story_id": sid, "title": r["region"].get("title", ""),
                           "saved_world": r["region"].get("world", ""),
                           "recognized_world": world, "n_names": len(r["names"]),
                           "n_candidates": len(r["cands"])})
        if not world:
            continue  # unrecognized/invented world -> the stage does nothing here
        cast = worldcast.correction_cast(wac["split"])
        if not cast:
            continue
        cast_clean = {clean(n): n for n in cast if clean(n)}
        prompt = build_prompt(world, cast)
        if world not in variant_maps:
            variant_maps[world] = {clean(k): v for k, v in
                                   worlddict.load_variant_map(world).items() if clean(k)}
        for cand in r["cands"]:
            jobs.append({"cand": cand, "world": world, "prompt": prompt,
                         "cast_clean": cast_clean})

    if not jobs:
        return {"auto": [], "pending": [], "worlds": worlds_out}

    if redecode is None:
        decisions = run_model(_redecode_worker, audio_path, jobs, sorted(singles),
                              variant_maps, timeout=timeout)
    else:
        decisions = redecode(audio_path, jobs, sorted(singles), variant_maps)

    grouped = group_decisions(decisions)
    return {**grouped, "worlds": worlds_out}


def write_pending(session_dir: Path, result: dict, transcript: dict) -> Path:
    """Persist the bless queue (atomic; mirrors the detections file conventions)."""
    path = Path(session_dir) / "pending-name-corrections.json"
    payload = {
        "_about": "Name corrections awaiting human review (the bless queue). Written by the "
                  "namefix stage; POST /api/sessions/<id>/name-corrections/bless applies one. "
                  "The transcript is never modified by this file's existence.",
        "namefix_version": VERSION,
        "config_fingerprint": config_fingerprint(),
        "transcript_fingerprint": transcript_fingerprint(transcript),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "worlds": result["worlds"],
        "pending": result["pending"],
        "_rejected": (json.loads(path.read_text()).get("_rejected", [])
                      if path.exists() else []),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    import os
    os.replace(tmp, path)
    return path
