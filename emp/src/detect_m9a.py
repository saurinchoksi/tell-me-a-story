#!/usr/bin/env python3
"""
L1 -- deterministic M9a (family-name) detector for the EMP.  READ-ONLY.

Faithful reconstruction of the in-sample phonetic probe validated in
`reference/career-build/emp.md` (see "M9a ground truth + detector validation").
The earlier probe was a one-off and was never saved; this is the reusable form.

A transcript token is flagged iff:
  (1) any Double Metaphone code (PRIMARY *or* SECONDARY) of the token matches
      any Double Metaphone code of a roster CANONICAL name      [phonetic layer]
  OR
  (2) the token exactly equals a known front-distortion mishearing that the
      phonetic layer cannot reach -- the one surname alias.      [alias layer]

Matching on BOTH metaphone codes is load-bearing. A name heard with a "th"
(e.g. a child's name rendered "Arthi"/"Earthy") encodes to a PRIMARY theta code
("AR0") but a SECONDARY "t" code ("ART"); it is the secondary code that links it
back to the plain-"t" canonical. A single-code matcher would miss these.

PRIVACY: no real names are hardcoded here. The roster (canonical spellings + the
in-sample variant tokens that seed the alias layer) is read at runtime from the
gitignored private sidecar `emp/results/pivot-notes.private.json`. This script is
therefore safe to commit; its OUTPUT echoes mis-rendered name tokens and is
written under `emp/results/visuals/<id>/` (gitignored).

Usage:
    python emp/src/detect_m9a.py 20260414-213156 20260211-210718
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from metaphone import doublemetaphone

ROOT = Path(__file__).resolve().parents[2]
SIDECAR = ROOT / "emp" / "results" / "pivot-notes.private.json"
SESSIONS_DIR = ROOT / "sessions"
VISUALS = ROOT / "emp" / "results" / "visuals"


def clean(tok: str) -> str:
    """Lowercase; strip surrounding punctuation and the possessive; keep letters."""
    t = tok.strip().lower()
    t = re.sub(r"[^a-z'’]+$", "", t)   # trailing punctuation (keep apostrophe)
    t = re.sub(r"^[^a-z'’]+", "", t)   # leading punctuation
    t = re.sub(r"['’]s?$", "", t)      # possessive 's or trailing apostrophe
    t = re.sub(r"[^a-z]", "", t)            # letters only
    return t


def codes(word: str) -> set:
    """The non-empty Double Metaphone codes (primary + secondary) of a word."""
    return {c for c in doublemetaphone(word) if c}


def load_roster():
    side = json.loads(SIDECAR.read_text())
    roster = side["m9a_roster"]
    people = roster["people"]

    canon = {p["id"]: p["canonical"] for p in people}        # id -> canonical spelling
    canon_forms = {clean(n) for n in canon.values()}         # for "is this the right spelling?"

    code_to_pid = {}                                          # dm code -> {person id}
    for pid, name in canon.items():
        for c in codes(clean(name)):
            code_to_pid.setdefault(c, set()).add(pid)
    roster_codes = set(code_to_pid)

    # In-sample rendering -> person id (the sidecar holds only the 5 in-sample
    # sessions, so this never leaks the held-out sessions). Used to label the
    # alias and nothing else.
    rend_to_pid = {}
    for p in people:
        for _sess, rends in p.get("renderings_by_session", {}).items():
            for r in rends:
                rend_to_pid[clean(r)] = p["id"]

    # Alias layer = in-sample variant tokens whose codes reach no canonical.
    alias = {}
    for v in side["m9_variants"]["M9a"]:
        cv = clean(v)
        if cv and not (codes(cv) & roster_codes):
            alias[cv] = rend_to_pid.get(cv)
    return canon, canon_forms, code_to_pid, roster_codes, alias


def detect(session_id, canon, canon_forms, code_to_pid, roster_codes, alias):
    path = SESSIONS_DIR / session_id / "transcript-rich.json"
    data = json.loads(path.read_text())
    flags, n_tokens = [], 0
    for seg in data["segments"]:
        for wi, w in enumerate(seg.get("words", [])):
            c = clean(w["word"])
            if not c:
                continue
            n_tokens += 1
            if c in canon_forms:
                continue   # correctly-spelled canonical -- not an error
            tok_codes = codes(c)
            shared = tok_codes & roster_codes
            if shared:
                matched = sorted({pid for code in shared for pid in code_to_pid[code]})
                mtype = "phonetic"
            elif c in alias:
                matched = [alias[c] or "<unknown>"]
                mtype = "alias"
            else:
                continue
            flags.append({
                "session": session_id,
                "segment_id": seg["id"],
                "word_index": wi,
                "start": w.get("start"),
                "end": w.get("end"),
                "token": w["word"].strip(),
                "cleaned": c,
                "dm_codes": sorted(tok_codes),
                "match_type": mtype,
                "matched_person_ids": matched,
                "matched_canonicals": [canon.get(pid, pid) for pid in matched],
            })
    return flags, n_tokens


def report(session_id, flags, n_tokens, canon):
    print(f"\n{'='*64}\nSESSION {session_id}  --  {n_tokens} word tokens, {len(flags)} flags")
    by_person = Counter(pid for f in flags for pid in f["matched_person_ids"])
    print("\n  flags by matched roster member:")
    for pid, n in by_person.most_common():
        print(f"     {n:>3}  {pid:<12} (canonical '{canon.get(pid, pid)}')")
    print("\n  distinct renderings flagged  (token  ->  count, codes, matched):")
    per = {}
    for f in flags:
        key = f["cleaned"]
        per.setdefault(key, {"n": 0, "token": f["token"], "codes": f["dm_codes"],
                             "who": f["matched_canonicals"], "type": f["match_type"]})
        per[key]["n"] += 1
    for key, info in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        who = "/".join(info["who"])
        print(f"     {info['n']:>3}  {info['token']:<12} {str(info['codes']):<16} "
              f"-> {who:<10} [{info['type']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="session ids under sessions/")
    args = ap.parse_args()

    canon, canon_forms, code_to_pid, roster_codes, alias = load_roster()
    print("L1 roster (canonical -> Double Metaphone codes):")
    for pid, name in canon.items():
        print(f"   {name:<10} ({pid:<12}) {sorted(codes(clean(name)))}")
    print(f"   alias layer (exact, front-distortion): "
          f"{ {k: canon.get(v, v) for k, v in alias.items()} }")

    for sid in args.sessions:
        flags, n_tokens = detect(sid, canon, canon_forms, code_to_pid, roster_codes, alias)
        report(sid, flags, n_tokens, canon)
        outdir = VISUALS / sid
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / "m9a-l1-flags.json"
        out.write_text(json.dumps({
            "_about": "L1 (deterministic M9a detector) flags -- the SEALED prediction "
                      "to be scored against an independent human ear pass. Gitignored "
                      "(contains mis-rendered name tokens).",
            "session": sid,
            "n_word_tokens": n_tokens,
            "n_flags": len(flags),
            "flags": flags,
        }, indent=2))
        print(f"\n  -> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
