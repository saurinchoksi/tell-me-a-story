#!/usr/bin/env python3
"""Score the Stage-1 per-story name auditor against the by-ear answer keys.  READ-ONLY.

This is the ONLY side that reads name-truth.json (the gold). The auditor
(audit_names.py) never does — that firewall keeps the eval honest.

Gold derivation, per reviewable name (an `items` spelling), per story it lands in:
  not-a-name / place-other         -> no-error (a negative; must NOT be flagged)
  roster                            -> M9a   (out of Stage-1 P/R; reported separately)
  tv-canon  & delivered == true     -> no-error (correctly-spelled canon; precision test)
  tv-canon  & delivered != true     -> M9c    (canon spelled wrong)
  improvised & delivered == true    -> no-error (correctly-spelled improvised name)
  improvised & delivered != true    -> M9b if delivered & true share a Double Metaphone
                                       code (a variant), else M9d (a substitution onto an
                                       unrelated valid word — Bacchus PKS vs Pataki PTK).
  (true_spelling blank)             -> no-error / unjudged (listed, not scored)

The auditor reads the DELIVERED word tokens (post-normalization), so the gold
errors are only those that SURVIVE normalization still wrong — that is the point
(we score the names the system actually ships). Recall denominators are small by
design; the headline is the confusion matrix and precision.

    ./venv/bin/python emp/src/score_names.py --scout      # gold-only diagnostics (no model)
    ./venv/bin/python emp/src/score_names.py              # full score (needs name-audit-*.json)
"""
import argparse
import json
from collections import Counter, defaultdict

from audit_common import (SESSIONS, ROOT, clean, load_rich, load_regions,
                          story_of_pos)
from detectors.phonetics import codes
from name_truth import proper_name_candidates, detect_phrases, sidecar_path, m9_coded_segments

# gold class constants
ERR_CASES = ("M9b", "M9c", "M9d")        # the three text-relevant name-error cases
STAGE1_CASES = ("M9b", "M9c")            # what Stage 1 is scored on (M9a shipped; M9d audio-only)


def share_dm(a, b):
    """Do two cleaned spellings share any Double Metaphone code? (variant vs substitution)."""
    return bool(codes(a) & codes(b))


def gold_class(category, cleaned, true_spelling):
    """Map an answer-key item to its gold class. `cleaned` is the delivered spelling
    (single token, or space-joined for a phrase card)."""
    cat = (category or "").strip()
    true_norm = clean(true_spelling or "")                  # letters-only, lowercased
    delivered_norm = cleaned.replace(" ", "")               # phrase keys carry a space
    if cat in ("", None):
        return "unjudged"
    if cat in ("not-a-name", "place-other"):
        return "no-error"
    if cat == "roster":
        return "M9a"
    if cat in ("tv-canon", "improvised"):
        if not true_norm:
            return "no-error"                               # name, no recorded correction
        if true_norm == delivered_norm:
            return "no-error"                               # correctly-spelled name
        if cat == "tv-canon":
            return "M9c"
        return "M9b" if share_dm(delivered_norm, true_norm) else "M9d"
    return "unjudged"


def find_occurrences(rich, cleaned):
    """All (seg_id, wi) of a cleaned spelling in the DELIVERED words. Handles a
    space-joined phrase key by matching consecutive cleaned tokens."""
    parts = cleaned.split(" ")
    out = []
    for s in rich["segments"]:
        cls = [clean(w["word"].strip()) for w in s.get("words", [])]
        if len(parts) == 1:
            out += [(s["id"], wi) for wi, c in enumerate(cls) if c == cleaned]
        else:
            n = len(parts)
            out += [(s["id"], wi) for wi in range(len(cls) - n + 1)
                    if cls[wi:wi + n] == parts]
    return out


def candidate_set(rich, region, pos_of):
    """The names the auditor would SEE for one story (transcript-only, cap-gated) —
    used here only to measure each gold error's reachability (the recall ceiling).
    Mirrors audit_names.py's candidate construction; does NOT read the answer key."""
    lo, hi = region["start_pos"], region["end_pos"]
    segs = [s for s in rich["segments"] if lo <= pos_of.get(s["id"], -1) <= hi]
    singles = proper_name_candidates(segs)
    cands = set(singles)
    for phrase in detect_phrases(segs, singles):
        cands.add(phrase)
    return cands


def build_gold(sid):
    """Per-(story, spelling) gold units for one session, plus session-level extras.

    Returns dict: {regions, units, occ_truth, m9_segs, uncovered, no_key}."""
    key_path = sidecar_path(sid)
    key = json.loads(key_path.read_text()) if key_path.exists() else {"items": {}, "occurrences": {}}
    items, occ_truth = key.get("items", {}), key.get("occurrences", {})
    rich = load_rich(sid)
    regions, pos_of = load_regions(sid)
    m9_segs = m9_coded_segments(ROOT / "sessions" / sid, rich["segments"])

    cand_by_story = {r["idx"]: candidate_set(rich, r, pos_of) for r in regions}
    # Phrase-member fold (the "names are spans" lesson): when a multi-word phrase card
    # exists ("rubber ducky", "cruel kid"), its member tokens are NOT separate names —
    # they are fragments of that one card. name_truth.py hard-dedups them at render; the
    # gold must do the same, or a correctly-transcribed "ducky" looks like an M9d. A
    # single-word item whose token is a member of any phrase key is marked phrase-member
    # and excluded from scoring (the phrase card carries the real verdict).
    phrase_members = {tok for k in items if " " in k for tok in k.split(" ")}
    units = []
    covered_segs = set()
    for cleaned, meta in items.items():
        if " " not in cleaned and cleaned in phrase_members:
            cls = "phrase-member"
        else:
            cls = gold_class(meta.get("category"), cleaned, meta.get("true_spelling", ""))
        occs = find_occurrences(rich, cleaned)
        # which stories does this spelling touch?
        by_story = defaultdict(list)
        for seg_id, wi in occs:
            covered_segs.add(seg_id)
            st = story_of_pos(pos_of.get(seg_id), regions)
            by_story[st].append((seg_id, wi))
        if not by_story:                       # spelling not found in delivered words (stale key)
            by_story[None] = []
        for st_idx, st_occs in by_story.items():
            reachable = (st_idx is not None and cleaned in cand_by_story.get(st_idx, set()))
            units.append({
                "cleaned": cleaned, "category": meta.get("category", ""),
                "true": meta.get("true_spelling", ""), "gold": cls,
                "story": st_idx, "n_occ": len(st_occs), "occs": st_occs,
                "in_story": st_idx is not None, "reachable": reachable,
            })
    # M9-coverage guarantee: hand-coded M9 segments with no surfaced name
    uncovered = sorted((s for s in m9_segs if s not in covered_segs), key=str)
    return {"regions": regions, "units": units, "occ_truth": occ_truth,
            "m9_segs": m9_segs, "uncovered": uncovered, "has_key": key_path.exists()}


# ----------------------------------- scout ------------------------------------
def scout():
    print("=" * 78)
    print("GOLD SCOUT — per-session answer-key derivation (model-free)")
    print("=" * 78)
    tot = Counter()
    for sid, name in SESSIONS.items():
        g = build_gold(sid)
        regions = g["regions"]
        print(f"\n##### {name} ({sid}) — {len(regions)} stor"
              f"{'y' if len(regions) == 1 else 'ies'} "
              f"{'' if g['has_key'] else '[NO KEY]'}")
        for r in regions:
            print(f"   story {r['idx']}: segs {r['start_id']}-{r['end_id']}  world={r['world']!r}")

        per_case = defaultdict(list)
        for u in g["units"]:
            per_case[u["gold"]].append(u)
        for cls in ("M9c", "M9b", "M9d", "M9a"):
            us = per_case.get(cls, [])
            if not us:
                continue
            in_story = [u for u in us if u["in_story"]]
            if cls in STAGE1_CASES:
                tot[cls] += len(in_story)                       # only in-story units are scorable
            print(f"   --- {cls}: {len(in_story)} in-story unit(s)"
                  f"{f' + {len(us) - len(in_story)} out-of-story' if len(us) > len(in_story) else ''} ---")
            for u in sorted(us, key=lambda x: (x["story"] is None, x["story"] or 0, x["cleaned"])):
                reach = "" if u["reachable"] else (
                    "  [OUT-OF-STORY — segmenter excluded]" if not u["in_story"]
                    else "  [NOT in cap-gated candidate set — fulltext-only]")
                arrow = f" -> {u['true']}" if u["true"] else ""
                print(f"       story {u['story']}  {u['cleaned']!r}{arrow}  ({u['n_occ']}×){reach}")
        n_noerr = len(per_case.get("no-error", []))
        n_member = len(per_case.get("phrase-member", []))
        print(f"   no-error/negative units: {n_noerr}"
              f"{f'  (+{n_member} phrase-member fragments folded)' if n_member else ''}")
        # occurrence-grain M9d ceiling (substitutions onto valid words)
        if g["occ_truth"]:
            print(f"   occurrence-grain truth marks (M9d / garbage / unintelligible): {len(g['occ_truth'])}")
        if g["uncovered"]:
            print(f"   ⚠ M9-coded segments with NO surfaced name (coverage gap): {g['uncovered'][:15]}")

    print("\n" + "=" * 78)
    print(f"STAGE-1 GOLD ERROR TOTALS (pooled, story-scope):  "
          f"M9c={tot['M9c']}  M9b={tot['M9b']}   "
          f"(M9a reported separately; M9d is the audio-only ceiling)")
    print("=" * 78)


# ============================ scoring the audit ===============================
ARCHS = ("worksheet", "fulltext", "hybrid", "v2")
GOLD_ROWS = ["M9c", "M9b", "M9d", "M9a", "no-error"]
DET_COLS = ["M9c", "M9b", "M9d", "none"]
# The two Pandavas canon names that are minor TRANSLITERATION variants, not clear
# mistranscriptions (Duryodhan~Duryodhana, Yudhisthir~Yudhishthira are accepted
# spellings). Reported as a lenient cut so the judgment is visible, not buried.
M9C_MINOR = {"duryodhan", "yudhisthir"}
PROVISIONAL = {"20251207-202105"}   # Cruel Baby: key predates the phrase-card feature


def _detcol(case):
    return {"M9b": "M9b", "M9c": "M9c", "M9d-suspect": "M9d"}.get(case, "none")


def gold_map(sid):
    """{(story_idx, cleaned) -> gold class} for in-story, non-phrase-member units."""
    g = build_gold(sid)
    m = {(u["story"], u["cleaned"]): u["gold"] for u in g["units"]
         if u["story"] is not None and u["gold"] != "phrase-member"}
    return m, g


def load_audit(sid, arch):
    p = ROOT / "emp" / "results" / "visuals" / sid / f"name-audit-{arch}.json"
    return json.loads(p.read_text()) if p.exists() else None


def predictions(audit):
    """{(story_idx, cleaned) -> case} exploded from an audit's per-flag wrong spellings."""
    preds = {}
    for st in audit.get("stories", []):
        sidx = st["story"]["idx"]
        for f in st.get("flags", []):
            for c in f.get("wrong_cleaned", []):
                preds.setdefault((sidx, c), f["case"])
    return preds


# Which real architectures the ensemble unions. We take worksheet (precise) ∪ fulltext
# (reaches the lowercase names the cap-gate hides): a name is "detected" if EITHER flags
# it — maximum recall, the goal for a detection-only monitor. (hybrid added nothing unique.)
ENSEMBLE_OF = ("worksheet", "fulltext")


def predictions_for(sid, arch):
    """Predictions for one architecture, or the union for the synthetic 'ensemble'."""
    if arch == "ensemble":
        merged = {}
        for a in ENSEMBLE_OF:
            au = load_audit(sid, a)
            if au:
                for k, case in predictions(au).items():
                    merged.setdefault(k, case)
        return merged
    au = load_audit(sid, arch)
    return predictions(au) if au else {}


def score_session(sid, arch):
    gm, g = gold_map(sid)
    preds = predictions_for(sid, arch)
    matrix = {r: Counter() for r in GOLD_ROWS + ["phantom"]}
    fps, misses, caught = [], [], []
    for key, gcls in gm.items():
        col = _detcol(preds.get(key))
        matrix[gcls][col] += 1
        if gcls in ("M9b", "M9c"):
            (caught if col != "none" else misses).append((key, gcls, col))
        if gcls == "no-error" and col != "none":
            fps.append((key, col, "flagged a correctly-spelled name"))
    for key, case in preds.items():
        if key not in gm:
            col = _detcol(case)
            matrix["phantom"][col] += 1
            fps.append((key, col, "flagged a non-reviewable token"))
    return {"matrix": matrix, "fps": fps, "misses": misses, "caught": caught,
            "preds": preds, "gm": gm}


def _safe(n, d):
    return f"{n}/{d} = {n / d:.2f}" if d else f"{n}/{d} = —"


def case_pr(pooled, case):
    """(name-recall, case-recall, name-precision, case-precision) counts for a case."""
    m = pooled
    gold_total = sum(m[case].values())                       # gold units of this case
    case_caught = m[case][case]                              # gold C predicted C
    name_caught = sum(v for k, v in m[case].items() if k != "none")  # gold C predicted anything
    pred_total = sum(m[r][case] for r in m)                  # all preds with column = case
    pred_real = sum(m[r][case] for r in ("M9c", "M9b", "M9d", "M9a"))  # preds landing on a real error
    return {"gold": gold_total, "name_rec": name_caught, "case_rec": case_caught,
            "pred_total": pred_total, "pred_real": pred_real, "case_prec_n": m[case][case]}


def print_matrix(pooled):
    print(f"      detector ->   {'  '.join(f'{c:>5}' for c in DET_COLS)}    (row = human gold)")
    for r in GOLD_ROWS + ["phantom"]:
        if sum(pooled[r].values()) == 0:
            continue
        cells = "  ".join(f"{pooled[r][c]:>5}" for c in DET_COLS)
        print(f"   {r:>9} |   {cells}")


def run_scoring(sids):
    print("=" * 80)
    print("STAGE-1 NAME AUDITOR — scored vs the by-ear answer keys (per architecture)")
    print("  spelling-grain; recall denominators are small by design (M9c=4, M9b=4);")
    print("  M9a = inconsistent family names (out of Stage-1 scope, shown); M9d = audio-only.")
    print("=" * 80)

    score_archs = list(ARCHS) + ["ensemble"]
    summary = {}
    for arch in score_archs:
        pooled = {r: Counter() for r in GOLD_ROWS + ["phantom"]}
        all_fps, all_miss, all_caught = [], [], []
        for sid in sids:
            r = score_session(sid, arch)
            for row in pooled:
                pooled[row] += r["matrix"][row]
            tag = "  (PROVISIONAL key)" if sid in PROVISIONAL else ""
            for f in r["fps"]:
                all_fps.append((SESSIONS[sid], *f))
            for mss in r["misses"]:
                all_miss.append((SESSIONS[sid] + tag, *mss))
            for c in r["caught"]:
                all_caught.append((SESSIONS[sid], *c))

        print(f"\n{'#' * 72}\n## ARCHITECTURE: {arch}\n{'#' * 72}")
        print_matrix(pooled)
        m9c, m9b = case_pr(pooled, "M9c"), case_pr(pooled, "M9b")
        # M9c lenient cut (drop the minor transliteration variants from the denominator)
        m9c_minor_caught = sum(1 for (sess, (st, cl), g, col) in
                               [(s, k, gc, co) for (s, k, gc, co) in all_caught if gc == "M9c"]
                               if cl in M9C_MINOR)
        m9c_minor_total = sum(1 for sid in sids for (k, gc) in gold_map(sid)[0].items()
                              if gc == "M9c" and k[1] in M9C_MINOR)
        m9c_strict_caught, m9c_strict_total = m9c["name_rec"], m9c["gold"]
        m9c_len_caught = m9c_strict_caught - m9c_minor_caught
        m9c_len_total = m9c_strict_total - m9c_minor_total

        print(f"\n  M9c (canon) recall  — strict  {_safe(m9c_strict_caught, m9c_strict_total)}"
              f"   lenient (drop minor variants) {_safe(m9c_len_caught, m9c_len_total)}")
        print(f"  M9b (improvised) recall (caught at all)  {_safe(m9b['name_rec'], m9b['gold'])}")
        print(f"  M9c precision (case-correct)  {_safe(m9c['case_prec_n'], m9c['pred_total'])}"
              f"   | name-precision {_safe(m9c['pred_real'], m9c['pred_total'])}")
        print(f"  M9b precision (case-correct)  {_safe(m9b['case_prec_n'], m9b['pred_total'])}"
              f"   | name-precision {_safe(m9b['pred_real'], m9b['pred_total'])}")
        n_fp = sum(pooled["no-error"][c] for c in DET_COLS if c != "none") + \
            sum(pooled["phantom"][c] for c in DET_COLS if c != "none")
        n_pred = sum(sum(pooled[r][c] for r in pooled) for c in DET_COLS if c != "none")
        print(f"  overall: {n_pred} flags, {n_fp} false positives "
              f"(landed on a no-error name or a non-name token)")

        print("  --- off-diagonals ---")
        for sess, (st, cl), gc, col in sorted(all_miss):
            print(f"    MISS   {sess:18s} story{st} {cl!r:18s} gold {gc}, detector said none")
        for sess, (st, cl), col, why in sorted(all_fps):
            row = next((g for s in sids for (k, g) in gold_map(s)[0].items()
                        if k == (st, cl) and SESSIONS[s] == sess), "phantom")
            print(f"    FP     {sess:18s} story{st} {cl!r:18s} detector {col}  ({why}; gold={row})")
        # bonus: M9a names the roster-free auditor surfaced (out of scope, but a real catch)
        m9a_hits = sum(pooled["M9a"][c] for c in DET_COLS if c != "none")
        if m9a_hits:
            print(f"  (+{m9a_hits} inconsistent FAMILY-name occurrence(s) flagged — M9a's job, caught for free)")

        summary[arch] = {
            "m9c_strict": (m9c_strict_caught, m9c_strict_total),
            "m9c_lenient": (m9c_len_caught, m9c_len_total),
            "m9b_rec": (m9b["name_rec"], m9b["gold"]),
            "flags": n_pred, "fp": n_fp,
        }

    print(f"\n{'=' * 80}\nSIDE-BY-SIDE (the 'which architecture' deliverable)\n{'=' * 80}")
    print(f"  {'arch':10s} | M9c rec (strict/lenient) | M9b rec | flags | false-pos")
    for a in score_archs:
        s = summary[a]
        print(f"  {a:10s} | {s['m9c_strict'][0]}/{s['m9c_strict'][1]} "
              f"/ {s['m9c_lenient'][0]}/{s['m9c_lenient'][1]}            "
              f"| {s['m9b_rec'][0]}/{s['m9b_rec'][1]}     | {s['flags']:>4}  | {s['fp']}")
    print("\n  Note: Cruel Baby key is PROVISIONAL (predates phrase cards); its only error-gold")
    print("  is M9d phrase-split artifacts (audio-only, unscored), so it does not move recall.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scout", action="store_true", help="gold-only diagnostics (no model needed)")
    ap.add_argument("sessions", nargs="*")
    args = ap.parse_args()
    sids = args.sessions or list(SESSIONS)
    if args.scout:
        scout()
    else:
        run_scoring(sids)


if __name__ == "__main__":
    main()
