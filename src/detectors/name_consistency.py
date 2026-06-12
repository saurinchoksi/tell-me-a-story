"""M9b — inconsistent-name detector.

Catches an improvised story name spelled several ways within one recording —
the engines Jiraki/Jameis/Jammus, or a name Whisper renders one way here and
another there. Unlike the M9a detector there is no roster and no truth source:
the *only* defect is inconsistency, so the detector flags it and leaves the
choice of a winning spelling to a separate fix.

Mechanism (code-only, no model): group the recording's capitalized tokens into
phonetic clusters by shared Double Metaphone code, then flag every occurrence in
any cluster that appears under more than one spelling. A name spelled one
consistent way is never flagged.

Roster-agnostic by design, so it also surfaces inconsistently-spelled *family*
names (an M9a case is, phonetically, just another multi-spelling cluster). That
is acceptable — an inconsistent name is worth flagging either way — but it means
the detector's output can echo family-name variants, so callers must treat the
flags as private (the per-session detections.json already lives under gitignored
sessions/).
"""

from collections import defaultdict
from pathlib import Path

from detectors.base import Detector, load_transcript
from detectors.phonetics import clean, codes, is_capitalized


class _UnionFind:
    """Minimal union-find over hashable items (the distinct cleaned spellings)."""

    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class NameConsistencyDetector(Detector):
    id = "m9b-name-consistency"
    label = "Inconsistent name spelling"
    failure_mode = "M9b"
    version = "0.1.0"  # pre-validation; bump to 1.0.0 once validated

    def run(self, session_dir: Path) -> dict:
        data = load_transcript(session_dir)

        # 1. Collect capitalized-token occurrences (names are capitalized; the
        #    gate drops lowercase common words).
        occurrences = []  # (segment_id, word_index, start, end, raw, cleaned)
        form_codes = {}   # cleaned form -> set of Double Metaphone codes
        n_tokens = 0
        for seg in data["segments"]:
            for wi, w in enumerate(seg.get("words", [])):  # gap segments have no words
                raw = w["word"].strip()
                c = clean(raw)
                if not c:
                    continue
                n_tokens += 1
                if not is_capitalized(raw):
                    continue
                occurrences.append((seg["id"], wi, w.get("start"), w.get("end"), raw, c))
                form_codes.setdefault(c, codes(c))

        # 2. Cluster the distinct spellings: union any two that share a DM code.
        uf = _UnionFind()
        for c in form_codes:
            uf.add(c)
        code_to_forms = defaultdict(list)
        for form, fcodes in form_codes.items():
            for code in fcodes:
                code_to_forms[code].append(form)
        for forms in code_to_forms.values():
            for other in forms[1:]:
                uf.union(forms[0], other)

        # 3. A cluster is inconsistent iff it holds >1 distinct cleaned spelling.
        cluster_forms = defaultdict(set)        # root -> {cleaned forms}
        cluster_surface = defaultdict(set)      # root -> {raw surface forms}
        cluster_count = defaultdict(int)        # root -> total occurrences
        for sid, wi, start, end, raw, c in occurrences:
            root = uf.find(c)
            cluster_forms[root].add(c)
            cluster_surface[root].add(raw)
            cluster_count[root] += 1
        inconsistent = {r for r, forms in cluster_forms.items() if len(forms) > 1}

        # 4. One flag per occurrence inside an inconsistent cluster.
        flags = []
        for sid, wi, start, end, raw, c in occurrences:
            root = uf.find(c)
            if root not in inconsistent:
                continue
            flags.append({
                "segment_id": sid,
                "word_index": wi,
                "start": start,
                "end": end,
                "token": raw,
                "cleaned": c,
                "dm_codes": sorted(form_codes[c]),
                "cluster_id": min(cluster_forms[root]),
                "cluster_spellings": sorted(cluster_surface[root]),
                "n_cluster_occurrences": cluster_count[root],
            })
        return {"n_word_tokens": n_tokens, "flags": flags}
