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

# Precision layer (see the score_m9b validation: code-only clustering had 0.94
# recall but 0.06 precision — it drowned in capitalized common words and
# interjections sharing metaphone codes). Two cheap deterministic gates:
DEFAULT_WORDLIST = Path("/usr/share/dict/words")  # the system English dictionary
MIN_NAME_LEN = 4   # improvised names are >=4 chars; I/Oh/We/No/Uh are not
# Fillers a formal wordlist may omit (the <4-char ones are already dropped by
# MIN_NAME_LEN; these are the >=4 stragglers).
EXTRA_COMMON = {"yeah", "whoa", "woah", "haha", "hehe", "uhhuh", "mmhmm", "okay"}


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
    version = "0.2.0"  # pre-validation; bump to 1.0.0 once validated

    def __init__(self, wordlist_path=DEFAULT_WORDLIST):
        # wordlist_path=None → stoplist-only (degraded, used by tests).
        self.wordlist_path = Path(wordlist_path) if wordlist_path else None
        self._common = None  # loaded lazily

    def _is_common(self, c: str) -> bool:
        """Is this cleaned token an ordinary English word? (in the dictionary,
        a known filler, or a plural/possessive/3sg of a dictionary word — the
        trailing-s strip catches cleaned contractions like 'whats')."""
        if self._common is None:
            common = set(EXTRA_COMMON)
            if self.wordlist_path and self.wordlist_path.exists():
                common |= {
                    line.strip().lower()
                    for line in self.wordlist_path.read_text(errors="ignore").splitlines()
                    if line.strip()
                }
            self._common = common
        return c in self._common or (c.endswith("s") and c[:-1] in self._common)

    def run(self, session_dir: Path) -> dict:
        data = load_transcript(session_dir)

        # 1. Collect name-candidate occurrences: capitalized (name-shaped) and at
        #    least MIN_NAME_LEN chars (drops short function words / interjections).
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
                if not is_capitalized(raw) or len(c) < MIN_NAME_LEN:
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
        # A cluster is a real inconsistent name iff it is spelled >1 way AND at
        # least one spelling is not an ordinary word. The "any non-common" rule
        # keeps a name even when one variant happens to be a dictionary word
        # (Pataki/Bacchus survives via "pataki"; the misspelling is the signal),
        # while dropping clusters that are *all* common (What/Whats, Yeah/Whoa).
        inconsistent = {
            r for r, forms in cluster_forms.items()
            if len(forms) > 1 and any(not self._is_common(f) for f in forms)
        }

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
