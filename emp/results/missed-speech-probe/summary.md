# Missed-speech recovery probe

Gaps probed (voice on the diarizer, no transcript trace, >= 1.0s): 20 across 2 Mahabharata-world sessions.
Gaps where a world+cast-primed re-decode produced words INSIDE the gap: 19 of 20.

A produced word is a CANDIDATE recovery, not a confirmed one — the prompt can
force text into noise. Every yielding gap has an ear-check card (gitignored
visuals). The decision this feeds: whether context-primed re-decoding earns a
place as a missed-speech treatment, where 8 prior approaches recovered nothing.

| session | gaps | yielded words |
|---|---|---|
| 20260211-210718 | 20 | 19 |

## Human verdicts (Choksi by ear, 2026-07-02)
Of the 19 yielding gaps: **17 "really said", 1 "partly right" (sentence right, one name
garbled), 1 "can't tell", 0 "not said".** The candidate recoveries are essentially all
genuine dropped speech — the context-primed re-decode is a working missed-speech treatment,
human-confirmed. (Per-gap hearings live in the gitignored sidecar.)
