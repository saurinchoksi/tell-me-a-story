# `[unintelligible]` gap classification — summary

Read-only. For each injected `[unintelligible]` gap, the recognizer is re-run on the isolated clip; `no_speech_prob` decides **M8 (nothing to transcribe)** vs **real speech**. Scored against the existing by-ear codes. Gaps a human coded as another mode are held out.

## What the signals can do

- **Loudness fails** — the three classes overlap completely in volume (a loud chuckle is M8, a loud mumble is NotA). A loudness rule scores 29%, worse than always guessing NotA (46%).
- **Whisper-on-clip separates M8 from real speech** — the mechanizable axis.
- **M3 vs NotA does not mechanize** — Whisper decodes both into plausible text, so that finer call stays an ear judgment; Whisper's transcript is shown as a decode aid, not a verdict.

**M8-vs-speech agreement: 23/28 = 82%.** M8 flag precision 6/7, recall 6/10.

| your bucket \ suggested | M8 | speech |
|---|---|---|
| M8 | 6 | 4 |
| speech (NotA+M3) | 1 | 17 |

## Per session (suggested)

| session | gaps | M8 | real speech | held out |
|---|---|---|---|---|
| Moon Story | 1 | 1 | 0 | 0 |
| Cruel Baby | 5 | 1 | 4 | 0 |
| Rubber Ducky | 3 | 1 | 2 | 0 |
| Pandavas | 3 | 0 | 3 | 0 |
| Portal Story | 22 | 6 | 16 | 6 |

## M8-vs-speech disagreements (5) — check by ear

- Cruel Baby `gap_606.198` (10:06.20, 0.03s): you coded **M8**, suggested **speech** (no-speech 0.45, 1 words heard)
- Moon Story `gap_386.705` (6:26.70, 0.78s): you coded **NotA**, suggested **M8** (no-speech 1.00, 0 words heard)
- Portal Story `gap_413.688` (6:53.69, 1.72s): you coded **M8**, suggested **speech** (no-speech 0.34, 1 words heard)
- Portal Story `gap_1083.254` (18:03.25, 0.02s): you coded **M8**, suggested **speech** (no-speech 0.88, 1 words heard)
- Portal Story `gap_1121.493` (18:41.49, 0.19s): you coded **M8**, suggested **speech** (no-speech 0.82, 1 words heard)

## Held out — another mode owns these (6)

- Portal Story `gap_18.425` (0:18.42) coded M2
- Portal Story `gap_56.967` (0:56.97) coded M7
- Portal Story `gap_361.240` (6:01.24) coded M7
- Portal Story `gap_367.484` (6:07.48) coded M7
- Portal Story `gap_782.052` (13:02.05) coded M4/M7
- Portal Story `gap_1020.142` (17:00.14) coded M7/M8
