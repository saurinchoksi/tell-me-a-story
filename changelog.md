# Changelog

Newest entries at top. Each entry is a few plain sentences — what changed, and what it taught when there's a real lesson. Exact diffs live in git history; decision context lives in Linear.

*Claude Code drafts these entries from each session's work; Saurin Choksi reviews and approves.*

## 2026-05-18 — Sessions page: pipeline-health warning indicator

Added a warning marker to the Sessions list that appears only when a session has a failed pipeline stage; hovering it names the stage. The pipeline had been recording each stage's success or failure all along, but the page only showed whether output files existed — and a file existing doesn't mean the stage that wrote it succeeded. Turning the marker on immediately surfaced two old sessions that had been quietly carrying a broken stage.

## 2026-05-15 — Sessions page: column headers + length, notes-count, validation-status

Gave the Sessions list a proper header row and three new columns: how long each recording is, how many validation notes it has, and a validation status you cycle through by clicking — not started, in progress, done. The status is something you set by hand, not something inferred, because whether a session has truly been reviewed is a human judgment the system can't make for you.

## 2026-03-12 — Gemma 3n discovered, experiment plan expanded

Digging through Hugging Face turned up Gemma 3n, Google's audio-capable model, as a strong candidate for the upcoming experiments — and the only one of the four that runs natively on Apple Silicon without dragging in a separate framework. Added it to the plan as the model to try first, since it has the lowest setup cost. A model only earns a look now if it can run on this stack as-is.

## 2026-03-10 — Pipeline processing statistics (TMAS-13)

Added timing to every pipeline stage, plus a summary block recording how long the whole run took, how many words, segments, and speakers came out, and how big the output files are. The pipeline now prints a timing summary when it finishes.

## 2026-03-10 — Tracker migration: SYNC files out, Linear in

Moved all project task-tracking into Linear and retired the old setup — a Notion database plus handoff files passed between Claude instances. Linear is now the single source of truth, with skills built for both Claude Desktop and Claude Code to work against it. The old system needed a dozen-plus calls just to list tickets; Linear does it in one — the right tool mattered more than the effort already sunk into the wrong one.

## 2026-03-03 — Cross-session identification works

Speaker identification now runs automatically at the end of the pipeline. On a completely different recording, it matched Saurin at 96% and Arti at 94% from just one earlier voice sample each — proof that a voice profile generalizes across sessions from a single enrollment. Odd variants, like a silly voice or a distant mic, correctly land in the lower "suggested" tier rather than being forced into a confident match.

## 2026-03-02 — Speaker identification: from experiment to working UI

Built the whole speaker-identification feature in a day, from voice-profile storage through to the web interface. The system now works end to end: the pipeline extracts a voice fingerprint, identification proposes matches, a person confirms them in a review screen, and profiles build up over time. A cold start — no profiles yet, everything unknown — was designed to feel like a starting point, not an error.

## 2026-02-26 — Session Reviewer enters the page

Added a Session Reviewer section to the portfolio page, with a screenshot. Until now the page showed the pipeline but no tools and no person in the loop — the reviewer is where the human enters the story.

## 2026-02-22 — Messy audio in, structured transcript out

Designed an interactive visualization of the pipeline for the portfolio page, and landed on a sharper framing for what it does: messy audio in, structured transcript out. Putting speaker detection before name correction in the walkthrough also tells a better story — meet the speakers first, then fix what they said.

## 2026-02-20 — Deep code review: 14 findings, 8 fixes shipped

Reviewed every source and test file end to end, turning up 14 items — and shipped all 8 of the actionable fixes in a single commit. The one big item deferred is architectural: splitting the Mahabharata-specific logic out of the core pipeline, which needs a design conversation before any code.

## 2026-02-20 — README, test coverage, and removing friction

Rewrote the README around a before-and-after sample — raw Whisper output becoming a clean, speaker-labeled transcript — which sells the project far better than any architecture description. Also added tests to the entry-point scripts and made the validator's note-saving instant. The lesson that kept recurring: building infrastructure for problems you don't have yet costs more than the friction it would remove.

## 2026-02-19 — Second real session and what it revealed

Ran the first genuinely real-world session through the pipeline — a ten-minute original moon story with several people in the room. It surfaced a new failure the filters don't catch: hallucinated text sitting on top of perfectly clean audio. Controlled test recordings, it turns out, don't predict how the pipeline behaves on messy household audio.

## 2026-02-18 — Gaps, inbox, and the pipeline running end to end

The pipeline now runs start to finish from a single command: drop audio in an inbox folder, run one script. It also detects gaps — moments where speaker detection hears someone but transcription produces nothing — and marks them honestly as unintelligible rather than guessing. Two AI models had been fighting over the Mac's GPU memory and crashing; running one of them in a separate process fixed it.

## 2026-02-18 — Diarization becomes visible: speaker indicators in validator

Made speaker information visible in the transcript validator — each segment now carries a speaker-colored border and badge, and word-level mismatches get a wavy underline. Checking it by eye caught three bugs that unit tests never would have, because border opacity, class names, and badge visibility are all things you can only really verify by looking.

## 2026-02-18 — Pipeline becomes fully self-contained: Ollama out, MLX-LM in

Swapped Ollama — a separate inference server the pipeline had to reach over the network — for a model library that loads straight into Python. Now every stage works the same way: load the model, use it, unload it. Ollama had really been solving a problem this project doesn't have — serving many users at once — and that mismatch kept showing up as silent failures.

## 2026-02-17 — 12 files become 8, and modules own their identity

Consolidated the source code from twelve files down to eight, and made each module responsible for its own identity instead of having the orchestrator stamp it in. The guiding question when something felt misplaced: follow it to the one piece of code that actually uses it, and put it there.

## 2026-02-17 — Three artifacts, not five: session folder redesign

Simplified what each session folder holds from five files down to three: the raw transcript exactly as Whisper produced it, the speaker data, and an enriched transcript that layers corrections on top. The raw version is kept separate and never overwritten — the honest record of what the machine actually heard has to survive.

## 2026-02-10 — Sub-agent removal and codebase hygiene

Removed a multi-agent setup — separate reviewer and coder agents with an orchestrator — after a review found it was solving a problem a fourteen-file project simply doesn't have. The takeaway worth keeping: before adopting any practice that sounds smart, ask whether this project actually has the problem it solves.

## 2026-02-09 — Workflow Revision for Opus 4.6

Reworked the whole project workflow around Claude's newer capabilities, and folded 29 separate journal files into this single changelog — 103KB of scattered notes down to 19KB in one place. Journals are dead now; the changelog is the single historical record. Settled on a clear division of labor: instructions for guaranteed behavior, memory for accumulated knowledge, files on disk for anything that must be read explicitly.

## 2026-02-08 — Autonomous Code Exploration for Quiet Speech

Three attempts to recover Arti's quietest speech all failed, so the work shifted to letting Claude Code experiment autonomously instead of running each idea by hand. The finding was firm: the voice is genuinely quiet — not masked by noise, just faint — and Whisper has a hard floor, not a soft edge it can be nudged past. The quality of the handoff turned out to decide the quality of the autonomous work: tight context, loose approach.

## 2026-02-08 — Quiet Speech Recovery: Gain + initial_prompt Trap

Tried fourteen combinations of audio processing and prompting to recover a moment where speaker detection heard Arti but transcription produced nothing. None worked — boosting volume can't create information that was never captured. It also exposed a trap: feeding Whisper narrative context as a hint makes it skip over matching audio entirely, so a hint should only ever be a plain list of vocabulary.

## 2026-02-08 — Re-enrich Script + Architecture Refactor

Built a way to re-run the correction and speaker-labeling steps on an existing session without redoing the expensive transcription and speaker detection from scratch. The shared logic lives in one function used by both fresh runs and re-runs, rather than being copied.

## 2026-02-07 — Hallucination Filters: Three Patterns, Zero False Positives

Replaced a blunt confidence threshold for catching hallucinated words with three targeted filters, which caught every known case with no false positives. The key discovery: a hallucination is almost always a one-word segment sitting alone — the shape of the segment, not the confidence number, is what gives it away.

## 2026-02-07 — Hallucination Ground Truth + Diarization Enrichment

Went through every low-confidence word in a session by hand, with Choksi confirming what was actually said. Only four were real hallucinations, and three of them sat in spots where speaker detection had heard silence — two independent systems disagreeing turned out to be a far better hallucination signal than low confidence alone.

## 2026-02-07 — Full Pipeline Validation Run

Ran the complete pipeline, every correction stage included, against the real recording for the first time. It made 33 name corrections with no false positives.

## 2026-02-06 — Normalization on Real Audio: Punctuation Bug

Ran the name-correction stages against the real Mahabharata recording for the first time and hit a small but blocking bug: Whisper attaches trailing punctuation to words, so a name with a comma stuck to it never matched the reference list. Fixing it on the spot beat noting it for later — seeing the real correction counts was worth more.

## 2026-02-06 — Inline Corrections Architecture

Designed how name corrections get recorded, working it out through conversation rather than up front. The transcript became a living document that each stage adds to and never destroys — the original Whisper text is always preserved underneath any correction. The architecture conversation, with its stream of "but what if?" questions, was the most valuable part of the work.

## 2026-02-06 — Full Transcript LLM Beats Segment-by-Segment

Compared correcting Sanskrit names one segment at a time versus handing the model the whole transcript at once. The whole-transcript approach won cleanly — it caught everything with no mistakes, while the piecemeal version invented wrong corrections like turning "dad" into a Sanskrit name, because it lacked the surrounding context to know better.

## 2026-02-05 — Segment-Level Filtering + Sanskrit Name Experiment

Wired hallucination filtering into the validator. The first attempt filtered word by word and wrongly stripped valid quiet words out of good sentences — the fix was to treat filtering as a whole-segment decision, because hallucinations are segment-shaped problems, not word-shaped ones.

## 2026-02-04 — Hallucination Filters Are Useless, Then Reverted

Tested whether Whisper's own quality signals could flag hallucinations — temperature and compression caught none of them. Built a confidence-based filter, then reverted it the same day after realizing an existing query-time filter already handled exactly those cases. Worth checking whether the tools you already have solve the problem before building new ones.

## 2026-02-03 — Architecture Clarification: Artifacts vs Queries

Stripped the pipeline down to three saved files, deleting several intermediate ones that could just as easily be computed on demand. The principle that emerged: pipeline stages should preserve complete raw data, and filtering should happen later at query time — which keeps the source material intact for experimentation.

## 2026-02-01 — Course Correction: Wrong Whisper Model

Discovered that a stretch of earlier findings had been measured against the wrong Whisper model, which invalidated them — most notably a supposed audio-length threshold that doesn't actually exist. Conclusions drawn from a misconfigured setup are worse than no conclusions, because they create false confidence.

## 2026-01-31 — Whisper Audio Length Threshold (Later Invalidated)

Pinned down what looked like a precise audio-length threshold where Whisper's behavior changes, narrowed in through repeated trials. The exact number was later thrown out — wrong model — but one piece survived: Whisper needs a few minutes of context to get quiet speakers' words right, and short clips produce worse results than full recordings.

## 2026-01-29 — Heuristics Dead End, Pivot to LLM

Built a set of timing rules to guess which speaker an ambiguous gap belongs to, then reverted all of it — a time threshold simply can't tell a listener's "uh-huh" apart from a sentence continuing, only the meaning of the words can. The work pivoted to using a language model instead. Use intelligence, not heuristics: stacking brittle rules feels like progress but doesn't generalize.

## 2026-01-27 — Validation Player Spec + Orchestration Research

Designed the validation player — a way to listen back to a recording with the transcript synced — and researched how much agent orchestration the project actually needs. The answer was: very little, save the elaborate setups for bigger systems.

## 2026-01-26 — Pipeline Testing on Real Recordings

Ran the pipeline on two more real bedtime-story recordings. Speaker separation held up and Arti's quiet voice came through in many places, but the Sanskrit name problem showed up consistently — Pandavas heard as "Fondos," "Bondos," or "Pondos" depending on the moment.

## 2026-01-25 — JSON Output Complete: Audio In → Structured JSON Out

The pipeline is now complete in its basic form: an audio file goes in, and a structured file comes out with speaker-labeled conversation and a timestamp on every word. The design saved generously — keeping room for things like splitting out individual stories later — while building only what's needed now. Those per-word timestamps leave the door open for caption-style playback, words highlighting as the audio plays.

## 2026-01-24 — Hallucination Filtering: Two Layers, Different Purposes

Worked out a two-layer approach to hallucinations that turns on one distinction: speech that was real but couldn't be decoded gets marked unintelligible and kept, while text the model fabricated outright gets deleted. Honest transcripts, not clean ones. There was also a build-versus-buy lesson here — the value of writing this yourself the first time isn't the code, it's the judgment you gain about the problem.

## 2026-01-23 — Leading Fragments + Model Clarity

Tracked down the last handful of unlabeled speech fragments and found most were turn-starts — a few words right before the next speaker, where speaker detection missed the boundary. Assigning those to the upcoming speaker cut unidentified fragments from 75 down to 1. The single stubborn one was a hallucination of the trickier kind: the model heard quiet speech and invented plausible words, which is far harder to spot than obvious repetition.

## 2026-01-22 — Alignment Pipeline: First Speaker-Labeled Transcript

Built the step that combines transcription with speaker detection, producing the first transcript that actually reads as a conversation — unlabeled fragments dropped from 75 to 7, and 159 scattered pieces consolidated into 23 real turns. Arti remembering Duryodhana and Yudhishthira, in print: this is what the project is for.

> SPEAKER_01: Dad, why do the Fondos and the Goros want to be king?
> SPEAKER_00: Uh-huh. Well, so the oldest brother of the Goros, his name was, do you remember?
> SPEAKER_01: Durioden.

## 2026-01-21 — Diarization Working + Model Size Matters for Child Speech

Got speaker detection running, and compared Whisper's small and large models on the same stretch of audio. The small model produced absolute silence where Arti speaks; the large model caught her full sentence. With the wrong model, ten seconds of a child's voice simply vanish — and her voice is the whole point of the project.

## 2026-01-20 — Day 1: Project Started

Set up the project and ran the first transcription: a 5.6-minute Mahabharata bedtime story through Whisper. It worked, but mangled every Sanskrit name — "Yudhishthira" became "you this there," "Pandavas" became "fondos." Fixing that mangling became the project's first real problem.
