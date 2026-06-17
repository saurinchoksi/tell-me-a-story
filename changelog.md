# Changelog

Newest entries at top. Each entry is a few plain sentences — what changed, and what it taught when there's a real lesson. Exact diffs live in git history; decision context lives in Linear.

*Claude Code drafts these entries from each session's work; Saurin Choksi reviews and approves.*

## 2026-06-16 — A faster, tidier pipeline that splits each recording into its stories

Reworked how the recording pipeline runs its AI steps and added story-splitting on top. Each recording is now split into its separate bedtime stories — saved right in the transcript and shown as labelled dividers on the screen — and the name-checker reads those instead of re-splitting every time. Re-cleaning a recording no longer redoes the slow AI steps when nothing they depend on changed; each heavy step runs as its own short-lived helper that frees the Mac's memory when done; and a long-standing glitch where a fixed-up name didn't reach the transcript line is gone. Every piece landed as its own tested step, and a multi-angle code review caught a real bug — a stuck AI step could hang forever instead of timing out — which is now fixed and guarded by a test.

## 2026-06-16 — One AI setup instead of two

The project kept two separate Python setups — one for most of the work, a second only for the Gemma model the name tools use — because an old library clash stopped them sharing one. That clash is gone: an experiment confirmed all four AI pieces (transcription, the speaker model, the word-fixer, and Gemma) now run together in a single setup, producing the very same results. Merged them into one, so there's one thing to install and keep current instead of two. The heavy steps still each run as their own short-lived helper process to keep the Mac's graphics memory clean — that isolation was never about having two setups, only about giving each model a fresh process.

## 2026-06-16 — The story-name auditor now runs in the Monitor

The per-story reader that catches a misheard name — built and scored on the eval side this week — now runs as a standing check in the Monitor, beside the family-name and inconsistency detectors. For each recording it splits the stories, reads each one with a small on-device model, and flags an invented name spelled several ways or a real character spelled wrong, while leaving correctly-spelled real names — a show's actual engine — alone. On the five test recordings it caught every clear mistake and stayed silent on the two clean ones, even where the older word-by-word checks fired dozens of false alarms. Because reading a whole recording takes minutes, it runs when a recording is added and on demand, never during a page view — a new "offline-only" setting keeps any slow model out of every web request — and it ships experimental until it has been checked against recordings it hasn't seen.

## 2026-06-16 — Auditing each story's names with a small on-device reader

With the recordings now split into their separate tales, built the reader that checks the names in each one — catching a well-known character whose name was misspelled, and an invented name the transcriber spelled several different ways. Tried three ways of feeding it the material and scored each against the by-ear answer keys: handing it a tidy list of the names with example lines beat making it read everything, which found one extra mistake but raised far more false alarms. A follow-up pass got the best of both — it now catches every misspelling (even one hiding in lowercase) while leaving correctly-spelled real names alone, by asking itself which names in a look-alike group are genuine before flagging. It also confirmed a hard limit worth being honest about — when a made-up name is written as a different real name that sounds like it, reading the text can't catch it; only the audio can.

## 2026-06-15 — Splitting a recording into its separate stories

Bedtime recordings often hold more than one story, and the per-story name checks can't begin until we know where each one starts and stops. Built a splitter: cheap text-and-timing signals propose likely boundaries, a small on-device model reads the recording in order to confirm them and name each story's world, and a second pass folds back together any story a milk break had split in two. Against the hand-marked answer key it gets the number of stories right on all five recordings — including the one with three — without wrongly splitting the four single-story ones. The lesson that shaped it: the second pass may only merge, never delete — the one time it was allowed to delete, it threw away a real story that happened to open with collaborative chatter.

## 2026-06-15 — A drag-to-mark tool for setting where each bedtime story starts and ends

One recording often holds several stories with wandering, milk-break gaps between them, and the name checks have to run per story — a made-up name can differ across stories, and which show or book world applies is a per-story fact — so that split must be marked by hand before it can be automated. The new tool shows a recording as a column of lines with each story drawn as a colored bar whose top and bottom you drag to the exact start and end; anything outside a bar counts as not-story and is skipped, and any line can be played to pin a fuzzy start by ear. A small direct-manipulation lesson surfaced along the way: the drag first hunted for the line under the cursor at a fixed spot near the left edge, which is empty margin on a centered page, so it had to find the line by height instead.

## 2026-06-15 — A by-ear review split name errors into four kinds, one of them invisible

Going through one recording's flagged names against the audio — with a small tool that plays each word and lets the real name be typed in — surfaced a failure no text check can catch: a made-up character was sometimes transcribed as a real, similar-sounding character from a TV show, a confident wrong name that reads as perfectly correct. That sits alongside three kinds already known — the family's own names, names invented on the spot, and names with an outside source like a show or book — so the project's name map now tracks four cases. It also corrected a long-standing example that had been filed as a simple inconsistent spelling, when it was really one of these silent swaps. The lesson that keeps surfacing: when a wrong name happens to be another valid name, only an ear or a model that follows the story can catch it; spelling, roster, and consistency checks all pass it through.

## 2026-06-12 — The Monitor now scans and views as separate steps

The Monitor used to re-run its checkers quietly every time you opened the page, which was clever but wonky: it forced every checker to be fast enough to finish inside a page load, which is why the slow AI model had to be bolted on as an awkward special case. Now scanning and viewing are split cleanly. Checkers run when there's a reason to — right after a recording is transcribed, and on demand from a "Re-scan" button — and run the full pass including the model. Opening the page just reads the last results; if a transcript changed since its scan, the row shows a small "re-scan" mark instead of silently recomputing. The same shape now fits every checker, including future slow ones, with no per-checker special-casing.

## 2026-06-12 — A second name checker ships, and this time a small AI model earned its place

The Monitor gained a second checker: it catches a made-up character name spelled several ways within one story, which would otherwise break tracking that character across the telling. Plain code finds the inconsistencies and a dictionary screens out the noise, but that screen is blind to invented names that are also ordinary words — so those go to a small local AI model that reads a few lines and decides whether the word is a character. A head-to-head of four local models, with the wording tuned to each, landed the opposite way from the first checker: the smallest deployable model won outright and beat one three times its size, lifting the share of inconsistent names caught from under half to most while keeping the flags trustworthy. The model is too slow to run on every page view, so it runs as an explicit upgrade pass; the lesson that keeps repeating is that you only learn whether code or a model wins by measuring, never by guessing up front.

The name-checker proven in the eval project now runs over every recording as part of the system, with a new Monitor screen showing flagged words across all sessions and within each one — the transcript itself is never touched, and the framework takes more failure-mode checkers as they get validated. Each flagged word has a play button that plays the audio around it and stops on its own, so a questionable catch can be checked by ear in one click; it plays the whole containing sentence rather than a tight slice around the word, since the word's own timestamps drift enough to clip the name. Saved results carry fingerprints of both things they depend on, the transcript and the family name list, so editing either re-scans automatically on the next page view; there is no manual step. A same-day code review caught that the first version watched only the transcript half, a reminder that cached results without an invalidation signal quietly turn into lies.

## 2026-06-10 — The eval project's working doc moved into the repo, kept private

The doc that holds the evaluation project's method and running findings now lives inside the project folder instead of a separate reference directory, after a workspace reorganization. Because it carries family names and job-search context, it is excluded from version control while the tools and write-up drafts around it stay public. A file that moves into a public repo's folder is published by default; keeping it private has to be done explicitly.

## 2026-06-01 — A one-line rule beat a local model at cleaning up name flags

The name-checker's one false alarm — an everyday word that sounds like a name — prompted a test: could the smallest on-device language model, reading each flag in context, drop the false alarms while keeping the real names? It proved reliable but not accurate — it also threw out a real name that doubles as a common word, a worse trade than it fixed. A single deterministic rule — keep only capitalized flags — removed the false alarm and kept every real name, so the checker now uses that instead of a model.

## 2026-06-01 — Name detector checked against recordings it had never seen

The deterministic name-checker — which flags when the family's own names come out mistranscribed — was scored by ear against two recordings held out of the earlier analysis. It caught every name error a fresh listen turned up, including spellings of the child's name it had never been shown, and raised a single false alarm on an ordinary word that merely sounds like her name. Checking it on recordings it had never seen, rather than the ones it was tuned on, is what turns a hopeful accuracy figure into a trustworthy one.

## 2026-06-01 — Playback speed is now a dropdown, with a 2× option

The validator's audio speed used to be a row of buttons topping out at 1.5×. It's now a single dropdown that also offers 2×, so you can move through a long recording faster without crowding the control bar.

## 2026-05-29 — Evals project files gathered into one folder

The scattered evals scripts and result files now live under one top-level folder, and the failure-mode counts render as a standalone page; the recordings and coding files stayed where they were.

## 2026-05-28 — Checking whether each word lands where it was actually spoken

Every word in a transcript carries a start and end time, but the transcriber guesses those rather than measuring them, so they drift — and by ear you can only catch the worst few. A new pass weighs each word's claimed moment against the actual sound across all five reviewed recordings and flags the clear misses: a near-silent spot at the claimed time with the real word audible right beside it. Even after setting aside everything that's really a different problem — filler-word loops, hallucinations, and wrong-speaker or wrong-word lines already marked elsewhere — the longest recording still held dozens of genuine cases against the handful catchable by ear, confirming the true rate is several times what hand-counting shows. The lesson that kept recurring through the build: cross-check the transcript against the sound and against the notes already taken, so each problem is counted once and nothing masquerades as a timing error.

## 2026-05-28 — Finding the speech that never made it onto the page

When someone speaks but the transcriber writes nothing at all, there's no line for a reviewer to flag, so counting by hand never sees it. A new pass lays who was talking against what got written down across all five reviewed recordings and lists every stretch that fell through, at both a strict and a loose cutoff. The longest, busiest session held far more of these silent drops than the rest — over a minute of speech with no trace at all — which a count built only from what's on the page would have badly understated.

## 2026-05-28 — A stuck-transcriber stretch now counts as one failure, not dozens

When the transcriber gets stuck it writes the same little word — "Hmm.", "Right." — over and over across a run of segments, sometimes burying real speech underneath, and tagging each one separately made a single broken moment look like dozens of failures. A new, tenth failure mode now marks the whole stretch once and folds the repeats away, while the real lines caught in the middle are left untouched. Reading the labels closely turned up two more of these loops than we knew about, both hiding in a session that had been reviewed past the end of its story.

## 2026-05-21 — The validator got friendlier for axial coding

To make axial coding transcript failures more friendly, the validator tool received the following upgrades:
- Segment cards now show a strip of easily selectable failure modes.
- Hovering a card or any word inside it lights up the matching audio range on the wave player.
- Segment header timestamps switched to m:ss to stop forcing mental math against the player's format.

## 2026-05-20 — The Sessions list can be sorted by column

Click a column header on the Sessions list — Date, Length, Validation, or Notes — to sort by that column; click again to flip the direction. The point is finding unvalidated recordings quickly: sorting by validation pushes those to the top. Untranscribed entries have no length, so they cluster at one end rather than scattering through the list.

## 2026-05-18 — The Sessions list flags failed pipeline stages

The Sessions list now shows a warning when part of a session's processing failed — hover it to see which step. Until now the page only checked that the output files existed, which isn't the same as the work behind them succeeding. The warning caught two old sessions that had been failing without a trace.

## 2026-05-15 — Length, notes, and validation columns added to the Sessions list

The Sessions list now has a header row and three new columns: each recording's length, its count of review notes, and a validation state. You set that state by hand — not started, in progress, done — because whether a session has really been checked is a human call, not something the system can infer.

## 2026-03-12 — Gemma 3n added to the model-experiment plan

A search through Hugging Face turned up Gemma 3n, Google's audio-capable model. It's the only one of the four candidates that runs natively on this Mac without a separate framework, so it goes first — lowest setup cost. New rule for what gets evaluated: it has to run on the existing stack as-is.

## 2026-03-10 — Per-stage timing added to the pipeline

Every pipeline stage is now timed, and a summary block records the totals: how long the whole run took, how many words, segments, and speakers came out, and the size of each output file. The pipeline prints that summary when it finishes.

## 2026-03-10 — Task tracking moved to Linear

All task tracking moved into Linear, retiring the old setup — a Notion database plus handoff files passed between Claude instances. It's the single source of truth now, with skills for Claude Desktop and Claude Code to work against it. Listing tickets took a dozen-plus calls before and takes one now — the right tool mattered more than the effort already sunk into the wrong one.

## 2026-03-03 — Speaker identification works across sessions

Speaker identification now runs automatically when the pipeline finishes. On a completely different recording it matched Saurin at 96% and my daughter at 94%, working from just one earlier voice sample each — a profile generalizes across sessions from a single enrollment. Odd cases, like a silly voice or a distant mic, land in the lower "suggested" tier instead of being forced into a confident match.

## 2026-03-02 — Speaker identification built, backend to web UI

The whole speaker-identification feature came together in a day, from profile storage to the web interface. It works end to end: the pipeline extracts a voice fingerprint, proposes matches, a person confirms them on a review screen, and profiles accumulate over time. A cold start — nothing identified yet — was designed to feel like a beginning, not an error.

## 2026-02-26 — Session Reviewer added to the portfolio page

The portfolio page gets a Session Reviewer section, with a screenshot. Until now it showed the pipeline but no tools and no person in the loop — the reviewer is where the human enters the story.

## 2026-02-22 — A pipeline visualization designed for the portfolio page

An interactive visualization of the pipeline is designed for the portfolio page, along with a sharper framing of what the project does: messy audio in, structured transcript out. Ordering the walkthrough with speaker detection before name correction also tells a better story — meet the speakers first, then fix what they said.

## 2026-02-20 — Code review: 14 findings, 8 fixes shipped

A review of every source and test file turned up 14 items, and all 8 of the actionable fixes shipped in a single commit. The one big deferral is architectural — splitting the Mahabharata-specific logic out of the core pipeline — and it needs a design conversation before any code.

## 2026-02-20 — README rewritten around a before-and-after sample

The README now opens with a before-and-after sample — raw Whisper output becoming a clean, speaker-labeled transcript — which sells the project far better than any architecture description. The entry-point scripts got tests, and the validator's note-saving became instant. The recurring lesson: building infrastructure for problems you don't have yet costs more than the friction it would remove.

## 2026-02-19 — A real-world recording exposes a new kind of hallucination

The first genuinely real-world session went through the pipeline — a ten-minute original moon story with several people in the room. It exposed a failure the filters don't catch: hallucinated text sitting on top of perfectly clean audio. Controlled test recordings, it turns out, don't predict how the pipeline behaves on messy household audio.

## 2026-02-18 — The pipeline runs end to end from one command

The pipeline now runs start to finish from one command — drop audio in an inbox folder, run one script. It also detects gaps, where speaker detection hears someone but transcription produces nothing, and marks them honestly as unintelligible rather than guessing. A crash where two AI models fought over the Mac's GPU memory got fixed by running one of them in a separate process.

## 2026-02-18 — Speaker labels become visible in the transcript validator

The transcript validator now shows who said what — each segment carries a speaker-colored border and badge, and word-level mismatches get a wavy underline. Checking it by eye caught three bugs unit tests never would have: border opacity, class names, and badge visibility are all things you can only verify by looking.

## 2026-02-18 — Ollama replaced by an in-process model library

Ollama — a separate inference server the pipeline reached over the network — is replaced by a model library that loads straight into Python. Every stage now works the same way: load the model, use it, unload it. Ollama had been solving a problem this project doesn't have, serving many users at once, and that mismatch kept showing up as silent failures.

## 2026-02-17 — Source files consolidated from 12 to 8

The source code went from twelve files to eight, and each module now owns its identity instead of the orchestrator stamping it in. The guiding question when something felt misplaced: follow it to the one piece of code that actually uses it, and put it there.

## 2026-02-17 — Session folders simplified from five files to three

Each session folder now holds three files instead of five: the raw transcript exactly as Whisper produced it, the speaker data, and an enriched transcript with corrections layered on top. The raw version stays separate and is never overwritten — the honest record of what the machine actually heard has to survive.

## 2026-02-10 — The multi-agent setup is removed

A multi-agent setup — separate reviewer and coder agents with an orchestrator — is removed, after a review found it solved a problem a fourteen-file project simply doesn't have. The takeaway: before adopting any practice that sounds smart, ask whether the project actually has the problem it solves.

## 2026-02-09 — Journals consolidated into a single changelog

Twenty-nine separate journal files folded into this one changelog — 103KB of scattered notes down to 19KB in one place — as part of reworking the workflow around Claude's newer capabilities. The changelog is the single historical record now; the journals are gone. The work also settled a clear division of labor: instructions for guaranteed behavior, memory for accumulated knowledge, files on disk for anything that must be read explicitly.

## 2026-02-08 — Claude Code takes over the quiet-speech experiments

After three hand-run attempts to recover my daughter's quietest speech all failed, the work shifted to letting Claude Code experiment on its own. The finding was firm: her voice is genuinely faint, not masked by noise, and Whisper has a hard floor that no decoding setting nudges past. What decided the quality of the autonomous work was the handoff — tight context, loose approach.

## 2026-02-08 — Audio boosting can't recover quiet speech

Fourteen combinations of audio processing and prompting were tried to recover a moment where speaker detection heard my daughter but transcription produced nothing. None worked — boosting volume can't create information the mic never captured. The attempt also exposed a trap: feeding Whisper narrative context as a hint makes it skip the matching audio entirely, so a hint should only ever be a plain word list.

## 2026-02-08 — Re-running corrections without re-transcribing

There's now a way to re-run the correction and labeling steps on an existing session without redoing the expensive transcription and speaker detection from scratch. The shared logic lives in one function, used by both fresh runs and re-runs rather than copied into each.

## 2026-02-07 — Three filters catch hallucinations with no false positives

A blunt confidence threshold for catching hallucinated words is replaced by three targeted filters, which caught every known case with no false positives. The key discovery: a hallucination is almost always a lone one-word segment — its shape, not its confidence number, is what gives it away.

## 2026-02-07 — Hallucinations show up where two systems disagree

Every low-confidence word in a session got checked by hand, with Choksi confirming what was actually said. Only four were real hallucinations — and three sat where speaker detection had heard silence. Two systems disagreeing turns out to be a far better signal than low confidence alone.

## 2026-02-07 — The first full pipeline run on real audio

The complete pipeline ran against the real recording for the first time, every correction stage included. It made 33 name corrections with no false positives.

## 2026-02-06 — Trailing punctuation breaks name correction

The name-correction stages ran against the real Mahabharata recording for the first time and hit a small but blocking bug: Whisper attaches trailing punctuation to words, so a name with a comma stuck to it never matched the reference list. Fixing it on the spot beat noting it for later — seeing the real correction counts was worth more.

## 2026-02-06 — Corrections recorded inline on the transcript

How name corrections get recorded was worked out through conversation rather than designed up front. The transcript became a living document each stage adds to and never destroys — the original Whisper text stays preserved underneath every correction. That design conversation, with its stream of "but what if?" questions, was the most valuable part of the work.

## 2026-02-06 — Whole-transcript name correction beats segment-by-segment

Two ways of correcting Sanskrit names were compared: one segment at a time, or the whole transcript handed to the model at once. The whole-transcript approach won cleanly — it caught everything with no mistakes, while the piecemeal version invented wrong corrections like turning "dad" into a Sanskrit name, for lack of surrounding context.

## 2026-02-05 — Hallucination filtering moves from words to whole segments

Hallucination filtering is wired into the validator. The first attempt worked word by word and wrongly stripped valid quiet words out of good sentences; the fix was to make it a whole-segment decision, because a hallucination is a segment-shaped problem, not a word-shaped one.

## 2026-02-04 — A hallucination filter is built and reverted the same day

Whisper's own quality signals were tested as hallucination flags — temperature and compression caught none. A confidence-based filter got built, then reverted the same day once it turned out an existing query-time filter already handled those cases. Check whether the tools you already have solve the problem before building new ones.

## 2026-02-03 — The pipeline keeps three files, computes the rest on demand

The pipeline is stripped to three saved files; several intermediate ones, easily computed on demand, are deleted. The principle that emerged: a stage should preserve complete raw data, and filtering should happen later at query time. That keeps the source material intact for experimentation.

## 2026-02-01 — Earlier findings invalidated: the wrong Whisper model

A stretch of earlier findings turned out to have been measured against the wrong Whisper model, which invalidates them — most of all a supposed audio-length threshold that doesn't actually exist. Conclusions drawn from a misconfigured setup are worse than no conclusions: they create false confidence.

## 2026-01-31 — A Whisper audio-length threshold (later invalidated)

Repeated trials seemed to pin down a precise audio-length threshold where Whisper's behavior changes. The exact number was later thrown out — wrong model — but one piece survived: Whisper needs a few minutes of context to get quiet speakers' words right, and short clips do worse than full recordings.

## 2026-01-29 — Speaker-gap heuristics dropped for an LLM

A set of timing rules tried to guess which speaker an ambiguous gap belongs to, then got reverted entirely — a time threshold can't tell a listener's "uh-huh" from a sentence continuing; only the meaning of the words can. The work pivoted to a language model. Use intelligence, not heuristics: stacking brittle rules feels like progress but doesn't generalize.

## 2026-01-27 — The validation player tool is designed

The validation player is designed — a way to listen back to a recording with the transcript synced along. A side question, how much agent orchestration the project needs, got an answer: very little, save the elaborate setups for bigger systems.

## 2026-01-26 — The pipeline tested on two more real recordings

The pipeline ran on two more real bedtime-story recordings. Speaker separation held up and my daughter's quiet voice came through in many places. But the Sanskrit name problem showed up consistently — "Pandavas" heard as "Fondos," "Bondos," or "Pondos" depending on the moment.

## 2026-01-25 — The pipeline produces a structured transcript from audio

The pipeline is complete in its basic form: an audio file goes in, a structured file comes out — speaker-labeled conversation with a timestamp on every word. The design saved generously, leaving room for things like splitting out individual stories later, while building only what's needed now. Those per-word timestamps open the door to caption-style playback, words lighting up as the audio plays.

## 2026-01-24 — Hallucination handling becomes two layers

Hallucination handling settles into two layers, on one distinction: speech that was real but couldn't be decoded gets marked unintelligible and kept, while text the model fabricated outright gets deleted. Honest transcripts, not clean ones. There's a build-versus-buy lesson too — the value of writing this yourself the first time isn't the code, it's the judgment you gain about the problem.

## 2026-01-23 — Unlabeled speech fragments cut from 75 to 1

The last handful of unlabeled speech fragments turned out to be mostly turn-starts — a few words right before the next person speaks, where speaker detection missed the boundary. Assigning those forward cut the unlabeled count from 75 to 1. The one that stayed was a trickier hallucination: the model heard quiet speech and invented plausible words, far harder to spot than obvious repetition.

## 2026-01-22 — The first speaker-labeled transcript

The step that combines transcription with speaker detection is built, and it produces the first transcript that reads as an actual conversation — unlabeled fragments down from 75 to 7, and 159 scattered pieces consolidated into 23 real turns. My daughter remembering Duryodhana and Yudhishthira, in print: this is what the project is for.

> SPEAKER_01: Dad, why do the Fondos and the Goros want to be king?
> SPEAKER_00: Uh-huh. Well, so the oldest brother of the Goros, his name was, do you remember?
> SPEAKER_01: Durioden.

## 2026-01-21 — Child speech needs the large Whisper model

Speaker detection is running, and Whisper's small and large models were compared on the same stretch of audio. The small one produced absolute silence where my daughter speaks; the large one caught her full sentence. With the wrong model, ten seconds of a child's voice simply vanish — and her voice is the whole point of the project.

## 2026-01-20 — Day one: the project begins

The project is set up, and the first transcription runs: a 5.6-minute Mahabharata bedtime story through Whisper. It works, but mangles every Sanskrit name — "Yudhishthira" becomes "you this there," "Pandavas" becomes "fondos." Fixing that became the project's first real problem.
