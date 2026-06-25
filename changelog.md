# Changelog

Newest entries at top. Each one is a few plain sentences: what changed, and what it taught when there's a real lesson. Exact diffs live in git history; decision context lives in Linear.

*Claude Code drafts these entries from each session's work; Saurin Choksi reviews and approves.*

## 2026-06-24 — The name check now shows its best guesses, not just the sure things

The check that catches a known character spelled wrong used to drop any catch whose suggested spelling didn't sound close enough to what was heard, quietly losing real mistakes it had already found (a badly garbled Dhritarashtra, say). It now keeps them, sorted by confidence: the sure ones on top, a "best guess" group below, a "Show all" link for the shakiest. A scored test on real and invented worlds backed the change. Its one cost, a child inventing names that sound real, the "made-up name" button already clears in a click.

## 2026-06-24 — You can now overrule the name monitor when it's wrong

Sometimes a name check gets a name wrong. The usual case: a child invents a character that sounds exactly like a real one (a made-up engine that sounds like Thomas's James), and the check for misspelled known characters insists it's the real one. You can now set it straight right on the page, either "this isn't that character" or "this spelling is fine, leave it," and the flags rearrange: the false claim drops, and the genuinely inconsistent invented name shows up where it belongs. The correction saves next to the transcript so it survives every re-scan, leaves the family-name check untouched, and doubles as a record of where the detectors get things wrong.

## 2026-06-24 — Each session now says what story it holds

The Sessions list and the Monitor used to label every recording by date alone, so you had to open one to see what was inside. Both now show a short content line: the story titles, plus the recognized world when there is one (the Mahabharata night, the Thomas the Tank Engine night). The same summary heads a session's own monitor page too, so it stays recognizable once you click in. Nothing new gets computed for it; the page already opens each transcript to read its length, so the summary rides along on that read.

## 2026-06-23 — The three name checks now look and read the same

The session page runs three name checks: a family name, a name spelled several ways, and a known character spelled wrong. Over time they'd grown three different looks, one grouped under a per-name heading, the other two flat lists, one's tags an unstyled grey. They now share one layout: each name gets a single heading with a colored tag and its spellings, and every mention sits underneath. Some dead code went out with it, and the all-zeros sample recording no longer shows up as a phantom "1899" row.

## 2026-06-23 — A mis-heard known name no longer shows up in two sections at once

When a known character's name was written several ways, it could show up in both the "inconsistent spelling" section and the "known character misspelled" section at once, the same character split across two lists. The part meant to hand that name to one section was working one spelling at a time, so it peeled off the few spellings the reader claimed and left the rest, including the correct one, in the other section. It now hands over the whole name at once, so the character lands in exactly one place. A small fix rode along: two recordings from the same day now show the time in their titles, so they stop looking like one session.

## 2026-06-21 — The known-character name check gets steadier and easier to read

The known-character name check now reads better and trusts itself more. Every spelling of one character sits under a single heading instead of five separate flags, and a catch shows only when the suggested spelling actually sounds like what was heard (a "Jammus" that's plainly James). It also runs several times over shuffled name lists and keeps what most runs agree on, because the model's answer on a borderline name turned out to depend on the list's order. That taught us its limit: told a story's world, the model will confidently read a child's made-up character as a misspelled real one, so we show only the sound-alike catches.

## 2026-06-19 — The name checker switched to a smarter model, and now it works

The check that catches a known character spelled wrong (Bhishma heard as "Bishma," Karna as "Garn") went from barely working to working, and the fix was a better model, not more code. The old small reader caught about 1 of 11 real misspellings on the held-out Mahabharata recording. A newer model that fits the same memory, asked the same simple question, caught 8 of 11, including the badly garbled name we'd written off as the audio floor. The story-splitting step moved to the new model too, and every recording was re-split and re-checked, so the monitor reflects the change everywhere. The lesson worth keeping: a capable model wants the simplest prompt, and one stray phrase ("names may be misspelled") quietly switched off world-recognition until we removed it.

## 2026-06-17 — The name checker stops flagging everyday words as misspelled characters

The check for misspelled known characters had a noisy flip side: on the held-out Mahabharata recording it read ordinary words like "arrows" and "wars" as misspellings of a character it invented called "Arrow," turning 2 real catches into 24 flags. It now ignores a word the dictionary already knows, unless the story uses it as a name, trimming those 24 back to the 2 true ones. The careful part was keeping the real names: this machine's word list includes much of the mythology itself (Krishna, Drona, even Dhritarashtra), so "it's in the dictionary" can't mean "it's ordinary" on its own. A word is dropped only when the dictionary knows it and the story never writes it as a capitalized name.

## 2026-06-17 — Bedtime stories now recognize their world from the whole story

The step that decides whether a recording comes from a known world (the Mahabharata) or is made up was reading only a dozen scattered lines and leaning on a few hard-coded example worlds. So on a long Mahabharata recording it never saw the telling names, called the whole thing made-up, and quietly switched the known-character check off. It now reads the whole story with a general prompt, no built-in list of our own worlds. To prove it I wrote a hundred fabricated bedtime stories, half from real worlds like Star Wars and He-Man, half made up, and scored the old wording against new ones: the best new prompt went from about two-thirds right to four-fifths. The honest surprise, caught only on the held-out Mahabharata recording the tuning never touched: the bigger cause was the dozen-line sampling, not the wording. Fed the whole story, even the old prompt recognized the Mahabharata, so both fixes ship. One limit stays: a small on-device reader still struggles to place less-famous shows from a few garbled names.

## 2026-06-17 — A recognized name no longer shows up under two checks

When the reader recognizes a known name spelled wrong (the Mahabharata's "Pandavas"), that name now shows only under the known-character check, not doubled under the made-up check as "inconsistent." The dedup happens at view time and does nothing when the reader didn't recognize the world, so a recording where it missed the world keeps those catches under the made-up check rather than losing them. Separately, the full detector set ran over every recording, so the monitor now reflects the current checks everywhere. One honest finding: the on-device reader recognized the world on the recordings it was built on, but misread the held-out Mahabharata recording as made-up and caught nothing there, so on that one the errors still fall to the made-up check.

## 2026-06-17 — The Monitor's three name checks now read cleanly

Fixed three things that made the Monitor's name flags confusing. Pages with a known-character flag were going blank, because the screen knew only two of the three flag shapes; it now handles all three without crashing. The made-up-name check was doing two jobs badly, re-flagging the family name that the family check already owns and flagging contractions like "you're" as names; it now leaves both alone, so the three checks finally mean three distinct things. The biggest change: the made-up check stopped flagging correct spellings. With no answer key it used to flag every spelling of a name used more than one way, so a name spelled right 46 times with one stray typo lit up 47 times. It now takes the spelling used most often in a recording as the intended one and flags only the odd ones, so you see the one typo to fix, not the 46 correct spellings. When no spelling clearly wins it still shows them all, and "most common wins" is only ever a within-recording call, never a claim about the one true spelling everywhere.

## 2026-06-17 — Misspelled known-character names now get their own monitor line

A name from a known source (a Thomas & Friends engine, a Mahabharata character) that the transcriber spelled wrong now shows up as its own check in the Monitor, beside the family-name and made-up-name checks. It was already being caught, bundled inside the combined story-name check; this splits it out by the tag it already carried. The reader still does the full combined pass underneath, so nothing about the accuracy changed, only what you see. It's marked experimental until it's checked against recordings it hasn't seen.

## 2026-06-16 — The pipeline now splits each recording into its stories

Reworked how the pipeline runs its AI steps, and added story-splitting on top. Each recording now splits into its separate bedtime stories, saved in the transcript and shown as labelled dividers, and the name checker reads those instead of re-splitting every time. Re-cleaning a recording no longer redoes the slow AI steps when nothing they depend on changed, each heavy step runs as its own short-lived helper that frees the Mac's memory, and an old glitch where a fixed-up name didn't reach the transcript line is gone. A multi-angle code review caught a real bug, a stuck AI step that could hang forever instead of timing out, now fixed and guarded by a test.

## 2026-06-16 — One AI setup instead of two

The project kept two separate Python setups, one for most of the work and a second only for the Gemma model the name tools use, because an old library clash stopped them sharing one. That clash is gone: an experiment confirmed all four AI pieces (transcription, the speaker model, the word-fixer, and Gemma) run together in one setup with the same results. Merged them, so there's one thing to install and keep current instead of two. The heavy steps still each run as their own short-lived helper to keep the Mac's graphics memory clean, which was never about having two setups, only about giving each model a fresh process.

## 2026-06-16 — The story-name auditor now runs in the Monitor

The per-story reader that catches a misheard name, built and scored on the eval side this week, now runs as a standing check in the Monitor, beside the family-name and inconsistency detectors. For each recording it splits the stories, reads each with a small on-device model, and flags an invented name spelled several ways or a real character spelled wrong, while leaving correctly-spelled real names alone. On the five test recordings it caught every clear mistake and stayed silent on the two clean ones, even where the older word-by-word checks fired dozens of false alarms. Reading a whole recording takes minutes, so it runs when a recording is added and on demand, never during a page view, and ships experimental until it's checked against recordings it hasn't seen.

## 2026-06-16 — Auditing each story's names with a small on-device reader

With the recordings now split into separate tales, built the reader that checks the names in each one: a well-known character misspelled, or an invented name the transcriber spelled several ways. Tried three ways of feeding it the material and scored each against the by-ear answer keys. Handing it a tidy list of the names with example lines beat making it read everything, which found one extra mistake but raised far more false alarms. A follow-up pass got the best of both: it now catches every misspelling, even one hiding in lowercase, while leaving correctly-spelled real names alone, by asking itself which names in a look-alike group are genuine before flagging. It also confirmed a hard limit: when a made-up name is written as a different real name that sounds like it, reading the text can't catch it, only the audio can.

## 2026-06-15 — Splitting a recording into its separate stories

Bedtime recordings often hold more than one story, and the per-story name checks can't begin until we know where each one starts and stops. Built a splitter: cheap text-and-timing signals propose likely boundaries, a small on-device model reads the recording in order to confirm them and name each story's world, and a second pass merges back any story a milk break had split in two. Against the hand-marked answer key it gets the story count right on all five recordings, including the one with three, without wrongly splitting the four single-story ones. The lesson that shaped it: the second pass may only merge, never delete. The one time it could delete, it threw away a real story that happened to open with collaborative chatter.

## 2026-06-15 — A drag-to-mark tool for where each bedtime story starts and ends

One recording often holds several stories with wandering, milk-break gaps between them, and the name checks run per story (a made-up name can differ across stories, and which show or book applies is a per-story fact), so the split has to be marked by hand before it can be automated. The new tool shows a recording as a column of lines, each story a colored bar whose top and bottom you drag to the exact start and end. Anything outside a bar counts as not-story and is skipped, and any line can be played to pin a fuzzy start by ear. A small lesson surfaced: the drag first hunted for the line under the cursor at a fixed spot near the left edge, which is empty margin on a centered page, so it had to find the line by height instead.

## 2026-06-15 — A by-ear review split name errors into four kinds, one of them invisible

Going through one recording's flagged names against the audio, with a small tool that plays each word and lets the real name be typed in, surfaced a failure no text check can catch: a made-up character sometimes transcribed as a real, similar-sounding character from a TV show, a confident wrong name that reads as perfectly correct. That joins three kinds already known (the family's own names, names invented on the spot, and names from an outside source like a show or book), so the name map now tracks four cases. It also corrected an old example filed as a simple inconsistent spelling, when it was really one of these silent swaps. The lesson that keeps surfacing: when a wrong name happens to be another valid name, only an ear or a model that follows the story can catch it; spelling, roster, and consistency checks all pass it through.

## 2026-06-12 — The Monitor now scans and views as separate steps

The Monitor used to re-run its checkers quietly every time you opened the page. Clever but wonky: every checker had to finish inside a page load, which is why the slow AI model was bolted on as an awkward special case. Scanning and viewing are now split. Checkers run when there's a reason to, right after a recording is transcribed and on demand from a "Re-scan" button, and run the full pass including the model. Opening the page reads the last results, and if a transcript changed since its scan, the row shows a small "re-scan" mark instead of silently recomputing. The same shape now fits every checker, slow ones included, with no special-casing.

## 2026-06-12 — A second name check ships, and a small AI model earned its place

The Monitor gained a second check: it catches a made-up character name spelled several ways within one story, which would otherwise break tracking that character across the telling. Plain code finds the inconsistencies and a dictionary screens out the noise, but that screen is blind to invented names that are also ordinary words, so those go to a small local AI model that reads a few lines and decides whether the word is a character. A head-to-head of four local models landed the opposite way from the first checker: the smallest deployable model won outright and beat one three times its size, lifting the share of inconsistent names caught from under half to most while keeping the flags trustworthy. It's too slow for every page view, so it runs as an explicit upgrade pass. The lesson that keeps repeating: you only learn whether code or a model wins by measuring, never by guessing up front.

## 2026-06-11 — The name monitor goes live with its first check

The first name check, proven in the eval project, now runs over every recording, with a new Monitor screen showing flagged words across all sessions and within each one. The transcript itself is never touched, and the framework takes more checkers as they get validated. Each flagged word has a play button that plays the audio around it and stops on its own, so a questionable catch can be checked by ear in one click; it plays the whole sentence, not a tight slice, since the word's own timestamps drift enough to clip the name. Saved results carry fingerprints of the two things they depend on, the transcript and the family-name list, so editing either re-scans on the next page view. A same-day code review caught that the first version watched only the transcript half, a reminder that cached results with no invalidation signal quietly turn into lies.

## 2026-06-10 — The eval project's working doc moved into the repo, kept private

The doc that holds the eval project's method and running findings now lives inside the project folder instead of a separate reference directory, after a workspace reorganization. Because it carries family names and job-search context, it stays out of version control while the tools and write-up drafts around it stay public. A file that moves into a public repo's folder is published by default; keeping it private has to be done on purpose.

## 2026-06-01 — A one-line rule beat a local model at cleaning up name flags

The name checker's one false alarm, an everyday word that sounds like a name, prompted a test: could the smallest on-device model, reading each flag in context, drop the false alarms while keeping the real names? It proved reliable but not accurate, throwing out a real name that doubles as a common word, a worse trade than it fixed. A single rule, keep only capitalized flags, removed the false alarm and kept every real name, so the checker uses that instead of a model.

## 2026-06-01 — Name detector checked against recordings it had never seen

The name checker, which flags when the family's own names come out mistranscribed, was scored by ear against two recordings held out of the earlier analysis. It caught every name error a fresh listen turned up, including spellings of the child's name it had never been shown, and raised one false alarm on an ordinary word that merely sounds like her name. Checking it on recordings it had never seen, rather than the ones it was tuned on, is what turns a hopeful accuracy figure into a trustworthy one.

## 2026-06-01 — Playback speed is now a dropdown, with a 2× option

The validator's audio speed used to be a row of buttons topping out at 1.5×. It's now a single dropdown that also offers 2×, so you can move through a long recording faster without crowding the control bar.

## 2026-05-29 — Evals project files gathered into one folder

The scattered evals scripts and result files now live under one top-level folder, and the failure-mode counts render as a standalone page; the recordings and coding files stayed where they were.

## 2026-05-28 — Checking whether each word lands where it was actually spoken

Every word in a transcript carries a start and end time, but the transcriber guesses those rather than measuring them, so they drift, and by ear you can only catch the worst few. A new pass weighs each word's claimed moment against the actual sound across all five reviewed recordings and flags the clear misses: a near-silent spot at the claimed time with the real word audible right beside it. Even after setting aside everything that's really a different problem (filler-word loops, hallucinations, and wrong-speaker or wrong-word lines marked elsewhere), the longest recording still held dozens of genuine cases against the handful catchable by ear, confirming the true rate is several times what hand-counting shows. The lesson: cross-check the transcript against the sound and against the notes already taken, so each problem is counted once and nothing masquerades as a timing error.

## 2026-05-28 — Finding the speech that never made it onto the page

When someone speaks but the transcriber writes nothing at all, there's no line for a reviewer to flag, so counting by hand never sees it. A new pass lays who was talking against what got written down across all five reviewed recordings and lists every stretch that fell through, at both a strict and a loose cutoff. The longest, busiest session held far more of these silent drops than the rest, over a minute of speech with no trace at all, which a count built only from what's on the page would have badly understated.

## 2026-05-28 — A stuck-transcriber stretch now counts as one failure, not dozens

When the transcriber gets stuck it writes the same little word ("Hmm.", "Right.") over and over across a run of segments, sometimes burying real speech underneath, and tagging each one separately made a single broken moment look like dozens of failures. A new, tenth failure mode marks the whole stretch once and folds the repeats away, while the real lines caught in the middle are left untouched. Reading the labels closely turned up two more of these loops than we knew about, both hiding in a session reviewed past the end of its story.

## 2026-05-21 — The validator got friendlier for axial coding

Three upgrades to make axial coding less of a slog:
- Segment cards now show a strip of failure modes you can click.
- Hovering a card or any word in it lights up the matching range on the wave player.
- Segment timestamps switched to m:ss, so there's no mental math against the player's format.

## 2026-05-20 — The Sessions list can be sorted by column

Click a column header on the Sessions list (Date, Length, Validation, or Notes) to sort by it; click again to flip the direction. The point is finding unvalidated recordings fast: sorting by validation pushes those to the top. Untranscribed entries have no length, so they cluster at one end rather than scattering through the list.

## 2026-05-18 — The Sessions list flags failed pipeline stages

The Sessions list now flags a session whose processing partly failed; hover to see which step. Checking that the output files exist isn't the same as checking the work behind them worked. The flag caught two old sessions that had been failing with no trace.

## 2026-05-15 — Length, notes, and validation columns added to the Sessions list

The Sessions list now has a header row and three new columns: each recording's length, its count of review notes, and a validation state. You set that state by hand (not started, in progress, done), because whether a session has really been checked is a human call, not something the system can infer.

## 2026-03-12 — Gemma 3n added to the model-experiment plan

A search through Hugging Face turned up Gemma 3n, Google's audio-capable model. It's the only one of the four candidates that runs natively on this Mac without a separate framework, so it goes first, lowest setup cost. New rule for what gets evaluated: it has to run on the existing stack as-is.

## 2026-03-10 — Per-stage timing added to the pipeline

Every pipeline stage is now timed, and a summary block records the totals: how long the whole run took, how many words, segments, and speakers came out, and the size of each output file. The pipeline prints that summary when it finishes.

## 2026-03-10 — Task tracking moved to Linear

All task tracking moved into Linear, retiring the old setup: a Notion database plus handoff files passed between Claude instances. It's the single source of truth now, with skills for Claude Desktop and Claude Code to work against it. Listing tickets took a dozen-plus calls before and takes one now, so the right tool mattered more than the effort already sunk into the wrong one.

## 2026-03-03 — Speaker identification works across sessions

Speaker identification now runs automatically when the pipeline finishes. On a completely different recording it matched Saurin at 96% and my daughter at 94%, from one earlier voice sample each, so a profile generalizes across sessions from a single enrollment. Odd cases, like a silly voice or a distant mic, land in the lower "suggested" tier instead of being forced into a confident match.

## 2026-03-02 — Speaker identification built, backend to web UI

The whole speaker-identification feature came together in a day, from profile storage to the web interface. It works end to end: the pipeline extracts a voice fingerprint, proposes matches, a person confirms them on a review screen, and profiles build up over time. A cold start, nothing identified yet, was designed to feel like a beginning, not an error.

## 2026-02-26 — Session Reviewer added to the portfolio page

The portfolio page gets a Session Reviewer section, with a screenshot. Until now it showed the pipeline but no tools and no person in the loop; the reviewer is where the human enters the story.

## 2026-02-22 — A pipeline visualization designed for the portfolio page

An interactive visualization of the pipeline is designed for the portfolio page, along with a sharper framing of what the project does: messy audio in, structured transcript out. Ordering the walkthrough with speaker detection before name correction tells a better story: meet the speakers first, then fix what they said.

## 2026-02-20 — Code review: 14 findings, 8 fixes shipped

A review of every source and test file turned up 14 items, and all 8 of the actionable fixes shipped in a single commit. The one big deferral is architectural, splitting the Mahabharata-specific logic out of the core pipeline, and it needs a design conversation before any code.

## 2026-02-20 — README rewritten around a before-and-after sample

The README now opens with a before-and-after sample, raw Whisper output becoming a clean, speaker-labeled transcript, which sells the project far better than any architecture description. The entry-point scripts got tests, and the validator's note-saving became instant. The recurring lesson: building infrastructure for problems you don't have yet costs more than the friction it would remove.

## 2026-02-19 — A real-world recording exposes a new kind of hallucination

The first genuinely real-world session went through the pipeline, a ten-minute original moon story with several people in the room. It exposed a failure the filters don't catch: hallucinated text sitting on top of perfectly clean audio. Controlled test recordings, it turns out, don't predict how the pipeline behaves on messy household audio.

## 2026-02-18 — The pipeline runs end to end from one command

The pipeline now runs start to finish from one command: drop audio in an inbox folder, run one script. It also detects gaps, where speaker detection hears someone but transcription produces nothing, and marks them honestly as unintelligible rather than guessing. A crash where two AI models fought over the Mac's GPU memory got fixed by running one of them in a separate process.

## 2026-02-18 — Speaker labels become visible in the transcript validator

The transcript validator now shows who said what: each segment carries a speaker-colored border and badge, and word-level mismatches get a wavy underline. Checking it by eye caught three bugs unit tests never would have, since border opacity, class names, and badge visibility are all things you can only verify by looking.

## 2026-02-18 — Ollama replaced by an in-process model library

Ollama, a separate inference server the pipeline reached over the network, is replaced by a model library that loads straight into Python. Every stage now works the same way: load the model, use it, unload it. Ollama had been solving a problem this project doesn't have, serving many users at once, and that mismatch kept showing up as silent failures.

## 2026-02-17 — Source files consolidated from 12 to 8

The source code went from twelve files to eight, and each module now owns its identity instead of the orchestrator stamping it in. The guiding question when something felt misplaced: follow it to the one piece of code that actually uses it, and put it there.

## 2026-02-17 — Session folders simplified from five files to three

Each session folder now holds three files instead of five: the raw transcript exactly as Whisper produced it, the speaker data, and an enriched transcript with corrections layered on top. The raw version stays separate and is never overwritten, because the honest record of what the machine actually heard has to survive.

## 2026-02-10 — The multi-agent setup is removed

A multi-agent setup, separate reviewer and coder agents with an orchestrator, is removed, after a review found it solved a problem a fourteen-file project doesn't have. The takeaway: before adopting any practice that sounds smart, ask whether the project actually has the problem it solves.

## 2026-02-09 — Journals consolidated into a single changelog

Twenty-nine separate journal files folded into this one changelog, 103KB of scattered notes down to 19KB in one place, as part of reworking the workflow around Claude's newer abilities. The changelog is the single historical record now; the journals are gone. The work also settled a clear division of labor: instructions for guaranteed behavior, memory for accumulated knowledge, files on disk for anything that must be read explicitly.

## 2026-02-08 — Claude Code takes over the quiet-speech experiments

After three hand-run attempts to recover my daughter's quietest speech all failed, the work shifted to letting Claude Code experiment on its own. The finding was firm: her voice is genuinely faint, not masked by noise, and Whisper has a hard floor that no decoding setting nudges past. What decided the quality of the autonomous work was the handoff: tight context, loose approach.

## 2026-02-08 — Audio boosting can't recover quiet speech

Fourteen combinations of audio processing and prompting were tried to recover a moment where speaker detection heard my daughter but transcription produced nothing. None worked: boosting volume can't create information the mic never captured. The attempt also exposed a trap: feeding Whisper narrative context as a hint makes it skip the matching audio entirely, so a hint should only ever be a plain word list.

## 2026-02-08 — Re-running corrections without re-transcribing

There's now a way to re-run the correction and labeling steps on an existing session without redoing the expensive transcription and speaker detection from scratch. The shared logic lives in one function, used by both fresh runs and re-runs rather than copied into each.

## 2026-02-07 — Three filters catch hallucinations with no false positives

A blunt confidence threshold for catching hallucinated words is replaced by three targeted filters, which caught every known case with no false positives. The key discovery: a hallucination is almost always a lone one-word segment, so its shape, not its confidence number, is what gives it away.

## 2026-02-07 — Hallucinations show up where two systems disagree

Every low-confidence word in a session got checked by hand, with Choksi confirming what was actually said. Only four were real hallucinations, and three sat where speaker detection had heard silence. Two systems disagreeing turns out to be a far better signal than low confidence alone.

## 2026-02-07 — The first full pipeline run on real audio

The complete pipeline ran against the real recording for the first time, every correction stage included. It made 33 name corrections with no false positives.

## 2026-02-06 — Trailing punctuation breaks name correction

The name-correction stages ran against the real Mahabharata recording for the first time and hit a small but blocking bug: Whisper attaches trailing punctuation to words, so a name with a comma stuck to it never matched the reference list. Fixing it on the spot beat noting it for later, since seeing the real correction counts was worth more.

## 2026-02-06 — Corrections recorded inline on the transcript

How name corrections get recorded was worked out through conversation rather than designed up front. The transcript became a living document each stage adds to and never destroys, with the original Whisper text preserved underneath every correction. That design conversation, with its stream of "but what if?" questions, was the most valuable part of the work.

## 2026-02-06 — Whole-transcript name correction beats segment-by-segment

Two ways of correcting Sanskrit names were compared: one segment at a time, or the whole transcript handed to the model at once. The whole-transcript approach won cleanly. It caught everything with no mistakes, while the piecemeal version invented wrong corrections like turning "dad" into a Sanskrit name, for lack of surrounding context.

## 2026-02-05 — Hallucination filtering moves from words to whole segments

Hallucination filtering is wired into the validator. The first attempt worked word by word and wrongly stripped valid quiet words out of good sentences; the fix was to make it a whole-segment decision, because a hallucination is a segment-shaped problem, not a word-shaped one.

## 2026-02-04 — A hallucination filter is built and reverted the same day

Whisper's own quality signals were tested as hallucination flags, but temperature and compression caught none. A confidence-based filter got built, then reverted the same day once it turned out an existing query-time filter already handled those cases. Check whether the tools you already have solve the problem before building new ones.

## 2026-02-03 — The pipeline keeps three files, computes the rest on demand

The pipeline is stripped to three saved files; several intermediate ones, easily computed on demand, are deleted. The principle that emerged: a stage should preserve complete raw data, and filtering should happen later at query time. That keeps the source material intact for experimentation.

## 2026-02-01 — Earlier findings invalidated: the wrong Whisper model

A stretch of earlier findings turned out to have been measured against the wrong Whisper model, which invalidates them, most of all a supposed audio-length threshold that doesn't actually exist. Conclusions drawn from a misconfigured setup are worse than no conclusions: they create false confidence.

## 2026-01-31 — A Whisper audio-length threshold (later invalidated)

Repeated trials seemed to pin down a precise audio-length threshold where Whisper's behavior changes. The exact number was later thrown out (wrong model), but one piece survived: Whisper needs a few minutes of context to get quiet speakers' words right, and short clips do worse than full recordings.

## 2026-01-29 — Speaker-gap heuristics dropped for an LLM

A set of timing rules tried to guess which speaker an ambiguous gap belongs to, then got reverted entirely: a time threshold can't tell a listener's "uh-huh" from a sentence continuing, only the meaning of the words can. The work pivoted to a language model. Use intelligence, not heuristics: stacking brittle rules feels like progress but doesn't generalize.

## 2026-01-27 — The validation player tool is designed

The validation player is designed, a way to listen back to a recording with the transcript synced along. A side question, how much agent orchestration the project needs, got an answer: very little, so save the elaborate setups for bigger systems.

## 2026-01-26 — The pipeline tested on two more real recordings

The pipeline ran on two more real bedtime-story recordings. Speaker separation held up and my daughter's quiet voice came through in many places. But the Sanskrit name problem showed up consistently: "Pandavas" heard as "Fondos," "Bondos," or "Pondos" depending on the moment.

## 2026-01-25 — The pipeline produces a structured transcript from audio

The pipeline is complete in its basic form: an audio file goes in, a structured file comes out, speaker-labeled conversation with a timestamp on every word. The design saved generously, leaving room for things like splitting out individual stories later, while building only what's needed now. Those per-word timestamps open the door to caption-style playback, words lighting up as the audio plays.

## 2026-01-24 — Hallucination handling becomes two layers

Hallucination handling settles into two layers, on one distinction: speech that was real but couldn't be decoded gets marked unintelligible and kept, while text the model fabricated outright gets deleted. Honest transcripts, not clean ones. There's a build-versus-buy lesson too: the value of writing this yourself the first time isn't the code, it's the judgment you gain about the problem.

## 2026-01-23 — Unlabeled speech fragments cut from 75 to 1

The last handful of unlabeled speech fragments turned out to be mostly turn-starts, a few words right before the next person speaks, where speaker detection missed the boundary. Assigning those forward cut the unlabeled count from 75 to 1. The one that stayed was a trickier hallucination: the model heard quiet speech and invented plausible words, far harder to spot than obvious repetition.

## 2026-01-22 — The first speaker-labeled transcript

The step that combines transcription with speaker detection is built, and it produces the first transcript that reads as an actual conversation: unlabeled fragments down from 75 to 7, and 159 scattered pieces consolidated into 23 real turns. My daughter remembering Duryodhana and Yudhishthira, in print: this is what the project is for.

> SPEAKER_01: Dad, why do the Fondos and the Goros want to be king?
> SPEAKER_00: Uh-huh. Well, so the oldest brother of the Goros, his name was, do you remember?
> SPEAKER_01: Durioden.

## 2026-01-21 — Child speech needs the large Whisper model

Speaker detection is running, and Whisper's small and large models were compared on the same stretch of audio. The small one produced absolute silence where my daughter speaks; the large one caught her full sentence. With the wrong model, ten seconds of a child's voice simply vanish, and her voice is the whole point of the project.

## 2026-01-20 — Day one: the project begins

The project is set up, and the first transcription runs: a 5.6-minute Mahabharata bedtime story through Whisper. It works, but mangles every Sanskrit name: "Yudhishthira" becomes "you this there," "Pandavas" becomes "fondos." Fixing that became the project's first real problem.
