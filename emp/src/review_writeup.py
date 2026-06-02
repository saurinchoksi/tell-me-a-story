#!/usr/bin/env python3
"""
Interactive review doc for the name-detection case study.
Generator + --serve + JSON sidecar (the interactive-artifact pattern; cf. emp/src/count.py).

    python emp/src/review_writeup.py           # regenerate the clean publish HTML + the review HTML
    python emp/src/review_writeup.py --serve    # serve the review version; margin notes save to disk

CONTENT below is the single source of truth for the prose. Margin notes live in
emp/writeup/name-detection-eval-notes.json (gitignored) and are NEVER baked into the
clean publish file. Regenerating never clobbers notes; each saved note is routed by
block id, so saving one note never wipes another.
"""
import argparse
import html as _html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WD = ROOT / "emp" / "writeup"
PUBLISH = WD / "name-detection-eval.html"            # clean, committable, no notes UI
REVIEW = WD / "name-detection-eval-review.html"      # interactive, gitignored
NOTES = WD / "name-detection-eval-notes.json"        # co-edited notes, gitignored
GH = "https://github.com/saurinchoksi/tell-me-a-story/tree/main/emp"

# (block id, kind, inner html).  kind: eyebrow h1 deck h2 p ul callout hr footnote
CONTENT = [
 ("eyebrow", "eyebrow", "Evaluation case study"),
 ("title", "h1", "The name the computer kept getting wrong"),
 ("deck", "deck", "I built a small tool that records my kid's bedtime stories and writes them down. Then I sat down to find out what it gets wrong, how often, and the best way to fix it."),
 ("intro1", "p", "At night, my kid and I make up stories, and I record them. A little pipeline on my own computer turns the audio into text, labels who said what, and pulls out the characters and names we invent along the way. The point is to keep a growing record of our made-up world. One rule sits on top of all of it: nothing leaves my machine. This is my family's voice, so it never goes to the cloud."),
 ("intro2", "p", "The tool is good, but it gets things wrong, and the thing it gets wrong most is names. It will spell my kid's name four different ways in a single story. That matters more than it sounds like it should. If the main character's name is a mess, the record I am trying to keep is mostly junk. So I stopped adding features and did something slower and more useful. I sat down to measure it. Two questions: how bad is it really, and what is the best way to catch it."),
 ("h_read", "h2", "First I read the transcripts and wrote down what was broken"),
 ("read1", "p", "I did not try to fix anything yet. That part is a trap. The moment you start rewriting lines, you stop noticing the failures and start polishing them away. So I just read, story by story, and every time I hit a new kind of mistake, I gave it a short, plain name. Missed a word. Put the words on the wrong person. I wrote down what I saw, not what I guessed had caused it. To keep this quick, I built a small tool for it: each line of the transcript shows up as a card with a row of buttons, one per kind of mistake, so I could tag a line with a click instead of writing a note every time."),
 ("read2", "p", "I kept going until I stopped finding new kinds. That took six stories. In the first one I found fifteen new kinds of mistake. By the fourth, I was finding one. It never quite dropped to zero, but the flattening was the signal to stop: I had clearly seen the common failures, and chasing the long tail across fifty more stories would have been a waste of time. I ended with eighteen kinds."),
 ("h_group", "h2", "Then I grouped them"),
 ("group1", "p", "A lot of those eighteen were the same problem in different clothes. So I sorted them into ten groups, and for each group I wrote down a rough idea of how you would actually fix it. Grouping by the fix is the useful part. Two mistakes that look different but get repaired the same way belong together. I also kept a &quot;none of these&quot; bucket open the whole time, in case something refused to fit any group. It stayed empty. That is a small thing, but it told me the ten groups really did cover everything I had seen."),
 ("h_count", "h2", "Then I counted"),
 ("count1", "p", "Finding the kinds of mistakes is one job. Counting how often each one happens is a different job, so I did it on its own pass. I built the count as something I could poke at and edit, not a frozen table. Each mistake type carries its own notes and a guess at how a fix would work, and the whole thing saves back to a file so I can keep refining it as I learn more. And because some mistakes are nearly impossible to catch by ear, I wrote a couple of small scripts that line the transcript up against the actual sound and against who was talking. They turned up a pile of mistakes I would have missed by hand. In the longest story alone, close to a minute of talking had been dropped with no trace at all."),
 ("count2", "p", "The result was not close. Names were the most common mistake by a wide margin, well ahead of plain wrong words and wrong-speaker mix-ups. So names is what I went after."),
 ("h_what", "h2", "What a name mistake actually is"),
 ("what1", "p", "Before building a catcher, I had to be honest about what I was even catching, and it turned out a name mistake is really four different problems:"),
 ("what_list", "ul", "<li>The same wrong spelling, every time.</li><li>The same name spelled a few different ways inside one story.</li><li>A name whose first sound gets misheard, so it stops sounding like the real name at all.</li><li>A name that comes out as an ordinary word, or as a different, perfectly normal name. This is the hard one. Nothing about the text looks wrong, so you can only catch it by listening.</li>"),
 ("what2", "p", "Those four do not give way to the same tool. Three of them leave a trace in the text you can grab onto. The fourth erases the only thing a text checker could ever look at, so no amount of cleverness in software will catch it. That is not a flaw to engineer around. It is a fact about where the clue lives."),
 ("h_key", "h2", "The answer key"),
 ("key1", "p", "To grade anything, I needed the right answers. So I went back through every story and wrote down each real person, the one correct spelling of their name, and every single way the computer had mangled it. That list is the answer key. Every catcher I try gets graded against it, the same way, so the comparison stays fair."),
 ("h_three", "h2", "Three ways to catch the mistakes"),
 ("three1", "p", "I tried three. The first is not a model at all. Turn every word into a rough code for how it sounds, then flag any word that sounds like one of our names but is not spelled like one of them. It is about fifteen lines of code. Graded against the answer key, it caught about 97% of the mistakes, with no model anywhere in sight."),
 ("three2", "p", "The second was small AI models, the kind you could run on a cheap home device without sending anything to the cloud. I tried a range of sizes. The small ones crashed on the longer stories. The medium ones stopped crashing but flagged far too much, forcing ordinary words onto the name list with total confidence. And the ones actually smart enough to help are too big to fit on the hardware I would put this on."),
 ("three3", "p", "The third was a big cloud model. This one did the hard cases. It reasoned them out, the way a person would, catching a name that had turned into a different name by noticing it appeared nowhere else and that a parent was clearly talking to the kid. But I cannot use it, for the same reason that shaped everything else here. My kid's voice does not go to the cloud."),
 ("h_decide", "h2", "The decision: ship the code"),
 ("decide1", "p", "The plain code wins. It is simple, it never crashes, it never invents a match, and it catches almost everything for the price of a tiny script. The cloud model is smarter, and it is off the table. The local models that would fit are worse than the code, and the ones that could compete do not fit."),
 ("decide_callout", "callout", "And one kind of mistake, the name that turns into an ordinary word, cannot be caught from the text by anything, big model or small, because the clue simply is not in the text. I would rather say that out loud, and point to the one or two cases it costs me, than ship something that looks impressive and quietly guesses."),
 ("decide2", "p", "That, more than the score, is what the exercise gave me. Not just a tool, but an honest account of what my own rules cost. Keeping this private and on a small device costs me a couple of hard cases, and now I can name them exactly instead of waving my hands."),
 ("h_test", "h2", "One more test, just to be sure"),
 ("test1", "p", "The code had exactly one false alarm: a word that happens to sound like the name. I wondered if the smallest local model could at least clean that one up, since judging a single word in context is a far easier job than reading a whole story. This time it did not crash. But it failed at the one thing I asked it to do. A real name that is also an everyday word, it threw out, reading it as the everyday word and ignoring the sentence around it. It traded away several real catches to remove one false alarm. That is a worse tool, not a better one."),
 ("test2", "p", "What actually fixed the false alarm was one more line of code: only flag words that start with a capital letter. The false alarm was lowercase, sitting in the middle of a sentence. The real names were all capitalized. Done."),
 ("h_work", "h2", "Did it actually work, or did I just fit my own test?"),
 ("work1", "p", "That is the question that matters, and it is the easiest one to skip. A score measured on the same stories you built the tool from can look great for the wrong reasons. So I ran the whole thing on two stories I had never opened. It caught every name mistake I could find by ear in them, including spellings it had never seen before, and it raised the same single harmless false alarm. The score held up on new material, which is the only kind of score worth trusting."),
 ("h_took", "h2", "What I took from this"),
 ("took1", "p", "The project is small and the names are made up, but the steps are not particular to either one:"),
 ("took_list", "ul", "<li><strong>Break the problem apart before you reach for a tool.</strong> &quot;Name mistakes&quot; was four different problems wearing one label, and one of them has no software answer at all.</li><li><strong>Plain code often beats an AI model.</strong> When the clue is sitting right there in the data, code is cheaper, faster, repeats perfectly, and does not make things up. Save the model for the cases that genuinely need reasoning, and make it prove it earns its place against a real answer key before you trust it.</li><li><strong>Be honest about what your limits cost you.</strong> Private, on-device, small hardware: each choice buys something and costs something, and a real evaluation turns that from a vague worry into a number you can point to.</li>"),
 ("took2", "p", "There is one more thing that carries, and it is the one I keep coming back to. This project turns sound into text. A voice assistant does the opposite. It turns an idea into speech. The thing you would be checking is completely different. But the work is the same: sit with the failures, name them, group them, count them, choose a check for each one, and test that check against a real answer key before you believe it. The tool you build will not survive the next model. The way you decided what to build will."),
 ("rule", "hr", ""),
 ("footnote", "footnote", 'Names and identifying details from the family recordings have been removed; the failures, the model results, and the method are unchanged. The code behind this (the detector, its scorer, the counting board, and the audio checks) is on <a href="' + GH + '">GitHub</a>. Written with help from an AI assistant and edited by me.'),
]

BASE_CSS = """
  :root{--ink:#1a1a1a;--muted:#5b5f66;--faint:#8a8f98;--rule:#e6e7ea;--bg:#fbfbfa;--accent:#3a5a78;--good:#1f7a4d;--mono:"SF Mono",ui-monospace,Menlo,Consolas,monospace;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.62;font-size:17px;-webkit-font-smoothing:antialiased}
  .eyebrow{font-size:.78rem;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);margin:0 0 .9rem}
  h1{font-size:2.05rem;line-height:1.18;letter-spacing:-.015em;margin:0 0 .7rem;font-weight:700}
  .deck{font-size:1.18rem;color:var(--muted);margin:0 0 2.4rem;line-height:1.5}
  h2{font-size:1.32rem;letter-spacing:-.01em;margin:3rem 0 .3rem;font-weight:680}
  p{margin:.95rem 0}
  a{color:var(--accent)}
  strong{font-weight:650}
  .callout{border-left:3px solid var(--accent);padding:.3rem 0 .3rem 1.2rem;margin:2rem 0;color:var(--ink);font-size:1.08rem;line-height:1.55}
  code{font-family:var(--mono);font-size:.86em;background:#f0f1f3;padding:.08em .35em;border-radius:4px}
  hr{border:none;border-top:1px solid var(--rule);margin:3rem 0}
  .footnote{font-size:.86rem;color:var(--faint);margin-top:2.5rem;line-height:1.55}
  ul{margin:.9rem 0;padding-left:1.2rem}
  ul li{margin:.5rem 0}
"""

CLEAN_CSS = """
  .wrap{max-width:720px;margin:0 auto;padding:4rem 1.25rem 6rem}
"""

REVIEW_CSS = """
  .doc{max-width:1140px;margin:0 auto;padding:2rem 1.5rem 7rem}
  .row{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:2.4rem;align-items:start;margin-top:.3rem}
  .row.sec{margin-top:2.6rem}
  .content{min-width:0}
  .content>*{margin:.55rem 0}
  .content>h1{margin:.2rem 0 .1rem}
  .content>.deck{margin:.2rem 0 1rem}
  .content>.callout{margin:.8rem 0}
  .note{width:100%;font:inherit;font-size:.83rem;line-height:1.45;color:var(--ink);background:#fffdf5;border:1px solid var(--rule);border-radius:8px;padding:.45rem .6rem;resize:vertical;min-height:2rem;opacity:.4;transition:opacity .15s,border-color .15s,background .15s}
  .note::placeholder{color:var(--faint)}
  .note:hover,.note:focus{opacity:1;outline:none;border-color:var(--accent)}
  .note:not(:placeholder-shown){opacity:1;background:#fff6df;border-color:#e3c98a}
  .note.saved{border-color:var(--good)}
  .savehint{display:none;position:sticky;top:0;z-index:5;background:#fff3cd;border-bottom:1px solid #e3c98a;color:#6b5512;padding:.6rem 1rem;font-size:.85rem;text-align:center}
  .topbar{position:sticky;top:0;z-index:4;background:rgba(251,251,250,.92);backdrop-filter:saturate(180%) blur(6px);border-bottom:1px solid var(--rule);padding:.55rem 1.5rem;font-size:.8rem;color:var(--faint);display:flex;justify-content:space-between;align-items:center}
  .topbar b{color:var(--ink);font-weight:650}
"""

REVIEW_JS = """
<script>
(function(){
  var served = location.protocol.indexOf('http') === 0;
  var hint = document.getElementById('savehint');
  var status = document.getElementById('savestatus');
  if(!served && hint){ hint.style.display='block'; }
  if(served && status){ status.textContent='editing live — notes save to disk'; }
  var timers = {};
  document.querySelectorAll('textarea.note').forEach(function(t){
    t.addEventListener('input', function(){
      if(!served) return;
      var id = t.dataset.id, val = t.value;
      clearTimeout(timers[id]);
      timers[id] = setTimeout(function(){
        fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({id:id, value:val})})
          .then(function(r){ if(r.ok){ t.classList.add('saved'); if(status){status.textContent='saved';}
                             setTimeout(function(){ t.classList.remove('saved'); }, 700); } });
      }, 450);
    });
  });
})();
</script>
"""


def block_inner(kind, h):
    return {
        "eyebrow": '<p class="eyebrow">' + h + '</p>',
        "h1": '<h1>' + h + '</h1>',
        "deck": '<p class="deck">' + h + '</p>',
        "h2": '<h2>' + h + '</h2>',
        "p": '<p>' + h + '</p>',
        "ul": '<ul>' + h + '</ul>',
        "callout": '<div class="callout">' + h + '</div>',
        "hr": '<hr>',
        "footnote": '<p class="footnote">' + h + '</p>',
    }[kind]


def page(css, body):
    return ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>The name the computer kept getting wrong</title>\n<style>\n'
            + css + '\n</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')


def render(mode, notes):
    if mode == "clean":
        body = ('<main class="wrap">\n'
                + "\n".join("  " + block_inner(k, h) for _, k, h in CONTENT)
                + "\n</main>")
        return page(BASE_CSS + CLEAN_CSS, body)

    rows = [
        '<div class="topbar"><span>Review draft &middot; add notes in the right margin</span>'
        '<span id="savestatus">read-only</span></div>',
        '<div class="savehint" id="savehint">This file is read-only. Run '
        '<code>python emp/src/review_writeup.py --serve</code> and open the localhost link to save your notes.</div>',
        '<main class="doc">',
    ]
    for bid, k, h in CONTENT:
        sec = " sec" if k == "h2" else ""
        note_val = _html.escape(notes.get(bid, "") or "")
        rows.append(
            '  <div class="row' + sec + '">'
            '<div class="content">' + block_inner(k, h) + '</div>'
            '<div class="margin"><textarea class="note" data-id="' + bid + '" rows="2" '
            'placeholder="note…">' + note_val + '</textarea></div>'
            '</div>'
        )
    rows.append('</main>')
    rows.append(REVIEW_JS)
    return page(BASE_CSS + REVIEW_CSS, "\n".join(rows))


def load_notes():
    if NOTES.exists():
        try:
            data = json.loads(NOTES.read_text())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"_about": "Margin notes for the name-detection case study review, keyed by block id. "
                      "Gitignored working file; Claude reads it between turns."}


def save_notes(notes):
    NOTES.write_text(json.dumps(notes, indent=2, ensure_ascii=False))


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, render("review", load_notes()).encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found")

        def do_POST(self):
            if self.path != "/save":
                self._send(404, b"not found")
                return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b"bad json")
                return
            bid = data.get("id")
            if bid:
                notes = load_notes()            # reload so we never clobber other fields
                notes[bid] = data.get("value", "")   # route the save by block id only
                save_notes(notes)
            self._send(200, b"ok")

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="serve the review doc and save notes to disk")
    ap.add_argument("--port", type=int, default=5057)
    args = ap.parse_args()

    notes = load_notes()
    PUBLISH.write_text(render("clean", notes))
    REVIEW.write_text(render("review", notes))
    print("wrote " + str(PUBLISH.relative_to(ROOT)) + " and " + str(REVIEW.relative_to(ROOT)))

    if args.serve:
        if not NOTES.exists():
            save_notes(notes)
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler())
        url = "http://127.0.0.1:" + str(args.port)
        print("serving review doc at " + url + "  (Ctrl-C to stop)")
        print("notes save to " + str(NOTES.relative_to(ROOT)))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
