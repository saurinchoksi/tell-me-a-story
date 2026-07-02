#!/usr/bin/env python3
"""Choksi's review checklist — the human-verdict queue from the namefix arc.  CO-EDITABLE.

Everything from the 2026-07-01/02 work that waits on Choksi's ear, eye, or taste, as one
checklist he and Claude both read and edit between turns. Each item: what it is, where it
lives, roughly how long, and why it matters. Status + free-text notes persist to a JSON
sidecar (the interactive-artifact pattern: a browser page can't write files, so --serve is
the save bridge; opened as a plain file it still reads, it just can't save).

    python emp/src/review_checklist.py            # regenerate the HTML
    python emp/src/review_checklist.py --serve    # serve with live, file-backed saving

Sidecar: emp/results/review-checklist-notes.json (committed, like pivot-notes.json — keep
family-private details out of the notes; item ids/descriptions are already repo-public).
Claude: read the sidecar to see his verdicts/notes; regenerating NEVER clobbers them.
"""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTES_PATH = ROOT / "emp" / "results" / "review-checklist-notes.json"
OUT_PATH = ROOT / "emp" / "results" / "review-checklist.html"
PORT = 8768

# The queue. Order = suggested order (highest-leverage ear work first).
ITEMS = [
    {"id": "missed-speech-cards", "kind": "ear", "mins": "~10 min",
     "title": "The 19 missed-speech recovery cards",
     "what": "A context-primed re-listen produced words inside 19 of 20 stretches where the "
             "pipeline had NO transcript at all (“Vishma's father was the king”, “met a woman "
             "and fell in love with her”). Each card: the candidate text + the audio. The call "
             "per card: is that really said?",
     "why": "If even half are real, this is the first traction ever on the dropped-speech "
            "problem (8 prior approaches recovered nothing) and becomes its own arc.",
     "where": "python emp/src/missed_speech_probe.py --serve  →  localhost:8769  (type what you hear per card; it saves)"},
    {"id": "thomas-queue", "kind": "tap", "mins": "~3 min",
     "title": "The Thomas night's 6 queued names (incl. the invented engines)",
     "what": "The gate wanted to write Jammus→Thomas and Jameis→Thomas; it was stopped and "
             "everything queued. Your taps decide: reject the invented-engine rewrites, rule "
             "on the other four (Thomas→Gordon, James→Thomas, Michael→Thomas, Sir→Thomas).",
     "why": "Clears the only session the rollout held back; confirms the queue handles the "
            "invented-copy class end to end.",
     "where": "cd ui && npm run dev → http://localhost:5174/name-review  (▶ plays each spot in place now)"},
    {"id": "garbage-cards", "kind": "ear", "mins": "~8 min",
     "title": "The 15 worst garbage-census finds",
     "what": "Non-name junk the census flagged mechanically: the non-Latin hallucinations, "
             "stuck repeats, zero-confidence clusters. Per card: was anything really said, "
             "and is the token junk?",
     "why": "Your verdicts turn the census counts into the next arc's ground truth (54 "
            "uncoded finds total).",
     "where": "python emp/src/garbage_census.py --serve  →  localhost:8770"},
    {"id": "case-study-edit", "kind": "edit", "mins": "~20 min",
     "title": "Case-study draft: your edit pass",
     "what": "The namefix eval story (Problem → Action → Result) drafted for the portfolio. "
             "My genuine first draft — cut and reshape freely; the before/after pair is "
             "already banked for the voice bible.",
     "why": "This is the EMP's Sierra deliverable — the eval-frameworks evidence, ready when "
            "you are.",
     "where": "open emp/writeup/namefix-eval.html"},
    {"id": "site-panel-read", "kind": "eye", "mins": "~3 min",
     "title": "Read the live “Fix names” panel",
     "what": "The rewritten stage description on the TMAS page (present-tense, your voice "
             "rules, clean-room numbers in the lanes). Does it land as you?",
     "why": "It's live on saurinchoksi.com; your final wording also completes the "
            "before/after voice pair.",
     "where": "https://saurinchoksi.com/portfolio/tell-me-a-story.html — Fix names stage"},
    {"id": "changelog-approve", "kind": "eye", "mins": "~1 min",
     "title": "Changelog entry: “The new name-fixer listens instead of guessing”",
     "what": "One entry covering the namefix arc, in the work-log voice.",
     "why": "The changelog header says you review and approve; it's already live via the "
            "site's changelog page.",
     "where": "changelog.md (top entry)"},
    {"id": "rekey-heldouts", "kind": "ear", "mins": "~30–45 min",
     "title": "Re-mark the name-truth keys for both held-out sessions",
     "what": "Fresh marking pages generated over the corrected/honest transcripts. Your "
             "by-ear pass rebuilds the answer keys the June scoring can no longer use.",
     "why": "Unblocks the deferred detector-dictionary migration (its ship-gate needs valid "
            "keys) and restores trustworthy precision numbers for M9c.",
     "where": "python emp/src/name_truth.py 20260211-210718 --serve  (then the same for "
              "20260414-213156)"},
    {"id": "pandavas-spotcheck", "kind": "ear", "mins": "~3 min (optional)",
     "title": "Spot-check the 7 applied fixes on the Pandavas night",
     "what": "Yudhisthir→Yudhishthira and Pondavas→Pandavas were auto-applied to the coded "
             "session (invariants held, backup kept). You already gave it an ad-hoc listen "
             "(‘all looks well’) — the artifact makes it per-spot and durable.",
     "why": "Optional belt-and-suspenders — the same chain scored 0-wrong on the held-out.",
     "where": "python emp/src/namefix_spotcheck.py 20260117-202237 --serve  →  localhost:8771  (7 cards, one per applied fix)"},
]

STATUSES = ["pending", "done", "skipped"]


def load_notes() -> dict:
    """Sidecar: {\"items\": {id: {status, notes}}, _*: preserved}. Missing file -> fresh."""
    if NOTES_PATH.exists():
        data = json.loads(NOTES_PATH.read_text())
    else:
        data = {}
    data.setdefault("_about", "Review-checklist verdicts + notes. Edited live via "
                              "review_checklist.py --serve, or by hand. Regenerating the "
                              "HTML never touches this file.")
    items = data.setdefault("items", {})
    for it in ITEMS:
        items.setdefault(it["id"], {"status": "pending", "notes": ""})
    return data


def save_notes(data: dict) -> None:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = NOTES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(NOTES_PATH)


def esc(x):
    return html.escape(str(x))


KIND_LABEL = {"ear": "🎧 listen", "tap": "👆 taps", "eye": "👀 read", "edit": "✍️ edit"}


def render_html(notes: dict) -> str:
    cards = ""
    n_done = sum(1 for it in ITEMS if notes["items"][it["id"]]["status"] == "done")
    for it in ITEMS:
        st = notes["items"][it["id"]]
        btns = "".join(
            f'<button class="st st-{s}{" on" if st["status"] == s else ""}" '
            f'data-id="{it["id"]}" data-status="{s}">{s}</button>'
            for s in STATUSES)
        cards += f"""
<div class="card k-{it['kind']} s-{esc(st['status'])}" id="card-{it['id']}">
  <div class="head">
    <span class="kind">{KIND_LABEL[it['kind']]}</span>
    <span class="mins">{esc(it['mins'])}</span>
    <div class="btns">{btns}</div>
  </div>
  <div class="title">{esc(it['title'])}</div>
  <div class="what">{esc(it['what'])}</div>
  <div class="why"><b>Why:</b> {esc(it['why'])}</div>
  <div class="where"><code>{esc(it['where'])}</code></div>
  <textarea class="note" data-id="{it['id']}" placeholder="your notes / verdicts (auto-saves when served)">{esc(st['notes'])}</textarea>
  <div class="stat" data-id="{it['id']}"></div>
</div>"""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review checklist — {n_done}/{len(ITEMS)} done</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:760px;margin:0 auto;padding:18px;background:#faf8f5;color:#1a1a1a;line-height:1.5}}
h1{{font-size:1.3rem;margin:.2em 0}}
.sub{{color:#666;font-size:.88rem;margin-bottom:1em}}
.card{{background:#fff;border-radius:12px;padding:14px 16px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,.07);border-left:5px solid #ccc}}
.card.k-ear{{border-left-color:#4a90d9}}.card.k-tap{{border-left-color:#2f9e44}}
.card.k-eye{{border-left-color:#a67c00}}.card.k-edit{{border-left-color:#8a5a3a}}
.card.s-done{{opacity:.55}}.card.s-skipped{{opacity:.4}}
.head{{display:flex;align-items:center;gap:.7rem}}
.kind{{font-size:.78rem}}.mins{{font-size:.75rem;color:#888}}
.btns{{margin-left:auto;display:flex;gap:.3rem}}
.st{{font-size:.72rem;border:1px solid #ccc;background:#fff;border-radius:6px;padding:.15rem .5rem;cursor:pointer}}
.st.on.st-done{{background:#2f9e44;color:#fff;border-color:#2f9e44}}
.st.on.st-pending{{background:#a67c00;color:#fff;border-color:#a67c00}}
.st.on.st-skipped{{background:#888;color:#fff;border-color:#888}}
.title{{font-weight:700;margin:.35em 0 .2em}}
.what{{font-size:.9rem}}.why{{font-size:.84rem;color:#555;margin:.3em 0}}
.where{{font-size:.8rem;margin:.3em 0}}code{{background:#f3efe9;padding:.1em .4em;border-radius:4px}}
.note{{width:100%;box-sizing:border-box;min-height:2.4em;border:1px solid #e5e0d8;border-radius:8px;padding:.4em .6em;font:inherit;font-size:.86rem;margin-top:.4em;background:#fffdf9}}
.stat{{font-size:.72rem;color:#888;min-height:1em}}.stat.warn{{color:#a67c00}}.stat.ok{{color:#2f9e44}}
.progress{{position:sticky;top:0;background:#faf8f5;padding:.4em 0;font-size:.9rem;font-weight:600;z-index:2}}
</style></head><body>
<h1>What's waiting on you</h1>
<p class="sub">The human-verdict queue from the name-fixer arc, roughly in leverage order. Statuses and notes save to <code>review-checklist-notes.json</code> when served (<code>python emp/src/review_checklist.py --serve</code>); Claude reads the same file, so your notes are the handoff.</p>
<div class="progress" id="prog">{n_done}/{len(ITEMS)} done</div>
{cards}
<script>
const served = location.protocol.startsWith('http');
function post(payload, statEl) {{
  if (!served) {{ if(statEl){{statEl.textContent='⚠ run --serve to save';statEl.className='stat warn';}} return; }}
  fetch('/save', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}})
    .then(r=>r.ok?r.json():Promise.reject())
    .then(()=>{{ if(statEl){{statEl.textContent='saved';statEl.className='stat ok';setTimeout(()=>statEl.textContent='',1200);}} }})
    .catch(()=>{{ if(statEl){{statEl.textContent='⚠ save failed';statEl.className='stat warn';}} }});
}}
document.querySelectorAll('.st').forEach(b=>b.addEventListener('click',()=>{{
  const id=b.dataset.id, status=b.dataset.status;
  const card=document.getElementById('card-'+id);
  card.className=card.className.replace(/s-(pending|done|skipped)/,'s-'+status);
  card.querySelectorAll('.st').forEach(x=>x.classList.toggle('on',x===b));
  const done=document.querySelectorAll('.card.s-done').length;
  document.getElementById('prog').textContent=done+'/{len(ITEMS)} done';
  post({{id:id, field:'status', value:status}}, card.querySelector('.stat'));
}}));
let timers={{}};
document.querySelectorAll('.note').forEach(t=>t.addEventListener('input',()=>{{
  const id=t.dataset.id;
  clearTimeout(timers[id]);
  timers[id]=setTimeout(()=>post({{id:id, field:'notes', value:t.value}},
    document.querySelector('.stat[data-id="'+id+'"]')), 600);
}}));
</script>
</body></html>"""


def serve(port: int) -> None:
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="text/html; charset=utf-8"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html", "/review-checklist.html"):
                self._send(200, render_html(load_notes()))
            elif path == "/review-checklist-notes.json":
                self._send(200, json.dumps(load_notes(), indent=2, ensure_ascii=False),
                           "application/json; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                self._send(404, "not found", "text/plain"); return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, '{"error":"bad json"}', "application/json"); return
            iid, field, value = body.get("id"), body.get("field"), body.get("value")
            data = load_notes()
            # field-routed save: one field at a time, so saving notes never clobbers status
            if iid in data["items"] and field in ("status", "notes"):
                if field == "status" and value not in STATUSES:
                    self._send(400, '{"error":"bad status"}', "application/json"); return
                data["items"][iid][field] = value
                save_notes(data)
                self._send(200, '{"ok":true}', "application/json")
            else:
                self._send(400, '{"error":"unknown item/field"}', "application/json")

        def log_message(self, *a):
            pass

    print(f"serving http://localhost:{port}/  (Ctrl-C to stop; edits save to {NOTES_PATH.name})")
    http.server.HTTPServer(("localhost", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    notes = load_notes()
    save_notes(notes)  # seed/refresh the sidecar with any new items
    OUT_PATH.write_text(render_html(notes))
    print(f"wrote {OUT_PATH}")
    if a.serve:
        serve(a.port)
