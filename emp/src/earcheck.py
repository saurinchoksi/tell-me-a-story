#!/usr/bin/env python3
"""Shared ear-check artifact library — card pages with audio, verdicts, and "what I hear".

Choksi's standing rule (2026-07-02): every listening/audio probe artifact gets a field where
he types WHAT HE HEARS, persisting to a file both he and Claude read — his transcription is
the ground truth, and it must land on disk, not in chat. This module is the one place that
shape lives (the third/fourth artifact wanted it; gap_classify.py grew the first hear-box).

An artifact = a list of cards -> one HTML page + a JSON sidecar + a --serve save bridge:
  card    = {id, header, body_html, clip (base64 mp3 str or None), verdicts: [labels]}
  sidecar = {"items": {id: {"verdict": "", "heard": ""}}, _keys preserved}
Interactive-artifact rules honored: current sidecar values are BAKED into the HTML (reads
fine from file://, shows "run --serve to save"), saves are debounced and field-routed (one
field per POST — saving `heard` can never clobber `verdict`), regeneration never clobbers
the sidecar. Clips are PRE-CUT mp3s embedded as data URIs (the eval-sweep audio lesson:
never stream the session m4a into a static page).

Sidecars live under gitignored emp/results/visuals/ — hearings are family content.
"""
import base64
import html
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def esc(x):
    return html.escape(str(x))


def cut_clip_b64(audio_path: Path, start: float, end: float, cache: Path,
                 lead: float = 1.5, tail: float = 1.0) -> str | None:
    """Cut [start-lead, end+tail] to a cached mono mp3; return base64 (None on failure)."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        ff = shutil.which("ffmpeg") or "ffmpeg"
        frm = max(0.0, (start or 0.0) - lead)
        dur = ((end or start or 0.0) - (start or 0.0)) + lead + tail
        cmd = [ff, "-y", "-ss", f"{frm:.3f}", "-i", str(audio_path), "-t", f"{dur:.3f}",
               "-ac", "1", "-ar", "22050", "-c:a", "libmp3lame", "-q:a", "6", str(cache)]
        if subprocess.run(cmd, capture_output=True).returncode != 0 or not cache.exists():
            return None
    return base64.b64encode(cache.read_bytes()).decode()


def load_sidecar(path: Path, ids) -> dict:
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault("_about", "Ear-check verdicts + what-I-hear transcriptions. Edited live "
                              "via the artifact's --serve, or by hand. Co-edited with Claude; "
                              "regeneration never touches this file.")
    items = data.setdefault("items", {})
    for i in ids:
        items.setdefault(i, {"verdict": "", "heard": ""})
    return data


def save_sidecar(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def build_page(title: str, intro: str, cards: list, sidecar_path: Path,
               serve_cmd: str) -> str:
    """Self-contained ear-check page. `cards` see module docstring; sidecar values baked in.
    Persists the (merged) sidecar so the save bridge always has the items on disk — existing
    verdicts/hearings survive, new card ids get seeded."""
    data = load_sidecar(sidecar_path, [c["id"] for c in cards])
    save_sidecar(sidecar_path, data)
    n_done = sum(1 for c in cards if data["items"][c["id"]]["verdict"])
    body = ""
    for c in cards:
        st = data["items"][c["id"]]
        btns = "".join(
            f'<button class="vd{" on" if st["verdict"] == v else ""}" '
            f'data-id="{esc(c["id"])}" data-v="{esc(v)}">{esc(v)}</button>'
            for v in c["verdicts"])
        audio = (f'<audio controls preload="none" src="data:audio/mpeg;base64,{c["clip"]}"></audio>'
                 if c.get("clip") else '<div class="noclip">clip unavailable</div>')
        body += f"""
<div class="card{' decided' if st['verdict'] else ''}" id="card-{esc(c['id'])}">
  <div class="h">{c['header']}</div>
  <div class="b">{c['body_html']}</div>
  {audio}
  <div class="row"><span class="lab">what I hear:</span>
    <textarea class="heard" data-id="{esc(c['id'])}" placeholder="type the words as you hear them (auto-saves when served)">{esc(st['heard'])}</textarea></div>
  <div class="row"><span class="lab">verdict:</span><div class="btns">{btns}</div>
    <span class="stat" data-id="{esc(c['id'])}"></span></div>
</div>"""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:680px;margin:0 auto;padding:16px;background:#faf8f5;color:#1a1a1a;line-height:1.5}}
h1{{font-size:1.2rem;margin:.2em 0}}
.sub{{color:#666;font-size:.86rem;margin-bottom:.4em}}
.prog{{position:sticky;top:0;background:#faf8f5;padding:.35em 0;font-weight:600;font-size:.9rem;z-index:2}}
.card{{background:#fff;border-radius:12px;padding:13px 16px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.card.decided{{opacity:.65}}
.h{{font-size:.76rem;color:#888}}
.b{{margin:.3em 0 .5em;font-size:.93rem}}
.b .tok{{font-family:ui-monospace,'SF Mono',Menlo,monospace;font-weight:600}}
audio{{width:100%}}
.row{{display:flex;align-items:flex-start;gap:.5rem;margin-top:.5em}}
.lab{{font-size:.72rem;color:#888;white-space:nowrap;padding-top:.35em}}
.heard{{flex:1;min-height:2.2em;border:1px solid #e5e0d8;border-radius:8px;padding:.35em .6em;font:inherit;font-size:.88rem;background:#fffdf9}}
.btns{{display:flex;gap:.35rem;flex-wrap:wrap}}
.vd{{font-size:.76rem;border:1px solid #ccc;background:#fff;border-radius:6px;padding:.2rem .6rem;cursor:pointer}}
.vd.on{{background:#2f9e44;color:#fff;border-color:#2f9e44}}
.stat{{font-size:.72rem;color:#888;padding-top:.3em}}.stat.warn{{color:#a67c00}}.stat.ok{{color:#2f9e44}}
.noclip{{font-size:.8rem;color:#a67c00}}
</style></head><body>
<h1>{esc(title)}</h1>
<p class="sub">{intro}</p>
<p class="sub">Answers save to <code>{esc(sidecar_path.name)}</code> when served: <code>{esc(serve_cmd)}</code>. Claude reads the same file — your typed hearings are the handoff.</p>
<div class="prog" id="prog">{n_done}/{len(cards)} decided</div>
{body}
<script>
const served = location.protocol.startsWith('http');
function post(payload, statEl) {{
  if (!served) {{ if(statEl){{statEl.textContent='⚠ run --serve to save';statEl.className='stat warn';}} return; }}
  fetch('/save', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}})
    .then(r=>r.ok?r.json():Promise.reject())
    .then(()=>{{ if(statEl){{statEl.textContent='saved';statEl.className='stat ok';setTimeout(()=>statEl.textContent='',1200);}} }})
    .catch(()=>{{ if(statEl){{statEl.textContent='⚠ save failed';statEl.className='stat warn';}} }});
}}
document.querySelectorAll('.vd').forEach(b=>b.addEventListener('click',()=>{{
  const id=b.dataset.id;
  const card=document.getElementById('card-'+id);
  card.querySelectorAll('.vd').forEach(x=>x.classList.toggle('on',x===b));
  card.classList.add('decided');
  document.getElementById('prog').textContent=document.querySelectorAll('.card.decided').length+'/{len(cards)} decided';
  post({{id:id, field:'verdict', value:b.dataset.v}}, card.querySelector('.stat'));
}}));
let timers={{}};
document.querySelectorAll('.heard').forEach(t=>t.addEventListener('input',()=>{{
  const id=t.dataset.id;
  clearTimeout(timers[id]);
  timers[id]=setTimeout(()=>post({{id:id, field:'heard', value:t.value}},
    document.querySelector('.stat[data-id="'+id+'"]')), 600);
}}));
</script>
</body></html>"""


def serve(render_fn, sidecar_path: Path, port: int) -> None:
    """Generic save bridge: GET / -> render_fn() (fresh HTML), POST /save -> field-routed
    write into the sidecar. render_fn re-reads the sidecar so a reload shows saved state."""
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
            if self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, render_fn())
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
            data = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {"items": {}}
            if field in ("verdict", "heard") and iid in data.get("items", {}):
                data["items"][iid][field] = value
                save_sidecar(sidecar_path, data)
                self._send(200, '{"ok":true}', "application/json")
            else:
                self._send(400, '{"error":"unknown item/field"}', "application/json")

        def log_message(self, *a):
            pass

    print(f"serving http://localhost:{port}/  (Ctrl-C to stop; saves -> {sidecar_path})")
    http.server.HTTPServer(("localhost", port), Handler).serve_forever()
