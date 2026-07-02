#!/usr/bin/env python3
"""Spot-check the fixes namefix APPLIED to a session — one ear-check card per corrected word.

After a rollout write, this is the verification artifact: every word the stage rewrote
(stage == "namefix" or "namefix_bless" in its correction trail), as a card with the
before -> after tokens, the segment line, the audio at that word, a verdict, and the
"what I hear" box (Choksi's standing rule for all listening artifacts). Verdicts persist
to a per-session co-edited sidecar.

READ-ONLY on the transcript. Usage:
    python emp/src/namefix_spotcheck.py <session-id>            # (re)generate the page
    python emp/src/namefix_spotcheck.py <session-id> --serve    # serve with live saving
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.helpers import get_session_dir  # noqa: E402
import earcheck  # noqa: E402

PORT = 8771


def collect_fixes(session_dir: Path) -> list[dict]:
    rich = json.loads((session_dir / "transcript-rich.json").read_text())
    fixes = []
    for seg in rich["segments"]:
        words = seg.get("words") or []
        for wi, w in enumerate(words):
            trail = [c for c in (w.get("_corrections") or [])
                     if (c.get("stage") or "").startswith("namefix")]
            if not trail:
                continue
            line = "".join(x["word"] for x in words).strip()
            fixes.append({
                "seg_id": seg["id"], "wi": wi,
                "start": w.get("start"), "end": w.get("end"),
                "before": w.get("_original", trail[0].get("from", "?")),
                "after": w["word"].strip(),
                "stage": trail[-1]["stage"],
                "line": line,
            })
    return fixes


def make_render(session_id: str):
    session_dir = get_session_dir(ROOT / "sessions", session_id)
    sidecar = ROOT / "emp/results/visuals" / session_id / "namefix-spotcheck.json"
    clipdir = ROOT / "emp/results/visuals" / session_id / "spotcheck-clips"
    serve_cmd = f"python emp/src/namefix_spotcheck.py {session_id} --serve"

    def render() -> str:
        fixes = collect_fixes(session_dir)
        cards = []
        for i, f in enumerate(fixes):
            cid = f"{f['seg_id']}-{f['wi']}"
            clip = None
            if f["start"] is not None:
                clip = earcheck.cut_clip_b64(session_dir / "audio.m4a", f["start"],
                                             f["end"] or f["start"],
                                             clipdir / f"fix{i}.mp3")
            mins = f"{int(f['start'] // 60)}:{int(f['start'] % 60):02d}" if f["start"] is not None else "?"
            cards.append({
                "id": cid,
                "header": f"{i+1} of {len(fixes)} · seg {earcheck.esc(f['seg_id'])} · {mins} "
                          f"· {earcheck.esc(f['stage'])}",
                "body_html": (f"<span class='tok'>{earcheck.esc(f['before'])}</span> &rarr; "
                              f"<span class='tok'>{earcheck.esc(f['after'])}</span><br>"
                              f"<span style='color:#666;font-size:.85em'>&ldquo;{earcheck.esc(f['line'])}&rdquo;</span>"),
                "clip": clip,
                "verdicts": ["fix is right", "fix is wrong", "can't tell"],
            })
        return earcheck.build_page(
            f"Applied fixes on {session_id}",
            f"Every word the name-fixer wrote on this session ({len(fixes)} total): the old "
            "token, the new one, and the audio at that spot. Play, type what you hear, verdict.",
            cards, sidecar, serve_cmd)

    return render, sidecar


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit("usage: namefix_spotcheck.py <session-id> [--serve]")
    sid = args[0]
    render, sidecar = make_render(sid)
    page = ROOT / "emp/results/visuals" / sid / "namefix-spotcheck.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render())
    print(f"wrote {page}")
    if "--serve" in sys.argv:
        earcheck.serve(render, sidecar, PORT)
