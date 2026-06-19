#!/usr/bin/env python3
"""Generate + cache Qwen3.5-4B casts with the working recipe: mlx_vlm loader (it's a VLM),
enable_thinking=False in the chat template (/no_think is ignored), and PLAIN-LINE output (no
JSON — cheaper and the model actually complies). Dedupe handles the temp-0 list loop.

    ./venv/bin/python emp/src/gen_cast_qwen35.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlx_vlm import load, generate   # noqa: E402

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
PROMPT = ('List the 25 best-known characters, families/groups, and important places of the '
          'fictional world "{world}". Output names only, one per line, English spelling, '
          'nothing else — no numbers, no JSON, no commentary.')

model, processor = load(MODEL)
tok = getattr(processor, "tokenizer", processor)


def cast_for(world):
    fmt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(world=world)}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    o = generate(model, processor, fmt, max_tokens=500, temperature=0.0, verbose=False)
    raw = getattr(o, "text", o) or ""
    out, seen = [], set()
    for line in raw.splitlines():
        s = re.sub(r"^[\s\-\*•\d\.\)]+", "", line).strip().strip('",')
        if not s or len(s) > 40 or s.lower() in seen:
            continue
        if re.search(r"[^\x00-\x7f]", s) or re.search(r"[:{}\[\]]", s):
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def main():
    spread = json.loads((ROOT / "emp" / "results" / "canon-spread" / "stories.json").read_text())["stories"]
    worlds = list(dict.fromkeys(["Mahabharata"] + [s["world"] for s in spread]))
    casts = {}
    for w in worlds:
        casts[w] = cast_for(w)
        has_p = "pandavas" in {n.lower() for n in casts[w]}
        has_k = "karna" in {n.lower() for n in casts[w]}
        print(f"  {w[:34]:36} {len(casts[w]):>2} names"
              + ("  [Pandavas]" if has_p else "") + ("  [Karna]" if has_k else ""), flush=True)
    out = ROOT / "emp" / "results" / "canon-spread" / "casts" / "qwen35.json"
    out.write_text(json.dumps({"model": MODEL, "casts": casts}, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
