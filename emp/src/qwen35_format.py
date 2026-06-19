#!/usr/bin/env python3
"""Qwen3.5-4B (via mlx_vlm) — does asking for PLAIN lines instead of JSON save tokens and/or
improve the cast? JSON makes a reasoning model spend output on quotes/brackets it doesn't need;
we can wrap to JSON on our side. Compares JSON vs one-name-per-line, measuring output length
(token proxy) and the names recovered. /no_think to skip the reasoning block; any leaked think
block is stripped before parsing.

    ./venv/bin/python emp/src/qwen35_format.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from story_segment import make_reader, extract_json   # noqa: E402

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
WORLDS = ["Mahabharata", "Star Wars", "KPop Demon Hunters"]

P_JSON = ('List 20 named characters of the fictional world "{world}". English spellings only. '
          'Return JSON only: {{"names": ["Name", ...]}}')
P_LINES = ('List the 20 best-known characters of the fictional world "{world}". '
           'Output the names ONLY, one per line — no numbers, no bullets, no JSON, no other '
           'words. Use the common English spelling.')


def strip_think(raw):
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw)
    raw = re.sub(r"(?is)^.*?thinking process:.*?(?=\n[A-Z\[{\"]|\n\n)", "", raw, count=1)
    return raw


def parse_json(raw):
    o = extract_json(raw)
    return o.get("names", []) if isinstance(o, dict) else []


def parse_lines(raw):
    out, seen = [], set()
    for line in strip_think(raw).splitlines():
        s = re.sub(r"^[\s\-\*•\d\.\)]+", "", line).strip().strip('",')
        if not s or len(s) > 40 or s.lower() in seen:
            continue
        if re.search(r"[^\x00-\x7f]", s) or re.search(r"[:{}\[\]]", s):
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def main():
    gen = make_reader(MODEL)
    print(f"{'world':22} {'format':6} {'out_chars':>9} {'names':>6}   first few")
    print("-" * 88)
    for w in WORLDS:
        for label, prompt, parse in [("json", P_JSON, parse_json), ("lines", P_LINES, parse_lines)]:
            raw = gen("/no_think\n" + prompt.format(world=w), max_tokens=600)
            names = parse(raw)
            karna = "  [Karna]" if any("karna" == n.lower() for n in names) else ""
            print(f"{w[:22]:22} {label:6} {len(raw):>9} {len(names):>6}   {names[:6]}{karna}", flush=True)


if __name__ == "__main__":
    main()
