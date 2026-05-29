#!/usr/bin/env python3
"""EMP Count (step 5): tally hand-coded axial-labels into the failure-mode pivot,
and dump the data needed for the two bookkeeping debts.

axial-labels.json schema: {"labels":[{segmentId, codes:[...], createdAt, updatedAt}]}.
`codes` is a list of mode tags ("M1".."M10","NotA"). Read-only on session data.
Run:  python3 scripts/emp_count.py
"""
import json, collections, re

S = [("20251207-195607","Moon"),("20251207-202105","Cruel Baby"),
     ("20251210-203654","Rubber Ducky"),("20260117-202237","Pandavas"),
     ("20260129-204404","Portal")]
NM=[n for _,n in S]
MODES=["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","NotA"]

def jload(p): return json.load(open(p))
def labels(sid): return jload(f"sessions/{sid}/axial-labels.json")["labels"]

def seg_text(sid):
    """segmentId -> transcript text, from transcript-rich.json (schema-tolerant)."""
    tr=jload(f"sessions/{sid}/transcript-rich.json")
    segs = tr["segments"] if isinstance(tr,dict) and "segments" in tr else (tr if isinstance(tr,list) else [])
    out={}
    for i,s in enumerate(segs):
        sid_key = s.get("id", s.get("segmentId", i)) if isinstance(s,dict) else i
        txt = s.get("text","") if isinstance(s,dict) else ""
        if not txt and isinstance(s,dict) and "words" in s:
            txt="".join(w.get("word",w.get("text","")) for w in s["words"])
        out[i]=txt           # index-based, matches segmentId convention
    return out

# ---------- name list ----------
md=jload("data/mahabharata.json")
names=[]
for e in md.get("entries",[]):
    for fld in ("canonical","name"):
        if isinstance(e.get(fld),str): names.append(e[fld])
    for fld in ("variants","aliases"):
        if isinstance(e.get(fld),list): names+=[x for x in e[fld] if isinstance(x,str)]
names_l=sorted({n.lower() for n in names if n and len(n)>2})
def is_name(t):
    t=(t or "").lower()
    return any(re.search(r"\b"+re.escape(n)+r"\b",t) for n in names_l)

# ---------- pivot ----------
piv={m:collections.Counter() for m in MODES}
multi=[]; recs=0; raw=collections.Counter()
for sid,name in S:
    for r in labels(sid):
        recs+=1
        codes=[c.strip() for c in r.get("codes",[]) if str(c).strip()]
        for c in codes: raw[c]+=1
        nn=sorted({c for c in codes if c!="NotA"})
        if len(nn)>1: multi.append((name,r.get("segmentId"),codes))
        for c in set(codes): piv.setdefault(c,collections.Counter())[name]+=1

print("### PIVOT — hand-coded mode instances (distinct code per segment)")
print("mode," + ",".join(NM) + ",TOTAL")
for m in MODES:
    rc=piv[m]; print(f"{m}," + ",".join(str(rc[n]) for n in NM) + f",{sum(rc.values())}")
for k in [x for x in piv if x not in MODES]:
    rc=piv[k]; print(f"!{k}," + ",".join(str(rc[n]) for n in NM) + f",{sum(rc.values())}")
print(f"records={recs} distinct_codes={sorted(raw)} dict_names={len(names_l)}")
print("multicoded=%d: "%len(multi) + "; ".join(f"{a}/s{b}/{c}" for a,b,c in multi))

# ---------- DEBT 1: M1 segments — name vs common (needs transcript text) ----------
print("\n### DEBT1 — every M1-coded segment, with its transcript text + name flag")
for sid,name in S:
    txt=seg_text(sid)
    for r in labels(sid):
        if "M1" not in r.get("codes",[]): continue
        sg=r.get("segmentId"); t=txt.get(sg,"")
        print(f"  {name}|s{sg}|{r.get('codes')}|{'NAME' if is_name(t) else 'common'}|{t[:80]!r}")

# ---------- DEBT 2: M2 segments — text (carve M2A silence vs M2B non-speech) ----------
print("\n### DEBT2 — every M2-coded segment, with its transcript text")
for sid,name in S:
    txt=seg_text(sid)
    for r in labels(sid):
        if "M2" not in r.get("codes",[]): continue
        sg=r.get("segmentId")
        print(f"  {name}|s{sg}|{r.get('codes')}|{txt.get(sg,'')[:80]!r}")

print("\n### M9 hand-coded per session")
for sid,name in S:
    print(f"  {name}: " + str(sum(1 for r in labels(sid) if 'M9' in r.get('codes',[]))))
