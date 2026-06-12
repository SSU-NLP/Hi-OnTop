#!/usr/bin/env python3
"""가설 검증: 문장 단위 분절이 긴 발화 안에 묻힌 경계를 잡아 turn 단위보다 나은가.
turn-level vs sentence-level(+filler drop) — 예측을 turn 으로 매핑 후 turn-level tolerance-F1 비교.
임베딩 MiniLM-int8. default = manifest 12 미팅(빠른 검증), --all 로 139.
"""
from __future__ import annotations
import json, re, sys, glob, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"src"))
from run_encoder_comparison import delta_eff_seq, boundaries
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from sentence_transformers import SentenceTransformer

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
    model_kwargs={"provider":"CPUExecutionProvider","file_name":"onnx/model_quint8_avx2.onnx"})
def embed(ts): return np.asarray(enc.encode(ts, normalize_embeddings=True, batch_size=128,
                                            show_progress_bar=False), dtype=np.float64)

TOPIC=REPO/"data"/"ami"/"topic"
ALL = "--all" in sys.argv
mids = sorted(os.path.basename(p)[:-5] for p in glob.glob(str(TOPIC/"*.json")) if "manifest" not in p)
if not ALL:
    man=set(m["meeting"] for m in json.load(open(TOPIC/"manifest.json"))); mids=[m for m in mids if m in man]

FILLER = re.compile(r"^(yeah|yep|yes|no|okay|ok|mm+|mm-?hmm|hmm+|uh+|um+|right|sure|"
                    r"alright|ah+|oh+|huh|kay|'kay|true|cool|well)$", re.I)
def split_sents(text):
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
def is_filler(s):
    toks = re.findall(r"[a-z'-]+", s.lower())
    return len(toks) > 0 and all(FILLER.match(t) for t in toks)

def tol_f1(gold_turns, pred_turns, tol):
    if not pred_turns or not gold_turns: return 0.0
    pr=sum(1 for i in pred_turns if any(abs(i-j)<=tol for j in gold_turns))/len(pred_turns)
    rc=sum(1 for j in gold_turns if any(abs(i-j)<=tol for i in pred_turns))/len(gold_turns)
    return 2*pr*rc/(pr+rc) if pr+rc>0 else 0.0

rows = {"turn": [], "sent": [], "sent+nofiller": []}
for mid in mids:
    d=json.load(open(TOPIC/f"{mid}.json")); turns=d["turns"]; bt=d["bnd_top"]
    gold_turns=[i for i,b in enumerate(bt) if b==1]

    # --- turn level ---
    e=embed([t["text"] for t in turns]); deff=delta_eff_seq(e)
    pt=[i for i,b in enumerate(adaptive_boundaries(deff,c=1.5,mode="ewma")) if b]
    rows["turn"].append((gold_turns, pt))

    # --- sentence level (+ optional filler drop) ---
    for variant, drop in [("sent", False), ("sent+nofiller", True)]:
        sents=[]; s2turn=[]
        for ti,t in enumerate(turns):
            for s in split_sents(t["text"]):
                if drop and is_filler(s): continue
                sents.append(s); s2turn.append(ti)
        if len(sents) < 3:
            rows[variant].append((gold_turns, [])); continue
        se=embed(sents); sdeff=delta_eff_seq(se)
        sp=[i for i,b in enumerate(adaptive_boundaries(sdeff,c=1.5,mode="ewma")) if b]
        pred_turns=sorted(set(s2turn[i] for i in sp if i < len(s2turn)))
        rows[variant].append((gold_turns, pred_turns))

print(f"AMI {len(mids)} 미팅 — turn vs sentence 단위 (ewma c=1.5), turn-level tolerance-F1\n")
print(f"  {'단위':14} {'±0':>6} {'±1':>6} {'±2':>6} {'±3':>6}")
for name, data in rows.items():
    r=[np.mean([tol_f1(g,p,tol) for g,p in data]) for tol in (0,1,2,3)]
    print(f"  {name:14} {r[0]:>6.3f} {r[1]:>6.3f} {r[2]:>6.3f} {r[3]:>6.3f}")
