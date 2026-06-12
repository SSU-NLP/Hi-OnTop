#!/usr/bin/env python3
"""GraphSeg(-window-d) 를 AMI 에 돌려 우리 cosine(merge+ewma) 과 tolerance-F1 비교."""
from __future__ import annotations
import json, pickle, sys, glob, os, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"src"))
from hi_ontop.baselines.graphseg_window import GraphSegWindowD
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

TOPIC = REPO/"data"/"ami"/"topic"; CACHE = REPO/"outputs"/"runs"/"_misc"/"ami_emb"
man = set(m["meeting"] for m in json.load(open(TOPIC/"manifest.json")))
mids = sorted(m for m in (os.path.basename(p)[:-5] for p in glob.glob(str(TOPIC/"*.json")) if "manifest" not in p) if m in man)

def tol_f1(yt_idx, yp_idx, tol):
    if not yp_idx or not yt_idx: return 0.0
    pr=sum(1 for i in yp_idx if any(abs(i-j)<=tol for j in yt_idx))/len(yp_idx)
    rc=sum(1 for j in yt_idx if any(abs(i-j)<=tol for i in yp_idx))/len(yt_idx)
    return 2*pr*rc/(pr+rc) if pr+rc>0 else 0.0

# GloVe 1회 로드 (공유)
prime = GraphSegWindowD(window_d=10, sim_threshold=0.25, min_seg_size=3)
print("[setup] loading GloVe...", flush=True); t0=time.perf_counter()
prime._ensure_resources()
print(f"[setup] glove vocab={len(prime._glove)} ({time.perf_counter()-t0:.0f}s)", flush=True)

gs={0:[],1:[],2:[],3:[]}; cos={0:[],1:[],2:[],3:[]}; gs_pk=[]; gs_wd=[]; npred=ngold=0
for k,mid in enumerate(mids):
    d=json.load(open(TOPIC/f"{mid}.json")); turns=d["turns"]; bt=d["bnd_top"]
    gold_idx=[i for i,b in enumerate(bt) if b==1]; n=len(turns)
    # --- GraphSeg ---
    seg=GraphSegWindowD(window_d=10, sim_threshold=0.25, min_seg_size=3)
    seg._glove=prime._glove; seg._glove_dim=prime._glove_dim; seg._ic_table=prime._ic_table
    pred=[]
    for u in turns: pred.extend(seg.push(u["text"]))
    pred.extend(seg.flush())
    pred_idx=sorted(set(p-1 for p in pred if 1<=p<=n))      # 1-based→0-based
    yp=[1 if i in set(pred_idx) else 0 for i in range(n)]
    pk,wd=official_pk_wd(bt,yp); gs_pk.append(pk); gs_wd.append(wd)
    npred+=len(pred_idx); ngold+=len(gold_idx)
    # --- 우리 cosine merge 없이 raw (공정: GraphSeg 도 raw turn) ewma ---
    e=np.asarray(pickle.load(open(CACHE/f"{mid}.pkl","rb")),dtype=np.float64)
    cp=[i for i,b in enumerate(adaptive_boundaries(delta_eff_seq(e),c=1.5,mode="ewma")) if b]
    for tol in (0,1,2,3):
        gs[tol].append(tol_f1(gold_idx,pred_idx,tol))
        cos[tol].append(tol_f1(gold_idx,cp,tol))
    if (k+1)%4==0: print(f"  ...{k+1}/{len(mids)}",flush=True)

print(f"\n=== GraphSeg vs cosine(ewma) — AMI {len(mids)}미팅, tolerance-F1 ===")
print(f"  {'method':16} {'±0':>6} {'±1':>6} {'±2':>6} {'±3':>6}")
print(f"  {'GraphSeg-win10':16} {np.mean(gs[0]):>6.3f} {np.mean(gs[1]):>6.3f} {np.mean(gs[2]):>6.3f} {np.mean(gs[3]):>6.3f}")
print(f"  {'cosine ewma(raw)':16} {np.mean(cos[0]):>6.3f} {np.mean(cos[1]):>6.3f} {np.mean(cos[2]):>6.3f} {np.mean(cos[3]):>6.3f}")
print(f"\n  GraphSeg Pk={np.mean(gs_pk):.3f} WD={np.mean(gs_wd):.3f} | pred경계 {npred} / gold {ngold} (밀도 비교)")
