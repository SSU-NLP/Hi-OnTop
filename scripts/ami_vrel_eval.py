#!/usr/bin/env python3
"""Hi-OnTop V_rel deploy: 상대신호(r_active − λ·r_global) + 적응 임계치 + detected-경계 reset 루프.
완전 online (gold 미사용). gold-reset oracle 천장 0.687에 deploy가 얼마나 다가가나. AMI 139미팅."""
from __future__ import annotations
import json, pickle, glob, sys, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling

TOPIC="data/ami/topic"; CACHE="outputs/runs/_misc/ami_emb"; LLMP="outputs/runs/_misc/ami_llm_pred"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))


class Welford:
    def __init__(self): self.n=0; self.mean=0.0; self.M2=0.0
    def push(self,x): self.n+=1; d=x-self.mean; self.mean+=d/self.n; self.M2+=d*(x-self.mean)
    def std(self): return math.sqrt(self.M2/(self.n-1)) if self.n>1 else 0.0


def segment_vrel(e, c=1.0, lam=0.6, g_rho=0.15, rho_min=0.05, R=4, warmup=8):
    """online: m=active prototype(detected 경계서 reset), g=global EWMA(누적).
    signal s=r_active-λ·r_global, 적응 임계치 μ_s+c·σ_s (non-boundary 표본)."""
    n=len(e); m=e[0].copy(); k=1; g=e[0].copy(); gk=1; W=Welford(); bounds=[]
    for t in range(1,n):
        x=e[t]; ra=1.0-float(x@m); rg=1.0-float(x@g); s=ra-lam*rg
        thr=(W.mean + c*W.std()) if W.n>=warmup else None
        if thr is not None and s>thr and k>=R:
            bounds.append(t); m=x.copy(); k=1                 # detected 경계 → active reset
        else:
            W.push(s)                                          # 임계치 표본 (경계 아닐 때만)
            rho=max(rho_min,1.0/(k+1)); m=(1-rho)*m+rho*x; m=m/(np.linalg.norm(m)+1e-12); k+=1
        gr=max(g_rho,1.0/(gk+1)); g=(1-gr)*g+gr*x; g=g/(np.linalg.norm(g)+1e-12); gk+=1  # global 누적
    return bounds


def tol_f1(gold,pred,t=2):
    if not pred or not gold: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=t for j in gold))/len(pred)
    r=sum(1 for j in gold if any(abs(i-j)<=t for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0

DATA=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); turns=d["turns"]; bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    gold=[i for i,b in enumerate(bt) if b==1]
    e=np.asarray(pickle.load(open(f"{CACHE}/{mid}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
    llm=[p for p in json.load(open(f"{LLMP}/{mid}.json"))["pred"] if 0<p<n] if glob.glob(f"{LLMP}/{mid}.json") else []
    DATA.append((mid,turns,bt,n,gold,e,llm))

def evalp(pred_fn):
    F2,PK,WD,SC=[],[],[],[]; tot=0
    for mid,turns,bt,n,gold,e,llm in DATA:
        pred=sorted(set(p for p in pred_fn(mid,turns,n,e,llm) if 0<p<n-1)); tot+=len(pred)
        f2=tol_f1(gold,pred); yt=[int(b) for b in bt]; yp=[1 if i in set(pred) else 0 for i in range(n)]
        pk,wd=official_pk_wd(yt,yp)
        F2.append(f2); PK.append(pk); WD.append(wd); SC.append(0.5*f2+0.25*(1-pk)+0.25*(1-wd))
    return tot,np.mean(F2),np.mean(PK),np.mean(WD),np.mean(SC)

def tt(mid,turns,n,e,llm):
    seg=StreamingTextTiling(); out=[]
    for t in turns: out+=seg.push(t["text"])
    out+=seg.flush(); return out

ngold=sum(len(d[4]) for d in DATA)
print(f"AMI {len(mids)}미팅 gold={ngold}. F1=±2tol, Pk/WD↓, Score=0.5F1+0.25(1-Pk)+0.25(1-WD)")
print(f"{'method':<30}{'pred':>6}{'  F1(±2)':>9}{'  Pk':>8}{'  WD':>8}{'  Score':>9}")
for name,fn in [("Hi-OnTop δ_eff ewma [기존]", lambda mid,turns,n,e,llm:[i for i,b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)),c=1.5,mode="ewma")) if b]),
                ("TextTiling", tt), ("LLM full-context", lambda mid,turns,n,e,llm: llm)]:
    tot,f2,pk,wd,sc=evalp(fn); print(f"{name:<30}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
print("-- V_rel deploy (online, detected-reset), c sweep --")
best=None
for c in (2.5,2.0,1.5,1.2,1.0,0.8,0.5):
    fn=lambda mid,turns,n,e,llm,c=c: segment_vrel(e,c=c)
    tot,f2,pk,wd,sc=evalp(fn)
    print(f"{('V_rel c='+str(c)):<30}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
print("(gold-reset oracle 천장 ±2 F1=0.687 / LLM Score=0.640)")
