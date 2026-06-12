#!/usr/bin/env python3
"""V_rel deploy 개선: reset 안정화(robust 갱신, anchor) + 임계치(local-max gating, refractory).
oracle 0.687 ↔ deploy 격차 메우기. 완전 online. AMI 139미팅 ablation."""
from __future__ import annotations
import json, pickle, glob, sys, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

TOPIC="data/ami/topic"; CACHE="outputs/runs/_misc/ami_emb"; LLMP="outputs/runs/_misc/ami_llm_pred"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))


class Welford:
    def __init__(self): self.n=0; self.mean=0.0; self.M2=0.0
    def push(self,x): self.n+=1; d=x-self.mean; self.mean+=d/self.n; self.M2+=d*(x-self.mean)
    def std(self): return math.sqrt(self.M2/(self.n-1)) if self.n>1 else 0.0


def segment_vrel2(e, c=1.0, lam=0.6, g_rho=0.15, rho_min=0.05, R=4, warmup=8,
                  robust=False, robust_z=2.0, anchor=0.0, peak=False):
    """robust: 이상치(s>μ+zσ) turn은 prototype 갱신서 제외. anchor: 화제 첫turn 비중.
    peak: s가 직전보다 상승(local rising)일 때만 발화."""
    n=len(e); m=e[0].copy(); first=e[0].copy(); k=1; g=e[0].copy(); gk=1; W=Welford(); bounds=[]
    s_prev=-1e9
    for t in range(1,n):
        x=e[t]
        proto = m if anchor<=0 else (anchor*first+(1-anchor)*m)
        if anchor>0: proto=proto/(np.linalg.norm(proto)+1e-12)
        ra=1.0-float(x@proto); rg=1.0-float(x@g); s=ra-lam*rg
        thr=(W.mean + c*W.std()) if W.n>=warmup else None
        fire = (thr is not None and s>thr and k>=R)
        if peak: fire = fire and (s>=s_prev)
        if fire:
            bounds.append(t); m=x.copy(); first=x.copy(); k=1
        else:
            is_out = robust and W.n>=warmup and (s > W.mean + robust_z*W.std())
            W.push(s)
            if not is_out:                          # 이상치면 prototype 오염 방지 위해 갱신 skip
                rho=max(rho_min,1.0/(k+1)); m=(1-rho)*m+rho*x; m=m/(np.linalg.norm(m)+1e-12); k+=1
            else:
                k+=1
        gr=max(g_rho,1.0/(gk+1)); g=(1-gr)*g+gr*x; g=g/(np.linalg.norm(g)+1e-12); gk+=1
        s_prev=s
    return bounds


def tol_f1(gold,pred,t=2):
    if not pred or not gold: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=t for j in gold))/len(pred)
    r=sum(1 for j in gold if any(abs(i-j)<=t for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0

DATA=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    gold=[i for i,b in enumerate(bt) if b==1]
    e=np.asarray(pickle.load(open(f"{CACHE}/{mid}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
    DATA.append((bt,n,gold,e))

def evalp(fn):
    F2,PK,WD,SC=[],[],[],[]; tot=0
    for bt,n,gold,e in DATA:
        pred=sorted(set(p for p in fn(e) if 0<p<n-1)); tot+=len(pred)
        f2=tol_f1(gold,pred); yt=[int(b) for b in bt]; yp=[1 if i in set(pred) else 0 for i in range(n)]
        pk,wd=official_pk_wd(yt,yp)
        F2.append(f2); PK.append(pk); WD.append(wd); SC.append(0.5*f2+0.25*(1-pk)+0.25*(1-wd))
    return tot,np.mean(F2),np.mean(PK),np.mean(WD),np.mean(SC)

ngold=sum(len(d[2]) for d in DATA)
print(f"AMI {len(mids)}미팅 gold={ngold}. oracle 천장 ±2F1=0.687 / LLM Score=0.640 / 기존 Score=0.203")
print(f"{'config':<40}{'pred':>6}{'  F1(±2)':>9}{'  Pk':>8}{'  WD':>8}{'  Score':>9}")
def show(label, **kw):
    best=(-1,None)
    for c in (2.0,1.5,1.2,1.0,0.8,0.6):
        tot,f2,pk,wd,sc=evalp(lambda e,kw=kw,c=c: segment_vrel2(e,c=c,**kw))
        if f2>best[0]: best=(f2,(label+f" c={c}",tot,f2,pk,wd,sc))
    _,row=best; nm,tot,f2,pk,wd,sc=row
    print(f"{nm:<40}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
show("base")
show("+robust", robust=True)
show("+peak", peak=True)
show("+robust+peak", robust=True, peak=True)
show("+robust+peak+anchor0.3", robust=True, peak=True, anchor=0.3)
show("+robust+peak R=6", robust=True, peak=True, R=6)
show("+robust+peak R=3", robust=True, peak=True, R=3)
print("(각 행은 ±2 F1 최대가 되는 c 선택. Pk/WD↓good)")
