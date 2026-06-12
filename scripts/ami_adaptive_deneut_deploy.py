#!/usr/bin/env python3
"""run-length 적응 de-neut V_rel — DEPLOY (detected-reset + 적응임계치, 완전 online).
β=clip(A−B·log(1+k/L0)) — 검출된 segment 길이 k 로 (oracle gold-reset 아님).
de-neut: prototype·발화에서 global 성분 β만큼 제거. AMI 139 + DTS. vs V_rel deploy(Score 0.358)."""
from __future__ import annotations
import json, pickle, glob, sys, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import load_dialogs, delta_eff_seq, official_pk_wd, CACHE
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

def nr1(v): return v/(np.linalg.norm(v)+1e-12)

def segment(e, c=1.0, A=2.0, B=1.0, L0=8, lam=0.6, g_rho=0.15, rho_min=0.05, R=4, warmup=8):
    """detected-reset + adaptive μ+cσ. β는 검출 segment 길이 k로."""
    n=len(e); m=e[0].copy(); k=1; g=e[0].copy(); gk=1
    Wm=0.0; WM2=0.0; Wn=0; pred=[]; last=-999
    for t in range(1,n):
        x=e[t]; beta=min(max(A-B*math.log(1+k/L0),0.0),1.0)
        mc=m-beta*float(m@g)*g; mc=mc/(np.linalg.norm(mc)+1e-9)
        xc=x-beta*float(x@g)*g; xc=xc/(np.linalg.norm(xc)+1e-9)
        V=(1-float(xc@mc))-lam*(1-float(x@g))
        sd=math.sqrt(WM2/(Wn-1)) if Wn>1 else 0.0
        thr=(Wm+c*sd) if Wn>=warmup else None
        if thr is not None and V>thr and k>=R and t-last>=R:
            pred.append(t); m=x.copy(); k=1; last=t
        else:
            Wn+=1; dd=V-Wm; Wm+=dd/Wn; WM2+=dd*(V-Wm)
            rho=max(rho_min,1.0/(k+1)); m=nr1((1-rho)*m+rho*x); k+=1
        g=nr1((1-g_rho)*g+g_rho*x); gk+=1
    return pred

def tol_f1(gold,pred,t=2):
    if not pred or not gold: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=t for j in gold))/len(pred)
    r=sum(1 for j in gold if any(abs(i-j)<=t for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0

def load_ami():
    TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
    mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
    out=[]
    for mid in mids:
        d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
        e=np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
        out.append((bt,n,[i for i,b in enumerate(bt) if b==1],e))
    return out

def ev(DATA, segfn):
    F2,PK,WD,SC=[],[],[],[]; tot=0
    for bt,n,gold,e in DATA:
        pred=sorted(set(p for p in segfn(e) if 0<p<n-1)); tot+=len(pred)
        f2=tol_f1(gold,pred); yt=[int(b) for b in bt]; yp=[1 if i in set(pred) else 0 for i in range(n)]
        pk,wd=official_pk_wd(yt,yp); F2.append(f2); PK.append(pk); WD.append(wd); SC.append(0.5*f2+0.25*(1-pk)+0.25*(1-wd))
    return tot,np.mean(F2),np.mean(PK),np.mean(WD),np.mean(SC)

if __name__ == "__main__":
    AMI=load_ami(); ng=sum(len(d[2]) for d in AMI)
    print(f"AMI {len(AMI)}미팅 gold={ng}. de-neut adaptive DEPLOY. 기존 δ_eff Score 0.203 / V_rel deploy 0.358 / oracle 천장 0.341(adaptive)")
    print(f"{'method':<28}{'pred':>6}{'  F1(±2)':>9}{'  Pk':>8}{'  WD':>8}{'  Score':>9}")
    tot,f2,pk,wd,sc=ev(AMI, lambda e:[i for i,b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)),c=1.5,mode="ewma")) if b])
    print(f"{'δ_eff ewma [기존]':<28}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
    for c in (2.0,1.5,1.2,1.0,0.8):
        tot,f2,pk,wd,sc=ev(AMI, lambda e,c=c: segment(e,c=c))
        print(f"{('adaptive-deneut c='+str(c)):<28}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}", flush=True)
