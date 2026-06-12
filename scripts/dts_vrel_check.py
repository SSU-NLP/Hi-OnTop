#!/usr/bin/env python3
"""DTS 무회귀 체크 — V_rel vs 기존 δ_eff, oracle/online 둘 다. TIAGE/Dialseg711/SuperSeg test.
인코더 minilm-int8 (enccmp 캐시, load_dialogs 정렬). exact F1(주) + Pk/WD/Score + ±2 F1."""
from __future__ import annotations
import sys, pickle, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from sklearn.metrics import f1_score
from run_encoder_comparison import load_dialogs, delta_eff_seq, official_pk_wd, CACHE

DSETS = [("tiage", "test"), ("dialseg711", "test"), ("superseg", "test")]
LAM, G_RHO, RHO_MIN, R = 0.6, 0.15, 0.05, 4


def norm_rows(e):
    return e/(np.linalg.norm(e, axis=1, keepdims=True)+1e-9)

def vrel_goldreset(e, gold, lam=LAM, g_rho=G_RHO, rho_min=RHO_MIN):
    n=len(e); gs=set(gold); s=np.zeros(n); m=e[0].copy(); k=1; g=e[0].copy(); gk=1
    for t in range(1,n):
        x=e[t]; s[t]=(1-float(x@m))-lam*(1-float(x@g))
        if t in gs: m=x.copy(); k=1
        else:
            rho=max(rho_min,1.0/(k+1)); m=(1-rho)*m+rho*x; m=m/(np.linalg.norm(m)+1e-12); k+=1
        gr=max(g_rho,1.0/(gk+1)); g=(1-gr)*g+gr*x; g=g/(np.linalg.norm(g)+1e-12); gk+=1
    return s

def vrel_online(e, tau, lam=LAM, g_rho=G_RHO, rho_min=RHO_MIN, R=R):
    """global threshold τ, detected-reset prototype."""
    n=len(e); m=e[0].copy(); k=1; g=e[0].copy(); gk=1; pred=[]; last=-999
    for t in range(1,n):
        x=e[t]; s=(1-float(x@m))-lam*(1-float(x@g))
        if s>tau and k>=R and t-last>=R:
            pred.append(t); m=x.copy(); k=1; last=t
        else:
            rho=max(rho_min,1.0/(k+1)); m=(1-rho)*m+rho*x; m=m/(np.linalg.norm(m)+1e-12); k+=1
        gr=max(g_rho,1.0/(gk+1)); g=(1-gr)*g+gr*x; g=g/(np.linalg.norm(g)+1e-12); gk+=1
    return pred

def tol_f1(gold,pred,t=2):
    if not pred or not gold: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=t for j in gold))/len(pred)
    r=sum(1 for j in gold if any(abs(i-j)<=t for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0

def metrics(dialogs, preds):
    """global exact F1 + mean Pk/WD + Score + mean ±2 F1."""
    g_all,p_all,PK,WD,F2=[],[],[],[],[]
    for (utts,yt),pred in zip(dialogs,preds):
        n=len(yt); ps=set(p for p in pred if 0<p<n-1)
        yp=[1 if i in ps else 0 for i in range(n)]
        g_all+=yt; p_all+=yp
        pk,wd=official_pk_wd(yt,yp); PK.append(pk); WD.append(wd)
        gold=[i for i,b in enumerate(yt) if b==1]; F2.append(tol_f1(gold,sorted(ps)))
    f1=f1_score(g_all,p_all,zero_division=0); pk=np.mean(PK); wd=np.mean(WD)
    return f1, pk, wd, 0.5*f1+0.25*(1-pk)+0.25*(1-wd), np.mean(F2)

def oracle_ceiling(dialogs, sigs, tol=False):
    """per-dialog best threshold. tol=False→exact F1, True→±2 F1."""
    F=[]
    for (utts,yt),s in zip(dialogs,sigs):
        n=len(yt); gold=[i for i,b in enumerate(yt) if b==1]
        cand=sorted(set(float(s[i]) for i in range(1,n-1))) if n>2 else []
        best=0.0
        for thr in cand:
            pred=[i for i in range(1,n-1) if s[i]>thr]
            if tol: f=tol_f1(gold,pred)
            else:
                yp=[1 if i in set(pred) else 0 for i in range(n)]; f=f1_score(yt,yp,zero_division=0)
            best=max(best,f)
        F.append(best)
    return np.mean(F)

print("DTS 무회귀: V_rel vs 기존 δ_eff (minilm-int8). exact F1 주지표, Score=0.5F1+0.25(1-Pk)+0.25(1-WD)\n")
for ds,split in DSETS:
    dialogs=load_dialogs(ds,split)
    embs=[norm_rows(np.asarray(a,dtype=np.float64)) for a in pickle.load(open(CACHE/f"enccmp_{ds}_{split}_minilm-int8.pkl","rb"))]
    golds=[[i for i,b in enumerate(yt) if b==1] for (utts,yt) in dialogs]
    deff=[np.asarray([0.0]+list(delta_eff_seq(e))[1:]) if False else np.asarray(delta_eff_seq(e)) for e in embs]
    vrg=[vrel_goldreset(e,g) for e,g in zip(embs,golds)]
    print(f"=== {ds} (dlg={len(dialogs)}) ===")
    # oracle 천장 (per-dialog best threshold)
    print(f"  [oracle 천장] δ_eff exactF1={oracle_ceiling(dialogs,deff):.3f}  V_rel exactF1={oracle_ceiling(dialogs,vrg):.3f}"
          f"   (±2: δ_eff {oracle_ceiling(dialogs,deff,tol=True):.3f} / V_rel {oracle_ceiling(dialogs,vrg,tol=True):.3f})")
    # online (best global τ): δ_eff threshold vs V_rel online
    def best_global(sig_pred_fn, grid):
        best=(-1,None)
        for tau in grid:
            preds=[sig_pred_fn(e,tau) for e in embs]
            f1,pk,wd,sc,f2=metrics(dialogs,preds)
            if sc>best[0]: best=(sc,(tau,f1,pk,wd,sc,f2))
        return best[1]
    grid=np.linspace(0.1,0.9,33)
    deff_pred=lambda e,tau: [i for i in range(1,len(e)) if delta_eff_seq(e)[i]>tau]  # 비효율: 아래서 캐시
    # δ_eff 는 시퀀스 캐시 사용
    def deff_pred_cached(idx,tau): s=deff[idx]; return [i for i in range(1,len(s)) if s[i]>tau]
    def online_eval(pred_by_idx, grid):
        best=(-1,None)
        for tau in grid:
            preds=[pred_by_idx(i,tau) for i in range(len(embs))]
            f1,pk,wd,sc,f2=metrics(dialogs,preds)
            if sc>best[0]: best=(sc,(round(float(tau),3),f1,pk,wd,sc,f2))
        return best[1]
    de=online_eval(lambda i,tau: deff_pred_cached(i,tau), np.linspace(0.3,0.95,27))
    vr=online_eval(lambda i,tau: vrel_online(embs[i],tau), np.linspace(0.0,0.6,31))
    print(f"  [online best-τ] δ_eff: τ={de[0]} F1={de[1]:.3f} Pk={de[2]:.3f} WD={de[3]:.3f} Score={de[4]:.3f} (±2 {de[5]:.3f})")
    print(f"  [online best-τ] V_rel: τ={vr[0]} F1={vr[1]:.3f} Pk={vr[2]:.3f} WD={vr[3]:.3f} Score={vr[4]:.3f} (±2 {vr[5]:.3f})")
    print()
