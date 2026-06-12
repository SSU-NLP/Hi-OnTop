#!/usr/bin/env python3
"""Hi-OnTop V_rel + BOCPD top-K run-length particle filter (codex 설계).
hard reset 폐기 → reset 가설 K개 병렬 유지(가설별 prototype 분리), boundary posterior mass로 emission.
완전 online. 목표: deploy가 clean+μcσ(0.554)/oracle(0.687)에 다가가나. AMI 139미팅."""
from __future__ import annotations
import json, pickle, glob, sys, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

TOPIC="data/ami/topic"; CACHE="outputs/runs/_misc/ami_emb"; LLMP="outputs/runs/_misc/ami_llm_pred"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))


def _sig(z): return 1.0/(1.0+math.exp(-z)) if z>-30 else 0.0
def _lse(a):
    mx=max(a); return mx+math.log(sum(math.exp(v-mx) for v in a))


def segment_bocpd(e, K=8, lam=0.6, c=2.0, beta=2.0, H0=0.03, L_min=8, tau_L=4.0,
                  eta=0.02, rho_min=0.05, rho_g=0.2, sig_min=0.05, theta=0.4, R=4, warmup=15):
    n=len(e); norm=lambda v: v/(np.linalg.norm(v)+1e-12)
    # --- phase1 warmup: 단일 prototype 으로 V 분포(μ0,σ0) 시드 (cold-start z 폭증 방지) ---
    m=e[0].copy(); g=e[0].copy(); Vs=[]
    W=min(warmup, max(3, n//6))
    for t in range(1, W+1):
        x=e[t]; rg=1.0-float(x@g); ra=1.0-float(x@m); Vs.append(ra-lam*rg)
        rho=max(rho_min,1.0/(t+1)); m=norm((1-rho)*m+rho*x); g=norm((1-rho_g)*g+rho_g*x)
    mu0=float(np.mean(Vs)); sg0=max(float(np.std(Vs)), sig_min); q0=mu0*mu0+sg0*sg0
    # --- phase2 particle filter (per-particle μ_h,q_h: run-length별 baseline; μ0 로 시드) ---
    parts=[[0.0, W, W, m.copy(), mu0, q0]]   # [lw, s, l, m, mu, q]
    bounds=[]; last=-999
    for t in range(W+1, n):
        x=e[t]; rg=1.0-float(x@g)
        children=[]; lw_bound_list=[]
        for lw,s,l,m_h,mu,q in parts:
            ra=1.0-float(x@m_h); V=ra-lam*rg
            sd=math.sqrt(max(q-mu*mu, sig_min*sig_min)); z=(V-mu)/sd
            H=min(max(H0*_sig((l-L_min)/tau_L), 1e-3), 0.15)
            pB=_sig(math.log(H/(1-H))+beta*(z-c)); ee=max(eta,1.0/l)*(1-pB)  # count-based 빠른 초기 적응
            rho=max(rho_min, 1.0/l)
            children.append([lw+math.log(1-H), s, l+1, norm((1-rho)*m_h+rho*x),
                             mu+ee*(V-mu), q+ee*(V*V-q)])
            lw_bound_list.append(lw+math.log(H)+beta*(z-c))
        children.append([_lse(lw_bound_list), t, 1, x.copy(), mu0, q0])  # reset child: μ0 시드
        Z=_lse([ch[0] for ch in children])
        for ch in children: ch[0]-=Z
        p_t=math.exp(children[-1][0])
        children.sort(key=lambda ch:-ch[0]); parts=children[:K]
        Z2=_lse([ch[0] for ch in parts])
        for ch in parts: ch[0]-=Z2
        g=norm((1-rho_g)*g+rho_g*x)
        if p_t>theta and t-last>=R:
            bounds.append(t); last=t
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
    llm=[p for p in json.load(open(f"{LLMP}/{mid}.json"))["pred"] if 0<p<n] if glob.glob(f"{LLMP}/{mid}.json") else []
    DATA.append((bt,n,gold,e,llm))

def evalp(fn):
    F2,PK,WD,SC=[],[],[],[]; tot=0
    for bt,n,gold,e,llm in DATA:
        pred=sorted(set(p for p in fn(e,llm) if 0<p<n-1)); tot+=len(pred)
        f2=tol_f1(gold,pred); yt=[int(b) for b in bt]; yp=[1 if i in set(pred) else 0 for i in range(n)]
        pk,wd=official_pk_wd(yt,yp)
        F2.append(f2); PK.append(pk); WD.append(wd); SC.append(0.5*f2+0.25*(1-pk)+0.25*(1-wd))
    return tot,np.mean(F2),np.mean(PK),np.mean(WD),np.mean(SC)

ngold=sum(len(d[2]) for d in DATA)
print(f"AMI {len(mids)}미팅 gold={ngold}. 목표: clean+μcσ 0.554 / oracle 0.687 / LLM Score 0.640")
print(f"{'method':<28}{'pred':>6}{'  F1(±2)':>9}{'  Pk':>8}{'  WD':>8}{'  Score':>9}")
for name,fn in [("Hi-OnTop δ_eff ewma [기존]", lambda e,llm:[i for i,b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)),c=1.5,mode="ewma")) if b]),
                ("LLM full-context", lambda e,llm: llm)]:
    tot,f2,pk,wd,sc=evalp(fn); print(f"{name:<28}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
print("-- BOCPD particle filter (online), θ sweep --")
for th in (0.5,0.4,0.3,0.25,0.2,0.15):
    fn=lambda e,llm,th=th: segment_bocpd(e,theta=th)
    tot,f2,pk,wd,sc=evalp(fn)
    print(f"{('BOCPD θ='+str(th)):<28}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
