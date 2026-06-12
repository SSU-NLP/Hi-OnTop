#!/usr/bin/env python3
"""BOCPD lagged changepoint-start emission (codex). p_t>θ 폐기 → C_t(τ)=start τ별 posterior,
L 지연 후 τ=t−L 확정(local-max). 같은 start=동일 prototype이라 start별 1입자로 정확 표현.
목표: deploy ±2F1을 clean+μcσ(0.554)에 다가가게. AMI 139미팅."""
from __future__ import annotations
import json, pickle, glob, sys, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

TOPIC="data/ami/topic"; CACHE="outputs/runs/_misc/ami_emb"; LLMP="outputs/runs/_misc/ami_llm_pred"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
def _sg(z): return 1.0/(1.0+math.exp(-z)) if -30<z<30 else (0.0 if z<0 else 1.0)
def _lse(a): mx=max(a); return mx+math.log(sum(math.exp(v-mx) for v in a))
def _nm(v): return v/(np.linalg.norm(v)+1e-12)


def segment_bocpd_lag(e, K=16, lam=0.6, c=2.0, beta=2.0, H0=0.03, L_min=8, tau_L=4.0,
                      eta=0.02, rho_min=0.05, g_rho=0.2, sig_min=0.05,
                      theta_start=0.15, R=4, warmup=15, L=3):
    n=len(e)
    # warmup: 단일 prototype 으로 V 분포 시드
    m=e[0].copy(); g=e[0].copy(); Vs=[]; W=min(warmup, max(3, n//6))
    for t in range(1, W+1):
        x=e[t]; rg=1-float(x@g); ra=1-float(x@m); Vs.append(ra-lam*rg)
        rho=max(rho_min,1.0/(t+1)); m=_nm((1-rho)*m+rho*x); g=_nm((1-g_rho)*g+g_rho*x)
    mu0=float(np.mean(Vs)); sg0=max(float(np.std(Vs)),sig_min); q0=mu0*mu0+sg0*sg0
    # particles: start τ별 1개 [s, lw, l, m, mu, q]
    P=[[W, 0.0, W, m.copy(), mu0, q0]]
    bounds=[]; last=-999
    for t in range(W+1, n):
        x=e[t]; rg=1-float(x@g); kids=[]; lwb=[]
        for s,lw,l,mh,mu,q in P:
            ra=1-float(x@mh); V=ra-lam*rg
            sd=math.sqrt(max(q-mu*mu,sig_min*sig_min)); z=(V-mu)/sd
            H=min(max(H0*_sg((l-L_min)/tau_L),1e-3),0.15)
            pB=_sg(math.log(H/(1-H))+beta*(z-c)); ee=max(eta,1.0/l)*(1-pB)
            kids.append([s, lw+math.log(1-H), l+1, _nm((1-max(rho_min,1.0/l))*mh+max(rho_min,1.0/l)*x),
                         mu+ee*(V-mu), q+ee*(V*V-q)])
            lwb.append(lw+math.log(H)+beta*(z-c))
        kids.append([t, _lse(lwb), 1, x.copy(), mu0, q0])      # 새 changepoint start=t
        Z=_lse([k[1] for k in kids])
        for k in kids: k[1]-=Z
        kids.sort(key=lambda k:-k[1])
        keep=kids[:K]
        if not any(k[0]==t for k in keep): keep[-1]=kids[[i for i,k in enumerate(kids) if k[0]==t][0]]
        Z2=_lse([k[1] for k in keep])
        for k in keep: k[1]-=Z2
        P=keep
        cmap={k[0]: math.exp(k[1]) for k in P}
        g=_nm((1-g_rho)*g+g_rho*x)
        # lagged changepoint-start emission: τ=t−L 확정
        tau=t-L
        if tau>=W and tau-last>=R:
            cv=cmap.get(tau,0.0)
            if cv>theta_start and cv>=max(cmap.get(u,0.0) for u in range(tau-R,tau+R+1)):
                bounds.append(tau); last=tau
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
        pk,wd=official_pk_wd(yt,yp); F2.append(f2); PK.append(pk); WD.append(wd); SC.append(0.5*f2+0.25*(1-pk)+0.25*(1-wd))
    return tot,np.mean(F2),np.mean(PK),np.mean(WD),np.mean(SC)
ngold=sum(len(d[2]) for d in DATA)
print(f"AMI {len(mids)}미팅 gold={ngold}. 목표 clean+μcσ ±2F1=0.554 / oracle 0.687 / LLM Score 0.640")
print(f"{'method':<30}{'pred':>6}{'  F1(±2)':>9}{'  Pk':>8}{'  WD':>8}{'  Score':>9}")
for name,fn in [("Hi-OnTop δ_eff ewma [기존]", lambda e,llm:[i for i,b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)),c=1.5,mode="ewma")) if b]),
                ("LLM full-context", lambda e,llm: llm)]:
    tot,f2,pk,wd,sc=evalp(fn); print(f"{name:<30}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
print("-- BOCPD lagged changepoint-start emission --")
for L in (2,3):
    for th in (0.1,0.15,0.25):
        fn=lambda e,llm,L=L,th=th: segment_bocpd_lag(e,theta_start=th,L=L)
        tot,f2,pk,wd,sc=evalp(fn)
        print(f"{('lag L=%d θ=%.2f'%(L,th)):<30}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}", flush=True)
