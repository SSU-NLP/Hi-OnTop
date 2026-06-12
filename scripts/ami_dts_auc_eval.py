#!/usr/bin/env python3
"""threshold-free 비교: AUC(신호 경계 랭킹력) + 고정-c=2.0 deploy. calib·cherry-pick 없음.
δ_eff vs adaptive-deneut(run-length β). AMI(±2도) + DTS."""
from __future__ import annotations
import sys, pickle, glob, json, math
import numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import roc_auc_score
from run_encoder_comparison import delta_eff_seq, CACHE, load_dialogs
def nr(e): return e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
def deneut_sig(e,gold,A=2.0,B=1.0,L0=8,g_rho=0.15,rho_min=0.05,lam=0.6):
    n=len(e); gs=set(gold); s=np.zeros(n); m=e[0].copy(); k=1; g=e[0].copy(); gk=1
    for t in range(1,n):
        x=e[t]; beta=min(max(A-B*math.log(1+k/L0),0.0),1.0)
        mc=m-beta*float(m@g)*g; mc=mc/(np.linalg.norm(mc)+1e-9)
        xc=x-beta*float(x@g)*g; xc=xc/(np.linalg.norm(xc)+1e-9)
        s[t]=(1-float(xc@mc))-lam*(1-float(x@g))
        if t in gs: m=x.copy(); k=1
        else:
            rho=max(rho_min,1.0/(k+1)); m=(1-rho)*m+rho*x; m=m/(np.linalg.norm(m)+1e-12); k+=1
        gr=max(g_rho,1.0/(gk+1)); g=(1-gr)*g+gr*x; g=g/(np.linalg.norm(g)+1e-12); gk+=1
    return s
def auc_set(sigs,golds,ns,tol):
    A=[]
    for s,gold,n in zip(sigs,golds,ns):
        if n<=3 or not gold or len(gold)>=n-2: continue
        if tol: y=[1 if any(abs(i-g)<=2 for g in gold) else 0 for i in range(1,n-1)]
        else: y=[1 if i in set(gold) else 0 for i in range(1,n-1)]
        if sum(y) in (0,len(y)): continue
        A.append(roc_auc_score(y,[s[i] for i in range(1,n-1)]))
    return np.mean(A)

# 데이터
DATA={}
for ds,sp in [("tiage","test"),("dialseg711","test"),("superseg","test")]:
    dl=load_dialogs(ds,sp); em=[nr(np.asarray(x,dtype=np.float64)) for x in pickle.load(open(CACHE/f"enccmp_{ds}_{sp}_minilm-int8.pkl","rb"))]
    DATA[ds]=(em,[[i for i,b in enumerate(yt) if b==1] for (u,yt) in dl],[len(yt) for (u,yt) in dl])
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
aem=[];ag=[];an=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    aem.append(nr(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))); ag.append([i for i,b in enumerate(bt) if b==1]); an.append(n)
DATA["AMI"]=(aem,ag,an)
order=["tiage","dialseg711","superseg","AMI"]
print("=== threshold-free AUC (gold-reset 신호, c·calib 없음). 경계 랭킹력 ===")
print(f"{'domain':<12}{'δ_eff(ex/±2)':>18}{'deneut(ex/±2)':>18}{'  Δ(±2)':>9}")
for ds in order:
    em,go,ns=DATA[ds]
    de=[np.asarray(delta_eff_seq(e)) for e in em]; dn=[deneut_sig(e,g) for e,g in zip(em,go)]
    de_e,de_t=auc_set(de,go,ns,False),auc_set(de,go,ns,True)
    dn_e,dn_t=auc_set(dn,go,ns,False),auc_set(dn,go,ns,True)
    print(f"{ds:<12}{f'{de_e:.3f}/{de_t:.3f}':>18}{f'{dn_e:.3f}/{dn_t:.3f}':>18}{dn_t-de_t:>+9.3f}", flush=True)
print("\n(AUC 0.5=무작위, 1.0=완벽. ±2=경계±2턴을 양성으로. cherry-pick 불가능한 신호 우열.)")
