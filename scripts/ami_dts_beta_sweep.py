"""부분 de-neutralize β sweep: ra=1-cos(x−β(x·g)g, m−β(m·g)g). β=0 mean, β=1 full deneut.
중간 β가 AMI+DTS 4개 다 δ_eff strict 우위인지. lam=0.6."""
import sys, pickle, glob, json, math
import numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import f1_score
from run_encoder_comparison import load_dialogs, CACHE
def nr(e): return e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
def sig(e,gold,beta,g_rho=0.15,rho_min=0.05,lam=0.6):
    n=len(e); gs=set(gold); s=np.zeros(n); m=e[0].copy(); k=1; g=e[0].copy(); gk=1
    for t in range(1,n):
        x=e[t]
        mc=m-beta*float(m@g)*g; mc=mc/(np.linalg.norm(mc)+1e-9)
        xc=x-beta*float(x@g)*g; xc=xc/(np.linalg.norm(xc)+1e-9)
        ra=1-float(xc@mc); rg=1-float(x@g); s[t]=ra-lam*rg
        if t in gs: m=x.copy(); k=1
        else:
            rho=max(rho_min,1.0/(k+1)); m=(1-rho)*m+rho*x; m=m/(np.linalg.norm(m)+1e-12); k+=1
        gr=max(g_rho,1.0/(gk+1)); g=(1-gr)*g+gr*x; g=g/(np.linalg.norm(g)+1e-12); gk+=1
    return s
def oc(sigs,golds,ns,tol):
    F=[]
    for s,gold,n in zip(sigs,golds,ns):
        if n<=2: F.append(0.0); continue
        cand=sorted(set(float(s[i]) for i in range(1,n-1)))
        if len(cand)>80: cand=list(np.quantile(cand,np.linspace(0,1,80)))
        gset=set(gold); best=0.0
        for thr in cand:
            pred=[i for i in range(1,n-1) if s[i]>thr]
            if tol:
                if not pred: continue
                p=sum(1 for i in pred if any(abs(i-j)<=2 for j in gold))/len(pred); r=sum(1 for j in gold if any(abs(i-j)<=2 for i in pred))/len(gold)
                f=2*p*r/(p+r) if p+r>0 else 0.0
            else:
                yp=[1 if i in set(pred) else 0 for i in range(n)]; f=f1_score([1 if i in gset else 0 for i in range(n)],yp,zero_division=0)
            if f>best: best=f
        F.append(best)
    return np.mean(F)
DATA={}
for ds,sp in [("tiage","test"),("dialseg711","test"),("superseg","test")]:
    dl=load_dialogs(ds,sp); em=[nr(np.asarray(x,dtype=np.float64)) for x in pickle.load(open(CACHE/f"enccmp_{ds}_{sp}_minilm-int8.pkl","rb"))]
    DATA[ds]=(em,[[i for i,b in enumerate(yt) if b==1] for (u,yt) in dl],[len(yt) for (u,yt) in dl],False)
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
aem=[];ag=[];an=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    aem.append(nr(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))); ag.append([i for i,b in enumerate(bt) if b==1]); an.append(n)
DATA["AMI"]=(aem,ag,an,True)
order=["tiage","dialseg711","superseg","AMI"]; BASE={"tiage":0.452,"dialseg711":0.313,"superseg":0.467,"AMI":0.235}
print("부분 de-neutralize β sweep (lam=0.6) oracle. δ_eff: .452 .313 .467 .235")
print(f"{'beta':<8}"+"".join(f"{d:>9}" for d in order)+"  strict")
for beta in (0.0,0.3,0.5,0.7,0.85,1.0):
    res={ds:oc([sig(e,g,beta) for e,g in zip(DATA[ds][0],DATA[ds][1])],DATA[ds][1],DATA[ds][2],DATA[ds][3]) for ds in order}
    strict=all(res[ds]>BASE[ds] for ds in order)
    print(f"{beta:<8.2f}"+"".join(f"{res[d]:>9.3f}" for d in order)+("  ✓✓STRICT" if strict else ""), flush=True)
