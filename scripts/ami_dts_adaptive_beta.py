"""run-length 적응 β (최적화: 한 패스). β=clip(A−B·log(1+l/L0)). 짧은 seg→β높(de-neut), 긴 seg→β낮(V_rel)."""
import sys, pickle, glob, json, math
import numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import f1_score
from run_encoder_comparison import load_dialogs, CACHE
def nr(e): return e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
def sig(e,gold,A,B,L0=8,g_rho=0.15,rho_min=0.05,lam=0.6):
    n=len(e); gs=set(gold); s=np.zeros(n); m=e[0].copy(); k=1; g=e[0].copy(); gk=1; bsum=0.0; bc=0
    for t in range(1,n):
        x=e[t]
        beta=min(max(A-B*math.log(1+k/L0),0.0),1.0); bsum+=beta; bc+=1
        mc=m-beta*float(m@g)*g; mc=mc/(np.linalg.norm(mc)+1e-9)
        xc=x-beta*float(x@g)*g; xc=xc/(np.linalg.norm(xc)+1e-9)
        s[t]=(1-float(xc@mc))-lam*(1-float(x@g))
        if t in gs: m=x.copy(); k=1
        else:
            rho=max(rho_min,1.0/(k+1)); m=(1-rho)*m+rho*x; m=m/(np.linalg.norm(m)+1e-12); k+=1
        gr=max(g_rho,1.0/(gk+1)); g=(1-gr)*g+gr*x; g=g/(np.linalg.norm(g)+1e-12); gk+=1
    return s, (bsum/bc if bc else 0)
def oc(pairs,golds,ns,tol):
    F=[]; bs=[]
    for (s,b),gold,n in zip(pairs,golds,ns):
        bs.append(b)
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
    return np.mean(F), np.mean(bs)
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
print("run-length 적응 β=clip(A−B·log(1+l/8)). δ_eff: .452 .313 .467 .235")
print(f"{'(A,B)':<12}"+"".join(f"{d:>9}" for d in order)+"  β평균(t/d/s/A) strict")
for A,B in [(1.5,0.5),(1.6,0.6),(1.4,0.45),(1.8,0.7)]:
    res={}; bb={}
    for ds in order:
        pairs=[sig(e,g,A,B) for e,g in zip(DATA[ds][0],DATA[ds][1])]
        res[ds],bb[ds]=oc(pairs,DATA[ds][1],DATA[ds][2],DATA[ds][3])
    strict=all(res[ds]>BASE[ds] for ds in order)
    print(f"{str((A,B)):<12}"+"".join(f"{res[d]:>9.3f}" for d in order)+f"  {bb['tiage']:.2f}/{bb['dialseg711']:.2f}/{bb['superseg']:.2f}/{bb['AMI']:.2f}"+("  ✓✓STRICT" if strict else ""), flush=True)
