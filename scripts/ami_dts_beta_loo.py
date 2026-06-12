"""leave-one-domain-out: (A,B)를 held-out 뺀 3 도메인에서 tuning(sum-margin 최대) → held-out에서 검증.
adaptive vs δ_eff oracle. grid×4 행렬 1회 계산 후 LOO 분석."""
import sys, pickle, glob, json, math
import numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import f1_score
from run_encoder_comparison import load_dialogs, delta_eff_seq, CACHE
def nr(e): return e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
def beta_sig(e,gold,A,B,L0=8,g_rho=0.15,rho_min=0.05,lam=0.6):
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
def oc1(s,gold,n,tol):
    if n<=2: return 0.0
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
    return best
def ocset(sigs,go,ns,tol): return np.mean([oc1(s,g,n,tol) for s,g,n in zip(sigs,go,ns)])
def load(ds,sp):
    dl=load_dialogs(ds,sp); em=[nr(np.asarray(x,dtype=np.float64)) for x in pickle.load(open(CACHE/f"enccmp_{ds}_{sp}_minilm-int8.pkl","rb"))]
    return em,[[i for i,b in enumerate(yt) if b==1] for (u,yt) in dl],[len(yt) for (u,yt) in dl]
D={ds:(*load(ds,"test"),False) for ds in ["tiage","dialseg711","superseg"]}
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
aem=[];ag=[];an=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    aem.append(nr(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))); ag.append([i for i,b in enumerate(bt) if b==1]); an.append(n)
D["AMI"]=(aem,ag,an,True)
order=["tiage","dialseg711","superseg","AMI"]
# δ_eff baseline (per dataset)
DEFF={ds:ocset([np.asarray(delta_eff_seq(e)) for e in D[ds][0]],D[ds][1],D[ds][2],D[ds][3]) for ds in order}
GRID=[(1.5,0.5),(2.0,1.0),(2.5,1.3),(3.0,1.5),(1.8,0.7)]
# adaptive oracle 행렬 [grid x dataset]
M={}
for AB in GRID:
    for ds in order:
        M[(AB,ds)]=ocset([beta_sig(e,g,*AB) for e,g in zip(D[ds][0],D[ds][1])],D[ds][1],D[ds][2],D[ds][3])
    print(f"  grid {AB} done", flush=True)
print("\nδ_eff baseline:", {ds:round(DEFF[ds],3) for ds in order})
print("\n=== leave-one-domain-out (held-out 뺀 3개서 sum-margin 최대 (A,B) 선택 → held-out 검증) ===")
print(f"{'held-out':<12}{'tuned(A,B)':<12}{'δ_eff':>8}{'adaptive':>10}{'margin':>9}{'  판정':>7}")
allok=True
for ho in order:
    train=[d for d in order if d!=ho]
    best=None
    for AB in GRID:
        marg=sum(M[(AB,d)]-DEFF[d] for d in train)
        if best is None or marg>best[0]: best=(marg,AB)
    AB=best[1]; ad=M[(AB,ho)]; de=DEFF[ho]; ok=ad>de; allok=allok and ok
    print(f"{ho:<12}{str(AB):<12}{de:>8.3f}{ad:>10.3f}{ad-de:>+9.3f}{('  ✓' if ok else '  ✗FAIL'):>7}", flush=True)
print(f"\n모든 held-out 도메인 adaptive>δ_eff : {'✓ cross-domain robust' if allok else '✗ 일부 도메인 회귀'}")
