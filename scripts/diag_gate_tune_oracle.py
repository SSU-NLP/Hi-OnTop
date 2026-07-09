import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import f1_score
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
EPS=1e-6
def gatew(dhist,SLOPE,W1,W2,BIAS):
    x=dhist[-64:]
    if len(x)<12: return 1.0
    q10,q50,q90=np.quantile(x,[.1,.5,.9])
    tail=(q90-q50)/(q50-q10+EPS); dx=np.abs(np.diff(x)); rough=float(np.median(dx))/(q90-q10+EPS)
    sc=W1*math.log(tail+EPS)+W2*math.log(rough+EPS)+BIAS
    return min(max(1/(1+math.exp(-SLOPE*sc)),0.05),0.95)
def sig_goldreset(e,gold,SLOPE,W1,W2,BIAS):
    n=len(e); gs=set(gold); s=np.zeros(n); m=e[0].copy(); k=1; g=e[0].copy(); recent=[e[0]]; dh=[]
    A,B,L0=dn.DEFAULTS['A'],dn.DEFAULTS['B'],dn.DEFAULTS['L0']
    for t in range(1,n):
        x=e[t]; xp=e[t-1]; gp=g.copy(); g=nrm(0.85*g+0.15*x)
        dprev=1-float(x@xp); dh.append(dprev)
        cc=np.zeros_like(x)
        for i,sv in enumerate(reversed(recent[-2:])): cc=cc+(0.7**i)*sv
        dctx=1-float(x@nrm(cc)); sharp=0.5*dprev+0.5*dctx
        beta=dn._beta(k,A,B,L0)                          # adaptive β
        xd=dn._deneut(x,gp,beta); md=dn._deneut(m,gp,beta)
        drift=(1-float(xd@md))-0.6*(1-float(x@gp))
        w=gatew(dh,SLOPE,W1,W2,BIAS); s[t]=w*sharp+(1-w)*drift
        if t in gs: m=x.copy(); k=1; recent=[x]
        else:
            rho=max(0.05,1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1; recent.append(x)
    return s
def oracle_dts(sigs,gy):
    ev=S.new_evaluation()
    for sig,yt in zip(sigs,gy):
        nn=len(yt); cand=sorted(set(float(sig[i]) for i in range(1,nn)))
        if len(cand)>100: cand=list(np.quantile(cand,np.linspace(0,1,100)))
        ytv=list(yt); ytv[-1]=0; best=None; bf=-1
        for thr in cand:
            yp=S.signal_to_pred(sig,thr); f=f1_score(ytv,yp,zero_division=0)
            if f>bf: bf=f; best=yp
        ev.add(ytv,best)
    return ev.compute()['total_score']
def oracle_ami(sigs,golds,ns):
    F=[]
    for s,gold,n in zip(sigs,golds,ns):
        if n<=2: F.append(0.0); continue
        cand=sorted(set(float(s[i]) for i in range(1,n-1)))
        if len(cand)>80: cand=list(np.quantile(cand,np.linspace(0,1,80)))
        best=0.0
        for thr in cand:
            pred=[i-1 for i in range(1,n-1) if s[i]>thr]
            if not pred: continue
            p=sum(1 for i in pred if any(abs(i-j)<=2 for j in gold))/len(pred); r=sum(1 for j in gold if any(abs(i-j)<=2 for i in pred))/len(gold)
            best=max(best,2*p*r/(p+r) if p+r>0 else 0.0)
        F.append(best)
    return float(np.mean(F))
# load
DTS={}
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    DTS[ds]=(embs,gy,[[i for i,b in enumerate(yt) if b==1] for yt in gy])
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
AMI=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    e=nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))
    AMI.append((e,[i for i,b in enumerate(bt) if b==1],n))
print("SLOPE BIAS | dialseg tiage superseg  AMI   | mean")
for SLOPE in (2.0,3.5,5.0):
    for BIAS in (0.0,1.0,2.0):
        row={}
        for ds in DTS:
            embs,gy,golds=DTS[ds]
            sigs=[sig_goldreset(e,g,SLOPE,1.15,0.85,BIAS) for e,g in zip(embs,golds)]
            row[ds]=oracle_dts(sigs,gy)
        asig=[sig_goldreset(e,g,SLOPE,1.15,0.85,BIAS) for e,g,n in AMI]
        amio=oracle_ami(asig,[g for _,g,_ in AMI],[n for *_,n in AMI])
        mean=np.mean([row['dialseg711'],row['tiage'],row['superseg'],amio])
        print(f"{SLOPE:4.1f} {BIAS:4.1f} | {row['dialseg711']:.3f} {row['tiage']:.3f} {row['superseg']:.3f}  {amio:.3f} | {mean:.3f}", flush=True)
