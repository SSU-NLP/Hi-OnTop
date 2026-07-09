import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import f1_score
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
EPS=1e-6
def gate_internals(dhist, SLOPE=2.0,W1=1.15,W2=0.85,BIAS=0.0):
    x=dhist[-64:]
    if len(x)<12: return 1.0,None,None
    q10,q50,q90=np.quantile(x,[.1,.5,.9])
    tail=(q90-q50)/(q50-q10+EPS); dx=np.abs(np.diff(x)); rough=float(np.median(dx))/(q90-q10+EPS)
    score=W1*math.log(tail+EPS)+W2*math.log(rough+EPS)+BIAS
    w=min(max(1/(1+math.exp(-SLOPE*score)),0.05),0.95)
    return w,tail,rough
def gated_goldreset(e,gold,gp_params=dict()):
    n=len(e); gs=set(gold); s=np.zeros(n); m=e[0].copy(); k=1; g=e[0].copy(); recent=[e[0]]; dhist=[]
    ws=[];tails=[];roughs=[]
    for t in range(1,n):
        x=e[t]; xp=e[t-1]; gp=g.copy()
        g=nrm((1-0.15)*g+0.15*x)
        dprev=1-float(x@xp); dhist.append(dprev)
        cc=np.zeros_like(x)
        for i,sv in enumerate(reversed(recent[-2:])): cc=cc+(0.7**i)*sv
        dctx=1-float(x@nrm(cc)); sharp=0.5*dprev+0.5*dctx
        xd=dn._deneut(x,gp,1.0); md=dn._deneut(m,gp,1.0)
        drift=(1-float(xd@md))-0.6*(1-float(x@gp))
        w,tl,rg=gate_internals(dhist,**gp_params); ws.append(w)
        if tl is not None: tails.append(tl); roughs.append(rg)
        s[t]=w*sharp+(1-w)*drift
        if t in gs: m=x.copy(); k=1; recent=[x]   # gold-reset (ctx도 reset: 천장 가정)
        else:
            rho=max(0.05,1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1; recent.append(x)
    return s, float(np.mean(ws)), (float(np.mean(tails)) if tails else 0), (float(np.mean(roughs)) if roughs else 0)
def oracle_dts(sigs,gy):
    ev=S.new_evaluation()
    for sig,yt in zip(sigs,gy):
        nn=len(yt); cand=sorted(set(float(sig[i]) for i in range(1,nn)))
        if len(cand)>120: cand=list(np.quantile(cand,np.linspace(0,1,120)))
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
            f=2*p*r/(p+r) if p+r>0 else 0.0; best=max(best,f)
        F.append(best)
    return float(np.mean(F))
print("=== gated blend ORACLE (gold-reset) — 현 게이트 상수 ===")
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    res=[gated_goldreset(e,[i for i,b in enumerate(yt) if b==1]) for e,yt in zip(embs,gy)]
    sigs=[r[0] for r in res]
    print(f"  {ds:11s}: oracle={oracle_dts(sigs,gy):.3f}  w̄={np.mean([r[1] for r in res]):.2f}  tail̄={np.mean([r[2] for r in res]):.2f}  rough̄={np.mean([r[3] for r in res]):.2f}", flush=True)
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
sigs=[];golds=[];ns=[];ws=[];tls=[];rgs=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    e=nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))
    g=[i for i,b in enumerate(bt) if b==1]; s,w,tl,rg=gated_goldreset(e,g); sigs.append(s);golds.append(g);ns.append(n);ws.append(w);tls.append(tl);rgs.append(rg)
print(f"  {'AMI':11s}: oracle={oracle_ami(sigs,golds,ns):.3f}  w̄={np.mean(ws):.2f}  tail̄={np.mean(tls):.2f}  rough̄={np.mean(rgs):.2f}")
