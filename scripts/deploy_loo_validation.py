import sys, glob, json, math, pickle, itertools, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
EPS=1e-6
def gatew(dh,BIAS,SLOPE=2.0,W1=1.15,W2=0.85):
    x=dh[-64:]
    if len(x)<12: return 1.0
    q10,q50,q90=np.quantile(x,[.1,.5,.9])
    tail=(q90-q50)/(q50-q10+EPS); dx=np.abs(np.diff(x)); rough=float(np.median(dx))/(q90-q10+EPS)
    return min(max(1/(1+math.exp(-SLOPE*(W1*math.log(tail+EPS)+W2*math.log(rough+EPS)+BIAS))),0.05),0.95)
def deploy(e,Wd,BIAS,beta=1.0,c=1.0,R=2,lam=0.6,g_rho=0.15,warmup=5,a=0.5):
    n=len(e); emit=[]; g=e[0].copy(); recent=[e[0]]; dh=[]; Wm=0.0;Wn=0;WM2=0.0; last=-999
    for t in range(1,n):
        x=e[t]; xp=e[t-1]; gp=g.copy(); g=nrm((1-g_rho)*g+g_rho*x)
        dprev=1-float(x@xp); dh.append(dprev)
        cc=np.zeros_like(x)
        for i,sv in enumerate(reversed(recent[-2:])): cc=cc+(0.7**i)*sv
        dctx=1-float(x@nrm(cc)); sharp=a*dprev+(1-a)*dctx
        win=recent[-Wd:]; mwin=nrm(np.sum(win,axis=0))
        drift=(1-float(dn._deneut(x,gp,beta)@dn._deneut(mwin,gp,beta)))-lam*(1-float(x@gp))
        w=gatew(dh,BIAS); V=w*sharp+(1-w)*drift
        sd=math.sqrt(WM2/(Wn-1)) if Wn>1 else 0.0; thr=(Wm+c*sd) if Wn>=warmup else None
        if thr is not None and V>thr and (t-last)>=R: emit.append(t); last=t
        d=V-Wm; Wn+=1; Wm+=d/Wn; WM2+=d*(V-Wm); recent.append(x)
    return emit
# data
DTS={}
for ds in ["tiage","dialseg711","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    DTS[ds]=(embs,gy)
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
AMI=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    e=nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))
    AMI.append((e,[i for i,b in enumerate(bt) if b==1]))
def f12(pred,gold):
    if not pred: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=2 for j in gold))/len(pred); r=sum(1 for j in gold if any(abs(i-j)<=2 for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0
def score_domain(dom,Wd,BIAS):
    if dom=="AMI":
        return float(np.mean([f12([i-1 for i in deploy(e,Wd,BIAS)],g) for e,g in AMI]))
    embs,gy=DTS[dom]
    preds=[[1 if t in {i-1 for i in deploy(e,Wd,BIAS)} else 0 for t in range(len(yt))] for e,yt in zip(embs,gy)]
    return S.score_dialogues(gy,preds)['score']
DOMS=["tiage","dialseg711","superseg","AMI"]
GRID=list(itertools.product([0.0,1.0,2.0],[6,10,16]))  # (BIAS, Wd)
cell={}
for dom in DOMS:
    for (BIAS,Wd) in GRID:
        cell[(dom,BIAS,Wd)]=score_domain(dom,Wd,BIAS)
    print(f"[done] {dom}", flush=True)
print("\n=== LOO (상수는 나머지 3개에서 mean-best, held-out에서 평가) ===")
loo=[]
for held in DOMS:
    others=[d for d in DOMS if d!=held]
    best=max(GRID, key=lambda cfg: np.mean([cell[(d,)+cfg] for d in others]))
    indom=max(GRID, key=lambda cfg: cell[(held,)+cfg])
    loo.append((held,cell[(held,)+best],best,cell[(held,)+indom]))
    print(f"  {held:11s}: LOO={cell[(held,)+best]:.3f} (chosen BIAS={best[0]},Wd={best[1]})  | in-domain-best={cell[(held,)+indom]:.3f} (overfit gap {cell[(held,)+indom]-cell[(held,)+best]:+.3f})", flush=True)
