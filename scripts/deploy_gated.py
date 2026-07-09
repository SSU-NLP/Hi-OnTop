import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
EPS=1e-6
def sharp_weight(dhist):
    x=dhist[-64:]
    if len(x)<12: return 1.0
    q10,q50,q90=np.quantile(x,[.1,.5,.9])
    tail=(q90-q50)/(q50-q10+EPS)
    dx=np.abs(np.diff(x)); rough=float(np.median(dx))/(q90-q10+EPS)
    score=1.15*math.log(tail+EPS)+0.85*math.log(rough+EPS)
    w=1/(1+math.exp(-2.0*score))
    return min(max(w,0.05),0.95)
def gated(e,c=1.0,R=2,lam=0.6,g_rho=0.15,rho_min=0.05,warmup=5,a=0.5,ret_w=False):
    n=len(e); emit=[]; g=e[0].copy(); m=e[0].copy(); k=1; recent=[e[0]]; dhist=[]
    Wm=0.0;Wn=0;WM2=0.0; last=-999; ws=[]
    for t in range(1,n):
        x=e[t]; xp=e[t-1]; gp=g.copy()
        g=nrm((1-g_rho)*g+g_rho*x)
        dprev=1-float(x@xp); dhist.append(dprev)
        cc=np.zeros_like(x)
        for i,sv in enumerate(reversed(recent[-2:])): cc=cc+(0.7**i)*sv
        dctx=1-float(x@nrm(cc)); sharp=a*dprev+(1-a)*dctx
        xd=dn._deneut(x,gp,1.0); md=dn._deneut(m,gp,1.0)
        drift=(1-float(xd@md))-lam*(1-float(x@gp))
        w=sharp_weight(dhist); ws.append(w)
        V=w*sharp+(1-w)*drift
        sd=math.sqrt(WM2/(Wn-1)) if Wn>1 else 0.0; thr=(Wm+c*sd) if Wn>=warmup else None
        if thr is not None and V>thr and (t-last)>=R:
            emit.append(t); m=x.copy(); k=1; last=t
        else:
            Wn+=1; d=V-Wm; Wm+=d/Wn; WM2+=d*(V-Wm)
            rho=max(rho_min,1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1
        recent.append(x)
    return (emit, float(np.mean(ws)) if ws else 1.0) if ret_w else emit
print("=== DTS deploy Score (gated, official) + 평균 w_sharp ===")
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    preds=[]; wm=[]
    for e,yt in zip(embs,gy):
        em,w=gated(e,ret_w=True); wm.append(w)
        bset={i-1 for i in em}; preds.append([1 if t in bset else 0 for t in range(len(yt))])
    r=S.score_dialogues(gy,preds)
    print(f"  {ds:11s}: gated={r['score']:.3f}  w̄={np.mean(wm):.2f}  (δ_eff base {{'dialseg711':0.602,'tiage':0.476,'superseg':0.405}}[ds])", flush=True)
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
def f1_2(pred,gold):
    if not pred: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=2 for j in gold))/len(pred); r=sum(1 for j in gold if any(abs(i-j)<=2 for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0
F=[];W=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    e=nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))
    gold=[i for i,b in enumerate(bt) if b==1]; em,w=gated(e,ret_w=True); W.append(w)
    F.append(f1_2([i-1 for i in em],gold))
print(f"\n=== AMI deploy ±2 F1: gated={np.mean(F):.3f}  w̄={np.mean(W):.2f}  (현행 DeNeut ~0.131) ===")
