import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
EPS=1e-6
A_,B_,L0_=dn.DEFAULTS['A'],dn.DEFAULTS['B'],dn.DEFAULTS['L0']
def gatew(dh,SLOPE=2.0,W1=1.15,W2=0.85,BIAS=1.0):
    x=dh[-64:]
    if len(x)<12: return 1.0
    q10,q50,q90=np.quantile(x,[.1,.5,.9])
    tail=(q90-q50)/(q50-q10+EPS); dx=np.abs(np.diff(x)); rough=float(np.median(dx))/(q90-q10+EPS)
    return min(max(1/(1+math.exp(-SLOPE*(W1*math.log(tail+EPS)+W2*math.log(rough+EPS)+BIAS))),0.05),0.95)
def deploy(e,c=1.0,R=2,lam=0.6,g_rho=0.15,rho_min=0.05,warmup=5,a=0.5):
    n=len(e); emit=[]; g=e[0].copy(); m=e[0].copy(); k=1; n_eff=1; recent=[e[0]]; dh=[]
    Wm=0.0;Wn=0;WM2=0.0; last=-999; pending=None; refr=0
    def dn_b(v,gg,b): return dn._deneut(v,gg,b)
    for t in range(1,n):
        x=e[t]; xp=e[t-1]; gp=g.copy(); g=nrm((1-g_rho)*g+g_rho*x)
        dprev=1-float(x@xp); dh.append(dprev)
        cc=np.zeros_like(x)
        for i,sv in enumerate(reversed(recent[-2:])): cc=cc+(0.7**i)*sv
        dctx=1-float(x@nrm(cc)); sharp=a*dprev+(1-a)*dctx
        beta=dn._beta(k,A_,B_,L0_)
        drift=(1-float(dn_b(x,gp,beta)@dn_b(m,gp,beta)))-lam*(1-float(x@gp))
        w=gatew(dh); V=w*sharp+(1-w)*drift
        sd=math.sqrt(WM2/(Wn-1)) if Wn>1 else 0.0; thr=(Wm+c*sd) if Wn>=warmup else None
        if thr is not None and V>thr and (t-last)>=R and refr==0:
            emit.append(t); last=t; refr=R
            pending={"old_m":m.copy(),"old_k":k,"old_neff":n_eff,"seed":x.copy()}
            recent.append(x)   # ctx 무상태 — boundary여도 계속
            continue
        if pending is not None:
            seed=pending["seed"]; om=pending["old_m"]
            d_new=1-float(dn_b(seed,gp,beta)@dn_b(x,gp,beta))
            d_os=1-float(dn_b(seed,gp,beta)@dn_b(om,gp,beta)); d_on=1-float(dn_b(x,gp,beta)@dn_b(om,gp,beta))
            if (d_os+d_on)>d_new:    # accept new segment
                m=nrm(seed+x); k=2; n_eff=2
            else:                     # rollback: old 복원 + 흡수
                m=om; m=nrm((1-rho_min)*m+rho_min*seed); m=nrm((1-rho_min)*m+rho_min*x)
                k=pending["old_k"]+2; n_eff=pending["old_neff"]+2
            pending=None
            d=V-Wm; Wn+=1; Wm+=d/Wn; WM2+=d*(V-Wm); recent.append(x)
            if refr>0: refr-=1
            continue
        rho=max(rho_min,1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1; n_eff+=1
        d=V-Wm; Wn+=1; Wm+=d/Wn; WM2+=d*(V-Wm); recent.append(x)
        if refr>0: refr-=1
    return emit
print("=== DTS deploy Score (gated+quarantine, official) ===  oracle천장: dialseg .711/tiage .648/superseg .591")
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    preds=[]
    for e,yt in zip(embs,gy):
        bset={i-1 for i in deploy(e)}; preds.append([1 if t in bset else 0 for t in range(len(yt))])
    r=S.score_dialogues(gy,preds)
    base={'dialseg711':0.602,'tiage':0.476,'superseg':0.405}[ds]
    print(f"  {ds:11s}: gated+Q={r['score']:.3f}  (이전 gated {{'dialseg711':0.480,'tiage':0.375,'superseg':0.345}}[ds] / δ_eff base {base})", flush=True)
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
def f12(pred,gold):
    if not pred: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=2 for j in gold))/len(pred); r=sum(1 for j in gold if any(abs(i-j)<=2 for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0
F=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    e=nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))
    F.append(f12([i-1 for i in deploy(e)],[i for i,b in enumerate(bt) if b==1]))
print(f"\n=== AMI deploy ±2 F1: gated+Q={np.mean(F):.3f}  (이전 gated 0.115 / 현행 DeNeut 0.131 / oracle천장 0.307) ===")
