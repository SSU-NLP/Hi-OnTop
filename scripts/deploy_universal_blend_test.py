import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)

def universal_deploy(e,c=1.0,R=4,warmup=8,LD=8.0,GAMMA=2.0,LP=3.0,LAMMAX=0.6,g_rho=0.15,rho_min=0.05):
    n=len(e); pred=[]; m=e[0].copy(); k=1; g=e[0].copy(); recent=[e[0]]
    Wm=0.0;Wn=0;WM2=0.0; last=-999
    for t in range(1,n):
        x=e[t]; xp=e[t-1]
        q=1-math.exp(-(k/LD)**GAMMA); a=0.35+0.25*math.exp(-k/LP)
        dprev=1-float(x@xp)
        cc=np.zeros_like(x)
        for i,sv in enumerate(reversed(recent[-2:])): cc=cc+(0.7**i)*sv
        dctx=1-float(x@nrm(cc)); S_sharp=a*dprev+(1-a)*dctx
        beta=q; lam=LAMMAX*q
        mc=dn._deneut(m,g,beta); xc=dn._deneut(x,g,beta)
        S_drift=(1-float(xc@mc))-lam*(1-float(x@g))
        V=(1-q)*S_sharp+q*S_drift
        sd=math.sqrt(WM2/(Wn-1)) if Wn>1 else 0.0; thr=(Wm+c*sd) if Wn>=warmup else None
        if thr is not None and V>thr and k>=R and t-last>=R:
            pred.append(t); m=x.copy(); k=1; last=t; recent=[x]
        else:
            Wn+=1; d=V-Wm; Wm+=d/Wn; WM2+=d*(V-Wm)
            rho=max(rho_min,1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1; recent.append(x)
        g=nrm((1-g_rho)*g+g_rho*x)
    return pred

print("=== DTS deploy Score (official per-dialogue, c=1.0 calibration-free) ===")
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test")
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    gylist=[list(yt) for _,yt in dl]
    preds=[]
    for e,yt in zip(embs,gylist):
        bset={i-1 for i in universal_deploy(e)}
        preds.append([1 if t in bset else 0 for t in range(len(yt))])
    r=S.score_dialogues(gylist,preds)
    # 현행 DeNeut deploy 비교
    dnp=[]
    for e,yt in zip(embs,gylist):
        bset={i-1 for i in dn.segment(e)}
        dnp.append([1 if t in bset else 0 for t in range(len(yt))])
    rdn=S.score_dialogues(gylist,dnp)
    print(f"  {ds:11s}: universal={r['score']:.3f}  현행DeNeut={rdn['score']:.3f}  (δ_eff best ≈ dialseg .602/tiage .476/superseg .405)", flush=True)

# AMI ±2 deploy
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
def f1_2(pred,gold):
    if not pred: return 0.0
    p=sum(1 for i in pred if any(abs(i-j)<=2 for j in gold))/len(pred)
    r=sum(1 for j in gold if any(abs(i-j)<=2 for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r>0 else 0.0
F=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    e=nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))
    gold=[i for i,b in enumerate(bt) if b==1]
    pred=[i-1 for i in universal_deploy(e)]
    F.append(f1_2(pred,gold))
print(f"\n=== AMI deploy ±2 F1: universal={np.mean(F):.3f}  (현행 DeNeut threshold deploy ≈ 0.131) ===")
