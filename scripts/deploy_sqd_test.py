import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)

def sqd(e, TH_S=2.5, TH_D=2.5, R=2, Ld=8.0, gamma=2.0, lam=0.6,
        alpha_m=0.10, alpha_g=0.02, warmup=5, a=0.5):
    """Sharp-led Quarantined Drift. emit 0-lag, drift prototype quarantine+verify."""
    n=len(e); emit=[]
    g=e[0].copy()
    M=e[0].copy(); n_eff=1; recent=[e[0]]
    # running stats (Welford) — sharp: 전 turn / drift: committed-only
    sm=0.0;sv=0.0;sn=0; dm=0.0;dv=0.0;dn_=0
    pending=None; refr=0
    def zof(val,m,v,nn): 
        sd=math.sqrt(v/(nn-1)) if nn>1 else 0.0
        return (val-m)/sd if sd>0 else 0.0
    for t in range(1,n):
        x=e[t]; xp=e[t-1]; g_prev=g.copy()
        g=nrm((1-alpha_g)*g+alpha_g*x)
        # --- sharp (stateless) ---
        dprev=1-float(x@xp)
        cc=np.zeros_like(x)
        for i,sv2 in enumerate(reversed(recent[-2:])): cc=cc+(0.7**i)*sv2
        dctx=1-float(x@nrm(cc)); S_sharp=a*dprev+(1-a)*dctx
        z_s=zof(S_sharp,sm,sv,sn)
        sharp_hit=(refr==0 and sn>=warmup and z_s>TH_S)
        # --- drift (committed prototype, mature&clean) ---
        mature=1-math.exp(-((n_eff/Ld)**gamma))
        drift_ready=(n_eff>=4 and dn_>=warmup)
        drift_hit=False
        if drift_ready:
            xd=dn._deneut(x,g_prev,1.0); md=dn._deneut(M,g_prev,1.0)
            S_drift=(1-float(xd@md))-lam*(1-float(x@g_prev))
            z_d=zof(S_drift,dm,dv,dn_)
            drift_hit=(mature>0.5 and z_d>TH_D)
        boundary=sharp_hit or drift_hit
        if boundary:
            emit.append(t)
            pending={"t":t,"old_M":M.copy(),"old_neff":n_eff,"seed":x.copy()}
            refr=R
            continue
        # verify pending (1발화 뒤, internal only)
        if pending is not None:
            seed=pending["seed"]; oldM=pending["old_M"]
            d_new=1-float(dn._deneut(seed,g_prev,1.0)@dn._deneut(x,g_prev,1.0))
            d_os=1-float(dn._deneut(seed,g_prev,1.0)@dn._deneut(oldM,g_prev,1.0))
            d_on=1-float(dn._deneut(x,g_prev,1.0)@dn._deneut(oldM,g_prev,1.0))
            if (d_os+d_on)>d_new:   # accept new segment
                M=nrm(seed+x); n_eff=2; recent=[seed,x]
                dm=0.0;dv=0.0;dn_=0
            else:                    # rollback (false boundary) — old 유지
                M=pending["old_M"]; n_eff=pending["old_neff"]
                M=nrm((1-alpha_m)*M+alpha_m*seed); M=nrm((1-alpha_m)*M+alpha_m*x); n_eff+=2
                recent.append(x)
            pending=None
            # stats 갱신
            d=S_sharp-sm; sn+=1; sm+=d/sn; sv+=d*(S_sharp-sm)
            if refr>0: refr-=1
            continue
        # 일반 update
        rho=max(0.05,1.0/(n_eff+1)); M=nrm((1-rho)*M+rho*x); n_eff+=1; recent.append(x)
        d=S_sharp-sm; sn+=1; sm+=d/sn; sv+=d*(S_sharp-sm)
        if drift_ready:
            xd=dn._deneut(x,g_prev,1.0); md=dn._deneut(M,g_prev,1.0)
            Sd=(1-float(xd@md))-lam*(1-float(x@g_prev))
            dd=Sd-dm; dn_+=1; dm+=dd/dn_; dv+=dd*(Sd-dm)
        if refr>0: refr-=1
    return emit

# DTS
print("=== DTS deploy Score (SQD, official per-dialogue) ===")
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test")
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    gylist=[list(yt) for _,yt in dl]
    preds=[]
    for e,yt in zip(embs,gylist):
        bset={i-1 for i in sqd(e)}
        preds.append([1 if t in bset else 0 for t in range(len(yt))])
    r=S.score_dialogues(gylist,preds)
    print(f"  {ds:11s}: SQD={r['score']:.3f}  (현행DeNeut {{'dialseg711':0.367,'tiage':0.302,'superseg':0.293}}[ds], δ_eff base 0.602/0.476/0.405)", flush=True)

# AMI ±2
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
    pred=[i-1 for i in sqd(e)]
    F.append(f1_2(pred,gold))
print(f"\n=== AMI deploy ±2 F1: SQD={np.mean(F):.3f}  (현행 DeNeut ~0.131) ===")
