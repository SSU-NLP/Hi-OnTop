import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]: os.environ[v]="1"
import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as DS, ami_scoring as AS
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
D=dn.DEFAULTS
def deneut_V(e, c=1.0, R=4, warmup=8):
    n=len(e); m=e[0].copy(); k=1; g=e[0].copy(); Wm=0.0;WM2=0.0;Wn=0;last=-999; V=np.full(n,np.nan)
    for t in range(1,n):
        x=e[t]; beta=dn._beta(k,D['A'],D['B'],D['L0'])
        mc=dn._deneut(m,g,beta); xc=dn._deneut(x,g,beta)
        V[t]=(1-float(xc@mc))-D['lam']*(1-float(x@g))
        sd=math.sqrt(WM2/(Wn-1)) if Wn>1 else 0.0; thr=(Wm+c*sd) if Wn>=warmup else None
        if thr is not None and V[t]>thr and k>=R and t-last>=R:
            m=x.copy(); k=1; last=t
        else:
            Wn+=1; dd=V[t]-Wm; Wm+=dd/Wn; WM2+=dd*(V[t]-Wm)
            rho=max(D['rho_min'],1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1
        g=nrm((1-D['g_rho'])*g+D['g_rho']*x)
    return V
def knee_thr(vals):
    v=np.sort(np.array([x for x in vals if not np.isnan(x)]))[::-1]  # 내림차순
    if len(v)<4: return np.inf
    x=np.linspace(0,1,len(v)); y=(v-v.min())/(v.max()-v.min()+1e-9)
    # chord (0,1)-(1,0) 아래 최대거리 = elbow
    d=y-(1-x); i=int(np.argmax(d))
    return v[i]
def bounds(V,n,thr,R=4):
    pred=[];last=-999
    for t in range(1,n-1):
        if not np.isnan(V[t]) and V[t]>thr and t-last>=R: pred.append(t); last=t
    return pred
# DTS
print("적응-c(knee, per-dialogue) vs 고정 c=1.0 vs best-c (확정 채점기)")
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    knee_p=[]; npred=0
    for e,yt in zip(embs,gy):
        V=deneut_V(e); n=len(yt); b=bounds(V,n,knee_thr(V[1:n-1])); npred+=len(b)
        knee_p.append([1 if t in {i-1 for i in b} else 0 for t in range(n)])
    print(f"  {ds:11s}: knee Score={DS.score_dialogues(gy,knee_p)['score']:.3f}  (pred={npred})", flush=True)
# AMI
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
AMI=[(nrm(np.asarray(pickle.load(open(f"{AC}/{m}.pkl","rb")),dtype=np.float64)),
      (lambda b:(b.__setitem__(-1,0) or b))(list(json.load(open(f"{TOPIC}/{m}.json"))["bnd_top"]))) for m in mids]
golds=[AS.boundaries_to_pred(len(bt),[i for i,b in enumerate(bt) if b==1]) for _,bt in AMI]
kp=[];npred=0
for e,bt in AMI:
    V=deneut_V(e); n=len(bt); b=bounds(V,n,knee_thr(V[1:n-1])); npred+=len(b)
    kp.append(AS.boundaries_to_pred(n,b))
print(f"  {'AMI':11s}: knee Score={AS.score_meetings(golds,kp)['score']:.3f}  (pred={npred}, gold={sum(sum(bt) for _,bt in AMI)})")
print("  [참고] 고정 c=1.0: dialseg .367/tiage .302/superseg .293/AMI .282 | best-c: dialseg .376/superseg .345/AMI .372(c1.5)")
