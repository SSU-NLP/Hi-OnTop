import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]: os.environ[v]="1"
import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as DS, ami_scoring as AS
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
D=dn.DEFAULTS
def deneut_V_and_reset(e, c=1.0, R=4, warmup=8):
    # de-neut deploy: V 시퀀스 + c=1.0 임계로 detected reset (현 디폴트와 동일 동작), V 기록
    n=len(e); m=e[0].copy(); k=1; g=e[0].copy(); Wm=0.0;Wn=0;WM2=0.0; last=-999; V=np.zeros(n); fixed=[]
    for t in range(1,n):
        x=e[t]; beta=dn._beta(k,D['A'],D['B'],D['L0'])
        mc=dn._deneut(m,g,beta); xc=dn._deneut(x,g,beta)
        V[t]=(1-float(xc@mc))-D['lam']*(1-float(x@g))
        sd=math.sqrt(WM2/(Wn-1)) if Wn>1 else 0.0; thr=(Wm+c*sd) if Wn>=warmup else None
        if thr is not None and V[t]>thr and k>=R and t-last>=R:
            fixed.append(t); m=x.copy(); k=1; last=t
        else:
            Wn+=1; d=V[t]-Wm; Wm+=d/Wn; WM2+=d*(V[t]-Wm)
            rho=max(D['rho_min'],1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1
        g=nrm((1-D['g_rho'])*g+D['g_rho']*x)
    return V, fixed
def otsu(vals):
    vals=np.array([v for v in vals if v!=0.0])
    if len(vals)<3: return np.inf
    qs=np.quantile(vals,np.linspace(0.05,0.95,30)); best=(None,-1)
    for thr in qs:
        a=vals[vals<=thr]; b=vals[vals>thr]
        if len(a)==0 or len(b)==0: continue
        w0,w1=len(a)/len(vals),len(b)/len(vals); var=w0*w1*(a.mean()-b.mean())**2
        if var>best[1]: best=(thr,var)
    return best[0] if best[0] is not None else np.inf
def otsu_bounds(V,n,R=4):
    thr=otsu(V[1:n-1]); pred=[]; last=-999
    for t in range(1,n-1):
        if V[t]>thr and t-last>=R: pred.append(t); last=t
    return pred
# DTS
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    g_fix=[];g_ots=[]
    for e,yt in zip(embs,gy):
        V,fx=deneut_V_and_reset(e); n=len(yt)
        g_fix.append([1 if t in {i-1 for i in fx} else 0 for t in range(n)])
        g_ots.append([1 if t in {i-1 for i in otsu_bounds(V,n)} else 0 for t in range(n)])
    sf=DS.score_dialogues(gy,g_fix)['score']; so=DS.score_dialogues(gy,g_ots)['score']
    print(f"  {ds:11s}: 고정c=1.0={sf:.3f}  Otsu(천장)={so:.3f}  Δ={so-sf:+.3f}", flush=True)
# AMI
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
golds=[];pf=[];po=[]
for mid in mids:
    bt=list(json.load(open(f"{TOPIC}/{mid}.json"))["bnd_top"]); bt[-1]=0; n=len(bt)
    e=nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))
    V,fx=deneut_V_and_reset(e)
    golds.append(AS.boundaries_to_pred(n,[i for i,b in enumerate(bt) if b==1]))
    pf.append(AS.boundaries_to_pred(n,fx)); po.append(AS.boundaries_to_pred(n,otsu_bounds(V,n)))
print(f"  {'AMI':11s}: 고정c=1.0={AS.score_meetings(golds,pf)['score']:.3f}  Otsu(천장)={AS.score_meetings(golds,po)['score']:.3f}")
