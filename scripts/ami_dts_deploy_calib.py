"""공정 deploy: AMI calib/test split. 각 방법 calib에서 best-c → test 보고 (c cherry-pick 제거).
δ_eff·adaptive-deneut 둘 다 best-c. Score 비교."""
import sys, pickle, glob, json, math
import numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
sys.path.insert(0,"scripts")
from ami_adaptive_deneut_deploy import segment, tol_f1
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
DATA=[]
for mid in mids:
    d=json.load(open(f"{TOPIC}/{mid}.json")); bt=list(d["bnd_top"]); n=len(bt); bt[-1]=0
    e=np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
    DATA.append((bt,n,[i for i,b in enumerate(bt) if b==1],e))
def score(bt,n,gold,pred):
    pred=sorted(set(p for p in pred if 0<p<n-1)); f2=tol_f1(gold,pred)
    yt=[int(b) for b in bt]; yp=[1 if i in set(pred) else 0 for i in range(n)]; pk,wd=official_pk_wd(yt,yp)
    return 0.5*f2+0.25*(1-pk)+0.25*(1-wd), f2
def deff_seg(e,c): return [i for i,b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)),c=c,mode="ewma")) if b]
calib=[DATA[i] for i in range(0,len(DATA),2)]; test=[DATA[i] for i in range(1,len(DATA),2)]
def mean_score(subset,segfn):
    S=[]; F=[]
    for bt,n,gold,e in subset:
        sc,f2=score(bt,n,gold,segfn(e)); S.append(sc); F.append(f2)
    return np.mean(S),np.mean(F)
print("AMI calib(even)/test(odd) — 각 방법 calib best-c → test 보고. Score / ±2F1")
for name,fn,cs in [("δ_eff ewma", deff_seg, [2.5,2.0,1.5,1.0,0.5]),
                   ("adaptive-deneut", lambda e,c: segment(e,c=c), [2.5,2.0,1.5,1.2,1.0,0.8])]:
    best=None
    for c in cs:
        sc,_=mean_score(calib, lambda e,c=c: fn(e,c))
        if best is None or sc>best[0]: best=(sc,c)
    c=best[1]; tsc,tf2=mean_score(test, lambda e,c=c: fn(e,c)); csc,_=mean_score(calib, lambda e,c=c: fn(e,c))
    print(f"  {name:<18} best-c={c} | calib Score={csc:.3f} | TEST Score={tsc:.3f} ±2F1={tf2:.3f}", flush=True)
