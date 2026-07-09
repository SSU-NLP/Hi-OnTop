import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]: os.environ[v]="1"
import sys, glob, json, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from hi_ontop import ami_scoring as AS
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
AMI=[]
for m in mids:
    bt=list(json.load(open(f"{TOPIC}/{m}.json"))["bnd_top"]); bt[-1]=0
    AMI.append((nrm(np.asarray(pickle.load(open(f"{AC}/{m}.pkl","rb")),dtype=np.float64)),bt))
ngold=sum(sum(bt) for _,bt in AMI)
print(f"AMI gold경계={ngold}. dn.segment c sweep (warmup=8, 확정 채점기)")
print(f"{'c':>5}{'  Score':>9}{'  ±2F1':>9}{'  Pk':>8}{'  WD':>8}{'  pred':>7}")
for c in (0.0,0.3,0.5,0.8,1.0,1.2,1.5):
    golds=[AS.boundaries_to_pred(len(bt),[i for i,b in enumerate(bt) if b==1]) for _,bt in AMI]
    npred=0; preds=[]
    for e,bt in AMI:
        idx=dn.segment(e,c=c); npred+=len(idx); preds.append(AS.boundaries_to_pred(len(bt),idx))
    r=AS.score_meetings(golds,preds)
    print(f"{c:>5}{r['score']:>9.3f}{r['f1']:>9.3f}{r['pk']:>8.3f}{r['wd']:>8.3f}{npred:>7}",flush=True)
