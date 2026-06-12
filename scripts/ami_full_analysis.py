#!/usr/bin/env python3
"""AMI 전체 139 미팅 재측정: raw/merge 임베딩 캐시 + tolerance-F1 + offset 프로파일 + combo Score.
임베딩 = MiniLM-int8 (ONNX quint8_avx2, 384d). raw→ami_emb/, merge→ami_merge_emb/ 캐시.
"""
from __future__ import annotations
import json, pickle, sys, glob, os, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"src"))
from run_encoder_comparison import delta_eff_seq, boundaries, best_score_dstar
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from sentence_transformers import SentenceTransformer

enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
    model_kwargs={"provider":"CPUExecutionProvider","file_name":"onnx/model_quint8_avx2.onnx"})
def embed(ts): return np.asarray(enc.encode(ts, normalize_embeddings=True, batch_size=128,
                                            show_progress_bar=False), dtype=np.float64)

TOPIC=REPO/"data"/"ami"/"topic"
RAW=REPO/"outputs"/"runs"/"_misc"/"ami_emb"; MRG=REPO/"outputs"/"runs"/"_misc"/"ami_merge_emb"
RAW.mkdir(parents=True,exist_ok=True); MRG.mkdir(parents=True,exist_ok=True)
mids=sorted(os.path.basename(p)[:-5] for p in glob.glob(str(TOPIC/"*.json")) if "manifest" not in p)

def geom_flag(E,m=0.0):
    e=[v/(np.linalg.norm(v)+1e-12) for v in E]; fl=[False]*len(e)
    for i in range(1,len(e)-1):
        sp,sn,sb=float(e[i-1]@e[i]),float(e[i]@e[i+1]),float(e[i-1]@e[i+1])
        if sb>sp and sb>sn: fl[i]=True
    return fl
def merge(utts,yt,fl):
    g,cur=[],None
    for i in range(len(utts)):
        if fl[i] and cur is not None: cur.append(i)
        else: cur=[i]; g.append(cur)
    return [1 if any(yt[k]==1 for k in gg) else 0 for gg in g],[" ".join(utts[k] for k in gg) for gg in g]
def tol_f1(yt,yp,tol):
    gp=[i for i,b in enumerate(yp) if b]; gg=[i for i,b in enumerate(yt) if b]
    if not gp or not gg: return 0.0
    pr=sum(1 for i in gp if any(abs(i-j)<=tol for j in gg))/len(gp)
    rc=sum(1 for j in gg if any(abs(i-j)<=tol for i in gp))/len(gg)
    return 2*pr*rc/(pr+rc) if pr+rc>0 else 0.0

raw_data=[]; mrg_data=[]; W=4; prof=[[] for _ in range(2*W+1)]; offs=[]
t0=time.perf_counter()
for k,mid in enumerate(mids):
    d=json.load(open(TOPIC/f"{mid}.json")); turns=d["turns"]; bt=d["bnd_top"]; utts=[t["text"] for t in turns]
    rp=RAW/f"{mid}.pkl"
    e=pickle.load(open(rp,"rb")) if rp.exists() else embed(utts)
    e=np.asarray(e,dtype=np.float64)
    if not rp.exists(): pickle.dump(e,open(rp,"wb"))
    fl=geom_flag(e); my,mt=merge(utts,bt,fl)
    mp=MRG/f"{mid}.pkl"
    me=pickle.load(open(mp,"rb")) if mp.exists() else embed(mt)
    me=np.asarray(me,dtype=np.float64)
    if not mp.exists(): pickle.dump(me,open(mp,"wb"))
    raw_data.append((bt,delta_eff_seq(e))); mrg_data.append((my,delta_eff_seq(me)))
    # offset/profile (raw, ewma1.5)
    deff=np.array(delta_eff_seq(e)); mu,sd=deff[1:].mean(),deff[1:].std()+1e-9
    golds=[i for i,b in enumerate(bt) if b==1]
    preds=[i for i,b in enumerate(adaptive_boundaries(deff,c=1.5,mode="ewma")) if b]
    for g in golds:
        for kk in range(-W,W+1):
            j=g+kk
            if 0<=j<len(deff): prof[kk+W].append((deff[j]-mu)/sd)
        if preds:
            nn=min(preds,key=lambda p:abs(p-g))
            if abs(nn-g)<=6: offs.append(nn-g)
    if (k+1)%30==0: print(f"  ...{k+1}/{len(mids)} ({time.perf_counter()-t0:.0f}s)",flush=True)

def combo(data,label):
    yts=[y for y,_ in data]; deffs=[s for _,s in data]
    pool=np.array([v for s in deffs for v in s[1:]]); d80=float(np.percentile(pool,80))
    dor=best_score_dstar([(None,y) for y in yts],deffs)
    def sc(yps):
        from run_encoder_comparison import official_pk_wd
        from sklearn.metrics import f1_score
        pks,wds,g,p=[],[],[],[]
        for y,yp in zip(yts,yps): pk,wd=official_pk_wd(y,yp); pks.append(pk); wds.append(wd); g+=y; p+=yp
        f1=float(f1_score(g,p,zero_division=0)); pk,wd=float(np.mean(pks)),float(np.mean(wds))
        return 0.5*f1+0.25*(1-pk)+0.25*(1-wd),f1
    for nm,yps in [("p80",[boundaries(s,d80) for s in deffs]),
                   ("ewma1.5",[adaptive_boundaries(s,c=1.5,mode="ewma") for s in deffs]),
                   ("otsu",[adaptive_boundaries(s,mode="otsu") for s in deffs]),
                   ("oracle",[boundaries(s,dor) for s in deffs])]:
        s,f=sc(yps); print(f"  {label:6} {nm:9} Score={s:.4f} exactF1={f:.4f}",flush=True)

print(f"\n=== AMI 전체 {len(mids)} 미팅 ===\n")
nb=sum(sum(y) for y,_ in raw_data)
print(f"정답 경계 총 {nb}개\n")
print("[tolerance boundary-F1, ewma c=1.5]")
for label,data in [("raw",raw_data),("merge",mrg_data)]:
    row=[np.mean([tol_f1(y,adaptive_boundaries(s,c=1.5,mode='ewma')[:len(y)],tol) for y,s in data]) for tol in (0,1,2,3)]
    print(f"  {label:6} F1@tol 0={row[0]:.3f} 1={row[1]:.3f} 2={row[2]:.3f} 3={row[3]:.3f}")
print("\n[δ_eff z-score around gold boundary]")
print("  offset "+" ".join(f"{k:+d}" for k in range(-W,W+1)))
print("  z      "+" ".join(f"{np.mean(prof[k+W]):+.2f}" for k in range(-W,W+1)))
o=np.array(offs); print(f"\n[signed offset] mean={o.mean():+.2f} med={np.median(o):+.0f} "
      f"<0:{np.mean(o<0)*100:.0f}% =0:{np.mean(o==0)*100:.0f}% >0:{np.mean(o>0)*100:.0f}% (n={len(o)})")
print("\n[combo Score]")
combo(raw_data,"raw"); combo(mrg_data,"merge")
