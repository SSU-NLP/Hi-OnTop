import sys, pickle, math, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
from hi_ontop.hi_ontop import HiOnTop
DSTAR_P80={"tiage":0.8223,"dialseg711":0.8029,"superseg":0.8409}
def deff_seq(e):
    s=HiOnTop(dim=e.shape[1],delta_star=1.0,ctx_window=2,ctx_decay=0.7,ctx_blend_a=0.5)
    for v in e: s.assign(v.astype(np.float64))
    return np.array([float(h["delta_eff"]) for h in s.history()])
def running_adaptive(seq,c,warmup=5):
    # μ+cσ 러닝 (무상태 신호라 reset 없음) → 끝-turn pred
    n=len(seq); m=0.0;v=0.0;cnt=0; pred=[0]*n
    for t in range(1,n):
        sd=math.sqrt(v/(cnt-1)) if cnt>1 else 0.0
        thr=(m+c*sd) if cnt>=warmup else None
        if thr is not None and seq[t]>=thr: pred[t-1]=1
        d=seq[t]-m; cnt+=1; m+=d/cnt; v+=d*(seq[t]-m)
    return pred
print("δ_eff 신호 고정: threshold만 비교 (DTS deploy, official Score)")
print(f"{'ds':11s}{'p80(고정percentile)':>20}{'running μ+1.0σ':>16}{'running μ+1.5σ':>16}{'running μ+2.0σ':>16}")
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    seqs=[deff_seq(e) for e in embs]
    p80=S.score_dialogues(gy,[S.signal_to_pred(s,DSTAR_P80[ds]) for s in seqs])['score']
    row=[p80]
    for c in (1.0,1.5,2.0):
        row.append(S.score_dialogues(gy,[running_adaptive(s,c) for s in seqs])['score'])
    print(f"{ds:11s}{row[0]:>20.3f}{row[1]:>16.3f}{row[2]:>16.3f}{row[3]:>16.3f}", flush=True)
