import sys, glob, json, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE, load_dialogs
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
def hl2a(hl): return 1-0.5**(1.0/hl)
AF,ASL=hl2a(8),hl2a(64)
def bg_trace(e):
    """turn별 centroid_shift = 1-cos(g_fast,g_slow), r_global=1-cos(x,g_slow)."""
    n=len(e); gf=e[0].copy(); gs=e[0].copy(); cs=[];rg=[]
    for t in range(1,n):
        x=e[t]; gf=nrm((1-AF)*gf+AF*x); gs=nrm((1-ASL)*gs+ASL*x)
        cs.append(1-float(gf@gs)); rg.append(1-float(x@gs))
    return np.array(cs),np.array(rg)
# AMI 임베딩
TOPIC="data/ami/topic"; AC="outputs/runs/_misc/ami_emb"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
def ami_emb(mid): return nrm(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl","rb")),dtype=np.float64))

# (a) 단일 미팅 centroid_shift 분포
single=[]
for mid in mids[:30]:
    cs,_=bg_trace(ami_emb(mid)); single.append(cs.mean())
print(f"[a] 단일 AMI 미팅 centroid_shift 평균: {np.mean(single):.4f} (낮을수록 안정배경)")

# (b) 두 미팅 concat — seam(±5) vs 비-seam
seam_vals=[]; nonseam_vals=[]
for i in range(0,20,2):
    e1=ami_emb(mids[i]); e2=ami_emb(mids[i+1]); L=len(e1)
    cat=np.vstack([e1,e2]); cs,_=bg_trace(cat)
    # cs index t-1 대응. seam at turn L (e2 첫 turn). cs[L-1] 근방
    seam=L-1
    seam_window=cs[max(0,seam-2):seam+3]
    nonseam=np.concatenate([cs[10:seam-5], cs[seam+5:-5]])
    seam_vals.append(seam_window.max()); nonseam_vals.append(np.median(nonseam))
    if i==0:
        print(f"\n[b] 예시 concat {mids[0]}+{mids[1]} (seam at turn {L}):")
        print("    seam±2 centroid_shift:", [round(float(v),3) for v in cs[seam-2:seam+3]])
        print(f"    비-seam median: {np.median(nonseam):.3f}")
print(f"\n[b] 두-미팅 concat: seam peak 평균={np.mean(seam_vals):.4f}  vs 비-seam median 평균={np.mean(nonseam_vals):.4f}  (배율 {np.mean(seam_vals)/np.mean(nonseam_vals):.1f}x)")

# (c) DTS dialseg711 centroid_shift (안정배경 없음 가정)
dl=load_dialogs("dialseg711","test")
embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/"enccmp_dialseg711_test_minilm-int8.pkl","rb"))]
dts=[bg_trace(e)[0].mean() for e in embs[:50] if len(e)>10]
print(f"\n[c] DTS dialseg711 centroid_shift 평균: {np.mean(dts):.4f} (단일미팅 {np.mean(single):.4f} 대비)")
