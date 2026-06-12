"""calibration 불필요 확인: δ_eff vs adaptive-deneut Score-vs-c (AMI 전체). + Otsu(파라미터-free).
모든 c에서 deneut≥δ_eff면 calibration 자체가 불필요."""
import sys, pickle, glob, json, math
import numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from ami_adaptive_deneut_deploy import segment, tol_f1, load_ami
AMI=load_ami()
def sc(bt,n,gold,pred):
    pred=sorted(set(p for p in pred if 0<p<n-1)); f2=tol_f1(gold,pred)
    yt=[int(b) for b in bt]; yp=[1 if i in set(pred) else 0 for i in range(n)]; pk,wd=official_pk_wd(yt,yp)
    return 0.5*f2+0.25*(1-pk)+0.25*(1-wd), f2
def mean(fn):
    S=[];F=[]
    for bt,n,gold,e in AMI:
        s,f=sc(bt,n,gold,fn(e)); S.append(s); F.append(f)
    return np.mean(S),np.mean(F)
print("AMI Score(±2F1) — calibration 불필요 검증. 모든 c에서 deneut≥δ_eff?")
print(f"{'c':>6}{'  δ_eff':>16}{'  deneut':>16}")
for c in (2.5,2.0,1.5,1.2,1.0,0.8):
    ds,df=mean(lambda e,c=c:[i for i,b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)),c=c,mode="ewma")) if b])
    ns,nf=mean(lambda e,c=c: segment(e,c=c))
    print(f"{c:>6}{f'{ds:.3f}/{df:.3f}':>16}{f'{ns:.3f}/{nf:.3f}':>16}")
# Otsu (파라미터-free): δ_eff만 직접 지원
od,odf=mean(lambda e:[i for i,b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)),mode="otsu")) if b])
print(f"{'Otsu':>6}{f'{od:.3f}/{odf:.3f}':>16}{'(reset연동 필요)':>16}")
print("\n(각 칸 = Score/±2F1. 모든 c에서 deneut Score≥δ_eff면 → c 안 골라도 됨 = calibration 불필요)")
