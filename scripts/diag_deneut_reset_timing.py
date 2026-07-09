import sys, pickle, math, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import f1_score
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
D=dn.DEFAULTS
# 1) DTS gold 규약 재확인 (topic_id)
import json
d=json.load(open("benchmarks/superdialseg_data/dialseg711/segmentation_file_test.json"))
t=d["dial_data"]["dialseg711"][0]["turns"]; tid=[x['topic_id'] for x in t]; lab=[x['segmentation_label'] for x in t]
ones=[i for i,l in enumerate(lab) if l==1]
print("dialseg711 d0 label=1 위치:",ones)
for g in ones[:3]:
    print(f"  idx{g}: topic[{g-1}]={tid[g-1]} topic[{g}]={tid[g]} topic[{g+1}]={tid[g+1] if g+1<len(tid) else '-'} → {'end-turn(다음과 다름)' if g+1<len(tid) and tid[g]!=tid[g+1] else '?'}")
# 2) de-neut oracle: reset 타이밍 비교
def deneut_sig(e,gold,reset_mode):
    n=len(e); gs=set(gold); s=np.zeros(n); m=e[0].copy(); k=1; g=e[0].copy(); gk=1
    A,B,L0,lam,g_rho,rho_min=D['A'],D['B'],D['L0'],D['lam'],D['g_rho'],D['rho_min']
    for t in range(1,n):
        x=e[t]; beta=dn._beta(k,A,B,L0)
        mc=dn._deneut(m,g,beta); xc=dn._deneut(x,g,beta)
        s[t]=(1-float(xc@mc))-lam*(1-float(x@g))
        # reset: 'end'=t in gs (현재,오염) / 'start'=(t-1) in gs (새 segment 시작에서 clean)
        do_reset = (t in gs) if reset_mode=='end' else ((t-1) in gs)
        if do_reset: m=x.copy(); k=1
        else:
            rho=max(rho_min,1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1
        gr=max(g_rho,1.0/(gk+1)); g=nrm((1-gr)*g+gr*x); gk+=1
    return s
def oracle(sigs,gy):
    ev=S.new_evaluation()
    for sig,yt in zip(sigs,gy):
        nn=len(yt); cand=sorted(set(float(sig[i]) for i in range(1,nn)))
        if len(cand)>100: cand=list(np.quantile(cand,np.linspace(0,1,100)))
        ytv=list(yt); ytv[-1]=0; best=None; bf=-1
        for thr in cand:
            yp=S.signal_to_pred(sig,thr); f=f1_score(ytv,yp,zero_division=0)
            if f>bf: bf=f; best=yp
        ev.add(ytv,best)
    return ev.compute()['total_score']
print("\n=== de-neut oracle: reset 타이밍 (DTS, 공식 per-dialogue) ===")
print(f"{'ds':11s}{'reset@end(현재)':>16}{'reset@start(수정)':>18}  (δ_eff oracle 참고)")
deff_ref={'dialseg711':0.720,'tiage':0.598,'superseg':0.567}
for ds in ["dialseg711","tiage","superseg"]:
    dl=load_dialogs(ds,"test"); gy=[list(yt) for _,yt in dl]
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    golds=[[i for i,b in enumerate(yt) if b==1] for yt in gy]
    se=[deneut_sig(e,g,'end') for e,g in zip(embs,golds)]
    ss=[deneut_sig(e,g,'start') for e,g in zip(embs,golds)]
    print(f"{ds:11s}{oracle(se,gy):>16.3f}{oracle(ss,gy):>18.3f}  ({deff_ref[ds]})", flush=True)
