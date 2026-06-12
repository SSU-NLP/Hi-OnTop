#!/usr/bin/env python3
"""3-way 분절 비교 trace: (1) clean reset(gold-reset prototype + μ+2σ, 천장 0.554)
(2) V_rel c=2.0 best(detected-reset) (3) 공식 δ_eff ewma. 같은 미팅 같은 구간."""
from __future__ import annotations
import json, pickle, sys, math
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/"scripts")); sys.path.insert(0,str(REPO/"src"))
from run_encoder_comparison import delta_eff_seq
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
CACHE=REPO/"outputs"/"runs"/"_misc"/"ami_emb"; TOPIC=REPO/"data"/"ami"/"topic"

MID="ES2002a"; N=62; LAM=0.6; G_RHO=0.15; RHO=0.05; R=4; WARM=8; C=2.0
d=json.load(open(TOPIC/f"{MID}.json")); turns=d["turns"]; bt=d["bnd_top"]
e=np.asarray(pickle.load(open(CACHE/f"{MID}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
gold=set(i for i,b in enumerate(bt) if b==1)
nm=lambda v: v/(np.linalg.norm(v)+1e-12)
class Wf:
    def __init__(s): s.n=0;s.m=0.0;s.M2=0.0
    def push(s,x): s.n+=1;dd=x-s.m;s.m+=dd/s.n;s.M2+=dd*(x-s.m)
    def std(s): return math.sqrt(s.M2/(s.n-1)) if s.n>1 else 0.0

# 공통 global EWMA
n=len(e); g=e[0].copy()
G=[None]*n
for t in range(1,n):
    G[t]=g.copy(); g=nm((1-G_RHO)*g+G_RHO*e[t])
def rg_at(t,x): return 1.0-float(x@G[t])

# (1) clean: gold-reset
mc=e[0].copy(); kc=1; Wc=Wf(); segc=0; clean=[]
for t in range(1,n):
    x=e[t]; ra=1-float(x@mc); rg=rg_at(t,x); V=ra-LAM*rg
    thr=(Wc.m+C*Wc.std()) if Wc.n>=WARM else None
    fire=(thr is not None and V>thr and kc>=R)
    clean.append((segc,ra,rg,V,thr,fire))
    if t in gold: mc=x.copy(); kc=1; segc=t            # 깨끗한 reset (정답에서)
    else: Wc.push(V); mc=nm((1-max(RHO,1.0/(kc+1)))*mc+max(RHO,1.0/(kc+1))*x); kc+=1

# (2) v2.0: detected-reset
mv=e[0].copy(); kv=1; Wv=Wf(); segv=0; v20=[]; lastv=-999
for t in range(1,n):
    x=e[t]; ra=1-float(x@mv); rg=rg_at(t,x); V=ra-LAM*rg
    thr=(Wv.m+C*Wv.std()) if Wv.n>=WARM else None
    fire=(thr is not None and V>thr and kv>=R and t-lastv>=R)
    v20.append((segv,ra,rg,V,thr,fire))
    if fire: mv=x.copy(); kv=1; segv=t; lastv=t
    else: Wv.push(V); mv=nm((1-max(RHO,1.0/(kv+1)))*mv+max(RHO,1.0/(kv+1))*x); kv+=1

# (3) δ_eff
deff=list(delta_eff_seq(e)); dbnd=set(i for i,b in enumerate(adaptive_boundaries(deff,c=1.5,mode="ewma")) if b)

L=[f"# 3-way 분절 비교 trace — {MID} (turn 0–{N-1})","",
   "| 방법 | reset 기준 | 신호 | 상태 |","|---|---|---|---|",
   "| **clean** | **정답경계(gold)** | V_rel, μ+2σ | 천장 ±2F1 0.554 (배포불가, 컨닝) |",
   "| **V_rel c2.0** | 자기 검출 | V_rel, μ+2σ | deploy best Score 0.358 |",
   "| **δ_eff** | (없음, 2턴윈도우) | δ_eff, ewma | 공식 main, Score 0.203 |","",
   "`★`=정답 · `▲`=그 방법이 친 경계(=reset) · `요약[a..b]`=현재 prototype 범위.","",
   "**핵심 관전**: clean은 요약이 *정답에서만* 새로 시작해 안 섞임 → V_rel이 경계서 또렷. "
   "V_rel c2.0은 자기 추측으로 reset해 요약 오염. δ_eff는 요약 없이 2턴만 봄.","","---",""]
for t in range(1,N):
    i=t-1; sc,rac,rgc,Vc,thc,fc=clean[i]; sv,rav,rgv,Vv,thv,fv=v20[i]
    mark="  ★정답" if t in gold else ""
    txt=turns[t]["text"]; txt=txt if len(txt)<=64 else txt[:61]+"..."
    L.append(f"**[{t}] ({turns[t]['speaker']})** {txt}{mark}")
    tc=f"{thc:.2f}" if thc is not None else "—"; tv=f"{thv:.2f}" if thv is not None else "—"
    L.append(f"  · **clean**   요약[{sc}..{t-1}] r_act={rac:.2f} V={Vc:.2f}/{tc} {'▲' if fc else ' '}")
    L.append(f"  · **V_rel2.0** 요약[{sv}..{t-1}] r_act={rav:.2f} V={Vv:.2f}/{tv} {'▲' if fv else ' '}")
    L.append(f"  · **δ_eff**    δ={deff[t]:.2f}                     {'▲' if t in dbnd else ' '}")
    L.append("")
gg=sorted(i for i in gold if i<N)
fc_=[t for t in range(1,N) if clean[t-1][5]]; fv_=[t for t in range(1,N) if v20[t-1][5]]; fd_=[t for t in range(1,N) if t in dbnd]
def hit(fs): return sum(1 for x in gg if any(abs(x-f)<=2 for f in fs))
L+=["---","",f"**구간 요약** (정답 {gg}):",
    f"- clean    경계 {fc_} — ±2 적중 {hit(fc_)}/{len(gg)}",
    f"- V_rel2.0 경계 {fv_} — ±2 적중 {hit(fv_)}/{len(gg)}",
    f"- δ_eff    경계 {fd_} — ±2 적중 {hit(fd_)}/{len(gg)}",""]
out=REPO/"outputs"/"reports"/"vrel_compare3_trace.md"; out.write_text("\n".join(L)+"\n")
print(f"WROTE {out}")
print("정답",gg); print("clean",fc_,"hit",hit(fc_)); print("v2.0",fv_,"hit",hit(fv_)); print("deff",fd_,"hit",hit(fd_))
