#!/usr/bin/env python3
"""V_rel deploy 분절 trace — turn별 prototype(요약)·r_active·r_global·V_rel·임계치·reset 을
실제 미팅에서 그대로 보여줌. '깨끗한 reset' 이 왜 안 되는지 눈으로 확인용. → markdown."""
from __future__ import annotations
import json, pickle, sys, math
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/"scripts")); sys.path.insert(0,str(REPO/"src"))
CACHE=REPO/"outputs"/"runs"/"_misc"/"ami_emb"; TOPIC=REPO/"data"/"ami"/"topic"

MID="ES2002a"; N=62; LAM=0.6; G_RHO=0.15; RHO_MIN=0.05; R=4; WARMUP=8; C=1.0
d=json.load(open(TOPIC/f"{MID}.json")); turns=d["turns"]; bt=d["bnd_top"]
e=np.asarray(pickle.load(open(CACHE/f"{MID}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
gold=set(i for i,b in enumerate(bt) if b==1)
nm=lambda v: v/(np.linalg.norm(v)+1e-12)

class Wf:
    def __init__(s): s.n=0;s.m=0.0;s.M2=0.0
    def push(s,x): s.n+=1;dd=x-s.m;s.m+=dd/s.n;s.M2+=dd*(x-s.m)
    def std(s): return math.sqrt(s.M2/(s.n-1)) if s.n>1 else 0.0

m=e[0].copy(); k=1; seg_start=0; g=e[0].copy(); gk=1; W=Wf()
rows=[]; last=-999
for t in range(1,len(e)):
    x=e[t]; ra=1.0-float(x@m); rg=1.0-float(x@g); V=ra-LAM*rg
    thr=(W.m+C*W.std()) if W.n>=WARMUP else None
    fire=(thr is not None and V>thr and k>=R)
    rows.append(dict(t=t, ra=ra, rg=rg, V=V, thr=thr, fire=fire, segfrom=seg_start, l=k))
    if fire:
        m=x.copy(); k=1; seg_start=t; last=t
    else:
        W.push(V); rho=max(RHO_MIN,1.0/(k+1)); m=nm((1-rho)*m+rho*x); k+=1
    g=nm((1-G_RHO)*g+G_RHO*x); gk+=1

L=[f"# V_rel deploy 분절 trace — {MID} (turn 0–{N-1})","",
   f"**설정**: prototype=화제 EWMA 요약(경계서 reset) · global=최근맥락 EWMA(g_rho={G_RHO}) · "
   f"V_rel=r_active−{LAM}·r_global · 적응임계치 μ+{C}σ · 최소화제길이 R={R}.","",
   "**기호**: `★`=정답경계, `▲reset`=우리가 친 경계(여기서 prototype 새로 시작), "
   "`요약[a..b]`=현재 prototype이 담은 화제 발화 범위.","",
   "**읽는 법**: `r_act`=이 발화가 *지금 화제 요약*에서 먼 정도(클수록 새 화제스러움). "
   "`r_glob`=최근 전체 맥락에서 먼 정도. `V_rel`=둘의 차(임계치 넘으면 경계). "
   "정답(★)과 우리(▲)가 어긋나고, 한 번 어긋나면 요약[a..b]에 딴 화제가 섞이는 걸 보라.","",
   "---",""]
for r in rows:
    t=r["t"]
    if t>=N: break
    pre=[]
    if t in gold: pre.append("★정답경계")
    if r["fire"]: pre.append(f"▲reset")
    tag=("  ← "+", ".join(pre)) if pre else ""
    thr=f"{r['thr']:.2f}" if r["thr"] is not None else "(warmup)"
    txt=turns[t]["text"]; txt=txt if len(txt)<=70 else txt[:67]+"..."
    L.append(f"**[{t}] ({turns[t]['speaker']})** {txt}")
    L.append(f"  · 요약[{r['segfrom']}..{t-1}] (len {r['l']}) | r_act={r['ra']:.2f} r_glob={r['rg']:.2f} "
             f"**V_rel={r['V']:.2f}** vs 임계 {thr}{tag}")
    L.append("")
# 요약 통계
fires=[r["t"] for r in rows if r["fire"] and r["t"]<N]; golds=sorted(i for i in gold if i<N)
L+=["---","",f"**이 구간 요약**: 정답경계 {golds}, 우리경계(▲) {fires}.",
    f"정답 {len(golds)}개 중 ±2 적중 {sum(1 for gg in golds if any(abs(gg-f)<=2 for f in fires))}개.",""]
out=REPO/"outputs"/"reports"/"vrel_segment_trace.md"; out.write_text("\n".join(L)+"\n")
print(f"WROTE {out}")
print(f"정답경계(구간내): {golds}")
print(f"우리경계 ▲(구간내): {fires}")
