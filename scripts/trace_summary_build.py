#!/usr/bin/env python3
"""요약(prototype)을 어떻게 만드는지 — 각 화제 segment 의 *구성 발화 전문* + EWMA 가중치 + reset.
prototype = 그 segment 발화들의 EWMA 가중평균 (recent 일수록 큼). 원문 비절단."""
from __future__ import annotations
import json, pickle, sys, math
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/"scripts")); sys.path.insert(0,str(REPO/"src"))
CACHE=REPO/"outputs"/"runs"/"_misc"/"ami_emb"; TOPIC=REPO/"data"/"ami"/"topic"

MID="ES2002a"; N=62; LAM=0.6; G_RHO=0.15; RHO=0.05; R=4; WARM=8; C=1.0  # c=1.0: 경계 여러개라 요약구조 잘 보임
d=json.load(open(TOPIC/f"{MID}.json")); turns=d["turns"]; bt=d["bnd_top"]
e=np.asarray(pickle.load(open(CACHE/f"{MID}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
gold=set(i for i,b in enumerate(bt) if b==1); nm=lambda v: v/(np.linalg.norm(v)+1e-12)
class Wf:
    def __init__(s): s.n=0;s.m=0.0;s.M2=0.0
    def push(s,x): s.n+=1;dd=x-s.m;s.m+=dd/s.n;s.M2+=dd*(x-s.m)
    def std(s): return math.sqrt(s.M2/(s.n-1)) if s.n>1 else 0.0

# deploy V_rel 돌리며 segment 별 (turn, role, text, rho, r_act, V, thr, fire, gold) 기록
m=e[0].copy(); k=1; W=Wf(); g=e[0].copy(); rec=[]; lastf=-999
for t in range(1,N):
    x=e[t]; ra=1.0-float(x@m); rg=1.0-float(x@g); V=ra-LAM*rg
    thr=(W.m+C*W.std()) if W.n>=WARM else None
    fire=(thr is not None and V>thr and k>=R and t-lastf>=R)
    rec.append([t,turns[t]["speaker"],turns[t]["text"],ra,rg,V,thr,fire,t in gold])
    if fire:
        m=x.copy(); k=1; lastf=t
    else:
        rho=max(RHO,1.0/(k+1)); W.push(V); m=nm((1-rho)*m+rho*x); k+=1
    g=nm((1-G_RHO)*g+G_RHO*x)

# segment 단위로 묶어서 출력
L=[f"# 요약(prototype) 구성 trace — {MID} (turn 0–{N-1}, V_rel deploy c={C})","",
   "**요약(prototype) = 한 화제 안 발화들의 EWMA 가중평균.** 새 발화가 들어올 때마다 "
   "`prototype ← (1−ρ)·prototype + ρ·새발화`. ρ=mixing 가중(처음엔 큼, 길어질수록 1/(k+1)→작아짐). "
   "경계(▲)를 치면 그 요약을 *버리고* 새 발화 하나로 다시 시작.","",
   "`★`=정답경계. 각 발화 옆 `ρ=`가 그 발화가 *그 시점 요약*에 섞인 비율. `r_act`=직전까지의 요약에서 먼 정도.","",
   "---",""]
seg_id=0; L.append(f"## ═══ 요약 #{seg_id} 시작 (turn {rec[0][0]} 부터) ═══")
for i,(t,role,text,ra,rg,V,thr,fire,isg) in enumerate(rec):
    star=" ★정답경계" if isg else ""
    th=f"{thr:.2f}" if thr is not None else "warmup"
    # 이 turn 이 요약에 섞이는 ρ (fire 면 안 섞이고 새 요약의 첫 발화가 됨)
    L.append("")
    L.append(f"**[{t}] ({role})**{star}")
    L.append(f"> {text}")
    if fire:
        L.append(f"  ↳ r_act={ra:.2f} → **V_rel={V:.2f} > 임계 {th} ⟹ ▲경계!** 위 요약 버림.")
        seg_id+=1
        L.append("")
        L.append(f"## ═══ 요약 #{seg_id} 시작 (turn {t} 이 발화가 새 요약의 첫 재료) ═══")
    else:
        L.append(f"  ↳ r_act={ra:.2f} V_rel={V:.2f} (임계 {th}, 경계아님) → 이 발화를 요약에 섞음")
L.append("")
L+=["---","","**정리**: 위에서 `═══ 요약 #n ═══` 블록 하나가 prototype 하나야. 그 블록 안 발화들의 "
    "EWMA 평균이 그 화제의 '요약'이고, 다음 발화가 그 요약에서 충분히 멀면(V_rel>임계) 경계를 치고 새 블록 시작.",""]
out=REPO/"outputs"/"reports"/"vrel_summary_build_trace.md"; out.write_text("\n".join(L)+"\n")
print(f"WROTE {out}  ({seg_id+1}개 요약블록)")
