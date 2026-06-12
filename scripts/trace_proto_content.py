#!/usr/bin/env python3
"""prototype(요약 벡터)이 실제로 뭘 담는지 — c=2.0 deploy 전체 미팅. segment별 구성발화 전문 +
'대표발화'(최종 prototype 벡터에 cos 최댓값인 멤버). prototype=임베딩 평균이라 텍스트요약 아님."""
from __future__ import annotations
import json, pickle, sys, math
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/"scripts")); sys.path.insert(0,str(REPO/"src"))
CACHE=REPO/"outputs"/"runs"/"_misc"/"ami_emb"; TOPIC=REPO/"data"/"ami"/"topic"

MID="ES2002a"; LAM=0.6; G_RHO=0.15; RHO=0.05; R=4; WARM=8; C=2.0
d=json.load(open(TOPIC/f"{MID}.json")); turns=d["turns"]; bt=d["bnd_top"]
e=np.asarray(pickle.load(open(CACHE/f"{MID}.pkl","rb")),dtype=np.float64); e=e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-9)
gold=sorted(i for i,b in enumerate(bt) if b==1); nm=lambda v: v/(np.linalg.norm(v)+1e-12)
n=len(e)
class Wf:
    def __init__(s): s.n=0;s.m=0.0;s.M2=0.0
    def push(s,x): s.n+=1;dd=x-s.m;s.m+=dd/s.n;s.M2+=dd*(x-s.m)
    def std(s): return math.sqrt(s.M2/(s.n-1)) if s.n>1 else 0.0

# deploy c=2.0 → 경계 검출
m=e[0].copy(); k=1; W=Wf(); g=e[0].copy(); bnds=[]; lastf=-999
for t in range(1,n):
    x=e[t]; ra=1-float(x@m); rg=1-float(x@g); V=ra-LAM*rg
    thr=(W.m+C*W.std()) if W.n>=WARM else None
    if thr is not None and V>thr and k>=R and t-lastf>=R:
        bnds.append(t); m=x.copy(); k=1; lastf=t
    else:
        rho=max(RHO,1.0/(k+1)); W.push(V); m=nm((1-rho)*m+rho*x); k+=1
    g=nm((1-G_RHO)*g+G_RHO*x)

# segment 경계: [0, b1, b2, ..., n]
cuts=[0]+bnds+[n]
segs=[(cuts[i],cuts[i+1]-1) for i in range(len(cuts)-1)]

L=[f"# prototype(요약벡터)이 담은 내용 — {MID} (V_rel deploy **c={C} best**)","",
   f"**총 {n}발화 → 검출 경계 {len(bnds)}개 → 요약(segment) {len(segs)}개.** 정답경계 {gold}.","",
   "⚠️ **prototype은 텍스트 요약이 아니라 발화 임베딩들의 EWMA 평균 *벡터*.** 사람이 읽게, 각 segment의 "
   "**최종 prototype 벡터에 가장 가까운 실제 발화 = 『대표발화』**로 그 요약이 '대략 무슨 내용인지' 표시. "
   "그 아래 그 요약을 *구성한 전체 발화*를 prototype과의 유사도(cos)와 함께 나열.","",
   "---",""]
for si,(a,b) in enumerate(segs):
    members=list(range(a,b+1))
    # 최종 prototype = 멤버 EWMA (deploy 와 동일 방식 재현)
    mm=e[a].copy(); kk=1
    for t in members[1:]:
        rho=max(RHO,1.0/(kk+1)); mm=nm((1-rho)*mm+rho*e[t]); kk+=1
    sims={t: float(e[t]@mm) for t in members}
    rep=max(members,key=lambda t:sims[t])
    isg=[x for x in gold if a<=x<=b]
    L.append(f"## ═══ 요약 #{si} : turn {a}–{b} ({len(members)}발화){'  ★정답경계포함: '+str(isg) if isg else ''} ═══")
    L.append("")
    L.append(f"**『대표발화』 (prototype에 가장 가까움, cos={sims[rep]:.2f}) — [{rep}] ({turns[rep]['speaker']}):**")
    L.append(f"> {turns[rep]['text']}")
    L.append("")
    L.append("**이 요약을 구성한 전체 발화 (prototype과의 cos 유사도):**")
    for t in members:
        flag=" ★" if t in gold else ""
        rp=" ◀대표" if t==rep else ""
        L.append(f"- `[{t}]` cos={sims[t]:.2f}{flag}{rp} ({turns[t]['speaker']}): {turns[t]['text']}")
    L.append("")
out=REPO/"outputs"/"reports"/"vrel_proto_content_c2.md"; out.write_text("\n".join(L)+"\n")
print(f"WROTE {out} | 경계 {len(bnds)} segment {len(segs)} | bnds={bnds}")
print("정답",gold)
