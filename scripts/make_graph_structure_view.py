#!/usr/bin/env python3
"""v4 그래프 구조를 실제로 덤프 → markdown. 노드=발화, 엣지=cos>τ (시간감쇠 가중), community."""
import json, pickle, sys, math
from pathlib import Path
import numpy as np
import networkx as nx

REPO = Path(__file__).resolve().parent.parent
mid = "ES2002a"; LO, HI = 0, 46
TAU, T, RES, SM = 0.4, 40.0, 0.5, 2

d = json.load(open(REPO/"data"/"ami"/"topic"/f"{mid}.json"))
turns, bt = d["turns"], d["bnd_top"]
emb = np.asarray(pickle.load(open(REPO/"outputs"/"runs"/"_misc"/"ami_emb"/f"{mid}.pkl","rb")), dtype=np.float64)
n = len(emb); S = emb @ emb.T

# build weighted graph
G = nx.Graph(); G.add_nodes_from(range(n))
edges = {}  # i -> [(j, cos, weight)]
for i in range(n):
    edges[i] = []
for i in range(n):
    for j in range(i+1, n):
        if S[i, j] > TAU:
            w = float(S[i, j]) * math.exp(-abs(i-j)/T)
            if w > 1e-6:
                G.add_edge(i, j, weight=w)
                edges[i].append((j, float(S[i,j]), w)); edges[j].append((i, float(S[i,j]), w))
comms = nx.community.louvain_communities(G, weight="weight", resolution=RES, seed=0)
lab = np.zeros(n, int)
for ci, c in enumerate(comms):
    for v in c: lab[v] = ci
lab_s = lab.copy()
for i in range(n):
    lo, hi = max(0,i-SM), min(n,i+SM+1); v,c = np.unique(lab[lo:hi],return_counts=True); lab_s[i]=v[np.argmax(c)]
remap = {}
for x in lab_s:
    if x not in remap: remap[x]=len(remap)

trunc = lambda s,k=42: s if len(s)<=k else s[:k-1]+"…"
L = [f"# v4 그래프 구조 실제 덤프 — AMI {mid} (turn {LO}–{HI-1})", "",
     "## 그래프 만드는 법 (레시피)",
     f"- **노드** = 발화 1개. **임베딩** = MiniLM-int8(384d).",
     f"- **엣지** = 두 발화의 cosine 유사도 `cos > τ({TAU})` 이면 연결.",
     f"- **엣지 가중** = `cos × exp(−|i−j|/T)`, T={int(T)} → **멀리 떨어진 발화일수록 약하게**(시간감쇠).",
     f"- **community** = Louvain(resolution={RES}) 로 가중그래프를 토픽 덩어리로 분할 → 그 뒤 median smooth({SM}).",
     f"- **경계** = 이웃 발화의 community 가 바뀌는 곳.", "",
     f"전체 미팅: 노드 {n}, 엣지 {G.number_of_edges()}, community {len(set(lab_s))}개 (정답 top화제 {sum(bt)}).",
     "", "## community(토픽 덩어리) 구성 — 어느 발화들이 묶였나", ""]
# community -> member turns (LO~HI 구간 표시)
from collections import defaultdict
mem = defaultdict(list)
for i in range(n): mem[remap[lab_s[i]]].append(i)
for t in sorted(mem):
    ms = mem[t]
    inwin = [x for x in ms if LO<=x<HI]
    L.append(f"- **T{t}** ({len(ms)} 발화): {ms[:25]}{' …' if len(ms)>25 else ''}")
L += ["", "→ **같은 T 안에 멀리 떨어진 발화가 섞여 있으면 = 그래프가 비연속(revisit) 링크를 만든 것.**", ""]

L += ["## 발화별 엣지 (turn 0–45) — 누가 누구랑 연결됐나", "",
      "`연결`= 그 발화와 엣지로 이어진 turn (cos). **★far**=10턴 이상 떨어진 long-range 링크(가짜 revisit 의심).", "",
      "| # | (화자) `T` | 발화 | 연결된 turn (cos) |",
      "|--:|:--|------|------|"]
for i in range(LO, HI):
    es = sorted(edges[i], key=lambda x:-x[1])[:6]   # 상위 6 by cos
    estr = ", ".join(f"{j}({cs:.2f}){'★far' if abs(i-j)>=10 else ''}" for j,cs,w in es) or "—(고립)"
    gm = "★" if bt[i]==1 else ""
    L.append(f"| {i}{gm} | ({turns[i]['speaker']}) `T{remap[lab_s[i]]}` | {trunc(turns[i]['text'])} | {estr} |")

L += ["", "## 읽는 법 / 왜 과분절되나",
      "- 한 발화가 **여러 다른 T 의 발화와 연결**되거나, **★far(먼) 링크**가 있으면 community 배정이 흔들림.",
      "- 예: 동물그리기 구간에서 beagle 발화가 intro(T0) 발화와 cos 높아 엮이면 → 그 turn 이 T0 로 튐 → 경계 오발.",
      "- 즉 **그래프 엣지가 '같은 토픽'이 아니라 '비슷한 표현/주제어'로 생겨서**, 한 토픽 안 내용 변화·먼 우연 유사도가 community 를 쪼갬."]

out = REPO/"outputs"/"reports"/"graph_structure_view.md"
out.write_text("\n".join(L)+"\n")
print(f"WROTE {out} | 노드 {n}, 엣지 {G.number_of_edges()}, community {len(set(lab_s))}")
