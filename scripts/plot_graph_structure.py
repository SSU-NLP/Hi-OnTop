#!/usr/bin/env python3
"""v4 그래프를 timeline arc diagram 으로 시각화. 발화=시간축 점, 엣지=호.
긴 호(먼 연결=filler-to-filler)가 보이게."""
import json, pickle, math
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

REPO = Path(__file__).resolve().parent.parent
mid = "ES2002a"; LO, HI = 0, 60
TAU, T, RES = 0.4, 40.0, 0.5
d = json.load(open(REPO/"data"/"ami"/"topic"/f"{mid}.json"))
turns, bt = d["turns"], d["bnd_top"]
emb = np.asarray(pickle.load(open(REPO/"outputs"/"runs"/"_misc"/"ami_emb"/f"{mid}.pkl","rb")), dtype=np.float64)
n = len(emb); S = emb @ emb.T

G = nx.Graph(); G.add_nodes_from(range(n))
elist = []
for i in range(n):
    for j in range(i+1, n):
        if S[i, j] > TAU:
            w = float(S[i, j]) * math.exp(-abs(i-j)/T)
            if w > 1e-6:
                G.add_edge(i, j, weight=w); elist.append((i, j, float(S[i,j])))
comms = nx.community.louvain_communities(G, weight="weight", resolution=RES, seed=0)
lab = np.zeros(n, int)
for ci, c in enumerate(comms):
    for v in c: lab[v] = ci
def is_filler(t):
    import re
    w = re.findall(r"[a-z']+", t.lower())
    F = {"yeah","yep","yes","no","okay","ok","mm","mmhmm","hmm","uh","um","right","sure","alright","ah","oh","kay","mmhm","huh"}
    return len(w)>0 and all(x in F or x=="hmm" for x in w)

fig, ax = plt.subplots(figsize=(18, 6))
cmap = plt.cm.tab10
# edges as arcs (both endpoints in window)
nfar = nloc = 0
for i, j, cs in elist:
    if not (LO<=i<HI and LO<=j<HI): continue
    far = abs(i-j) >= 10
    mid_x = (i+j)/2.0; width = abs(i-j); height = width*0.5
    col = "#d62728" if far else "#999999"
    a = Arc((mid_x, 0), width, height, theta1=0, theta2=180,
            color=col, lw=0.6, alpha=0.55 if far else 0.30)
    ax.add_patch(a)
    nfar += far; nloc += (not far)
# nodes
for i in range(LO, HI):
    fil = is_filler(turns[i]["text"])
    ax.scatter(i, 0, s=120 if fil else 60, c=[cmap(lab[i]%10)],
               edgecolors="black" if fil else "none", linewidths=1.4, zorder=5)
    if fil:
        ax.annotate(turns[i]["text"][:8], (i,0), (i,-max(1,HI-LO)*0.12),
                    ha="center", fontsize=6, rotation=90, color="black")
# gold boundaries
for i in range(LO, HI):
    if bt[i]==1:
        ax.axvline(i, color="green", ls="--", lw=1.5, alpha=0.7)
        ax.annotate("gold", (i, (HI-LO)*0.5*0.5), color="green", fontsize=8, rotation=90)
ax.set_xlim(LO-1, HI); ax.set_ylim(-(HI-LO)*0.18, (HI-LO)*0.5*0.55)
ax.set_yticks([]); ax.set_xlabel("turn index")
ax.set_title(f"v4 graph structure (AMI {mid}, turn {LO}-{HI-1})  |  "
             f"RED arc = far edge (>=10 turns, filler-to-filler), GRAY = local  |  "
             f"black-ring node = filler  |  node color = community  |  green line = gold boundary\n"
             f"edges in window: far {nfar} / local {nloc}  -> {nfar/(nfar+nloc)*100:.0f}% are far (long-range)")
plt.tight_layout()
out = REPO/"outputs"/"reports"/"graph_structure.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"WROTE {out} | far-edges {nfar}, local {nloc}")
