#!/usr/bin/env python3
"""AMI 대화 전문 + 정답경계 + forward-merge 분절 → markdown.
filler(내용어 0)를 *다음* 실질 발화에 합침. ami_conversation_boundaries.md(v4 graph)와 비교용."""
import json, pickle, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"src"))
from run_encoder_comparison import delta_eff_seq
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

STOP = set("the a an and or but to of in on at for is are was were be been being i you he she it we they "
    "my your his her our their me him them this that these those do does did have has had will would "
    "can could should so um uh yeah okay ok oh ah mm hmm right yes no well just like get got going "
    "know think mean really very much lot here there what who how when where why yep sure alright kay".split())
def info(t):
    w = re.findall(r"[a-zA-Z']+", t.lower()); return sum(1 for x in w if x not in STOP and len(x) > 2)

mid = "ES2002a"; N = 105
d = json.load(open(REPO/"data"/"ami"/"topic"/f"{mid}.json"))
turns, bt, levels = d["turns"], d["bnd_top"], d["topic_levels"]
top_label = {t["start_turn"]: t["label"] for t in levels if t["depth"] == 1}
tx = [t["text"] for t in turns]; lens = [max(1, len(t.split())) for t in tx]
e = np.asarray(pickle.load(open(REPO/"outputs"/"runs"/"_misc"/"ami_emb"/f"{mid}.pkl", "rb")), dtype=np.float64)

# forward merge: filler(info0) → 다음 실질 발화 그룹
G = []; buf = []
for i in range(len(tx)):
    if info(tx[i]) == 0: buf.append(i)
    else: G.append(buf + [i]); buf = []
if buf: G.append(buf)
emb = np.array([np.average(e[g], axis=0, weights=[lens[k] for k in g]) for g in G])
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
predg = set(k for k, b in enumerate(adaptive_boundaries(delta_eff_seq(emb), c=1.5, mode="ewma")) if b)
turn2grp = {}; grp_start = {}
for gi, g in enumerate(G):
    grp_start[gi] = g[0]
    for k in g: turn2grp[k] = gi
v4pred_turns = set(G[gi][0] for gi in predg)        # 예측 경계 = 그룹 시작 turn

golds = [i for i, b in enumerate(bt) if b == 1 and i < N]
L = [f"# AMI {mid} — 대화 전문 + 정답경계(★) + **forward-merge** 분절 (turn 0–{N-1})", "",
     "filler(내용어 0개 발화)를 **다음 실질 발화에 합침**(forward). `G{n}`=합쳐진 그룹 번호 "
     "(같은 G번호 = 한 단위로 묶임). `▲`=예측 경계(그룹 시작). `★`=정답 경계. `(filler→)`=뒤로 흡수된 filler.",
     "", "정답 화제(top-level):"]
for st in sorted(top_label):
    if st < N: L.append(f"- turn {st}{' ★' if st in golds else ''}: **{top_label[st]}**")
L += ["", "---", ""]
for i in range(min(N, len(turns))):
    if bt[i] == 1:
        L += ["", f"### ━━━━━ ★ 정답경계 (turn {i}) → **{top_label.get(i,'?')}** ━━━━━", ""]
    gi = turn2grp.get(i, -1)
    isfill = info(tx[i]) == 0
    tag = "(filler→) " if isfill else ""
    gm = " ★" if bt[i] == 1 else ""
    pm = " ▲" if i in v4pred_turns else ""
    L.append(f"**[{i}] ({turns[i]['speaker']}) `G{gi}`{gm}{pm}** {tag}{turns[i]['text']}")
    L.append("")

# 요약
hit1 = sum(1 for g in golds if any(abs(g-p) <= 1 for p in v4pred_turns))
L += ["---", "",
      f"**forward-merge 요약(이 구간)**: 그룹 {len([gi for gi in range(len(G)) if G[gi][0] < N])}개, "
      f"예측경계 {len([p for p in v4pred_turns if p < N])}, 정답 {len(golds)}, 정답 ±1 적중 {hit1}/{len(golds)}.",
      "",
      "**읽는 법**: filler 발화(예 \"Yes.\")가 *뒤* 내용 발화와 같은 `G번호`로 묶임 → topic-opener filler가 "
      "새 그룹(=새 토픽) 시작으로 보존됨. v4(graph)의 `[Tn]` revisit 과 비교."]
out = REPO/"outputs"/"reports"/"ami_forward_merge_view.md"
out.write_text("\n".join(L)+"\n")
print(f"WROTE {out} | 그룹 {len(G)}, 예측경계 {len(predg)}, gold {sum(bt)}, ±1hit(구간) {hit1}/{len(golds)}")
