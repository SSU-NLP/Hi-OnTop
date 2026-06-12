#!/usr/bin/env python3
"""AMI 대화 전문 + 정답경계 + *최고 embedding 방법* 분절 → markdown.
최고 방법 = geometry-merge(cos-isolation 탐지 + 재인코딩) + bottom10% low-info 제외 + ewma 적응임계치.
AMI ±2 F1 ≈ 0.195 (embedding 천장). 비교: LLM ±2 0.526 / exact는 누구도 불가."""
import json, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"src"))
from run_encoder_comparison import delta_eff_seq
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
    model_kwargs={"provider":"CPUExecutionProvider","file_name":"onnx/model_quint8_avx2.onnx"})
embed = lambda ts: np.asarray(enc.encode(ts, normalize_embeddings=True), dtype=np.float64)
FW = ["yeah","okay","mm-hmm","right","yes","uh-huh","mm","hmm","sure","alright","yep","oh","ah","mhm","uh","um"]
ref = embed(FW).mean(0); ref /= np.linalg.norm(ref)+1e-9

mid = "ES2002a"; N = 105
d = json.load(open(REPO/"data"/"ami"/"topic"/f"{mid}.json"))
turns, bt, levels = d["turns"], d["bnd_top"], d["topic_levels"]
top_label = {t["start_turn"]: t["label"] for t in levels if t["depth"] == 1}
e = np.asarray(pickle.load(open(REPO/"outputs"/"runs"/"_misc"/"ami_emb"/f"{mid}.pkl","rb")), dtype=np.float64)

# --- 최고 방법 ---
def gflag(E):
    ee = [v/(np.linalg.norm(v)+1e-12) for v in E]; fl = [False]*len(ee)
    for i in range(1, len(ee)-1):
        sp, sn, sb = float(ee[i-1]@ee[i]), float(ee[i]@ee[i+1]), float(ee[i-1]@ee[i+1])
        if sb > sp and sb > sn: fl[i] = True
    return fl
fl = gflag(e); G = []
for i in range(len(turns)):
    if fl[i] and G: G[-1].append(i)
    else: G.append([i])
gemb = embed([" ".join(turns[k]["text"] for k in g) for g in G])      # 재인코딩
ginfo = 1 - gemb@ref
thr = np.percentile(ginfo, 10); sel = [k for k in range(len(G)) if ginfo[k] >= thr]   # bottom10% 제외
sub = gemb[sel]; sub = sub/(np.linalg.norm(sub, axis=1, keepdims=True)+1e-9)
sp = set(k for k, b in enumerate(adaptive_boundaries(list(delta_eff_seq(sub)), c=1.5, mode="ewma")) if b)
pred_turns = set(G[sel[k]][0] for k in sp if k < len(sel))
turn2grp = {}
for gi, g in enumerate(G):
    for k in g: turn2grp[k] = gi
gold_excluded = set(k for k in range(len(G)) if k not in set(sel))

golds = [i for i, b in enumerate(bt) if b == 1 and i < N]
L = [f"# AMI {mid} — 대화 전문 + 정답경계(★) + **최고 embedding 방법** 분절 (turn 0–{N-1})", "",
     "**최고 방법** = geometry-merge(cos-isolation 탐지 + 합친 텍스트 *재인코딩*) + bottom10% low-info 그룹 제외 "
     "+ ewma 적응임계치. **AMI ±2 F1 ≈ 0.195 (embedding 천장).** `★`=정답, `▲`=예측경계, "
     "`G{n}`=merge 그룹, `[제외]`=low-info로 거리계산서 빠진 그룹.",
     "", "비교 (12미팅): **embedding ±2 0.195 / LLM ±2 0.526 / exact F1 은 둘 다 ~0.03(누구도 불가).**",
     "", "정답 화제(top-level):"]
for st in sorted(top_label):
    if st < N: L.append(f"- turn {st}{' ★' if st in golds else ''}: **{top_label[st]}**")
L += ["", "---", ""]
for i in range(min(N, len(turns))):
    if bt[i] == 1:
        L += ["", f"### ━━━━━ ★ 정답경계 (turn {i}) → **{top_label.get(i,'?')}** ━━━━━", ""]
    gi = turn2grp.get(i, -1)
    gm = " ★" if bt[i] == 1 else ""
    pm = " ▲" if i in pred_turns else ""
    ex = " `[제외]`" if gi in gold_excluded else ""
    L.append(f"**[{i}] ({turns[i]['speaker']}) `G{gi}`{ex}{gm}{pm}** {turns[i]['text']}")
    L.append("")

# 요약
hit2 = sum(1 for g in golds if any(abs(g-p) <= 2 for p in pred_turns))
L += ["---", "",
      f"**최고 방법 요약(이 구간)**: 그룹 {len([gi for gi in range(len(G)) if G[gi][0] < N])}개, "
      f"예측경계 {len([p for p in pred_turns if p < N])}, 정답 {len(golds)}, 정답 ±2 적중 {hit2}/{len(golds)}.",
      "",
      "**읽는 법**: filler 발화는 cos-isolation으로 잡혀 앞 그룹에 흡수(같은 G번호). low-info 그룹은 `[제외]`되어 "
      "거리계산서 빠짐. ▲(예측)가 ★(정답)과 정확히 안 맞고 ±1~2 어긋나는 게 보일 것 — 정답이 응답/filler에 찍혀서. "
      "이게 embedding 천장(±2 0.195)의 모습이고, LLM(±2 0.53)은 이 어긋남을 담화이해로 줄임."]
out = REPO/"outputs"/"reports"/"ami_conversation_boundaries.md"
out.write_text("\n".join(L)+"\n")
print(f"WROTE {out} | 그룹 {len(G)}, 예측 {len(pred_turns)}, gold {sum(bt)}, ±2hit(구간) {hit2}/{len(golds)}")
