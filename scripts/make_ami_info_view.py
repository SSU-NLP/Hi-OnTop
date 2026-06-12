#!/usr/bin/env python3
"""AMI 대화 전문 + 각 발화의 임베딩-정보량(info_emb) 값 + tier → markdown."""
import json, pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
mid = "ES2002a"; N = 105
d = json.load(open(REPO/"data"/"ami"/"topic"/f"{mid}.json"))
turns, bt, levels = d["turns"], d["bnd_top"], d["topic_levels"]
top_label = {t["start_turn"]: t["label"] for t in levels if t["depth"] == 1}
e = np.asarray(pickle.load(open(REPO/"outputs"/"runs"/"_misc"/"ami_emb"/f"{mid}.pkl", "rb")), dtype=np.float64)
mean = e.mean(0); mean = mean/(np.linalg.norm(mean)+1e-9)
ie = 1 - e @ mean                       # info_emb (전체 미팅 기준)
p30, p60 = np.percentile(ie, 30), np.percentile(ie, 60)
def tier(v):
    if v < p30: return "🔵FILLER"
    if v < p60: return "⚪약함"
    return "🟢내용"

golds = [i for i, b in enumerate(bt) if b == 1 and i < N]
L = [f"# AMI {mid} — 발화별 임베딩-정보량(info_emb) + 전문 (turn 0–{N-1})", "",
     "**info_emb = 1 − cos(발화, 미팅 전체 평균 임베딩)** — 평균에 가까우면 낮음(generic=filler), 멀면 높음(독특=내용). 사전 0개.",
     f"이 미팅 임계: **p30={p30:.3f}** (이하=🔵FILLER, forward-merge 대상) · **p60={p60:.3f}** (이상=🟢내용, 경계탐지 대상).",
     f"info_emb 범위: min {ie.min():.3f} / median {np.median(ie):.3f} / max {ie.max():.3f}. `★`=정답경계.", "",
     "| # | (화자) | **info_emb** | tier | 발화 |",
     "|--:|:--|--:|:--|------|"]
for i in range(min(N, len(turns))):
    gm = " ★" if bt[i] == 1 else ""
    L.append(f"| {i}{gm} | ({turns[i]['speaker']}) | **{ie[i]:.3f}** | {tier(ie[i])} | {turns[i]['text']} |")

# 경계 발화들의 info_emb (★가 filler tier에 걸리는지)
L += ["", "## 정답 경계 발화의 info_emb (★가 어느 tier?)", ""]
for g in golds:
    L.append(f"- turn {g} **{ie[g]:.3f}** {tier(ie[g])} — \"{turns[g]['text'][:50]}\" → {top_label.get(g,'?')}")
L += ["",
      "→ 경계가 🔵FILLER tier(낮은 info)에 걸리면 = forward-merge로 *뒤* 내용에 흡수되어 그 그룹 시작으로 보존. "
      "🟢내용 tier면 그 자체가 경계 후보."]
out = REPO/"outputs"/"reports"/"ami_info_content_view.md"
out.write_text("\n".join(L)+"\n")
print(f"WROTE {out} | p30={p30:.3f} p60={p60:.3f} | 경계 info: "+", ".join(f'{g}:{ie[g]:.2f}' for g in golds))
