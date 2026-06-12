#!/usr/bin/env python3
"""Figure S — Score vs latency (AMI 139). δ_eff/threshold/commit-refine + (LLM 버퍼곡선, 인가 시).

figure_R(±2F1)의 Score 짝. ±2F1 은 과분절을 보상하지만 Score(Pk/WD 포함)는 분절 품질을 보므로 그림이
뒤집힌다: commit-refine 이 전 지연구간에서 δ_eff(0-지연)를 Score 로 지배. commit-refine 의 우위는 lag 매입분.
LLM 버퍼 Score 는 Crts(Cloudflare Access) 인가 후 `ami_llm_buffer_eval.py` 로 채워 LLM_SCORE 에 넣으면 자동 표시.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)

# 측정값 (AMI 139, best-c by Score) — 2026-06-12 deploy_ceiling_anatomy
THRESHOLD = (0.0, 0.372)                      # 0-lag
DELTA_EFF = (0.0, 0.203)                      # 0-lag baseline (5.5× 과분절)
COMMIT_REFINE = [(6, 0.364), (9, 0.371), (12, 0.379), (19, 0.389),
                 (26, 0.401), (54, 0.393), (111, 0.392)]   # L=2,3,4,6,8,16,32 → 지연(s)
# LLM 버퍼 Score (Crts 403 으로 미산출) — 인가 후 (버퍼초, Score) 채우면 자동 표시. 예: [(10,?),(30,?),...,(1e3,?)]
LLM_SCORE: list[tuple[float, float]] = []

fig, ax = plt.subplots(figsize=(6.4, 4.2))
xs = [x for x, _ in COMMIT_REFINE]; ys = [y for _, y in COMMIT_REFINE]
ax.plot(xs, ys, "o-", color="#1f77b4", lw=2, label="Hi-OnTop commit-refine (de-neut)")
ax.scatter([THRESHOLD[0]], [THRESHOLD[1]], color="#1f77b4", marker="*", s=180, zorder=5,
           label="Hi-OnTop threshold (0-lag)")
ax.scatter([DELTA_EFF[0]], [DELTA_EFF[1]], color="#7f7f7f", marker="s", s=70, zorder=5,
           label="δ_eff (0-lag, 5.5× over-seg)")
if LLM_SCORE:
    lx = [x for x, _ in LLM_SCORE]; ly = [y for _, y in LLM_SCORE]
    ax.plot(lx, ly, "^--", color="#d62728", lw=2, label="LLM polling (buffer)")
else:
    ax.text(0.97, 0.06, "LLM Score: pending (Crts 403)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#d62728", style="italic")

ax.set_xlabel("Latency / buffer (seconds)")
ax.set_ylabel("Score  (0.5·F1±2 + 0.25·(1−Pk) + 0.25·(1−WD))")
ax.set_title("AMI topic segmentation — Score vs latency")
ax.set_xlim(-4, 116); ax.set_ylim(0.15, 0.45)
ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
ax.annotate("commit-refine gain = latency purchase\n(L<=3 ~ threshold)", xy=(26, 0.401),
            xytext=(40, 0.36), fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure_S_score_latency.{ext}", dpi=150)
print(f"saved {OUT}/figure_S_score_latency.{{pdf,png}}  (LLM points: {len(LLM_SCORE)})")
