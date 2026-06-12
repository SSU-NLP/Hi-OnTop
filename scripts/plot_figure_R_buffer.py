#!/usr/bin/env python3
"""figure_R — 버퍼-지연 곡선. LLM은 큰 버퍼 필요, Hi-OnTop은 0-지연. (AMI 12미팅, ±2 boundary-F1)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
# 12미팅 ±2 F1 (실측)
llm_buf = {10: 0.256, 30: 0.237, 60: 0.254, 120: 0.372}   # LLM 버퍼-폴링
llm_full = 0.526                                            # LLM 전체 transcript (offline)
hiontop = 0.227                                            # Hi-OnTop raw ewma (0 버퍼)
texttiling = 0.264                                         # TextTiling streaming (0 버퍼)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
xs = sorted(llm_buf)
ax.plot(xs, [llm_buf[x] for x in xs], "o-", color="#d62728", lw=2, ms=8,
        label="LLM polling (Qwen3.5-27B), buffer-limited")
# full-context LLM (offline, infinite buffer)
ax.axhline(llm_full, ls="--", color="#d62728", alpha=0.7, lw=1.6)
ax.annotate(f"LLM full-context (offline, ∞ buffer) = {llm_full:.2f}",
            (122, llm_full), (60, llm_full+0.018), color="#d62728", fontsize=9)
# 0-buffer streaming methods
ax.scatter([0], [hiontop], color="#1f77b4", s=130, zorder=6, marker="*",
           label=f"Hi-OnTop (0-buffer, streaming) = {hiontop:.2f}")
ax.scatter([0], [texttiling], color="#2ca02c", s=90, zorder=6, marker="s",
           label=f"TextTiling (0-buffer, streaming) = {texttiling:.2f}")
ax.annotate("0-latency\n(per-utterance)", (0, hiontop), (8, 0.10),
            fontsize=9, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.6))

ax.set_xlabel("LLM polling buffer (seconds of context before each segmentation call)")
ax.set_ylabel("Topic boundary F1 (±2-turn tolerance)")
ax.set_title("LLM segmentation needs a large context buffer; Hi-OnTop runs at 0-latency\n"
             "AMI meetings (n=12). LLM gains only emerge with near-full context; "
             "at streaming buffers it ties cheap 0-latency methods.", fontsize=10)
ax.set_xlim(-8, 135); ax.set_ylim(0, 0.60)
ax.grid(alpha=0.3); ax.legend(loc="center right", fontsize=9)
plt.tight_layout()
for ext in ("pdf", "png"):
    out = REPO/"outputs"/"figures"/f"figure_R_buffer_latency.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
print("WROTE outputs/figures/figure_R_buffer_latency.{pdf,png}")
