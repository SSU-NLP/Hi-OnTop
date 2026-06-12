"""Figure Q: MTB+ boundary F1 vs percentile curve (Fig L counterpart for MTB+).

Mirrors ``figure_L_percentile_score_curve.{pdf,png}`` (DTS Score vs percentile)
but uses MTB+ as evaluation domain and LLM segmenter boundaries as
pseudo-reference. y-axis = pairwise boundary F1.

Layout: 3 panels (one per LLM ref) × 3 lines (one per encoder).
x ∈ [5, 95], yellow band = [60, 80], stars mark best_p per (encoder × ref).

Data sources:
- δ_eff cache: ``outputs/experiments/2026-05-25_llm_distillation_calib/results.json``
  (full-pool sweep, n=666 δ_eff).

Output: ``outputs/figures/figure_Q_mtbp_percentile_curve.{pdf,png}``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "outputs/experiments/2026-05-25_llm_distillation_calib/results.json"
OUT_FIG = REPO / "outputs/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

ENCODERS = ("MPNet", "MiniLM", "MiniLM-int8")
LLM_REFS = ("GPT-5", "Qwen3.5-27B", "Qwen3.5-122B-A10B")
ENC_COLOR = {"MPNet": "#1f77b4", "MiniLM": "#ff7f0e", "MiniLM-int8": "#a30000"}
ENC_MARKER = {"MPNet": "o", "MiniLM": "s", "MiniLM-int8": "^"}


def setup_rc():
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9.5, "axes.spines.top": True,
        "axes.spines.right": True, "axes.spines.left": True,
        "axes.spines.bottom": True, "axes.edgecolor": "#666",
        "axes.linewidth": 0.7, "axes.grid": True,
        "grid.alpha": 0.35, "grid.linestyle": ":",
        "grid.color": "#bbb", "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def main():
    r = json.loads(RESULTS.read_text())
    setup_rc()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True,
                               gridspec_kw={"wspace": 0.10})

    for ai, ref in enumerate(LLM_REFS):
        ax = axes[ai]
        # yellow band p60-p80
        ax.axvspan(60, 80, alpha=0.22, color="#ffd966", zorder=0,
                    label="p60-p80 band" if ai == 0 else None)
        for enc in ENCODERS:
            sweep = r[enc]["sweep"]
            ps = np.array([s["p"] for s in sweep])
            f1s = np.array([s[f"f1_vs_{ref}"] for s in sweep])
            ax.plot(ps, f1s, "-", marker=ENC_MARKER[enc],
                     color=ENC_COLOR[enc], label=enc, zorder=3)
            # star at best p
            best_idx = int(np.argmax(f1s))
            ax.plot(ps[best_idx], f1s[best_idx], "*", color=ENC_COLOR[enc],
                     markersize=11, markeredgecolor="black",
                     markeredgewidth=0.5, zorder=5)
        ax.set_title(ref, fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel(r"Percentile $p$ (for $\delta^\ast$ calibration)",
                       fontsize=10)
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.set_xlim(0, 100)
        if ai == 0:
            ax.set_ylabel(r"Pairwise boundary $F_1$ vs LLM",
                           fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    # add star marker for best legend
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                               markerfacecolor="gray", markeredgecolor="black",
                               markersize=11, markeredgewidth=0.5,
                               label="best percentile"))
    labels.append("best percentile")
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.02),
                handlelength=1.8, columnspacing=1.5, handletextpad=0.5)
    fig.suptitle(r"$F_1$ vs percentile across (encoder $\times$ LLM ref) "
                  r"on Long-MT-Bench+ — best percentile inside p60-p80 band",
                  fontsize=10, y=1.10)
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"figure_Q_mtbp_percentile_curve.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
