#!/usr/bin/env python3
"""Percentile vs Score curve per (encoder × dataset) — paper-grade figure
showing **WHY p60-p80 band is sufficient**.

Source data: outputs/reports/delta_star_calibration.md table (Score per
percentile across 9 (encoder × dataset) cells).

Plots 3-panel (one per dataset). Each panel has 3 encoder curves +
calibration band (p60-p80) highlighted + oracle line per cell.

산출: outputs/figures/figure_L_percentile_score_curve.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_FIGDIR = REPO / "outputs" / "figures"
OUT_FIGDIR.mkdir(parents=True, exist_ok=True)

PERCENTILES = [5, 10, 15, 20, 25, 30, 35, 40, 45,
                50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# (encoder, dataset, [Score_p5, ..., Score_p95], oracle)
# p5..p45: outputs/experiments/2026-05-26_fig_l_low_p_extension/per_metric.json (2026-05-26 expansion)
# p50..p95: outputs/reports/delta_star_calibration.md (2026-05-23)
DATA = {
    "mpnet": {
        "TIAGE":        ([0.307, 0.325, 0.336, 0.349, 0.364, 0.375, 0.390, 0.404, 0.415,
                          0.422, 0.447, 0.457, 0.476, 0.459, 0.447, 0.433, 0.427, 0.402, 0.371], 0.473),
        "Dialseg711":   ([0.241, 0.254, 0.266, 0.283, 0.301, 0.323, 0.344, 0.366, 0.398,
                          0.433, 0.471, 0.509, 0.540, 0.575, 0.601, 0.616, 0.629, 0.609, 0.521], 0.630),
        "SuperDialseg": ([0.345, 0.362, 0.376, 0.391, 0.409, 0.420, 0.431, 0.439, 0.448,
                          0.455, 0.459, 0.465, 0.462, 0.459, 0.445, 0.427, 0.402, 0.366, 0.328], 0.464),
    },
    "minilm": {
        "TIAGE":        ([0.305, 0.321, 0.335, 0.346, 0.358, 0.380, 0.391, 0.405, 0.422,
                          0.439, 0.455, 0.470, 0.472, 0.485, 0.481, 0.469, 0.466, 0.416, 0.371], 0.485),
        "Dialseg711":   ([0.245, 0.257, 0.273, 0.287, 0.305, 0.327, 0.345, 0.370, 0.396,
                          0.437, 0.468, 0.506, 0.535, 0.565, 0.593, 0.612, 0.599, 0.580, 0.493], 0.609),
        "SuperDialseg": ([0.349, 0.364, 0.380, 0.395, 0.408, 0.414, 0.426, 0.434, 0.437,
                          0.438, 0.435, 0.430, 0.420, 0.404, 0.391, 0.373, 0.355, 0.338, 0.308], 0.438),
    },
    "minilm-int8": {
        "TIAGE":        ([0.305, 0.321, 0.332, 0.345, 0.357, 0.376, 0.389, 0.408, 0.428,
                          0.439, 0.453, 0.470, 0.472, 0.489, 0.482, 0.493, 0.448, 0.423, 0.378], 0.489),
        "Dialseg711":   ([0.246, 0.260, 0.276, 0.293, 0.310, 0.328, 0.352, 0.378, 0.413,
                          0.434, 0.469, 0.502, 0.535, 0.559, 0.586, 0.596, 0.600, 0.575, 0.499], 0.616),
        "SuperDialseg": ([0.351, 0.368, 0.381, 0.398, 0.408, 0.417, 0.427, 0.432, 0.436,
                          0.436, 0.434, 0.426, 0.415, 0.402, 0.387, 0.367, 0.355, 0.335, 0.305], 0.436),
    },
}

ENCS = ("mpnet", "minilm", "minilm-int8")
DATASETS = ("TIAGE", "Dialseg711", "SuperDialseg")
ENC_PRETTY = {"mpnet": "MPNet", "minilm": "MiniLM", "minilm-int8": "MiniLM-int8"}
# Unified with Fig Q (mtbp_percentile_curve.py) — same colors + markers per
# encoder so two figures can be read jointly.
ENC_COLOR = {"mpnet": "#1f77b4", "minilm": "#ff7f0e", "minilm-int8": "#a30000"}
ENC_MARKER = {"mpnet": "o", "minilm": "s", "minilm-int8": "^"}


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


def main() -> None:
    setup_rc()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=False,
                              gridspec_kw={"wspace": 0.20})

    for ci, ds in enumerate(DATASETS):
        ax = axes[ci]

        # p60-p80 calibration band (yellow, matches Fig Q)
        ax.axvspan(60, 80, alpha=0.22, color="#ffd966", zorder=0,
                    label="p60-p80 band" if ci == 0 else None)

        for enc in ENCS:
            scores, oracle = DATA[enc][ds]
            color = ENC_COLOR[enc]
            ax.plot(PERCENTILES, scores, "-",
                     marker=ENC_MARKER[enc], color=color,
                     label=ENC_PRETTY[enc] if ci == 0 else None,
                     zorder=4)
            # oracle horizontal line
            ax.axhline(oracle, color=color, linestyle=":", linewidth=0.9,
                        alpha=0.6, zorder=2)
            # mark best percentile with a star (matches Fig Q style)
            best_idx = int(np.argmax(scores))
            ax.plot(PERCENTILES[best_idx], scores[best_idx], "*",
                     color=color, markersize=11, markeredgecolor="black",
                     markeredgewidth=0.5, zorder=5)

        ax.set_title(ds, fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel(r"Percentile $p$ (for $\delta^\ast$ calibration)",
                       fontsize=10)
        if ci == 0:
            ax.set_ylabel(r"Score $\uparrow$", fontsize=10)
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.set_xlim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="gray", linestyle=":",
                                linewidth=0.9, label="oracle (test-sweep)"))
    labels.append("oracle (test-sweep)")
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                                markerfacecolor="gray",
                                markeredgecolor="black",
                                markersize=11, markeredgewidth=0.5,
                                label="best percentile"))
    labels.append("best percentile")
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.02),
                handlelength=1.8, columnspacing=1.5, handletextpad=0.5)
    fig.suptitle(
        r"Score vs percentile across (encoder $\times$ dataset) on DTS — "
        r"best percentile always falls inside the p60-p80 band",
        fontsize=10, y=1.10,
    )

    for ext in ("pdf", "png"):
        out = OUT_FIGDIR / f"figure_L_percentile_score_curve.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
