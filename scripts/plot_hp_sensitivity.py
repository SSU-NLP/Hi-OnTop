"""1D marginal sensitivity figure for (m, rho, a) at oracle delta*.

Reads: outputs/experiments/2026-05-25_hiontop_mra_sweep/sweep.json
Outputs: outputs/figures/figure_I_hp_sensitivity.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Unified style with Fig L (percentile_score_curve) and Fig Q (mtbp_percentile_curve).
mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9.5,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.edgecolor": "#666", "axes.linewidth": 0.7,
    "axes.grid": True, "grid.alpha": 0.35,
    "grid.linestyle": ":", "grid.color": "#bbb", "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04, "pdf.fonttype": 42, "ps.fonttype": 42,
})

REPO = Path(__file__).resolve().parent.parent
SWEEP_JSON = REPO / "outputs" / "experiments" / "2026-05-25_hiontop_mra_sweep" / "sweep.json"
FIG_DIR = REPO / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT = dict(m=2, rho=0.7, a=0.5)


def main() -> None:
    data = json.loads(SWEEP_JSON.read_text())
    grid = data["grid"]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True,
                              gridspec_kw={"wspace": 0.10})
    # Grid slightly stronger than default — user calibration.
    for ax in axes:
        ax.grid(True, alpha=0.42, linestyle=":", color="#aaa", linewidth=0.55)
        ax.set_axisbelow(True)

    metrics_orac = "oracle_mean3"
    metrics_px   = "bestpx_mean3"
    color_orac = "#d62728"  # red (original)
    color_px   = "#1f77b4"  # blue (original)

    hps = [("m", r"context window $m$", [2, 3, 4, 5, 6, 8]),
            ("rho", r"context decay $\rho$", [0.5, 0.7, 0.9]),
            ("a", r"blend weight $a$", [0.0, 0.3, 0.5, 0.7, 1.0])]

    for ax, (hp, xlabel, values) in zip(axes, hps):
        max_orac, max_px, mean_orac, mean_px = [], [], [], []
        for v in values:
            sub = [r for r in grid if r[hp] == v]
            max_orac.append(max(r[metrics_orac] for r in sub))
            max_px.append(max(r[metrics_px] for r in sub))
            mean_orac.append(np.mean([r[metrics_orac] for r in sub]))
            mean_px.append(np.mean([r[metrics_px] for r in sub]))

        ax.plot(values, max_orac, "o-", color=color_orac, lw=1.3,
                  markersize=6, label="Oracle (max)", zorder=4)
        ax.plot(values, mean_orac, "o--", color=color_orac, lw=0.8,
                  markersize=4, alpha=0.55, label="Oracle (mean)", zorder=3)
        ax.plot(values, max_px, "s-", color=color_px, lw=1.3,
                  markersize=5.5, label=r"Best-$p_x$ (max)", zorder=4)
        ax.plot(values, mean_px, "s--", color=color_px, lw=0.8,
                  markersize=4, alpha=0.55, label=r"Best-$p_x$ (mean)", zorder=3)

        # default value: dotted vertical line (user request — no yellow band)
        default_v = DEFAULT[hp]
        if default_v in values:
            ax.axvline(default_v, color="black", linestyle=":",
                          linewidth=0.9, alpha=0.55, zorder=1)

        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", labelpad=6)
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])

    axes[0].set_ylabel("Mean Score across 3 DTS benches", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.02),
                handlelength=2.0, columnspacing=1.4, handletextpad=0.5)
    fig.suptitle(
        r"Hyperparameter sensitivity at per-bench best $\delta^\ast$ "
        r"(default $m{=}2,\rho{=}0.7,a{=}0.5$ marked with dotted line)",
        fontsize=10, y=1.10,
    )

    out = FIG_DIR / "figure_I_hp_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
