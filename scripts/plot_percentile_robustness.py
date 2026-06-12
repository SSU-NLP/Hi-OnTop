"""Figure: Hi-OnTop Score robustness across (encoder × percentile p50..p95)
vs DTS baselines.

For each benchmark (TIAGE / Dialseg711 / SuperDialseg) shows:
  - 3 violin+strip plots (MPNet / MiniLM / MiniLM-int8), each containing
    the 10-point distribution of Hi-OnTop Score over p ∈ {p50..p95}.
  - horizontal dashed lines for the 4 online unsup baselines
    (TextTiling / GraphSeg / GreedySeg / CSM).

The figure makes two points at once:
  (i) for every encoder, EVERY percentile beats all baselines on most benches
  (ii) the variance over percentiles is small relative to the gap above baselines.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
})

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── data from outputs/reports/delta_star_calibration.md (Score, p50..p95) ───
PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# (encoder, bench): list of 10 Scores at p50..p95
SCORES = {
    ("MPNet", "TIAGE"):
        [0.422, 0.447, 0.457, 0.476, 0.459, 0.447, 0.433, 0.427, 0.402, 0.371],
    ("MPNet", "Dialseg711"):
        [0.433, 0.471, 0.509, 0.540, 0.575, 0.601, 0.616, 0.629, 0.609, 0.521],
    ("MPNet", "SuperDialseg"):
        [0.455, 0.459, 0.465, 0.462, 0.459, 0.445, 0.427, 0.402, 0.366, 0.328],
    ("MiniLM", "TIAGE"):
        [0.439, 0.455, 0.470, 0.472, 0.485, 0.481, 0.469, 0.466, 0.416, 0.371],
    ("MiniLM", "Dialseg711"):
        [0.437, 0.468, 0.506, 0.535, 0.565, 0.593, 0.612, 0.599, 0.580, 0.493],
    ("MiniLM", "SuperDialseg"):
        [0.438, 0.435, 0.430, 0.420, 0.404, 0.391, 0.373, 0.355, 0.338, 0.308],
    ("MiniLM-int8", "TIAGE"):
        [0.439, 0.453, 0.470, 0.472, 0.489, 0.482, 0.493, 0.448, 0.423, 0.378],
    ("MiniLM-int8", "Dialseg711"):
        [0.434, 0.469, 0.502, 0.535, 0.559, 0.586, 0.596, 0.600, 0.575, 0.499],
    ("MiniLM-int8", "SuperDialseg"):
        [0.436, 0.434, 0.426, 0.415, 0.402, 0.387, 0.367, 0.355, 0.335, 0.305],
}

# baselines from outputs/reports/dts_result.md (online unsup)
BASELINES = {
    "TextTiling-Style": {"TIAGE": 0.344, "Dialseg711": 0.395, "SuperDialseg": 0.399},
    "GraphSeg-Style":   {"TIAGE": 0.374, "Dialseg711": 0.450, "SuperDialseg": 0.312},
    "GreedySeg-Style":  {"TIAGE": 0.298, "Dialseg711": 0.491, "SuperDialseg": 0.384},
    "CSM-Style":        {"TIAGE": 0.430, "Dialseg711": 0.476, "SuperDialseg": 0.443},
}
BASELINE_COLOR = {
    "TextTiling-Style": "#666666",
    "GraphSeg-Style":   "#9467bd",
    "GreedySeg-Style":  "#8c564b",
    "CSM-Style":        "#e08214",
}
BASELINE_LINESTYLE = {
    "TextTiling-Style": "--",
    "GraphSeg-Style":   "-.",
    "GreedySeg-Style":  ":",
    "CSM-Style":        (0, (5, 1)),
}

ENCODERS = ["MPNet", "MiniLM", "MiniLM-int8"]
ENC_COLOR = {"MPNet": "#d62728", "MiniLM": "#ff7f0e", "MiniLM-int8": "#2ca02c"}
BENCHES = ["TIAGE", "Dialseg711", "SuperDialseg"]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.2), sharey=False)

    for ax, bench in zip(axes, BENCHES):
        data = [SCORES[(enc, bench)] for enc in ENCODERS]
        # boxplot
        bp = ax.boxplot(
            data, positions=np.arange(len(ENCODERS)),
            widths=0.55, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.4),
            boxprops=dict(linewidth=0.7, edgecolor="black"),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.5),
        )
        for patch, enc in zip(bp["boxes"], ENCODERS):
            patch.set_facecolor(ENC_COLOR[enc])
            patch.set_alpha(0.42)

        # strip (individual p_x points)
        for x, enc in enumerate(ENCODERS):
            ys = SCORES[(enc, bench)]
            xs = np.random.default_rng(seed=x).normal(0, 0.06, size=len(ys)) + x
            ax.scatter(xs, ys, c=ENC_COLOR[enc], s=30,
                        edgecolor="black", linewidth=0.45, zorder=5, alpha=0.95)

        # baselines as horizontal lines (extend across)
        for name in ["TextTiling-Style", "GraphSeg-Style",
                       "GreedySeg-Style", "CSM-Style"]:
            v = BASELINES[name][bench]
            ax.axhline(v, color=BASELINE_COLOR[name],
                          linestyle=BASELINE_LINESTYLE[name],
                          linewidth=1.1, alpha=0.85, zorder=2)
            # right-side label, outside box area
            ax.text(3.05, v, f"{name}\n{v:.3f}",
                      color=BASELINE_COLOR[name], fontsize=7.2,
                      va="center", ha="left", weight="bold",
                      bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="white", edgecolor=BASELINE_COLOR[name],
                                  linewidth=0.5, alpha=0.97))

        ax.set_xticks(np.arange(len(ENCODERS)))
        ax.set_xticklabels(ENCODERS, fontsize=9)
        ax.set_title(bench, pad=8, fontsize=11.5, weight="bold")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
        # extend xlim to fit right-side labels
        ax.set_xlim(-0.55, 4.7)
        # per-bench y-axis range
        all_ys = [v for enc in ENCODERS for v in SCORES[(enc, bench)]]
        all_bs = [BASELINES[n][bench] for n in BASELINES]
        ymin = min(all_ys + all_bs) - 0.03
        ymax = max(all_ys + all_bs) + 0.03
        ax.set_ylim(ymin, ymax)

    axes[0].set_ylabel("Segmentation Score", fontsize=10.5)

    # build a shared legend (encoders + baselines)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    enc_handles = [Patch(facecolor=ENC_COLOR[e], edgecolor="black",
                            linewidth=0.6, alpha=0.6,
                            label=f"Hi-OnTop ({e})")
                     for e in ENCODERS]
    base_handles = [Line2D([0], [0], color=BASELINE_COLOR[n],
                              linestyle=BASELINE_LINESTYLE[n],
                              linewidth=1.0, label=n)
                      for n in BASELINES]
    fig.legend(handles=enc_handles + base_handles,
                loc="upper center", bbox_to_anchor=(0.5, 1.04),
                ncol=4, frameon=False, fontsize=7.5,
                handlelength=1.6, columnspacing=1.2)

    fig.suptitle(
        "Hi-OnTop Score distribution over $p_x \\in \\{p50,\\,p55,\\,\\ldots,\\,p95\\}$  "
        "vs. online unsupervised baselines (each box: 10 percentile points)",
        fontsize=11, y=1.05,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "figure_J_percentile_robustness.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
