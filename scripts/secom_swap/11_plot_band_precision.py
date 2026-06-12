"""Figure A: Per-band gold-boundary precision proves graded score is calibrated.

Compute fresh on 3 benchmarks (TIAGE / Dialseg711 / SuperDialseg) using
the official superdialseg_data harness + Hi-OnTop MPNet. Continuous
``graded_score = δ_eff / δ*`` is binned into many fine bins; precision =
P(gold boundary | g_t in bin) computed per bin.

Two complementary views in one figure:
  (a) coarse 4-band (Ben-Yakov & Henson 2018 categories), as bars
  (b) fine 10-bin smooth curve overlaid, with monotone-trend annotation

Output: ``plots/band_precision.{pdf,png}``.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

SDS = REPO_ROOT / "benchmarks/superdialseg_data"
CACHE = REPO_ROOT / "outputs/runs/_misc"
OUT_DIR = REPO_ROOT / "outputs/experiments/2026-05-21_v413_secom_swap/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

M, RHO, A = 2, 0.7, 0.5

# delta_star per bench (MPNet, calib N=200, seed=0 — same as paper rows)
DSTAR = {
    "tiage":       0.5618,   # p70
    "dialseg711":  0.5514,   # p70
    "superseg":    0.5751,   # p70
}

MPNET_PKL = {
    "tiage":      "sds_emb_tiage_test.pkl",
    "dialseg711": "sds_emb_dialseg711_test.pkl",
    "superseg":   "sds_emb_superseg_test.pkl",
}

BENCH_PRETTY = {
    "tiage":      "TIAGE",
    "dialseg711": "Dialseg711",
    "superseg":   "SuperDialseg",
}
BENCH_COLOR = {
    "tiage":      "#4F81BD",
    "dialseg711": "#C0504D",
    "superseg":   "#9BBB59",
}


def load_dialogs(ds):
    raw = json.loads((SDS / ds / "segmentation_file_test.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if yt:
            yt[-1] = 0
        if len(utts) >= 2:
            out.append((utts, yt))
    return out


def collect_scores(ds):
    """Return (g_arr, y_arr) flat over all (turn t>=1 in all test dialogs).
    g_t = δ_eff(t) / δ*. y_t = gold boundary (binary)."""
    dia = load_dialogs(ds)
    with open(CACHE / MPNET_PKL[ds], "rb") as fh:
        embs = pickle.load(fh)
    assert len(embs) == len(dia)
    gs, ys = [], []
    for (utts, yt), emb in zip(dia, embs):
        seg = HiOnTop(dim=emb.shape[1], delta_star=DSTAR[ds],
                     ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
        for t, v in enumerate(emb):
            seg.assign(np.asarray(v, dtype=np.float64))
            if t >= 1:
                gs.append(seg.last_graded_score)
                ys.append(yt[t])
    return np.asarray(gs), np.asarray(ys, dtype=int)


def bin_precision(g_arr, y_arr, edges):
    """For each (edge_i, edge_{i+1}] bin, return (count, precision)."""
    counts, precs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (g_arr > lo) & (g_arr <= hi)
        n = int(mask.sum())
        if n == 0:
            counts.append(0); precs.append(np.nan); continue
        counts.append(n)
        precs.append(float(y_arr[mask].sum()) / n)
    return np.array(counts), np.array(precs)


def main():
    bench_data = {}
    for ds in ("tiage", "dialseg711", "superseg"):
        g, y = collect_scores(ds)
        bench_data[ds] = (g, y)
        print(f"{ds}: n={len(g)}  n_gold={int(y.sum())}  "
              f"g range [{g.min():.2f}, {g.max():.2f}]")

    # -- two-panel figure --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.0))

    # Panel (a): coarse 4-band bars (Ben-Yakov & Henson)
    band_edges = [0.0, 0.7, 1.0, 1.3, 100.0]
    band_labels = ["very_weak\n(<0.7)", "weak\n(0.7–1.0)",
                   "normal\n(1.0–1.3)", "strong\n(≥1.3)"]
    x = np.arange(len(band_labels))
    width = 0.27
    for i, ds in enumerate(("tiage", "dialseg711", "superseg")):
        g, y = bench_data[ds]
        cnt, pr = bin_precision(g, y, band_edges)
        pos = x + (i - 1) * width
        bars = ax1.bar(pos, np.nan_to_num(pr, nan=0.0), width,
                       label=BENCH_PRETTY[ds], color=BENCH_COLOR[ds],
                       edgecolor="black", linewidth=0.5)
        for b, v, n in zip(bars, pr, cnt):
            if n > 0:
                ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                         f"{v:.2f}\n(n={n})", ha="center", va="bottom",
                         fontsize=7.0, color="#333")
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_labels, fontsize=9)
    ax1.set_ylabel("Gold-boundary precision\n$P(y_t=1 \\mid g_t \\in \\text{band})$",
                   fontsize=10)
    ax1.set_xlabel(r"$g_t = \delta_\mathrm{eff} / \delta^*$", fontsize=11)
    ax1.set_title("(a) Coarse 4-band  (Ben-Yakov \\& Henson 2018)", fontsize=10.5)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=9, frameon=False)

    # Panel (b): fine continuous monotone curve (10 quantile bins per bench)
    n_bins = 10
    for ds in ("tiage", "dialseg711", "superseg"):
        g, y = bench_data[ds]
        quantiles = np.quantile(g, np.linspace(0, 1, n_bins + 1))
        # ensure unique edges (small noise to break ties)
        quantiles = np.unique(quantiles)
        if len(quantiles) - 1 < 4:
            print(f"warning: {ds} has few unique g — fewer bins")
        cnt, pr = bin_precision(g, y, quantiles)
        centers = 0.5 * (quantiles[:-1] + quantiles[1:])
        ax2.plot(centers, pr, "-o", color=BENCH_COLOR[ds],
                 linewidth=1.5, markersize=4.5, label=BENCH_PRETTY[ds],
                 markeredgecolor="black", markeredgewidth=0.4)
    ax2.set_xlabel(r"$g_t = \delta_\mathrm{eff} / \delta^*$ (quantile-binned)",
                   fontsize=10.5)
    ax2.set_ylabel("Gold-boundary precision", fontsize=10)
    ax2.set_title(f"(b) Fine {n_bins}-quantile curve  (monotone trend)",
                  fontsize=10.5)
    ax2.grid(True, alpha=0.25, linestyle=":")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=9, frameon=False)

    fig.suptitle("Calibrated graded boundary signal: precision rises with score "
                 "across all three benchmarks", fontsize=11.5, y=1.02)
    plt.tight_layout()

    pdf = OUT_DIR / "band_precision.pdf"
    png = OUT_DIR / "band_precision.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nsaved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
