#!/usr/bin/env python3
"""Paper figure — calib N convergence (gap-to-oracle).

inputs: outputs/experiments/2026-05-24_calib_n_option_a/results.json
outputs:
    main figure (bench-averaged, 3-panel): calib_n_convergence_main.pdf/.png
    appendix figure (9-cell small multiples): calib_n_convergence_appendix.pdf/.png

Design choices (from codex viz consult):
- y = gap to test-side oracle (Score) — lower=better, baseline at 0
- x = log N (calibration dialogs)
- threshold dashed line at gap=0.005 (= "near oracle")
- lines = p60 / p70 / p80 (label-free percentile), best=gray dashed (deemphasized)
- σ band (3-5 seed bootstrap) as shaded fill at α=0.18
- main = bench-averaged (3 encoders → 1 line per percentile per bench)
- appendix = per-cell (3×3 small multiples)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "outputs" / "experiments" / "2026-05-24_calib_n_option_a" / "results.json"
OUT_DIR = REPO / "outputs" / "experiments" / "2026-05-24_calib_n_option_a"

ENCS = ("mpnet", "minilm", "minilm-int8")
ENC_PRETTY = {"mpnet": "MPNet", "minilm": "MiniLM",
              "minilm-int8": "MiniLM-int8"}
BENCHES = ("tiage", "dialseg711", "superseg")
BENCH_PRETTY = {"tiage": "TIAGE", "dialseg711": "Dialseg711",
                "superseg": "SuperDialseg"}

# tab10 palette — matches appendix figures (user request 2026-05-25)
PCOLORS = {
    "p60": "#1f77b4",   # blue
    "p70": "#ff7f0e",   # orange (yellow side, unchanged)
    "p80": "#a30000",   # red (deepened for dashed-line distinguishability)
    "best": "#666666",  # gray
}
PSTYLES = {
    "p60": ("-",  "o", 4),
    "p70": ("-",  "s", 4),   # solid square for default
    "p80": ("-",  "^", 4),
    "best": ("--", "x", 4),
}

GAP_THRESHOLD = 0.005
CONV_TOL = 0.002  # gap within ±CONV_TOL of N_max value → considered converged


def _convergence_n(ns, mean, tol=CONV_TOL):
    """Smallest N at which mean gap stays within ±tol of the final-N value
    for the remaining sweep. Returns int(N) or None."""
    final = mean[-1]
    for i in range(len(ns)):
        if all(abs(mean[j] - final) <= tol for j in range(i, len(ns))):
            return int(ns[i]), float(final)
    return None, float(final)


def _setup_rc():
    """Unified style shared by figure H and figure P (EMNLP main quality)."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#666",
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "grid.color": "#bbb",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _curve(ax, ns, mean, std, label, key, *, show_band=False):
    color = PCOLORS[key]
    style, marker, ms = PSTYLES[key]
    mean = np.asarray(mean)
    std = np.asarray(std)
    ax.plot(ns, mean, style, color=color, marker=marker,
            markersize=ms, markeredgecolor="white", markeredgewidth=0.5,
            label=label, zorder=4)
    if show_band:
        ax.fill_between(ns, mean - std, mean + std,
                        color=color, alpha=0.13, linewidth=0, zorder=1)


def _annotate_threshold(ax):
    ax.axhline(GAP_THRESHOLD, color="#444444", linestyle=":", linewidth=0.8,
               alpha=0.7, zorder=2)
    ax.text(3.2, GAP_THRESHOLD + 0.0015, f"gap = {GAP_THRESHOLD}",
            fontsize=7, color="#444444", alpha=0.85, ha="left", va="bottom")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.6, zorder=2)


def _bench_avg(results, bench, ref="oracle"):
    """encoder-averaged gap for a bench. ref = 'oracle' or 'sup' (best at same N)."""
    n_set = sorted({r["N"]
                    for enc in ENCS
                    for r in results[f"{enc}/{bench}"]["rows"]})
    avg = {p: {n: [] for n in n_set} for p in ("p60", "p70", "p80", "best")}
    std = {p: {n: [] for n in n_set} for p in ("p60", "p70", "p80", "best")}
    for enc in ENCS:
        oracle_val = results[f"{enc}/{bench}"]["oracle"]
        for row in results[f"{enc}/{bench}"]["rows"]:
            ref_val = oracle_val if ref == "oracle" else row["best_mean"]
            for p in ("p60", "p70", "p80", "best"):
                gap = ref_val - row[f"{p}_mean"]
                avg[p][row["N"]].append(gap)
                std[p][row["N"]].append(row[f"{p}_std"])
    out = {}
    for p in ("p60", "p70", "p80", "best"):
        ns_p = sorted(n for n in n_set if avg[p][n])
        out[p] = (np.array(ns_p),
                  np.array([np.mean(avg[p][n]) for n in ns_p]),
                  np.array([np.mean(std[p][n]) for n in ns_p]))
    return out


def plot_main(results):
    """Main paper figure — 2 rows (oracle, sup) × 3 bench, polished aesthetics."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.5), sharex=True)
    summary = []

    for col, bench in enumerate(BENCHES):
        # row 0 = sup (top), row 1 = oracle (bottom) — user request 2026-05-25
        for row, ref in enumerate(("sup", "oracle")):
            ax = axes[row, col]
            data = _bench_avg(results, bench, ref=ref)

            # data lines
            for key in ("p60", "p70", "p80"):
                ns, mean, std = data[key]
                _curve(ax, ns, mean, std, label=key, key=key, show_band=False)

            # baselines
            ax.axhline(0, color="#222", linestyle="-", linewidth=0.7,
                        alpha=0.85, zorder=2)
            if ref == "oracle":
                ax.axhline(GAP_THRESHOLD, color="#444", linestyle=":",
                            linewidth=0.9, alpha=0.7, zorder=2)

            # robust y-range
            all_vals = []
            for key in ("p60", "p70", "p80"):
                all_vals.extend(data[key][1].tolist())
            data_min = min(all_vals)
            data_max = max(all_vals)
            span = data_max - min(data_min, 0)
            # leave HEADROOM for annotation box at top-right
            ymin = min(data_min, 0.0) - 0.10 * span
            ymax = data_max + 0.55 * span

            # convergence markers — vertical dashed line only (no rings)
            conv_marks = []
            for key in ("p60", "p70", "p80"):
                ns, mean, std = data[key]
                n_conv, gap_final = _convergence_n(ns, mean)
                if n_conv is not None:
                    ax.axvline(n_conv, color=PCOLORS[key], linestyle="--",
                                linewidth=0.9, alpha=0.8, zorder=3)
                    conv_marks.append((key, n_conv, float(mean[list(ns).index(n_conv)]),
                                        gap_final))

            ax.set_xlim(-5, 315)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
            # ALL rows show xtick labels
            ax.tick_params(axis="x", labelbottom=True)
            if row == 0:
                ax.set_title(BENCH_PRETTY[bench], pad=6, fontsize=11)
            if row == 1:
                # xlabel per column on bottom row
                ax.set_xlabel(r"Calibration dialogs $N$", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    "Gap to oracle" if ref == "oracle" else "Gap to sup",
                    fontsize=10,
                )
            ax.tick_params(axis="both", which="major", labelsize=9)

            # annotation box (no "N_conv (knee)" header — user request 2026-05-25)
            if conv_marks:
                rows_str = "\n".join(
                    f"$p_{{{k[1:]}}}$ → $N{{=}}${n}"
                    for k, n, _, _ in conv_marks
                )
                ax.text(0.97, 0.96, rows_str,
                          transform=ax.transAxes, ha="right", va="top",
                          fontsize=8, family="serif",
                          linespacing=1.4,
                          bbox=dict(boxstyle="round,pad=0.4",
                                      facecolor="white", edgecolor="#444",
                                      linewidth=0.8, alpha=0.98),
                          zorder=10)

            summary.append(
                f"{BENCH_PRETTY[bench]} / {ref}: " +
                ", ".join(f"{k}={n}" for k, n, _, _ in conv_marks)
            )

    # global legend
    from matplotlib.lines import Line2D
    legend_handles = []
    for key in ("p60", "p70", "p80"):
        color = PCOLORS[key]
        style, marker, ms = PSTYLES[key]
        legend_handles.append(Line2D([0], [0], color=color, marker=marker,
                                       linestyle=style, linewidth=1.8,
                                       markersize=ms + 1,
                                       markeredgecolor="white",
                                       markeredgewidth=0.5,
                                       label=f"$p_{{{key[1:]}}}$"))
    legend_handles.append(Line2D([0], [0], color="#444", linestyle=":",
                                    linewidth=1.0,
                                    label="threshold (gap = 0.005)"))
    legend_handles.append(Line2D([0], [0], color="#222", linestyle="-",
                                    linewidth=0.8,
                                    label="gap = 0  (= reference)"))
    fig.legend(handles=legend_handles,
                loc="upper center", bbox_to_anchor=(0.5, 1.00),
                ncol=5, frameon=False, fontsize=9.5,
                handlelength=1.8, columnspacing=1.6,
                handletextpad=0.5)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    main_pdf = OUT_DIR / "calib_n_convergence_main.pdf"
    main_png = OUT_DIR / "calib_n_convergence_main.png"
    fig.savefig(main_pdf)
    fig.savefig(main_png)
    plt.close(fig)
    print(f"main → {main_pdf}", flush=True)
    print("convergence summary:")
    for ln in summary:
        print(f"  {ln}")


def _plot_appendix_with_ref(results, ref: str):
    """3×3 small multiples — uniform style with main figure.

    rows = encoder (MPNet / MiniLM / MiniLM-int8)
    cols = bench (TIAGE / Dialseg711 / SuperDialseg)
    each cell: 3 percentile curves + N_conv ring markers + annotation box
    """
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.2),
                              sharex=True, sharey=False)
    summary = []

    for r, enc in enumerate(ENCS):
        for c, bench in enumerate(BENCHES):
            ax = axes[r, c]
            cell = results[f"{enc}/{bench}"]
            oracle = cell["oracle"]
            ns = np.array([row["N"] for row in cell["rows"]])

            curves = {}
            for key in ("p60", "p70", "p80"):
                mean = np.array([(oracle if ref == "oracle" else row["best_mean"])
                                 - row[f"{key}_mean"]
                                 for row in cell["rows"]])
                std = np.array([row[f"{key}_std"] for row in cell["rows"]])
                _curve(ax, ns, mean, std, label=key, key=key, show_band=False)
                curves[key] = (ns, mean, std)

            # baselines
            ax.axhline(0, color="#222", linestyle="-", linewidth=0.7,
                        alpha=0.85, zorder=2)
            if ref == "oracle":
                ax.axhline(GAP_THRESHOLD, color="#444", linestyle=":",
                            linewidth=0.9, alpha=0.7, zorder=2)

            # robust y-range with headroom
            all_vals = []
            for _, m, _ in curves.values():
                all_vals.extend(m.tolist())
            data_min, data_max = min(all_vals), max(all_vals)
            span = data_max - min(data_min, 0)
            ymin = min(data_min, 0.0) - 0.10 * span
            ymax = data_max + 0.55 * span

            # convergence markers — vertical dashed line only (no rings)
            conv_marks = []
            for key in ("p60", "p70", "p80"):
                ns_k, mean_k, _ = curves[key]
                n_conv, gap_final = _convergence_n(ns_k, mean_k)
                if n_conv is not None:
                    ax.axvline(n_conv, color=PCOLORS[key], linestyle="--",
                                linewidth=0.9, alpha=0.8, zorder=3)
                    conv_marks.append((key, n_conv,
                                        float(mean_k[list(ns_k).index(n_conv)]),
                                        gap_final))

            ax.set_xlim(-5, 315)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
            ax.tick_params(axis="both", which="major", labelsize=9)

            ax.tick_params(axis="x", labelbottom=True)
            if r == 0:
                ax.set_title(BENCH_PRETTY[bench], pad=6, fontsize=11)
            if r == 2:
                ax.set_xlabel(r"Calibration dialogs $N$", fontsize=10)
            if c == 0:
                ylab = "Gap to oracle" if ref == "oracle" else "Gap to sup"
                ax.set_ylabel(f"{ENC_PRETTY[enc]}\n{ylab}", fontsize=10)

            # annotation box (no "N_conv (knee)" header — user request 2026-05-25)
            if conv_marks:
                rows_str = "\n".join(
                    f"$p_{{{k[1:]}}}$ → $N{{=}}${n}"
                    for k, n, _, _ in conv_marks
                )
                ax.text(0.97, 0.96, rows_str,
                          transform=ax.transAxes, ha="right", va="top",
                          fontsize=8, family="serif",
                          linespacing=1.4,
                          bbox=dict(boxstyle="round,pad=0.4",
                                      facecolor="white", edgecolor="#444",
                                      linewidth=0.8, alpha=0.98),
                          zorder=10)

            summary.append(
                f"{enc}/{bench}/{ref}: " +
                ", ".join(f"{k}={n}" for k, n, _, _ in conv_marks)
            )

    # global legend — same items as main
    from matplotlib.lines import Line2D
    legend_handles = []
    for key in ("p60", "p70", "p80"):
        color = PCOLORS[key]
        style, marker, ms = PSTYLES[key]
        legend_handles.append(Line2D([0], [0], color=color, marker=marker,
                                       linestyle=style, linewidth=1.8,
                                       markersize=ms + 1,
                                       markeredgecolor="white",
                                       markeredgewidth=0.5,
                                       label=f"$p_{{{key[1:]}}}$"))
    legend_handles.append(Line2D([0], [0], color="#444", linestyle=":",
                                    linewidth=1.0,
                                    label="threshold (gap = 0.005)"))
    legend_handles.append(Line2D([0], [0], color="#222", linestyle="-",
                                    linewidth=0.8,
                                    label="gap = 0  (= reference)"))
    fig.legend(handles=legend_handles,
                loc="upper center", bbox_to_anchor=(0.5, 1.00),
                ncol=5, frameon=False, fontsize=9.5,
                handlelength=1.8, columnspacing=1.6,
                handletextpad=0.5)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    suffix = "appendix_oracle" if ref == "oracle" else "appendix_sup"
    apx_pdf = OUT_DIR / f"calib_n_convergence_{suffix}.pdf"
    apx_png = OUT_DIR / f"calib_n_convergence_{suffix}.png"
    fig.savefig(apx_pdf); fig.savefig(apx_png)
    plt.close(fig)
    print(f"appendix ({ref}) → {apx_pdf}", flush=True)
    for ln in summary:
        print(f"  {ln}")


def plot_appendix(results):
    _plot_appendix_with_ref(results, ref="oracle")
    _plot_appendix_with_ref(results, ref="sup")


def main() -> None:
    _setup_rc()
    if not RESULTS.exists():
        raise SystemExit(f"results.json not yet — run diag_calib_n_option_a.py first ({RESULTS})")
    results = json.loads(RESULTS.read_text())
    plot_main(results)
    plot_appendix(results)


if __name__ == "__main__":
    main()
