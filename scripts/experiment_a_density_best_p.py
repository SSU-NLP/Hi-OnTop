#!/usr/bin/env python3
"""A — Density-conditioned best-p sensitivity on SuperDialseg.

Protocol (codex 권고, 2026-05-25):
- Sub-corpus = SuperDialseg test (1322 dialogues, largest pool)
- Density def: per-dialogue boundary/turn = sum(yt) / len(utts) for utts in dialogue
- 3-bin: Low (bottom 25%), Mid (37.5-62.5%), High (top 25%) by density
- Bootstrap: R=100 iterations
- Each iter: sample N=100 from bin as calib pool, remaining (~230) as eval
- For each bootstrap: compute δ_eff on calib → percentile(p60/p70/p80) → δ*_p
- Score eval set with each δ*_p → argmax_p = best-p
- Output: P(best=p|bin) for p ∈ {p60, p70, p80} with 95% CI

Encoder = MPNet (paper main encoder).

산출: outputs/figures/figure_K_density_best_p.{pdf,png}
       outputs/experiments/2026-05-25_density_best_p/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from diag_calib_n_convergence import load_calib, delta_eff_seq  # noqa: E402
from run_encoder_comparison import score_set  # noqa: E402

ENCODER = "mpnet"
DATASET = "superseg"
PERCENTILES = (60, 70, 80)
R = 100  # bootstrap iterations
N_CALIB = 100  # fixed calibration sample size per bin

# bin boundaries on density quantile (low / mid / high)
BIN_DEFS = [
    ("Low",  0.00, 0.25, "#1f77b4"),
    ("Mid",  0.375, 0.625, "#2ca02c"),
    ("High", 0.75, 1.00, "#d62728"),
]

OUT_FIGDIR = REPO / "outputs" / "figures"
OUT_EXPDIR = REPO / "outputs" / "experiments" / "2026-05-25_density_best_p"
OUT_EXPDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("[1/4] loading SuperDialseg test (MPNet)", flush=True)
    _, _, test_dia, test_emb = load_calib(ENCODER, DATASET)
    n = len(test_dia)
    print(f"  n_test_dialogues={n}", flush=True)

    print("[2/4] computing per-dialogue density + turn count", flush=True)
    density = np.array([sum(yt) / max(1, len(utts))
                         for utts, yt in test_dia])
    turn_counts = np.array([len(utts) for utts, _ in test_dia])
    deffs = [delta_eff_seq(e) for e in test_emb]

    bin_qs = np.quantile(density, [0.25, 0.375, 0.625, 0.75])
    print(f"  density quantiles: 25%={bin_qs[0]:.3f}  37.5%={bin_qs[1]:.3f}  "
           f"62.5%={bin_qs[2]:.3f}  75%={bin_qs[3]:.3f}", flush=True)

    bin_idx = {}
    for name, lo, hi, _ in BIN_DEFS:
        lo_v = np.quantile(density, lo)
        hi_v = np.quantile(density, hi)
        if lo == 0.0:
            mask = density <= hi_v
        elif hi == 1.0:
            mask = density >= lo_v
        else:
            mask = (density >= lo_v) & (density <= hi_v)
        bin_idx[name] = np.where(mask)[0]
        print(f"  {name:4s} bin: n={bin_idx[name].size:4d}  "
              f"density ∈ [{density[mask].min():.3f}, {density[mask].max():.3f}]  "
              f"mean turns={turn_counts[mask].mean():.1f}", flush=True)

    print(f"[3/4] bootstrap R={R} (N_calib={N_CALIB}) per bin", flush=True)
    rng = np.random.default_rng(0)
    results = {}
    for name, _, _, _ in BIN_DEFS:
        idx = bin_idx[name]
        if idx.size < N_CALIB + 50:
            print(f"  {name}: SKIP (only {idx.size} dialogues, need ≥{N_CALIB+50})")
            continue

        best_p_count = {p: 0 for p in PERCENTILES}
        scores_per_p = {p: [] for p in PERCENTILES}

        for r_i in range(R):
            perm = rng.permutation(idx)
            calib = perm[:N_CALIB]
            eval_ = perm[N_CALIB:]

            allv = np.array([d for i in calib for d in deffs[i][1:]])
            scores = {}
            for p in PERCENTILES:
                dstar = float(np.percentile(allv, p))
                ev_dia = [test_dia[i] for i in eval_]
                ev_deff = [deffs[i] for i in eval_]
                sc = score_set(ev_dia, ev_deff, dstar)["score"]
                scores[p] = sc
                scores_per_p[p].append(sc)
            best_p = max(scores, key=scores.get)
            best_p_count[best_p] += 1

        # bootstrap CI on best-p frequency: Wilson 95% CI
        from math import sqrt
        z = 1.96
        bin_result = {}
        for p in PERCENTILES:
            k = best_p_count[p]
            phat = k / R
            denom = 1 + z**2 / R
            centre = (phat + z**2 / (2 * R)) / denom
            half = (z * sqrt(phat * (1 - phat) / R + z**2 / (4 * R**2))) / denom
            bin_result[f"p{p}"] = {
                "best_freq": phat,
                "ci_lo": max(0.0, centre - half),
                "ci_hi": min(1.0, centre + half),
                "mean_score": float(np.mean(scores_per_p[p])),
                "std_score": float(np.std(scores_per_p[p])),
            }
        results[name] = dict(
            n_dialogues=int(idx.size),
            density_min=float(density[idx].min()),
            density_max=float(density[idx].max()),
            density_mean=float(density[idx].mean()),
            mean_turns=float(turn_counts[idx].mean()),
            best_p=bin_result,
        )
        msg = f"  {name}: "
        for p in PERCENTILES:
            r = bin_result[f"p{p}"]
            msg += (f"p{p}: best={r['best_freq']:.2f} "
                     f"(CI[{r['ci_lo']:.2f},{r['ci_hi']:.2f}], "
                     f"Score={r['mean_score']:.3f}±{r['std_score']:.3f})  ")
        print(msg, flush=True)

    (OUT_EXPDIR / "results.json").write_text(json.dumps(results, indent=2))

    print("[4/4] plotting figure_K", flush=True)
    fig, (ax_freq, ax_score) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # ── panel 1: P(best=p|bin) stacked bar with CI ──
    bins_present = [b for b in BIN_DEFS if b[0] in results]
    x = np.arange(len(bins_present))
    width = 0.27
    p_colors = {60: "#1f77b4", 70: "#ff7f0e", 80: "#d62728"}
    for i, p in enumerate(PERCENTILES):
        freqs = [results[b[0]]["best_p"][f"p{p}"]["best_freq"]
                  for b in bins_present]
        lo = [results[b[0]]["best_p"][f"p{p}"]["ci_lo"] for b in bins_present]
        hi = [results[b[0]]["best_p"][f"p{p}"]["ci_hi"] for b in bins_present]
        yerr = np.maximum(
            np.array([np.array(freqs) - np.array(lo),
                       np.array(hi) - np.array(freqs)]),
            0.0,
        )
        ax_freq.bar(x + (i - 1) * width, freqs, width, color=p_colors[p],
                     yerr=yerr, capsize=3, alpha=0.85,
                     label=f"$p_{{{p}}}$", edgecolor="white", linewidth=0.5)
    ax_freq.set_xticks(x)
    ax_freq.set_xticklabels([f"{b[0]}\n(n={results[b[0]]['n_dialogues']}, "
                              f"$\\bar\\rho$={results[b[0]]['density_mean']:.2f})"
                              for b in bins_present], fontsize=9)
    ax_freq.set_ylabel(r"$P(\mathrm{best}\!=\!p\;|\;\mathrm{bin})$", fontsize=10)
    ax_freq.set_title("Best-percentile frequency across density bins",
                       fontsize=10.5, pad=4)
    ax_freq.set_ylim(0, 1.05)
    ax_freq.legend(loc="upper right", fontsize=9, frameon=True)
    ax_freq.grid(True, alpha=0.25, axis="y", linestyle=":")
    ax_freq.tick_params(axis="y", labelsize=8.5)

    # ── panel 2: mean Score per p per bin ──
    for p in PERCENTILES:
        scs = [results[b[0]]["best_p"][f"p{p}"]["mean_score"] for b in bins_present]
        stds = [results[b[0]]["best_p"][f"p{p}"]["std_score"] for b in bins_present]
        ax_score.errorbar(range(len(bins_present)), scs, yerr=stds,
                          marker="o", linewidth=1.4, markersize=6,
                          color=p_colors[p], label=f"$p_{{{p}}}$",
                          capsize=3)
    ax_score.set_xticks(range(len(bins_present)))
    ax_score.set_xticklabels([b[0] for b in bins_present], fontsize=9)
    ax_score.set_xlabel("Boundary density bin", fontsize=10)
    ax_score.set_ylabel(r"Score $\pm$ std (over $R=$" + str(R)
                          + " bootstraps)", fontsize=10)
    ax_score.set_title("Calibration robustness across density bins",
                        fontsize=10.5, pad=4)
    ax_score.legend(loc="best", fontsize=9, frameon=True)
    ax_score.grid(True, alpha=0.25, linestyle=":")
    ax_score.tick_params(axis="both", labelsize=8.5)

    fig.suptitle(
        "Density-conditioned percentile sensitivity (SuperDialseg, MPNet, "
        f"R={R}, $N_{{\\mathrm{{calib}}}}=$ {N_CALIB})",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    for ext in ("pdf", "png"):
        out = OUT_FIGDIR / f"figure_K_density_best_p.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
