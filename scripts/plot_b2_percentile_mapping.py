#!/usr/bin/env python3
"""B2 — Cross-encoder percentile mapping (per DTS dataset).

For each dataset d ∈ {TIAGE, Dialseg711, SuperDialseg} and each pair of
encoders (A, B), plot the function:

    f(p) = CDF_B( percentile_A(p) ) × 100

i.e., the δ_eff threshold τ_A = percentile_A(p) for encoder A, then ask
"what percentile of encoder B's δ_eff distribution is this τ_A?".

If the percentile rank is encoder-invariant → curve = identity (y=x).
If raw δ scales shift but ordering preserved → monotone smooth curve.

Plots 3-panel figure (one per dataset). Each panel has 3 curves (all
encoder pairs) + y=x diagonal + p60-p80 calibration band highlighted.

산출: outputs/figures/figure_J_percentile_mapping.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from diag_calib_n_convergence import load_calib, delta_eff_seq  # noqa: E402

ENCS = ("mpnet", "minilm", "minilm-int8")
DATASETS = ("tiage", "dialseg711", "superseg")
ENC_PRETTY = {"mpnet": "MPNet", "minilm": "MiniLM", "minilm-int8": "MiniLM-int8"}
DS_PRETTY = {"tiage": "TIAGE", "dialseg711": "Dialseg711", "superseg": "SuperDialseg"}

PAIRS = [
    ("mpnet", "minilm",      "#1f77b4"),
    ("mpnet", "minilm-int8", "#d62728"),
    ("minilm", "minilm-int8", "#2ca02c"),
]

OUT_DIR = REPO / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_deltas(enc: str, ds: str) -> np.ndarray:
    cal_dia, cal_emb, _, _ = load_calib(enc, ds)
    seqs = [delta_eff_seq(e) for e in cal_emb]
    return np.array([d for s in seqs for d in s[1:]])


def main() -> None:
    print("[1/2] collecting δ_eff arrays per (encoder × dataset)", flush=True)
    arrays = {}
    for enc in ENCS:
        for ds in DATASETS:
            arr = collect_deltas(enc, ds)
            arrays[(enc, ds)] = np.sort(arr)
            print(f"  {enc:13s}/{ds:11s}: n={arr.size}", flush=True)

    print("[2/2] plotting", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True, sharey=True)

    ps = np.arange(0, 100.5, 0.5)

    for ci, ds in enumerate(DATASETS):
        ax = axes[ci]
        # p60-p80 calibration band
        ax.axvspan(60, 80, color="#FFF3CD", alpha=0.65, zorder=1,
                    label="p60–p80 band" if ci == 0 else None)
        # y=x reference
        ax.plot([0, 100], [0, 100], color="black", linestyle="--",
                 linewidth=0.8, alpha=0.5, zorder=2,
                 label="identity (y=x)" if ci == 0 else None)

        for (a, b, color) in PAIRS:
            arr_a = arrays[(a, ds)]
            arr_b = arrays[(b, ds)]
            # τ_A at percentile p
            tau_a = np.percentile(arr_a, ps)
            # CDF_B (τ_A) × 100  →  use searchsorted on sorted arr_b
            q = np.searchsorted(arr_b, tau_a) / arr_b.size * 100.0
            label = f"{ENC_PRETTY[a]} → {ENC_PRETTY[b]}" if ci == 0 else None
            ax.plot(ps, q, color=color, linewidth=1.4, zorder=4, label=label)

        ax.set_title(DS_PRETTY[ds], fontsize=11, pad=4)
        ax.set_xlabel("Percentile in source encoder", fontsize=9.5)
        if ci == 0:
            ax.set_ylabel("Matched percentile in target encoder",
                          fontsize=9.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, linestyle=":")

    axes[0].legend(loc="lower right", fontsize=8, frameon=True,
                   framealpha=0.9, handlelength=2.0)

    fig.suptitle(
        "Cross-encoder percentile mapping — coarse band (p60–p80) "
        "transfers across encoders within a dataset",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"figure_J_percentile_mapping.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
