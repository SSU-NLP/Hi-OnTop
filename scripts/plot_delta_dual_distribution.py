#!/usr/bin/env python3
"""δ_eff (dual-channel blend) 분포를 (encoder × DTS dataset) cell 별로 보여주는 진단 figure.

각 cell 에서:
- histogram (n turn-level δ_eff samples)
- mean / variance (텍스트 annotation)
- percentile curve (overlay, 우측 y축)

cells: 3 encoder (mpnet, minilm, minilm-int8) × 3 dataset (tiage, dialseg711, superseg).
data source: calib (= 각 벤치 train split) 의 모든 turn-level δ_eff.

산출:
- outputs/experiments/2026-05-25_delta_eff_distribution/distribution.{pdf,png}
- outputs/experiments/2026-05-25_delta_eff_distribution/stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from diag_calib_n_convergence import ENCS, load_calib, delta_eff_seq  # noqa: E402

DATASETS = ("tiage", "dialseg711", "superseg")
ENC_PRETTY = {"mpnet": "MPNet", "minilm": "MiniLM", "minilm-int8": "MiniLM-int8"}
DS_PRETTY = {"tiage": "TIAGE", "dialseg711": "Dialseg711", "superseg": "SuperDialseg"}
PERCENTILES_GRID = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95)

OUT_DIR = REPO / "outputs" / "experiments" / "2026-05-25_delta_eff_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_deltas(enc: str, ds: str) -> np.ndarray:
    """calib (=train) split 의 모든 turn 의 δ_eff (turn 0 제외)."""
    cal_dia, cal_emb, _, _ = load_calib(enc, ds)
    seqs = [delta_eff_seq(e) for e in cal_emb]
    # turn 0 의 δ_eff=0 은 제외 (boundary 결정에 쓰지 않음)
    return np.array([d for s in seqs for d in s[1:]])


def main() -> None:
    stats = {}
    arrays: dict[tuple[str, str], np.ndarray] = {}

    print("[1/2] computing δ_eff per (encoder, dataset) ...", flush=True)
    for enc in ENCS:
        for ds in DATASETS:
            arr = collect_deltas(enc, ds)
            arrays[(enc, ds)] = arr
            stats[f"{enc}/{ds}"] = {
                "n_samples": int(arr.size),
                "mean": float(arr.mean()),
                "variance": float(arr.var()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "percentiles": {
                    f"p{p}": float(np.percentile(arr, p))
                    for p in PERCENTILES_GRID
                },
            }
            print(
                f"  {enc:13s}/{ds:11s}: n={arr.size:5d}  "
                f"μ={arr.mean():.4f}  σ²={arr.var():.4f}  "
                f"σ={arr.std():.4f}  p70={np.percentile(arr, 70):.4f}",
                flush=True,
            )

    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    print("[2/2] plotting 3x3 grid ...", flush=True)
    fig, axes = plt.subplots(
        len(ENCS), len(DATASETS),
        figsize=(13.0, 8.0), sharex=False,
    )

    for r, enc in enumerate(ENCS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            arr = arrays[(enc, ds)]

            # histogram (left y-axis)
            n, bins, _ = ax.hist(arr, bins=40, color="#4F81BD",
                                  alpha=0.6, edgecolor="white", linewidth=0.3)
            ax.set_ylabel("count" if c == 0 else "", fontsize=8.5)
            ax.tick_params(axis="y", labelsize=7.5, colors="#4F81BD")

            # percentile curve (right y-axis, overlay)
            ax2 = ax.twinx()
            ps = np.arange(0, 100.5, 1.0)
            qs = np.percentile(arr, ps)
            ax2.plot(qs, ps, color="#C0504D", linewidth=1.2, zorder=5)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("percentile" if c == len(DATASETS) - 1 else "",
                            fontsize=8.5, color="#C0504D")
            ax2.tick_params(axis="y", labelsize=7.5, colors="#C0504D")
            ax2.spines["right"].set_color("#C0504D")
            ax2.spines["right"].set_alpha(0.7)

            # mean / variance annotation
            ax.axvline(arr.mean(), color="black", linestyle="--",
                        linewidth=0.8, alpha=0.7, zorder=4)
            ax.text(
                0.97, 0.97,
                f"$\\mu$={arr.mean():.3f}\n$\\sigma^2$={arr.var():.4f}\n"
                f"$\\sigma$={arr.std():.3f}\n$n$={arr.size}",
                transform=ax.transAxes,
                fontsize=7.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.85),
            )

            # title
            if r == 0:
                ax.set_title(DS_PRETTY[ds], fontsize=10, pad=4)
            if c == 0:
                ax.set_ylabel(f"{ENC_PRETTY[enc]}\ncount",
                              fontsize=9, color="black")
            if r == len(ENCS) - 1:
                ax.set_xlabel(r"$\delta_{\mathrm{eff}}$", fontsize=9)

            ax.tick_params(axis="x", labelsize=7.5)
            ax.grid(True, alpha=0.2, linestyle=":")

    fig.suptitle(
        r"$\delta_{\mathrm{eff}}$ distribution by encoder × DTS dataset "
        r"(histogram + percentile curve, calib pool)",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.975))

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"distribution.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
