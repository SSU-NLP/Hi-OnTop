#!/usr/bin/env python3
"""DTS validation of label-free best-p calibration (codex option 1).

For each (encoder, DTS dataset) = 9 cells:
1. Compute δ_eff on test split (deploy domain analog) — **no labels used**.
2. Sweep p ∈ {60, 65, 70, 75, 80}: δ*_p = percentile_p(test_pool),
   apply δ*_p, count predicted segments per dialogue → avg_seg_len(p).
3. Get prior L from train split's gold avg seg len (labeled calibration corpus).
4. p̂(label-free) = argmin_p |avg_seg_len(p) − L|.
5. True best p (oracle) = argmax_p Score(p) on test split (uses labels).
6. Report: p̂ vs true best, Score@p̂ vs Score@best (label-free regret).

산출:
- outputs/experiments/2026-05-25_dts_label_free_best_p/results.json
- outputs/figures/figure_N_dts_label_free_best_p.{pdf,png}
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

ENCS = ("mpnet", "minilm", "minilm-int8")
DATASETS = ("tiage", "dialseg711", "superseg")
ENC_PRETTY = {"mpnet": "MPNet", "minilm": "MiniLM", "minilm-int8": "MiniLM-int8"}
DS_PRETTY = {"tiage": "TIAGE", "dialseg711": "Dialseg711", "superseg": "SuperDialseg"}
P_GRID = [60, 65, 70, 75, 80]

OUT_EXP = REPO / "outputs" / "experiments" / "2026-05-25_dts_label_free_best_p"
OUT_FIG = REPO / "outputs" / "figures"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def gold_avg_seg_len(dialogs):
    """Gold average segment length = total_turns / total_gold_segments."""
    total_turns = sum(len(u) for u, _ in dialogs)
    total_segs = sum(1 + sum(yt) for _, yt in dialogs)
    return total_turns / max(1, total_segs)


def predicted_avg_seg_len(deffs, dstar):
    total_turns = sum(len(d) for d in deffs)
    total_segs = 0
    for d in deffs:
        arr = np.asarray(d[1:])  # skip turn 0
        n_bnd = int((arr >= dstar).sum())
        total_segs += 1 + n_bnd
    return total_turns / max(1, total_segs)


def main() -> None:
    results = {}

    for enc in ENCS:
        for ds in DATASETS:
            cell = f"{enc}/{ds}"
            print(f"\n=== {cell} ===", flush=True)

            cal_dia, cal_emb, te_dia, te_emb = load_calib(enc, ds)
            print(f"  cal={len(cal_dia)}  test={len(te_dia)}", flush=True)

            # Prior (from train/calib gold)
            L_prior = gold_avg_seg_len(cal_dia)
            L_test = gold_avg_seg_len(te_dia)
            print(f"  L_train={L_prior:.3f}  (L_test={L_test:.3f} for ref)",
                  flush=True)

            # δ_eff on TEST (label-free)
            te_deffs = [delta_eff_seq(e) for e in te_emb]
            pool = np.concatenate([np.asarray(d[1:]) for d in te_deffs])

            sweep = []
            for p in P_GRID:
                dstar = float(np.percentile(pool, p))
                avg_len = predicted_avg_seg_len(te_deffs, dstar)
                # Score (with labels — only used for ORACLE comparison)
                sc = score_set(te_dia, te_deffs, dstar)
                sweep.append({
                    "p": p,
                    "dstar": dstar,
                    "predicted_avg_seg_len": avg_len,
                    "score": sc["score"],
                    "f1": sc["f1"],
                    "pk": sc["pk"],
                    "wd": sc["wd"],
                })

            # p_hat via prior matching (label-free)
            p_hat_lf = min(sweep, key=lambda s: abs(s["predicted_avg_seg_len"] - L_prior))
            # True best (oracle, uses test labels)
            p_best_oracle = max(sweep, key=lambda s: s["score"])

            regret = p_best_oracle["score"] - p_hat_lf["score"]
            results[cell] = {
                "L_prior_train": float(L_prior),
                "L_test_ref": float(L_test),
                "p_grid": P_GRID,
                "sweep": sweep,
                "p_hat_label_free": p_hat_lf["p"],
                "p_best_oracle": p_best_oracle["p"],
                "score_at_p_hat": p_hat_lf["score"],
                "score_at_p_best": p_best_oracle["score"],
                "score_regret": float(regret),
                "agrees": p_hat_lf["p"] == p_best_oracle["p"],
            }
            print(f"  p̂ (label-free, prior L={L_prior:.2f}) = p{p_hat_lf['p']}  "
                  f"Score@p̂={p_hat_lf['score']:.4f}", flush=True)
            print(f"  best (oracle, with labels) = p{p_best_oracle['p']}  "
                  f"Score@best={p_best_oracle['score']:.4f}", flush=True)
            print(f"  regret = {regret:.4f}  "
                  f"{'✓ agrees' if p_hat_lf['p']==p_best_oracle['p'] else '✗ mismatch'}",
                  flush=True)

    (OUT_EXP / "results.json").write_text(json.dumps(results, indent=2))

    # ── Figure: 9-cell grid, Score vs p with markers for p̂ vs oracle ──
    fig, axes = plt.subplots(len(ENCS), len(DATASETS), figsize=(13.5, 7.5),
                              sharex=True)
    for ri, enc in enumerate(ENCS):
        for ci, ds in enumerate(DATASETS):
            ax = axes[ri, ci]
            cell = f"{enc}/{ds}"
            data = results[cell]
            ps = [s["p"] for s in data["sweep"]]
            scores = [s["score"] for s in data["sweep"]]

            ax.plot(ps, scores, marker="o", linewidth=1.6, markersize=6,
                     color="#1f77b4", zorder=4)

            # mark p̂ (label-free) and best (oracle)
            p_hat = data["p_hat_label_free"]
            p_best = data["p_best_oracle"]
            sc_hat = data["score_at_p_hat"]
            sc_best = data["score_at_p_best"]

            ax.plot(p_hat, sc_hat, marker="*", markersize=18, color="#d62728",
                     markeredgecolor="black", markeredgewidth=0.6, zorder=6,
                     label="label-free p̂" if (ri, ci) == (0, 0) else None)
            if p_hat != p_best:
                ax.plot(p_best, sc_best, marker="^", markersize=12,
                         color="#2ca02c", markeredgecolor="black",
                         markeredgewidth=0.6, zorder=5,
                         label="oracle best" if (ri, ci) == (0, 0) else None)
            ax.annotate(f"p̂=p{p_hat}\nbest=p{p_best}\nΔ={data['score_regret']:.3f}",
                         xy=(0.04, 0.04), xycoords="axes fraction",
                         fontsize=8, ha="left", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.25",
                                   facecolor="white", edgecolor="gray", alpha=0.85))

            if ri == 0:
                ax.set_title(DS_PRETTY[ds], fontsize=11, pad=4)
            if ri == len(ENCS) - 1:
                ax.set_xlabel(r"Percentile $p$", fontsize=9.5)
            if ci == 0:
                ax.set_ylabel(f"{ENC_PRETTY[enc]}\nScore $\\uparrow$",
                               fontsize=9.5)
            ax.set_xticks(P_GRID)
            ax.tick_params(labelsize=8.5)
            ax.grid(True, alpha=0.25, linestyle=":")

    axes[0, 0].legend(loc="lower right", fontsize=8.5, frameon=True,
                       framealpha=0.92)

    fig.suptitle(
        "DTS validation of label-free best-$p$ via segment-length prior matching\n"
        r"(prior $L$ = train-gold avg segment length; $\star$ = label-free pick, "
        r"$\triangle$ = oracle best)",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.955))

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"figure_N_dts_label_free_best_p.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)

    # summary table to stdout
    print("\n" + "=" * 65, flush=True)
    print(f"{'cell':25s}  {'L_pri':>5s}  {'p̂':>4s}  {'best':>4s}  "
           f"{'Δ_Score':>7s}  match", flush=True)
    print("-" * 65, flush=True)
    agree_count = 0
    regret_sum = 0.0
    for cell, d in results.items():
        m = "✓" if d["agrees"] else "✗"
        if d["agrees"]:
            agree_count += 1
        regret_sum += d["score_regret"]
        print(f"{cell:25s}  {d['L_prior_train']:>5.2f}  "
               f"p{d['p_hat_label_free']:<3d}  p{d['p_best_oracle']:<3d}  "
               f"{d['score_regret']:>7.4f}  {m}", flush=True)
    print("-" * 65, flush=True)
    print(f"agreement: {agree_count}/9 cells  ·  mean regret = "
           f"{regret_sum / 9:.4f}", flush=True)


if __name__ == "__main__":
    main()
