#!/usr/bin/env python3
"""Label-free best-p calibration via segment-length prior matching.

Codex 권고 옵션 1 (2026-05-25). target domain (MTB+, unlabeled) 에서 사전에 정한
segment-length prior L 와 가장 가까운 predicted seg len 을 내는 percentile p
를 선택.

Protocol:
1. MTB+ 11 conv (54 session, 720 turn) — int8 ONNX encoder 로 embedding
2. session 단위로 HiOnTop δ_eff sequence 계산
3. 모든 δ_eff pool → percentile grid {p50, p55, ..., p90} 에서 δ*_p
4. 각 p 에 대해 session 별 boundary count → conv 별 predicted segment count →
   avg predicted segment length = total_turns / total_predicted_segments
5. Prior L ∈ {3.0, 3.77, 4.06, 5.40} (DTS 3 벤치 gold 평균 + 셋 평균):
   p̂(L) = argmin_p |avg_len(p) − L|
6. Plot + report — 어떤 prior 에서 어떤 p 가 선택되는지

산출:
- outputs/experiments/2026-05-25_label_free_best_p/results.json
- outputs/figures/figure_M_label_free_best_p.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

MTBP_PATH = REPO / "benchmarks" / "SeCom" / "experiment" / "data" / "mtbp" / "mtbp.jsonl"
OUT_EXP = REPO / "outputs" / "experiments" / "2026-05-25_label_free_best_p"
OUT_FIG = REPO / "outputs" / "figures"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Hi-OnTop HP (paper main)
M, RHO, A = 2, 0.7, 0.5
P_GRID = list(range(50, 91, 5))  # 50, 55, 60, ..., 90

# Priors from DTS 3 벤치 gold avg seg lengths (outputs/reports/boundary_density.md)
PRIORS = {
    "SuperDialseg gold (3.00)": 3.00,
    "TIAGE gold (3.77)": 3.77,
    "DTS-3 avg (4.06)": 4.06,
    "Dialseg711 gold (5.40)": 5.40,
}


def load_mtbp():
    with MTBP_PATH.open() as f:
        return [json.loads(line) for line in f]


def compute_delta_eff_per_session(encoder, mtbp):
    """Returns list of np.ndarray (one per session), each = δ_eff[1:] of that session."""
    all_deff = []
    sess_index = []  # (conv_idx, sess_idx, n_turns)
    for ci, conv in enumerate(mtbp):
        for si, sess in enumerate(conv["sessions"]):
            if not sess:
                continue
            vecs = encoder.encode(sess, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            seg = HiOnTop(dim=vecs.shape[1], delta_star=1.0,
                          ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
            for v in vecs:
                seg.assign(v.astype(np.float64))
            deffs = np.array([float(h["delta_eff"]) for h in seg.history()])
            # turn 0 의 δ_eff=0 제외
            all_deff.append(deffs[1:])
            sess_index.append((ci, si, len(sess)))
    return all_deff, sess_index


def count_segments(delta_eff_sessions, sess_index, dstar):
    """Returns (total_turns, total_segments, avg_seg_len) over all sessions/conv."""
    total_turns = sum(t for _, _, t in sess_index)
    # session 별 predicted boundaries → segments = 1 + n_boundaries
    total_segs = 0
    for deff, (_, _, _) in zip(delta_eff_sessions, sess_index):
        n_bnd = int((deff >= dstar).sum())
        total_segs += 1 + n_bnd
    return total_turns, total_segs, total_turns / max(1, total_segs)


def main() -> None:
    print("[1/4] loading MTB+ and int8 encoder", flush=True)
    mtbp = load_mtbp()
    print(f"  n_conv={len(mtbp)}", flush=True)
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"})

    print("[2/4] computing δ_eff per session", flush=True)
    all_deff, sess_idx = compute_delta_eff_per_session(encoder, mtbp)
    pool = np.concatenate(all_deff)
    print(f"  n_sessions={len(sess_idx)}  n_delta_samples={pool.size}  "
           f"mean={pool.mean():.4f} std={pool.std():.4f}", flush=True)

    print("[3/4] sweep p and compute avg seg len", flush=True)
    sweep = []
    for p in P_GRID:
        dstar = float(np.percentile(pool, p))
        tot_turns, tot_segs, avg_len = count_segments(all_deff, sess_idx, dstar)
        sweep.append({
            "p": p,
            "dstar": dstar,
            "total_turns": tot_turns,
            "total_segments": tot_segs,
            "avg_seg_len": avg_len,
        })
        print(f"  p={p}: δ*={dstar:.4f}  segs={tot_segs:4d}  "
               f"avg_len={avg_len:.3f}", flush=True)

    # prior matching
    p_hat_by_prior = {}
    for label, L in PRIORS.items():
        best = min(sweep, key=lambda s: abs(s["avg_seg_len"] - L))
        p_hat_by_prior[label] = {
            "L": L,
            "p_hat": best["p"],
            "avg_len_at_p_hat": best["avg_seg_len"],
            "dstar_at_p_hat": best["dstar"],
        }
        print(f"  prior {label} → p̂ = p{best['p']} "
               f"(avg_len {best['avg_seg_len']:.2f}, δ*={best['dstar']:.4f})",
               flush=True)

    results = {
        "encoder": "minilm-int8 ONNX quint8_avx2",
        "n_conv": len(mtbp),
        "n_sessions": len(sess_idx),
        "n_delta_samples": int(pool.size),
        "delta_eff_pool": {"mean": float(pool.mean()),
                            "std": float(pool.std())},
        "p_grid": P_GRID,
        "sweep": sweep,
        "priors": p_hat_by_prior,
    }
    (OUT_EXP / "results.json").write_text(json.dumps(results, indent=2))

    print("[4/4] plotting figure_M", flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ps = [s["p"] for s in sweep]
    lens = [s["avg_seg_len"] for s in sweep]
    ax.plot(ps, lens, marker="o", linewidth=1.8, markersize=6,
             color="#1f77b4", zorder=4, label="MTB+ predicted avg seg len")

    # prior horizontal lines + p̂ vertical markers
    prior_colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    for (label, L), color in zip(PRIORS.items(), prior_colors):
        ax.axhline(L, color=color, linestyle="--", linewidth=1.0,
                    alpha=0.7, zorder=2,
                    label=f"prior $L$ = {label}")
        # mark p_hat
        p_hat = p_hat_by_prior[label]["p_hat"]
        avg_at = p_hat_by_prior[label]["avg_len_at_p_hat"]
        ax.plot(p_hat, avg_at, marker="*", markersize=18, color=color,
                 markeredgecolor="black", markeredgewidth=0.8, zorder=6)
        ax.annotate(f"p̂=p{p_hat}",
                     xy=(p_hat, avg_at), xytext=(6, 8),
                     textcoords="offset points",
                     fontsize=9, color=color, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="white", edgecolor=color, alpha=0.85))

    ax.set_xlabel(r"Candidate percentile $p$", fontsize=11)
    ax.set_ylabel(r"Predicted avg segment length (turns/segment)", fontsize=11)
    ax.set_xticks(P_GRID)
    ax.set_title(
        "Label-free best-$p$ via segment-length prior matching\n"
        "(MTB+, MiniLM-int8 encoder, no deployment labels used)",
        fontsize=11.5, pad=8,
    )
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.92)
    ax.tick_params(labelsize=9.5)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"figure_M_label_free_best_p.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
