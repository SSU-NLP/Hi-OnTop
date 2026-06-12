#!/usr/bin/env python3
"""Label-free best-p calibration via LLM-distillation.

For each (encoder ∈ {MPNet, int8}, LLM-reference ∈ {GPT-5, Qwen-27B, Qwen-122B-A10B}):
1. Compute MTB+ δ_eff (per session) with that encoder.
2. Sweep p ∈ {60, 61, ..., 80} (21 values).
3. For each p: δ*_p = percentile_p(pool), apply to per-session δ_eff → boundary indices.
4. Compute pairwise boundary F1 vs LLM-reference's saved segments.jsonl.
5. Pick p̂(encoder, llm-ref) = argmax_p mean_F1 across 11 conv.

산출:
- outputs/experiments/2026-05-25_llm_distillation_calib/results.json
- outputs/experiments/2026-05-25_llm_distillation_calib/sweep_<encoder>_<ref>.csv
- outputs/figures/figure_O_llm_distillation_best_p.{pdf,png}
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

# Limit thread contention with other claude/node sessions (user request 2026-05-25).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(4)
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

MTBP_PATH = REPO / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"
RDIR = REPO / "benchmarks/SeCom/experiment/result/mtbp"

OUT_EXP = REPO / "outputs/experiments/2026-05-25_llm_distillation_calib"
OUT_FIG = REPO / "outputs/figures"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

P_GRID = list(range(5, 96))  # 5..95 inclusive, step 1 (expanded 2026-05-26)
M, RHO, A = 2, 0.7, 0.5

ENCODERS = {
    # MiniLM-int8 first (faster, lower contention risk)
    "MiniLM-int8": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "backend": "onnx",
        "file_name": "onnx/model_quint8_avx2.onnx",
    },
    "MiniLM": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "backend": "default",
    },
    "MPNet": {
        "model": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "backend": "default",
    },
}

LLM_REFS = {
    "GPT-5":              "gpt5seg",
    "Qwen3.5-27B":        "qwen27bseg",
    "Qwen3.5-122B-A10B":  "qwen35_122bseg",
}


def load_mtbp():
    with MTBP_PATH.open() as f:
        return [json.loads(line) for line in f]


def make_encoder(cfg):
    if cfg["backend"] == "onnx":
        return SentenceTransformer(
            cfg["model"], backend="onnx",
            model_kwargs={"provider": "CPUExecutionProvider",
                          "file_name": cfg["file_name"]})
    return SentenceTransformer(cfg["model"], device="cpu")


def encode_mtbp(encoder, mtbp, cache_path: Path):
    """Returns {conv_id: list of (session_idx, np.ndarray embeddings)}.
    Caches to disk to avoid re-encoding on rerun."""
    if cache_path.exists():
        print(f"  using cached embeddings {cache_path.name}", flush=True)
        return pickle.loads(cache_path.read_bytes())
    per_conv = {}
    for conv in mtbp:
        cid = conv["conversation_id"]
        sess_emb = []
        for si, sess in enumerate(conv["sessions"]):
            if not sess:
                continue
            vecs = encoder.encode(sess, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            sess_emb.append((si, vecs))
        per_conv[cid] = sess_emb
        print(f"    encoded conv {cid}: {len(sess_emb)} sessions", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(per_conv))
    print(f"  cached embeddings → {cache_path.name}", flush=True)
    return per_conv


def compute_delta_eff_from_embeddings(per_conv_emb):
    """Returns {conv_id: list of (session_idx, n_turns, np.ndarray δ_eff[1:])}."""
    per_conv = {}
    for cid, sess_emb in per_conv_emb.items():
        sess_deff = []
        for si, vecs in sess_emb:
            seg = HiOnTop(dim=vecs.shape[1], delta_star=1.0,
                          ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
            for v in vecs:
                seg.assign(v.astype(np.float64))
            deffs = np.array([float(h["delta_eff"]) for h in seg.history()])
            sess_deff.append((si, len(vecs), deffs[1:]))
        per_conv[cid] = sess_deff
    return per_conv


def hiontop_global_boundary_indices(per_conv_deff, dstar):
    """For each conv, return list of GLOBAL turn indices that are boundaries
    (i.e. boundary AFTER that turn).

    Matches boundary_after_indices() convention used in figure F:
    Each session is segmented independently → cross-session transitions are
    implicit boundaries (last turn of non-final session).
    Within session: boundary at turn t (1..n-1) iff δ_eff[t-1] >= δ*.
    """
    out = {}
    for cid, sess_deff in per_conv_deff.items():
        bnd = []
        offset = 0
        n_sess = len(sess_deff)
        for sess_i, (si, n_turns, deff) in enumerate(sess_deff):
            for t_local in range(1, n_turns):
                if deff[t_local - 1] >= dstar:
                    bnd.append(offset + t_local - 1)
            # cross-session boundary at end of this session (except final session)
            if sess_i < n_sess - 1:
                bnd.append(offset + n_turns - 1)
            offset += n_turns
        out[cid] = bnd
    return out


def boundary_after_indices_from_segments(segments):
    """Same as figure F: indices of last turn of each segment, except final."""
    bnd, cursor = [], -1
    for i, seg in enumerate(segments):
        cursor += len(seg)
        if i < len(segments) - 1 and len(seg) > 0:
            bnd.append(cursor)
    return bnd


def pairwise_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs)
    p = tp / max(1, len(ps))
    r = tp / max(1, len(gs))
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def load_llm_segments(subdir):
    """Returns {conv_id: list of boundary global indices}."""
    p = RDIR / subdir / "segments.jsonl"
    out = {}
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        out[r["conversation_id"]] = boundary_after_indices_from_segments(r["segments"])
    return out


def main() -> None:
    print(f"[1/4] loading MTB+ + {len(LLM_REFS)} LLM segments", flush=True)
    mtbp = load_mtbp()
    llm_bnd = {ref: load_llm_segments(sub) for ref, sub in LLM_REFS.items()}
    cids = sorted(b["conversation_id"] for b in mtbp)

    all_results = {}
    sweep_cache = {}  # for plotting

    for enc_name, enc_cfg in ENCODERS.items():
        print(f"\n[2/4] encoding MTB+ with {enc_name}", flush=True)
        cache = OUT_EXP / f"emb_{enc_name}.pkl"
        if not cache.exists():
            enc = make_encoder(enc_cfg)
            per_conv_emb = encode_mtbp(enc, mtbp, cache)
            del enc
        else:
            per_conv_emb = encode_mtbp(None, mtbp, cache)
        per_conv_deff = compute_delta_eff_from_embeddings(per_conv_emb)
        del per_conv_emb

        # pool δ_eff across all sessions
        pool = np.concatenate([d for sess in per_conv_deff.values()
                                 for _, _, d in sess])
        print(f"  pool size: {pool.size}  mean={pool.mean():.4f} "
               f"std={pool.std():.4f}", flush=True)

        # sweep
        print(f"[3/4] sweep p={P_GRID[0]}..{P_GRID[-1]} (step 1)", flush=True)
        sweep_rows = []
        for p in P_GRID:
            dstar = float(np.percentile(pool, p))
            hiontop_bnd = hiontop_global_boundary_indices(per_conv_deff, dstar)
            row = {"p": p, "dstar": dstar}
            for ref, llm_b in llm_bnd.items():
                f1s = [pairwise_f1(hiontop_bnd[c], llm_b[c]) for c in cids]
                row[f"f1_vs_{ref}"] = float(np.mean(f1s))
            sweep_rows.append(row)
        sweep_cache[enc_name] = sweep_rows

        # pick best-p per LLM-ref
        cell_results = {}
        for ref in LLM_REFS:
            best = max(sweep_rows, key=lambda r: r[f"f1_vs_{ref}"])
            cell_results[ref] = {
                "best_p": best["p"],
                "dstar_at_best": best["dstar"],
                "f1_at_best": best[f"f1_vs_{ref}"],
            }
        all_results[enc_name] = {
            "pool_mean": float(pool.mean()),
            "pool_std": float(pool.std()),
            "sweep": sweep_rows,
            "best_p_per_ref": cell_results,
        }

        # summary
        print(f"  → best-p per LLM ref ({enc_name}):", flush=True)
        for ref, info in cell_results.items():
            print(f"     vs {ref:20s}: p̂=p{info['best_p']}  "
                   f"F1={info['f1_at_best']:.4f}  δ*={info['dstar_at_best']:.4f}",
                   flush=True)

    (OUT_EXP / "results.json").write_text(json.dumps(all_results, indent=2))

    # ── Figure: 1×N grid (N=#encoders), per-encoder panel with 3 ref curves ──
    print(f"\n[4/4] plotting figure_O", flush=True)
    n_enc = len(sweep_cache)
    fig, axes = plt.subplots(1, n_enc, figsize=(4.6 * n_enc, 4.6), sharey=True)
    if n_enc == 1:
        axes = [axes]
    ref_colors = {"GPT-5": "#8E44AD",
                   "Qwen3.5-27B": "#1F8E3F",
                   "Qwen3.5-122B-A10B": "#0D5E1F"}
    for ax, (enc_name, sweep) in zip(axes, sweep_cache.items()):
        ps = [r["p"] for r in sweep]
        for ref, color in ref_colors.items():
            f1s = [r[f"f1_vs_{ref}"] for r in sweep]
            ax.plot(ps, f1s, marker="o", linewidth=1.6, markersize=4,
                     color=color, label=ref, zorder=4)
            best = max(sweep, key=lambda r: r[f"f1_vs_{ref}"])
            ax.plot(best["p"], best[f"f1_vs_{ref}"], marker="*",
                     markersize=16, color=color, markeredgecolor="black",
                     markeredgewidth=0.6, zorder=6)
            ax.annotate(f"p{best['p']}",
                         xy=(best["p"], best[f"f1_vs_{ref}"]),
                         xytext=(4, 8), textcoords="offset points",
                         fontsize=8, color=color, fontweight="bold")
        ax.set_xlabel("Hi-OnTop percentile $p$", fontsize=10)
        ax.set_ylabel("Pairwise boundary F1 vs LLM-ref", fontsize=10)
        ax.set_title(f"Encoder: {enc_name}", fontsize=10.5, pad=4)
        ax.set_xticks([60, 65, 70, 75, 80])
        ax.tick_params(labelsize=8.5)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="lower center", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Label-free best-$p$ via LLM distillation — pick $p$ maximizing pairwise "
        "boundary F1 vs strong LLM segmenter",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"figure_O_llm_distillation_best_p.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)

    # final summary table
    print("\n" + "=" * 70, flush=True)
    print("LABEL-FREE BEST-P via LLM distillation — summary", flush=True)
    print("=" * 70, flush=True)
    for enc_name, info in all_results.items():
        print(f"\n[{enc_name}]")
        for ref, r in info["best_p_per_ref"].items():
            print(f"  vs {ref:22s}  → p̂=p{r['best_p']:<2d}  "
                   f"F1={r['f1_at_best']:.4f}  δ*={r['dstar_at_best']:.4f}")


if __name__ == "__main__":
    main()
