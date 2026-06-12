#!/usr/bin/env python3
"""Figure H-style F1 gap curves for LLM-distillation calibration on MTB+.

Structure mirrors Figure H (calib N convergence on DTS):
- 2 rows × 3 cols. Rows = gap-to-oracle, gap-to-sup. Cols = 3 encoders.
- 3 lines per panel = 3 LLM references (GPT-5, Qwen-27B, Qwen-122B-A10B).
- x = N (number of train-side δ_eff samples used for percentile calibration).
- y = F1 gap (computed on TEST).

Definitions:
- MTB+ 11 conv split 5 train / 6 test (conv-level, fixed seed=0).
- For each (encoder, LLM-ref):
    * Pool δ_eff per side (train, test, full).
    * oracle_F1 = max over p∈[60..80] of F1(test) where
      δ*_p = percentile_p(test_pool) → Hi-OnTop on test.
    * sup_F1 = F1(test) at p*_sup = argmax F1(train) where
      δ*_p = percentile_p(full_train_pool).
    * For each N in N_GRID, for each seed:
        - Subsample N indices from train_pool.
        - For each p in [60..80]: δ*_p = percentile_p(subsample) → boundaries
          on test → F1 vs LLM-ref(test).
        - argmax_p F1(test) = p-best at this (N, seed).
        - record F1@p-best.
    * Aggregate over seeds: mean ± std of (oracle_F1 - F1@p-best) and
      (sup_F1 - F1@p-best).

Output: outputs/figures/figure_P_distill_n_convergence_mtbp.{pdf,png}
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
                                    TextArea, VPacker)
from matplotlib.patches import Circle
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "outputs/experiments/2026-05-25_llm_distillation_calib"
MTBP_PATH = REPO / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"
RDIR = REPO / "benchmarks/SeCom/experiment/result/mtbp"
OUT_FIG = REPO / "outputs/figures"
OUT_EXP = REPO / "outputs/experiments/2026-05-25_dts_distill_n_convergence"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

ENCODERS = ("MiniLM-int8", "MiniLM", "MPNet")
LLM_REFS = {
    "GPT-5":              "gpt5seg",
    "Qwen3.5-27B":        "qwen27bseg",
    "Qwen3.5-122B-A10B":  "qwen35_122bseg",
}
# Figure-P palette — distinct family from Figure H (blue/orange/red).
# Use purple/teal/magenta for high contrast + bright (not dreary).
LLM_COLOR = {"GPT-5":              "#7B3FB7",   # purple
              "Qwen3.5-27B":        "#00ACC1",   # teal
              "Qwen3.5-122B-A10B":  "#E91E63"}   # magenta
P_GRID = list(range(5, 96))  # expanded: 5..95 (2026-05-26)
N_GRID = [3, 10, 30, 50, 75, 100, 150, 200]  # train samples for calibration
SEEDS = list(range(50))
M, RHO, A = 2, 0.7, 0.5
CONV_TOL = 0.005  # gap within ±tol of final-N gap → considered converged

# Converged best_p / δ* / F1 (full MTB+ pool, n=666) — from
# outputs/experiments/2026-05-25_llm_distillation_calib/results.json.
# Used as in-figure summary (user request 2026-05-26).
BEST_P_TABLE = {
    "MPNet": {
        "GPT-5":              (73, 0.5037, 0.920),
        "Qwen3.5-27B":        (71, 0.4854, 0.918),
        "Qwen3.5-122B-A10B":  (71, 0.4854, 0.917),
    },
    "MiniLM": {
        "GPT-5":              (72, 0.7638, 0.921),
        "Qwen3.5-27B":        (72, 0.7638, 0.922),
        "Qwen3.5-122B-A10B":  (72, 0.7638, 0.913),
    },
    "MiniLM-int8": {
        "GPT-5":              (72, 0.7678, 0.914),
        "Qwen3.5-27B":        (72, 0.7678, 0.916),
        "Qwen3.5-122B-A10B":  (68, 0.6511, 0.909),
    },
}
# Short labels for ref names in summary box
REF_SHORT = {
    "GPT-5":              "GPT-5",
    "Qwen3.5-27B":        "Q-27B",
    "Qwen3.5-122B-A10B":  "Q-122B",
}


def convergence_n(ns, means, tol=CONV_TOL):
    """Smallest N where mean gap stays within ±tol of final-N gap for
    all remaining N values. Returns (N_conv, final_gap) or (None, final)."""
    final = means[-1]
    for i in range(len(ns)):
        if all(abs(means[j] - final) <= tol for j in range(i, len(ns))):
            return int(ns[i]), float(final)
    return None, float(final)

# Conv-level split: deterministic random 5/6 from 11 with seed=0
SPLIT_SEED = 0
TRAIN_RATIO = 5  # train_conv = 5, test_conv = 6


def load_mtbp_conv_ids():
    cids = []
    with MTBP_PATH.open() as f:
        for line in f:
            cids.append(json.loads(line)["conversation_id"])
    return cids


def boundary_after_indices_from_segments(segments):
    bnd, cursor = [], -1
    for i, seg in enumerate(segments):
        cursor += len(seg)
        if i < len(segments) - 1 and len(seg) > 0:
            bnd.append(cursor)
    return bnd


def load_llm_segments(subdir):
    out = {}
    p = RDIR / subdir / "segments.jsonl"
    for ln in p.read_text().splitlines():
        if not ln.strip(): continue
        r = json.loads(ln)
        out[r["conversation_id"]] = boundary_after_indices_from_segments(r["segments"])
    return out


def compute_delta_eff_per_session(per_conv_emb):
    per_conv = {}
    for cid, sess_emb in per_conv_emb.items():
        sd = []
        for si, vecs in sess_emb:
            seg = HiOnTop(dim=vecs.shape[1], delta_star=1.0,
                          ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
            for v in vecs:
                seg.assign(v.astype(np.float64))
            deffs = np.array([float(h["delta_eff"]) for h in seg.history()])
            sd.append((si, len(vecs), deffs[1:]))
        per_conv[cid] = sd
    return per_conv


def pool_deff(per_conv_deff, cids):
    parts = []
    for cid in cids:
        for _, _, deff in per_conv_deff[cid]:
            parts.append(deff)
    return np.concatenate(parts) if parts else np.array([])


def hiontop_boundaries(per_conv_deff, cids, dstar):
    out = {}
    for cid in cids:
        sess = per_conv_deff[cid]
        bnd = []
        offset = 0
        n_sess = len(sess)
        for sess_i, (_, n_turns, deff) in enumerate(sess):
            for t in range(1, n_turns):
                if deff[t-1] >= dstar:
                    bnd.append(offset + t - 1)
            if sess_i < n_sess - 1:
                bnd.append(offset + n_turns - 1)
            offset += n_turns
        out[cid] = bnd
    return out


def f1(pred, gold):
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs)
    p = tp / len(ps); r = tp / len(gs)
    return 2*p*r/(p+r) if (p+r)>0 else 0.0


def mean_f1(per_conv_deff, llm_bnd, cids, dstar):
    hiontop = hiontop_boundaries(per_conv_deff, cids, dstar)
    return float(np.mean([f1(hiontop[c], llm_bnd[c]) for c in cids]))


def main():
    cache_results = OUT_EXP / "mtbp_gap_results.json"
    if cache_results.exists():
        print(f"[cache] loading {cache_results}", flush=True)
        results = json.loads(cache_results.read_text())
        _plot(results)
        return

    cids = load_mtbp_conv_ids()
    print(f"MTB+ n_conv = {len(cids)}", flush=True)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = list(rng.permutation(cids))
    train_cids = perm[:TRAIN_RATIO]
    test_cids = perm[TRAIN_RATIO:]
    print(f"  train_cids ({len(train_cids)}): {train_cids}", flush=True)
    print(f"  test_cids  ({len(test_cids)}): {test_cids}", flush=True)

    llm_bnd = {ref: load_llm_segments(sub) for ref, sub in LLM_REFS.items()}
    deff_by_enc = {}
    for enc in ENCODERS:
        cache_path = CACHE_DIR / f"emb_{enc}.pkl"
        per_conv_emb = pickle.loads(cache_path.read_bytes())
        deff_by_enc[enc] = compute_delta_eff_per_session(per_conv_emb)
        print(f"  {enc}: loaded", flush=True)

    results = {}
    for enc in ENCODERS:
        results[enc] = {}
        train_pool = pool_deff(deff_by_enc[enc], train_cids)
        test_pool = pool_deff(deff_by_enc[enc], test_cids)
        print(f"\n[{enc}] train_pool={train_pool.size}  test_pool={test_pool.size}",
              flush=True)

        for ref in LLM_REFS:
            # oracle_F1: max F1(test) over p, where δ*_p = percentile_p(test_pool)
            oracle_F1 = -1.0; oracle_p = None
            for p in P_GRID:
                d = float(np.percentile(test_pool, p))
                v = mean_f1(deff_by_enc[enc], llm_bnd[ref], test_cids, d)
                if v > oracle_F1:
                    oracle_F1, oracle_p = v, p
            # sup_F1: F1(test) at p* = argmax F1(train) where δ*_p = percentile_p(train_pool, full)
            sup_p, sup_train_F1 = None, -1.0
            for p in P_GRID:
                d = float(np.percentile(train_pool, p))
                v_train = mean_f1(deff_by_enc[enc], llm_bnd[ref], train_cids, d)
                if v_train > sup_train_F1:
                    sup_train_F1, sup_p = v_train, p
            # apply sup_p (calibrated on train) → eval on test
            sup_F1 = mean_f1(deff_by_enc[enc], llm_bnd[ref], test_cids,
                              float(np.percentile(train_pool, sup_p)))
            print(f"  {ref:22s}: oracle p={oracle_p} F1={oracle_F1:.4f}  "
                   f"sup p={sup_p} F1(test)={sup_F1:.4f}", flush=True)

            ref_res = dict(oracle_F1=oracle_F1, oracle_p=oracle_p,
                            sup_F1=sup_F1, sup_p=sup_p,
                            per_N=[])
            for N in N_GRID:
                f1_best_list = []
                p_best_list = []
                if N >= train_pool.size:
                    # use full train (deterministic)
                    pbest, fbest = None, -1.0
                    for p in P_GRID:
                        d = float(np.percentile(train_pool, p))
                        v = mean_f1(deff_by_enc[enc], llm_bnd[ref],
                                     test_cids, d)
                        if v > fbest:
                            fbest, pbest = v, p
                    f1_best_list.append(fbest); p_best_list.append(pbest)
                else:
                    for s in SEEDS:
                        rng2 = np.random.default_rng(s)
                        idx = rng2.choice(train_pool.size, size=N, replace=False)
                        sub = train_pool[idx]
                        pbest, fbest = None, -1.0
                        for p in P_GRID:
                            d = float(np.percentile(sub, p))
                            v = mean_f1(deff_by_enc[enc], llm_bnd[ref],
                                         test_cids, d)
                            if v > fbest:
                                fbest, pbest = v, p
                        f1_best_list.append(fbest)
                        p_best_list.append(pbest)
                gap_oracle = [oracle_F1 - x for x in f1_best_list]
                gap_sup = [sup_F1 - x for x in f1_best_list]
                ref_res["per_N"].append(dict(
                    N=N,
                    p_best_mean=float(np.mean(p_best_list)),
                    p_best_std=float(np.std(p_best_list)),
                    f1_best_mean=float(np.mean(f1_best_list)),
                    gap_oracle_mean=float(np.mean(gap_oracle)),
                    gap_oracle_std=float(np.std(gap_oracle)),
                    gap_sup_mean=float(np.mean(gap_sup)),
                    gap_sup_std=float(np.std(gap_sup)),
                ))
                print(f"     N={N:4d}: F1@pbest={np.mean(f1_best_list):.4f}±"
                       f"{np.std(f1_best_list):.4f}  "
                       f"gap_oracle={np.mean(gap_oracle):+.4f}  "
                       f"gap_sup={np.mean(gap_sup):+.4f}", flush=True)
            results[enc][ref] = ref_res

    (OUT_EXP / "mtbp_gap_results.json").write_text(json.dumps(results, indent=2))
    _plot(results)


def _plot(results):
    print("\n[plot] figure_P (Figure H style, EMNLP main quality)", flush=True)
    # Unified style with Figure H (see plot_calib_n_convergence._setup_rc).
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

    # Same figsize as Figure H main (2 rows × 3 cols).
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.5), sharex=True,
                              gridspec_kw={"hspace": 0.30})
    GAP_THRESHOLD = 0.005  # same as Figure H

    for ai, enc in enumerate(ENCODERS):
        ax_top = axes[0, ai]
        ax_bot = axes[1, ai]
        conv_marks_top = []   # for above-panel annotation
        for ref in LLM_REFS:
            color = LLM_COLOR[ref]
            per_N = results[enc][ref]["per_N"]
            ns = np.array([r["N"] for r in per_N])
            g_or = np.array([r["gap_oracle_mean"] for r in per_N])
            g_or_s = np.array([r["gap_oracle_std"] for r in per_N])
            g_su = np.array([r["gap_sup_mean"] for r in per_N])
            g_su_s = np.array([r["gap_sup_std"] for r in per_N])
            ax_top.plot(ns, g_or, "-o", color=color, label=ref, zorder=4)
            ax_top.fill_between(ns, g_or - g_or_s, g_or + g_or_s,
                                 color=color, alpha=0.16, linewidth=0, zorder=1)
            ax_bot.plot(ns, g_su, "-o", color=color, zorder=4)
            ax_bot.fill_between(ns, g_su - g_su_s, g_su + g_su_s,
                                 color=color, alpha=0.16, linewidth=0, zorder=1)
            n_or, _ = convergence_n(ns, g_or, tol=CONV_TOL)
            n_su, _ = convergence_n(ns, g_su, tol=CONV_TOL)
            if n_or is not None:
                ax_top.axvline(n_or, color=color, linestyle="--",
                                linewidth=0.8, alpha=0.75, zorder=3)
                conv_marks_top.append((ref, n_or))
            if n_su is not None:
                ax_bot.axvline(n_su, color=color, linestyle="--",
                                linewidth=0.8, alpha=0.75, zorder=3)
        for ax in (ax_top, ax_bot):
            ax.axhline(0, color="black", linestyle="-", linewidth=0.6, zorder=2)
        ax_top.axhline(GAP_THRESHOLD, color="#444", linestyle=":",
                        linewidth=0.7, alpha=0.7, zorder=2)
        ax_top.set_title(enc, fontsize=11, pad=6, fontweight="bold")
        # In-panel annotation box: colored bullet + ref name + N/p/δ*/F1
        tbl = BEST_P_TABLE[enc]
        if conv_marks_top:
            ref_to_n = {ref_: n for ref_, n in conv_marks_top}
            rows = []
            for ref in LLM_REFS:
                n_ = ref_to_n.get(ref, None)
                p_, d_, f_ = tbl[ref]
                n_str = f"$N{{=}}${n_}" if n_ is not None else r"$N{=}$—"
                bullet = DrawingArea(7, 7, 0, 0)
                bullet.add_artist(Circle((3.5, 3.5), 2.6,
                                            fc=LLM_COLOR[ref], ec="none"))
                txt = TextArea(
                    f"{REF_SHORT[ref]}: {n_str}, "
                    rf"$p{{=}}${p_}, $\delta^\ast{{=}}${d_:.3f}, "
                    f"$F_1{{=}}${f_:.3f}",
                    textprops=dict(fontsize=6.2, family="serif",
                                    color="black"),
                )
                rows.append(HPacker(children=[bullet, txt],
                                     pad=0, sep=2, align="center"))
            content = VPacker(children=rows, pad=0.5, sep=1.0, align="left")
            ab = AnchoredOffsetbox(loc="upper right", child=content,
                                     pad=0.22, borderpad=0.45, frameon=True)
            ab.patch.set_boxstyle("round,pad=0.22")
            ab.patch.set_facecolor("white")
            ab.patch.set_edgecolor("#444")
            ab.patch.set_linewidth(0.7)
            ab.patch.set_alpha(0.97)
            ab.set_zorder(10)
            ax_top.add_artist(ab)
        for ax in (ax_top, ax_bot):
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_xlim(0, 210)
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.tick_params(axis="x", labelbottom=True)
        ax_bot.set_xlabel(r"Calibration turns $N$", fontsize=10)
        if ai == 0:
            ax_top.set_ylabel("Gap to oracle", fontsize=10)
            ax_bot.set_ylabel("Gap to sup", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(LLM_REFS),
                frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.00),
                handlelength=1.8, columnspacing=1.6, handletextpad=0.5)
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"figure_P_distill_n_convergence_mtbp.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
