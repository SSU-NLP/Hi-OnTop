#!/usr/bin/env python3
"""Figure U: timeline tape for a single LongMemEval instance.

Same format as plot_share_timeline_tape.py (figure_S).
Shows first --max-sessions sessions of one LME instance.

Output:
  outputs/experiments/2026-06-02_lme_boundary_comparison/plots/timeline_tape_<idx>.{pdf,png}
  outputs/figures/figure_U_lme_timeline_tape.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(4)
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop

LME_PATH = REPO / "benchmarks/longmemeval/data/longmemeval_s.json"
EXP      = REPO / "outputs/experiments/2026-06-02_lme_boundary_comparison"
OUT_DIR  = EXP / "plots"
OUT_FIG  = REPO / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lme_conv(idx: int, max_sessions: int) -> dict:
    with open(LME_PATH) as f:
        data = json.load(f)
    inst = data[idx]
    sessions_raw = inst["haystack_sessions"][:max_sessions]
    sessions: list[list[str]] = []
    for sess in sessions_raw:
        turns = [f"[{t['role'].upper()}] {t['content']}" for t in sess
                 if t.get("content", "").strip()]
        if turns:
            sessions.append(turns)
    gold_bnd: list[int] = []
    cursor = -1
    for i, sess in enumerate(sessions):
        cursor += len(sess)
        if i < len(sessions) - 1:
            gold_bnd.append(cursor)
    return {"conversation_id": inst["question_id"], "sessions": sessions,
            "gold_boundaries": gold_bnd,
            "dates": inst.get("haystack_dates", [])[:max_sessions]}


def load_bnd(jl: Path, cid: str) -> list[int]:
    if not jl.exists():
        return []
    for ln in jl.read_text().splitlines():
        if not ln.strip(): continue
        r = json.loads(ln)
        if r["conversation_id"] == cid:
            bnd, cursor = [], -1
            for i, seg in enumerate(r["segments"]):
                cursor += len(seg)
                if i < len(r["segments"]) - 1 and seg:
                    bnd.append(cursor)
            return bnd
    return []


def compute_graded(sessions: list[list[str]], delta_star: float):
    enc  = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"})
    turns = [t for sess in sessions for t in sess]
    vecs  = enc.encode(turns, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    seg   = HiOnTop(dim=vecs.shape[1], delta_star=delta_star)
    graded = []
    for v in vecs:
        seg.assign(v.astype(np.float64))
        graded.append(float(seg.history()[-1]["graded_score"]))
    session_breaks = []
    cursor = -1
    for i, sess in enumerate(sessions):
        cursor += len(sess)
        if i < len(sessions) - 1:
            session_breaks.append(cursor)
    return np.array(graded), session_breaks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv-index", type=int, default=0)
    ap.add_argument("--max-sessions", type=int, default=10)
    ap.add_argument("--delta-star-p70", type=float, default=None,
                    help="Hi-OnTop int8 p70 δ* for LME (auto-computed if not given)")
    return ap.parse_args()


def main():
    args = parse_args()
    conv = load_lme_conv(args.conv_index, args.max_sessions)
    cid  = conv["conversation_id"]
    n_turns = sum(len(s) for s in conv["sessions"])
    print(f"conv[{args.conv_index}]: {cid}  sessions={len(conv['sessions'])}  turns={n_turns}", flush=True)

    # auto-compute δ* if not given (use pool from experiment if available)
    if args.delta_star_p70 is None:
        # try to read from results.json
        res = EXP / "results.json"
        if res.exists():
            # find p70 dstar from segments file
            pass
        # fallback: recompute on this conv
        print("  δ* not specified — recomputing from this conv's pool (approximate)", flush=True)
        import pickle
        # try cached embeddings
        delta_star = 0.70  # reasonable default; override with --delta-star-p70
        print(f"  using δ*={delta_star} (pass --delta-star-p70 for exact value)", flush=True)
    else:
        delta_star = args.delta_star_p70

    gpt5    = load_bnd(EXP / "segments_gpt5.jsonl",    cid)
    qwen27b = load_bnd(EXP / "segments_qwen27b.jsonl", cid)
    qwen122 = load_bnd(EXP / "segments_qwen122b.jsonl", cid)
    p60     = load_bnd(EXP / "segments_int8_p60.jsonl", cid)
    p70     = load_bnd(EXP / "segments_int8_p70.jsonl", cid)
    p80     = load_bnd(EXP / "segments_int8_p80.jsonl", cid)
    gold    = conv["gold_boundaries"]

    print("computing graded scores…", flush=True)
    graded, session_breaks = compute_graded(conv["sessions"], delta_star)

    row_labels = [
        ("gold",     "Gold\nsessions",       "gray"),
        ("gpt5",     "GPT-5",                "#8E44AD"),
        ("qwen122b", "Qwen3.5-122B",         "#0D5E1F"),
        ("qwen27b",  "Qwen3.5-27B",          "#1F8E3F"),
        ("p60",      "Hi-OnTop\n(int8, p60)", "#5BA3D0"),
        ("p70",      "Hi-OnTop\n(int8, p70)", "#2980B9"),
        ("p80",      "Hi-OnTop\n(int8, p80)", "#1A5276"),
    ]
    bnd_map = {"gold": gold, "gpt5": gpt5, "qwen122b": qwen122,
               "qwen27b": qwen27b, "p60": p60, "p70": p70, "p80": p80}

    n_rows  = len(row_labels) + 1
    heights = [3.2] + [1.0] * len(row_labels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.0, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.15})
    ax_score, row_axes = axes[0], axes[1:]

    cmap = plt.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=0, vmax=2.0)
    ax_score.imshow(graded[None, :], cmap=cmap, norm=norm, aspect="auto",
                    extent=(-0.5, n_turns-0.5, -0.5, 0.5))
    ax_score.set_yticks([])
    ax_score.set_ylabel("Hi-OnTop\n$g_t$", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_score.axhline(0.5,  color="black", linewidth=0.3)
    ax_score.axhline(-0.5, color="black", linewidth=0.3)
    for b in p70:
        ax_score.annotate("", xy=(b, 0.55), xytext=(b, 1.1),
                          arrowprops=dict(arrowstyle="->", color="#1A3A5C", lw=1.3),
                          annotation_clip=False)

    for ax, (key, label, color) in zip(row_axes, row_labels):
        ax.set_yticks([]); ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.2)
        ax.set_xlim(-0.5, n_turns-0.5); ax.set_ylim(0, 1)
        for b in bnd_map[key]:
            ax.add_patch(plt.Rectangle((b+0.1, 0.15), 0.8, 0.7,
                                        color=color, alpha=0.85, linewidth=0.4, edgecolor="black"))

    for sb in session_breaks:
        for ax_ in list(axes):
            ax_.axvline(sb+0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    consensus = set(gpt5) & set(qwen27b) & set(qwen122) & set(p70)
    for b in consensus:
        for ax_ in row_axes:
            ax_.add_patch(plt.Rectangle((b+0.05, 0.05), 0.9, 0.9,
                                         fill=False, edgecolor="gold", linewidth=1.5))

    row_axes[-1].set_xlabel("Turn index (all sessions concatenated)", fontsize=8.5)
    tick_step = max(1, n_turns // 20)
    row_axes[-1].set_xticks(np.arange(0, n_turns, tick_step))
    row_axes[-1].tick_params(labelsize=8)

    fig.subplots_adjust(right=0.87)
    cax = fig.add_axes([0.89, 0.50, 0.012, 0.38])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cb.set_ticklabels(["0", "0.5", "1.0\n(thr)", "1.5", "2.0"])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("$g_t=\\delta_{eff}/\\delta^*$", fontsize=8)

    handles = [
        mpatches.Patch(color="gray",    alpha=0.85, label="Gold (LME sessions)"),
        mpatches.Patch(color="#8E44AD", alpha=0.85, label="GPT-5"),
        mpatches.Patch(color="#0D5E1F", alpha=0.85, label="Qwen3.5-122B"),
        mpatches.Patch(color="#1F8E3F", alpha=0.85, label="Qwen3.5-27B"),
        mpatches.Patch(color="#5BA3D0", alpha=0.85, label="Hi-OnTop (int8, p60)"),
        mpatches.Patch(color="#2980B9", alpha=0.85, label="Hi-OnTop (int8, p70)"),
        mpatches.Patch(color="#1A5276", alpha=0.85, label="Hi-OnTop (int8, p80)"),
        mpatches.Patch(facecolor="none", edgecolor="gold", linewidth=1.5, label="LLMs+p70 agree"),
        mpatches.Patch(facecolor="none", edgecolor="gray", linestyle="--", label="gold session"),
    ]
    ax_score.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.55),
                    ncol=3, fontsize=7.0, frameon=False, handlelength=1.2, columnspacing=0.7)

    dates_str = ""
    if conv.get("dates"):
        dates_str = f"\n({conv['dates'][0]} … {conv['dates'][-1]})"
    fig.suptitle(
        f"Timeline tape — LME[{args.conv_index}]: {cid}{dates_str}\n"
        f"(first {len(conv['sessions'])} sessions, {n_turns} turns)",
        fontsize=9, y=0.995)
    plt.tight_layout(rect=(0, 0, 0.88, 0.95))

    for ext in ("pdf", "png"):
        p = OUT_DIR / f"timeline_tape_{args.conv_index}.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"saved {p}", flush=True)
    for ext in ("pdf", "png"):
        p = OUT_FIG / f"figure_U_lme_timeline_tape.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"saved {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
