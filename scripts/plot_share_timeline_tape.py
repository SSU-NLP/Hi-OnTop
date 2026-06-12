#!/usr/bin/env python3
"""Figure: timeline tape for a single SHARE conversation.

Same format as scripts/secom_swap/13_plot_timeline_tape.py (figure_C) but for
SHARE data with the 6 methods from run_share_boundary_comparison.py:
  GPT-5, Qwen3.5-27B, Qwen3.5-122B, Hi-OnTop int8 (p60, p70, p80).

Gold SHARE session boundaries shown as dashed vertical lines.

Output:
  outputs/experiments/2026-06-02_share_boundary_comparison/
      plots/timeline_tape_<idx>.{pdf,png}
  outputs/figures/figure_S_share_timeline_tape.{pdf,png}
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

EXP     = REPO / "outputs/experiments/2026-06-02_share_boundary_comparison"
OUT_DIR = EXP / "plots"
OUT_FIG = REPO / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(p: Path) -> dict:
    """Load segments.jsonl → {conv_id: boundary_indices}."""
    out = {}
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r   = json.loads(ln)
        cid = r["conversation_id"]
        bnd = []
        cursor = -1
        for i, seg in enumerate(r["segments"]):
            cursor += len(seg)
            if i < len(r["segments"]) - 1 and seg:
                bnd.append(cursor)
        out[cid] = bnd
    return out


def load_share_conv(idx: int) -> dict:
    """Return the idx-th conversation from SHARE test split.

    Returns dict with:
        conversation_id, sessions (List[List[str]]), gold_boundaries (List[int])
    """
    data = json.loads((REPO / "benchmarks/SHARE/data/test.json").read_text())
    for i, (pair_key, info) in enumerate(data.items()):
        if i != idx:
            continue
        sessions: list[list[str]] = []
        for sess_info in info.get("dialogue", []):
            turns = [
                f"{d['speaker']}: {d['text']}"
                for d in sess_info.get("dialogues", [])
                if d.get("text", "").strip()
            ]
            if turns:
                sessions.append(turns)
        gold_bnd: list[int] = []
        cursor = -1
        for j, sess in enumerate(sessions):
            cursor += len(sess)
            if j < len(sessions) - 1:
                gold_bnd.append(cursor)
        return {"conversation_id": pair_key, "sessions": sessions,
                "gold_boundaries": gold_bnd}
    raise IndexError(f"SHARE test split has fewer than {idx+1} conversations")


def compute_graded_scores(sessions: list[list[str]],
                           delta_star: float) -> tuple[np.ndarray, list[int]]:
    """Encode all turns, run HiOnTop across the full conversation.

    Returns (graded_scores array, gold session break indices).
    Same flat-conversation approach as run_share_boundary_comparison.py
    (single HiOnTop instance, not per-session reset).
    """
    enc  = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"})
    turns = [t for sess in sessions for t in sess]
    vecs  = enc.encode(turns, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    seg   = HiOnTop(dim=vecs.shape[1], delta_star=delta_star)

    graded: list[float] = []
    for v in vecs:
        seg.assign(v.astype(np.float64))
        hist = seg.history()
        graded.append(float(hist[-1]["graded_score"]))  # = δ_eff / δ*

    # session break positions (end of each session except last)
    session_breaks: list[int] = []
    cursor = -1
    for j, sess in enumerate(sessions):
        cursor += len(sess)
        if j < len(sessions) - 1:
            session_breaks.append(cursor)

    return np.array(graded), session_breaks


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv_index", type=int, default=0,
                    help="Index into SHARE test split (default: 0 = LOLA/NEFF, "
                         "4 = DR.SHIRLEY/LIP, 7 = KYLE/MIKE)")
    ap.add_argument("--delta_star_p70", type=float, default=0.6914,
                    help="Hi-OnTop int8 p70 δ* for SHARE test split "
                         "(computed in run_share_boundary_comparison.py)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"loading SHARE conv[{args.conv_index}]…", flush=True)
    conv = load_share_conv(args.conv_index)
    cid  = conv["conversation_id"]
    print(f"  conv_id: {cid}", flush=True)
    print(f"  sessions: {len(conv['sessions'])}  "
          f"turns: {sum(len(s) for s in conv['sessions'])}", flush=True)

    # load segment files
    seg_dir = EXP
    gpt5_bnd    = load_jsonl(seg_dir / "segments_gpt5.jsonl").get(cid, [])
    qwen27b_bnd = load_jsonl(seg_dir / "segments_qwen27b.jsonl").get(cid, [])
    qwen122b_bnd= load_jsonl(seg_dir / "segments_qwen122b.jsonl").get(cid, [])
    p60_bnd     = load_jsonl(seg_dir / "segments_int8_p60.jsonl").get(cid, [])
    p70_bnd     = load_jsonl(seg_dir / "segments_int8_p70.jsonl").get(cid, [])
    p80_bnd     = load_jsonl(seg_dir / "segments_int8_p80.jsonl").get(cid, [])

    gold_bnd    = conv["gold_boundaries"]

    print("computing Hi-OnTop graded scores…", flush=True)
    graded, session_breaks = compute_graded_scores(conv["sessions"],
                                                    args.delta_star_p70)
    n_turn = len(graded)
    print(f"  n_turn={n_turn}  session_breaks={session_breaks}", flush=True)

    # ── plot ──────────────────────────────────────────────────────────────────
    # 7 rows: score + gold + gpt5 + qwen122b + qwen27b + p60 + p70 + p80
    row_labels = [
        ("gold",     "Gold\nsessions",      "gray"),
        ("gpt5",     "GPT-5",               "#8E44AD"),
        ("qwen122b", "Qwen3.5-122B",        "#0D5E1F"),
        ("qwen27b",  "Qwen3.5-27B",         "#1F8E3F"),
        ("p60",      "Hi-OnTop\n(int8, p60)", "#5BA3D0"),
        ("p70",      "Hi-OnTop\n(int8, p70)", "#2980B9"),
        ("p80",      "Hi-OnTop\n(int8, p80)", "#1A5276"),
    ]
    bnd_map = {
        "gold":     gold_bnd,
        "gpt5":     gpt5_bnd,
        "qwen122b": qwen122b_bnd,
        "qwen27b":  qwen27b_bnd,
        "p60":      p60_bnd,
        "p70":      p70_bnd,
        "p80":      p80_bnd,
    }

    n_rows  = len(row_labels) + 1  # +1 for score heatmap
    heights = [3.2] + [1.0] * len(row_labels)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(7.5, 5.0), sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": 0.15},
    )
    ax_score = axes[0]
    row_axes = axes[1:]

    # ── top: graded score heatmap ──
    cmap = plt.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=0, vmax=2.0)  # graded_score=1 → threshold, 2 → 2×threshold
    ax_score.imshow(graded[None, :], cmap=cmap, norm=norm, aspect="auto",
                    extent=(-0.5, n_turn - 0.5, -0.5, 0.5))
    ax_score.set_yticks([])
    ax_score.set_ylabel("Hi-OnTop\n$\\delta_{eff}$",
                         rotation=0, ha="right", va="center", fontsize=8.5)
    ax_score.axhline(0.5,  color="black", linewidth=0.3)
    ax_score.axhline(-0.5, color="black", linewidth=0.3)
    # mark p70 boundaries with arrows
    for b in p70_bnd:
        ax_score.annotate("", xy=(b, 0.55), xytext=(b, 1.1),
                          arrowprops=dict(arrowstyle="->", color="#1A3A5C", lw=1.3),
                          annotation_clip=False)

    # ── rows: boundary strips ──
    for ax, (key, label, color) in zip(row_axes, row_labels):
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.2)
        ax.set_xlim(-0.5, n_turn - 0.5)
        ax.set_ylim(0, 1)
        for b in bnd_map[key]:
            ax.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                        color=color, alpha=0.85,
                                        linewidth=0.4, edgecolor="black"))

    # ── gold session break dashed lines on all panels ──
    for sb in session_breaks:
        for ax_ in list(axes):
            ax_.axvline(sb + 0.5, color="gray", linestyle="--",
                        linewidth=0.7, alpha=0.6)

    # ── highlight consensus: all 3 LLMs AND p70 agree ──
    consensus = set(gpt5_bnd) & set(qwen27b_bnd) & set(qwen122b_bnd) & set(p70_bnd)
    for b in consensus:
        for ax_ in row_axes:
            ax_.add_patch(plt.Rectangle((b + 0.05, 0.05), 0.9, 0.9,
                                         fill=False, edgecolor="gold",
                                         linewidth=1.5))

    row_axes[-1].set_xlabel("Turn index (all sessions concatenated)", fontsize=8.5)
    row_axes[-1].set_xticks(np.arange(0, n_turn, max(1, n_turn // 15)))
    row_axes[-1].tick_params(labelsize=8)

    # ── colorbar ──
    fig.subplots_adjust(right=0.87)
    cax = fig.add_axes([0.89, 0.50, 0.012, 0.38])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cb.set_ticklabels(["0", "0.5", "1.0\n(threshold)", "1.5", "2.0"])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("$g_t = \\delta_{eff}/\\delta^*$", fontsize=8)

    # ── legend ──
    handles = [
        mpatches.Patch(color="gray",    alpha=0.85, label="Gold (SHARE sessions)"),
        mpatches.Patch(color="#8E44AD", alpha=0.85, label="GPT-5"),
        mpatches.Patch(color="#0D5E1F", alpha=0.85, label="Qwen3.5-122B"),
        mpatches.Patch(color="#1F8E3F", alpha=0.85, label="Qwen3.5-27B"),
        mpatches.Patch(color="#5BA3D0", alpha=0.85, label="Hi-OnTop (int8, p60)"),
        mpatches.Patch(color="#2980B9", alpha=0.85, label="Hi-OnTop (int8, p70)"),
        mpatches.Patch(color="#1A5276", alpha=0.85, label="Hi-OnTop (int8, p80)"),
        mpatches.Patch(facecolor="none", edgecolor="gold", linewidth=1.5,
                       label="LLMs + p70 all agree"),
        mpatches.Patch(facecolor="none", edgecolor="gray", linestyle="--",
                       label="gold session"),
    ]
    ax_score.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, 1.55), ncol=3, fontsize=7.0,
                    frameon=False, handlelength=1.2, columnspacing=0.7)

    fig.suptitle(
        f"Timeline tape — SHARE: {cid}\n"
        f"({len(conv['sessions'])} sessions, {n_turn} turns)",
        fontsize=9.5, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 0.88, 0.95))

    # save to experiment plots dir
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"timeline_tape_{args.conv_index}.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"saved {p}", flush=True)

    # save as figure_S (canonical figure)
    for ext in ("pdf", "png"):
        p = OUT_FIG / f"figure_S_share_timeline_tape.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"saved {p}", flush=True)

    plt.close(fig)


if __name__ == "__main__":
    main()
