"""Figure: per-turn graded score heatmap + boundary overlays for a single conversation.

Picks a conversation from MTB+ and visualizes:
1. (top row) Hi-OnTop graded score as a horizontal heatmap (one cell per turn).
2. (next rows) boundary positions from baseline gpt-4o-mini and ours.

Highlights where the two methods agree / disagree and shows the calibrated
continuous signal underneath ours' binary decisions.

Output: ``plots/timeline_tape.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXP = REPO_ROOT / "outputs/experiments/2026-05-21_v413_secom_swap"
OUT_DIR = EXP / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(p: Path):
    with p.open() as f:
        return [json.loads(line) for line in f]


def boundary_after_indices(segments: list[list[str]]) -> list[int]:
    """Returns the 0-based turn indices AFTER which a boundary occurs
    (i.e. the last turn of each segment except the final segment).

    For visualization purposes ALL utterances are concatenated across the
    conversation's sessions in order, so the indices are global.
    """
    bnd: list[int] = []
    cursor = -1
    for i, seg in enumerate(segments):
        cursor += len(seg)
        if i < len(segments) - 1 and len(seg) > 0:
            bnd.append(cursor)
    return bnd


def get_v413_graded_scores(conv: dict, delta_star: float):
    """Re-run Hi-OnTop on the conversation to harvest per-turn graded_score.

    Returns: (flat_scores, session_break_indices, ours_boundaries_global).
    Each session is processed with a fresh segmenter (same as adapter).
    """
    from hi_ontop.hi_ontop import HiOnTop

    # int8 ONNX encoder to match the int8 p70 boundary overlay in the figure.
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"})
    flat_scores: list[float] = []
    session_breaks: list[int] = []
    ours_bnd: list[int] = []
    offset = 0
    for sess in conv["sessions"]:
        if not sess:
            continue
        vecs = encoder.encode(sess, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        seg = HiOnTop(dim=384, delta_star=delta_star)
        for t, (v, _) in enumerate(zip(vecs, sess)):
            _, is_bnd = seg.assign(v.astype(np.float64))
            flat_scores.append(seg.last_graded_score)
            if is_bnd:
                ours_bnd.append(offset + t)
        offset += len(sess)
        if offset < sum(len(s) for s in conv["sessions"]):
            session_breaks.append(offset - 1)
    return np.array(flat_scores), session_breaks, ours_bnd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtbp_path",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"))
    ap.add_argument("--gpt5_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/gpt5seg/segments.jsonl"))
    ap.add_argument("--qwen122b_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/qwen35_122bseg/segments.jsonl"))
    ap.add_argument("--qwen_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/qwen27bseg/segments.jsonl"))
    ap.add_argument("--gpt4omini_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/gpt4omini/segments_fixed.jsonl"))
    ap.add_argument("--ours_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/ours_int8_p70/segments.jsonl"))
    ap.add_argument("--ours_p60_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/ours_int8_p60/segments.jsonl"))
    ap.add_argument("--ours_p80_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/ours_int8_p80/segments.jsonl"))
    ap.add_argument("--delta_star", type=float, default=0.7049,
                    help="int8 p70 MTB+ δ* (default).")
    ap.add_argument("--conv_index", type=int, default=3,
                    help="Which conversation from MTB+ to visualize (0-10).")
    args = ap.parse_args()

    mtbp = load_jsonl(Path(args.mtbp_path))
    gpt5_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.gpt5_segs))}
    qwen122b_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.qwen122b_segs))}
    qwen_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.qwen_segs))}
    gpt_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.gpt4omini_segs))}
    ours_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.ours_segs))}
    ours_p60_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.ours_p60_segs))}
    ours_p80_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.ours_p80_segs))}

    conv = mtbp[args.conv_index]
    cid = conv["conversation_id"]
    print(f"plotting conv {args.conv_index}: {cid}")

    flat_scores, session_breaks, ours_bnd_runtime = get_v413_graded_scores(conv, args.delta_star)
    n_turn = len(flat_scores)

    gpt5_bnd = boundary_after_indices(gpt5_data[cid]["segments"])
    qwen122b_bnd = boundary_after_indices(qwen122b_data[cid]["segments"])
    qwen_bnd = boundary_after_indices(qwen_data[cid]["segments"])
    gpt_bnd = boundary_after_indices(gpt_data[cid]["segments"])
    ours_bnd_saved = boundary_after_indices(ours_data[cid]["segments"])
    ours_p60_bnd = boundary_after_indices(ours_p60_data[cid]["segments"])
    ours_p80_bnd = boundary_after_indices(ours_p80_data[cid]["segments"])

    # EMNLP two-column friendly: 7in wide. 8 rows (score + 7 method rows).
    fig, (ax_score, ax_gpt5, ax_qwen122b, ax_qwen, ax_gpt,
            ax_p60, ax_ours, ax_p80) = plt.subplots(
        8, 1, figsize=(7.0, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       "hspace": 0.18},
    )

    # Top: graded score heatmap
    cmap = plt.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=0, vmax=2.0)
    ax_score.imshow(flat_scores[None, :], cmap=cmap, norm=norm, aspect="auto",
                     extent=(-0.5, n_turn - 0.5, -0.5, 0.5))
    ax_score.set_yticks([])
    ax_score.set_ylabel("Hi-OnTop\n$g_t$",
                         rotation=0, ha="right", va="center", fontsize=8.5)
    ax_score.axhline(0.5, color="black", linewidth=0.3)
    ax_score.axhline(-0.5, color="black", linewidth=0.3)
    # threshold annotations
    thresh_levels = [(0.7, "very_weak"), (1.0, "weak"), (1.3, "normal/strong")]

    # Mark ours boundaries with arrows on top row
    for b in ours_bnd_runtime:
        ax_score.annotate("", xy=(b, 0.55), xytext=(b, 1.1),
                          arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.3),
                          annotation_clip=False)

    # Session boundaries (dashed verticals)
    for sb in session_breaks:
        for ax_ in (ax_score, ax_gpt5, ax_qwen122b, ax_qwen, ax_gpt,
                      ax_p60, ax_ours, ax_p80):
            ax_.axvline(sb + 0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.65)

    # Row 1: GPT-5 boundary row
    ax_gpt5.set_yticks([])
    ax_gpt5.set_ylabel("GPT-5", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_gpt5.set_xlim(-0.5, n_turn - 0.5)
    ax_gpt5.set_ylim(0, 1)
    for b in gpt5_bnd:
        ax_gpt5.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                         color="#8E44AD", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Row 2: Qwen3.5-122B-A10B boundary row
    ax_qwen122b.set_yticks([])
    ax_qwen122b.set_ylabel("Qwen3.5-122B-A10B", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_qwen122b.set_xlim(-0.5, n_turn - 0.5)
    ax_qwen122b.set_ylim(0, 1)
    for b in qwen122b_bnd:
        ax_qwen122b.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                             color="#0D5E1F", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Row 2: Qwen3.5-27B boundary row
    ax_qwen.set_yticks([])
    ax_qwen.set_ylabel("Qwen3.5-27B", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_qwen.set_xlim(-0.5, n_turn - 0.5)
    ax_qwen.set_ylim(0, 1)
    for b in qwen_bnd:
        ax_qwen.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                         color="#C0504D", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Row 2: GPT-4o-mini boundary row
    ax_gpt.set_yticks([])
    ax_gpt.set_ylabel("GPT-4o-mini", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_gpt.set_xlim(-0.5, n_turn - 0.5)
    ax_gpt.set_ylim(0, 1)
    for b in gpt_bnd:
        ax_gpt.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                        color="#9B59B6", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Row 5: Hi-OnTop (p60)
    ax_p60.set_yticks([])
    ax_p60.set_ylabel("Hi-OnTop (p60)", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_p60.set_xlim(-0.5, n_turn - 0.5)
    ax_p60.set_ylim(0, 1)
    for b in ours_p60_bnd:
        ax_p60.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                         color="#7BAEDB", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Row 6: Hi-OnTop (p70 — main)
    ax_ours.set_yticks([])
    ax_ours.set_ylabel("Hi-OnTop (p70)", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_ours.set_xlim(-0.5, n_turn - 0.5)
    ax_ours.set_ylim(0, 1)
    for b in ours_bnd_saved:
        ax_ours.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                          color="#4F81BD", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Row 7: Hi-OnTop (p80)
    ax_p80.set_yticks([])
    ax_p80.set_ylabel("Hi-OnTop (p80)", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_p80.set_xlim(-0.5, n_turn - 0.5)
    ax_p80.set_ylim(0, 1)
    for b in ours_p80_bnd:
        ax_p80.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                         color="#2E5C8A", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Highlight pairwise agreement turns (only intersection of all LLM methods + Ours p70)
    agree = set(gpt5_bnd) & set(qwen122b_bnd) & set(qwen_bnd) & set(gpt_bnd) & set(ours_bnd_saved)
    for b in agree:
        for ax_ in (ax_gpt5, ax_qwen122b, ax_qwen, ax_gpt, ax_ours):
            ax_.add_patch(plt.Rectangle((b + 0.05, 0.05), 0.9, 0.9,
                                          fill=False, edgecolor="gold", linewidth=1.5))

    ax_p80.set_xlabel("Turn index (all sessions)", fontsize=8.5)
    ax_p80.set_xticks(np.arange(0, n_turn, 10))
    ax_p80.tick_params(labelsize=8)

    # Colorbar
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.45, 0.012, 0.42])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 0.7, 1.0, 1.3, 2.0])
    cbar.set_ticklabels(["0", "0.7\nv.weak", "1.0\nweak/normal", "1.3\nstrong", "2.0"])
    cbar.ax.tick_params(labelsize=7)

    handles = [
        mpatches.Patch(color="#8E44AD", alpha=0.85, label="GPT-5"),
        mpatches.Patch(color="#0D5E1F", alpha=0.85, label="Qwen3.5-122B-A10B"),
        mpatches.Patch(color="#C0504D", alpha=0.85, label="Qwen3.5-27B"),
        mpatches.Patch(color="#9B59B6", alpha=0.85, label="GPT-4o-mini"),
        mpatches.Patch(color="#7BAEDB", alpha=0.85, label="Hi-OnTop (int8, p60)"),
        mpatches.Patch(color="#4F81BD", alpha=0.85, label="Hi-OnTop (int8, p70)"),
        mpatches.Patch(color="#2E5C8A", alpha=0.85, label="Hi-OnTop (int8, p80)"),
        mpatches.Patch(facecolor="none", edgecolor="gold", linewidth=1.5,
                          label="all agree (LLMs $+$ p70)"),
        mpatches.Patch(facecolor="none", edgecolor="gray", linestyle="--", label="session"),
    ]
    ax_score.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.42),
                     ncol=3, fontsize=7.2, frameon=False, handlelength=1.2,
                     columnspacing=0.8)

    plt.tight_layout(rect=(0, 0, 0.89, 0.96))

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"timeline_tape_conv{args.conv_index}.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
