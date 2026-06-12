"""Figure F: N-method × N-method boundary agreement heatmap.

Each cell = boundary F1 between two methods (treating one as gold, the
other as pred; symmetric). Per-conversation F1 averaged.

Includes ALL 17 methods: 4 algorithmic + 1 supervised + 6 LLM + 6 Hi-OnTop.

Output: ``plots/boundary_agreement.{pdf,png}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RDIR = REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp"
OUT_DIR = REPO_ROOT / "outputs/experiments/2026-05-21_v413_secom_swap/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Order aligned with downstream_task.md (2026-05-24).
# Order: LLM top-left → algorithmic unsup → sup → Hi-OnTop (ours) bottom-right.
METHODS = [
    # LLM — top-left (GPT4Score desc)
    ("gpt5seg",          "GPT-5"),
    ("qwen35_122bseg",   "Qwen3.5-122B-A10B"),
    ("qwen27bseg",       "Qwen3.5-27B"),
    ("gpt4omini",        "GPT-4o-mini"),
    ("qwen35_4bseg",     "Qwen3.5-4B"),
    ("llama32_3bseg",    "Llama3.2-3B"),
    ("ministral3_3bseg", "Mistral3-3B"),
    ("qwen35_2bseg",     "Qwen3.5-2B"),
    # algorithmic unsupervised
    ("texttiling",       "TextTiling"),
    ("graphseg",         "GraphSeg"),
    ("greedyseg",        "GreedySeg"),
    ("csm",              "CSM"),
    # supervised
    ("roberta",          "RoBERTa"),
    # Hi-OnTop (ours) — bottom-right
    ("ours_p60",         r"Hi-OnTop (MPNet, $p_{60}$)"),
    ("ours_p70",         r"Hi-OnTop (MPNet, $p_{70}$)"),
    ("ours_hiontop",      r"Hi-OnTop (MPNet, $p_{80}$)"),
    ("ours_int8_p60",    r"Hi-OnTop (int8, $p_{60}$)"),
    ("ours_int8_p70",    r"Hi-OnTop (int8, $p_{70}$)"),
    ("ours_int8_p80",    r"Hi-OnTop (int8, $p_{80}$)"),
]


def boundary_indices(segments):
    bnd = []; c = -1
    for i, s in enumerate(segments):
        c += len(s)
        if i < len(segments) - 1 and len(s) > 0:
            bnd.append(c)
    return bnd


def load_per_conv(path):
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        out[r.get("conversation_id")] = boundary_indices(r.get("segments", []))
    return out


def f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs)
    p = tp / max(1, len(ps)); r = tp / max(1, len(gs))
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    boundaries = {}
    for sub, _ in METHODS:
        boundaries[sub] = load_per_conv(RDIR / sub / "segments.jsonl")
        print(f"{sub}: {len(boundaries[sub])} conv loaded")
    common = set.intersection(*[set(b.keys()) for b in boundaries.values()])
    print(f"common conversations: {len(common)}")

    n = len(METHODS)
    mat = np.zeros((n, n))
    for i, (a, _) in enumerate(METHODS):
        for j, (b, _) in enumerate(METHODS):
            f1s = [f1(boundaries[a][c], boundaries[b][c]) for c in common]
            mat[i, j] = float(np.mean(f1s))

    fig, ax = plt.subplots(figsize=(10.8, 8.6))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels([m[1] for m in METHODS], rotation=42, ha="right",
                       fontsize=8.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([m[1] for m in METHODS], fontsize=8.2)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            color = "white" if v > 0.5 else "#222222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.0, color=color)
    ax.set_title("Pairwise boundary F1 on Long-MT-Bench+ "
                 f"({len(common)} conv, {n} methods)", fontsize=10.5, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cbar.set_label("Pairwise boundary F1", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()

    pdf = OUT_DIR / "boundary_agreement.pdf"
    png = OUT_DIR / "boundary_agreement.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nsaved {pdf}\nsaved {png}")


if __name__ == "__main__":
    main()
