"""Figure E: Segment length distribution (violin) across ALL methods.

Each violin = distribution of segment lengths (# turns per segment) over
all conversations × all segments produced by that method.

Includes 4 algorithmic + 1 supervised + 6 LLM + 6 Hi-OnTop variants.

Output: ``plots/segment_length_violin.{pdf,png}``.
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
# (subdir, pretty label, color)
METHODS = [
    ("texttiling",       "TextTiling",          "#666666"),
    ("graphseg",         "GraphSeg",            "#9467BD"),
    ("greedyseg",        "GreedySeg",           "#8C564B"),
    ("csm",              "CSM",                 "#17BECF"),
    ("ours_p60",         r"Hi-OnTop (MPNet, $p_{60}$)", "#D62728"),
    ("ours_p70",         r"Hi-OnTop (MPNet, $p_{70}$)", "#E45756"),
    ("ours_hiontop",      r"Hi-OnTop (MPNet, $p_{80}$)", "#F37C7E"),
    ("ours_int8_p60",    r"Hi-OnTop (int8, $p_{60}$)",  "#FF7F0E"),
    ("ours_int8_p70",    r"Hi-OnTop (int8, $p_{70}$)",  "#FF9F4D"),
    ("ours_int8_p80",    r"Hi-OnTop (int8, $p_{80}$)",  "#FFC689"),
    ("roberta",          "RoBERTa",             "#BCBD22"),
    ("gpt5seg",          "GPT-5",               "#8E44AD"),
    ("qwen35_122bseg",   "Qwen3.5-122B-A10B",   "#0D5E1F"),
    ("qwen27bseg",       "Qwen3.5-27B",         "#1F8E3F"),
    ("gpt4omini",        "GPT-4o-mini",         "#2CA02C"),
    ("qwen35_4bseg",     "Qwen3.5-4B",          "#75B775"),
    ("llama32_3bseg",    "Llama3.2-3B",         "#3D9970"),
    ("ministral3_3bseg", "Mistral3-3B",         "#52B788"),
    ("qwen35_2bseg",     "Qwen3.5-2B",          "#A3D6A3"),
]


def seg_lengths(path: Path) -> list[int]:
    if not path.exists():
        return []
    lens = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        for s in r.get("segments", []):
            if len(s) > 0:
                lens.append(len(s))
    return lens


def main():
    data, labels, colors, means = [], [], [], []
    for sub, label, color in METHODS:
        lens = seg_lengths(RDIR / sub / "segments.jsonl")
        if not lens:
            print(f"skip {sub}: no data")
            continue
        data.append(lens)
        labels.append(label)
        colors.append(color)
        means.append(float(np.mean(lens)))
        print(f"{label:32s}: n={len(lens):4d}  mean={np.mean(lens):5.2f}  "
              f"median={np.median(lens):4.1f}  p95={np.percentile(lens, 95):4.0f}")

    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    positions = np.arange(len(data))
    vp = ax.violinplot(data, positions=positions, widths=0.82,
                       showmeans=False, showmedians=True, showextrema=False)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_edgecolor("black")
        body.set_alpha(0.72); body.set_linewidth(0.6)
    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.0)

    ax.scatter(positions, means, marker="D", s=26, facecolor="white",
               edgecolor="black", zorder=4, label="mean")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=9.0)
    ax.set_ylabel("Segment length (# turns / segment)", fontsize=11)
    ax.set_title("Per-method segment-length distribution on Long-MT-Bench+ "
                 f"(11 conv, 720 turns; {len(data)} methods)",
                 fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    plt.tight_layout()

    pdf = OUT_DIR / "segment_length_violin.pdf"
    png = OUT_DIR / "segment_length_violin.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nsaved {pdf}\nsaved {png}")


if __name__ == "__main__":
    main()
