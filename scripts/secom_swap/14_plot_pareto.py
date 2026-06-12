"""Figure G: Pareto frontier of QA quality vs context length.

x = avg retrieved tokens per query. y = GPT4Score (1-10, x10 -> 0-100).
Includes ALL available methods on Long-MT-Bench+: 4 algorithmic, 1
supervised, 6 LLM-based, 6 Hi-OnTop variants (3 MPNet × 3 int8). Pareto
frontier highlighted.

Output: ``plots/pareto_qa_context.{pdf,png}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP = REPO_ROOT / "outputs/experiments/2026-05-21_v413_secom_swap"
OUT_DIR = EXP / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (metrics_filename_stem, pretty label, color, marker, category)
# Order aligned with downstream_task.md (2026-05-24).
METHODS = [
    # Algorithmic baselines
    ("texttiling",            "TextTiling",     "#666666", "o", "alg"),
    ("graphseg",              "GraphSeg",       "#9467BD", "s", "alg"),
    ("greedyseg",             "GreedySeg",      "#8C564B", "^", "alg"),
    ("csm",                   "CSM",            "#17BECF", "D", "alg"),
    # Hi-OnTop MPNet
    ("ours_p60",              r"Hi-OnTop (MPNet, $p_{60}$)", "#D62728", "o", "ours"),
    ("ours_p70",              r"Hi-OnTop (MPNet, $p_{70}$)", "#E45756", "o", "ours"),
    ("ours_hiontop",           r"Hi-OnTop (MPNet, $p_{80}$)", "#F37C7E", "o", "ours"),
    # Hi-OnTop int8
    ("ours_int8_p60",         r"Hi-OnTop (int8, $p_{60}$)",  "#FF7F0E", "X", "ours"),
    ("ours_int8_p70",         r"Hi-OnTop (int8, $p_{70}$)",  "#FF9F4D", "X", "ours"),
    ("ours_int8_p80",         r"Hi-OnTop (int8, $p_{80}$)",  "#FFC689", "X", "ours"),
    # Supervised
    ("roberta",               "RoBERTa",        "#BCBD22", "P", "sup"),
    # LLM-based (GPT4Score desc)
    ("segsweep_gpt5seg",      "GPT-5",          "#8E44AD", "*", "llm"),
    ("segsweep_qwen35_122bseg", "Qwen3.5-122B-A10B", "#0D5E1F", "*", "llm"),
    ("segsweep_qwen27bseg",   "Qwen3.5-27B",    "#1F8E3F", "*", "llm"),
    ("baseline",              "GPT-4o-mini",    "#2CA02C", "*", "llm"),
    ("segsweep_qwen35_4b",    "Qwen3.5-4B",     "#75B775", "*", "llm"),
    ("segsweep_llama32_3b",   "Llama3.2-3B",    "#3D9970", "*", "llm"),
    ("segsweep_ministral3_3b","Mistral3-3B",    "#52B788", "*", "llm"),
    ("segsweep_qwen35_2b",    "Qwen3.5-2B",     "#A3D6A3", "*", "llm"),
]


def main():
    points = []
    for stem, label, color, marker, cat in METHODS:
        path = EXP / f"metrics_{stem}.json"
        if not path.exists():
            print(f"skip {stem}: no metrics")
            continue
        d = json.loads(path.read_text())
        points.append(dict(
            label=label, color=color, marker=marker, cat=cat,
            x=d["n_tokens"], y=d["gpt4_score_x10"],
        ))

    # EMNLP two-column 친화 (single-column ~3.3in, two-col ~7in). 7in 폭 채택.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for p in points:
        s = 80 if p["cat"] == "ours" else 55
        ax.scatter(p["x"], p["y"], s=s, color=p["color"],
                   marker=p["marker"], edgecolor="black", linewidth=0.5,
                   label=p["label"], zorder=3, alpha=0.92)

    # Pareto frontier
    sorted_p = sorted(points, key=lambda p: p["x"])
    front = []
    best_y = -1
    for p in sorted_p:
        if p["y"] > best_y:
            front.append(p); best_y = p["y"]
    if len(front) >= 2:
        fx = [p["x"] for p in front]; fy = [p["y"] for p in front]
        ax.plot(fx, fy, "--", color="#444444", linewidth=0.9, alpha=0.5,
                zorder=2, label="Pareto frontier")

    ax.set_xlabel("Avg retrieved tokens per query", fontsize=10)
    ax.set_ylabel(r"GPT4Score $\uparrow$", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Annotate ONLY the top-1 (avoid overlap clutter)
    top1 = max(points, key=lambda p: p["y"])
    ax.annotate(f"{top1['label']} ({top1['y']:.1f})",
                (top1["x"], top1["y"]),
                xytext=(top1["x"] + 200, top1["y"] - 0.5),
                fontsize=8.5, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.6),
                color="#222")

    # Legend OUTSIDE plot to right
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=7.8, ncol=1,
                    handlelength=1.0, labelspacing=0.32,
                    handletextpad=0.4, borderaxespad=0)

    plt.tight_layout()

    pdf = OUT_DIR / "pareto_qa_context.pdf"
    png = OUT_DIR / "pareto_qa_context.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
