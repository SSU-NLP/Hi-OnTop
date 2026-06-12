"""Figure B: Per-turn latency comparison across ALL segmentation methods.

A single panel:
  x = average ms / turn (log scale)
  y = method (categorical, sorted by latency)
  marker = category color
  Each method shown as ONE point (mean ms/turn across 11 MTB+ conversations).

This is the "ranked latency" view — much cleaner than scatter with overlapping bands.

Output: ``plots/latency_scatter.{pdf,png}``.
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

# (json_filename_stem, pretty label, color, category)
# Order aligned with downstream_task.md LaTeX table (2026-05-24):
# unsup-alg → ours-MPNet → ours-int8 → sup → LLM (GPT4Score desc).
METHODS = [
    # Algorithmic (unsup)
    ("texttiling",            "TextTiling",     "#666666", "alg"),
    ("graphseg",              "GraphSeg",       "#9467BD", "alg"),
    ("greedyseg",             "GreedySeg",      "#8C564B", "alg"),
    ("csm",                   "CSM",            "#17BECF", "alg"),
    # Hi-OnTop (MPNet)
    ("ours_p60",              r"Hi-OnTop (MPNet, $p_{60}$)", "#D62728", "ours"),
    ("ours_p70",              r"Hi-OnTop (MPNet, $p_{70}$)", "#E45756", "ours"),
    ("ours_hiontop",           r"Hi-OnTop (MPNet, $p_{80}$)", "#F37C7E", "ours"),
    # Hi-OnTop (int8)
    ("ours_int8_p60",         r"Hi-OnTop (int8, $p_{60}$)",  "#FF7F0E", "ours"),
    ("ours_int8_p70",         r"Hi-OnTop (int8, $p_{70}$)",  "#FF9F4D", "ours"),
    ("ours_int8_p80",         r"Hi-OnTop (int8, $p_{80}$)",  "#FFC689", "ours"),
    # Supervised
    ("roberta",               "RoBERTa",        "#BCBD22", "sup"),
    # LLM-based (GPT4Score desc)
    ("gpt5seg",               "GPT-5",          "#8E44AD", "llm"),
    ("qwen35_122bseg",        "Qwen3.5-122B-A10B", "#0D5E1F", "llm"),
    ("qwen27bseg",            "Qwen3.5-27B",    "#1F8E3F", "llm"),
    ("baseline",              "GPT-4o-mini",    "#2CA02C", "llm"),
    ("qwen35_4bseg",          "Qwen3.5-4B",     "#75B775", "llm"),
    ("llama32_3bseg",         "Llama3.2-3B",    "#3D9970", "llm"),
    ("ministral3_3bseg",      "Mistral3-3B",    "#52B788", "llm"),
    ("qwen35_2bseg",          "Qwen3.5-2B",     "#A3D6A3", "llm"),
]


def ms_per_turn(d):
    """Extract a single canonical ms/turn from a latency json."""
    if "ms_per_exchange" in d:
        return float(d["ms_per_exchange"])
    if "total_sec" in d and "n_exchanges" in d and d["n_exchanges"]:
        return 1000.0 * d["total_sec"] / d["n_exchanges"]
    return None


def main():
    rows = []
    for stem, label, color, cat in METHODS:
        path = EXP / f"latency_{stem}.json"
        if not path.exists():
            print(f"skip {stem}: no file")
            continue
        d = json.loads(path.read_text())
        ms = ms_per_turn(d)
        if ms is None:
            continue
        rows.append(dict(label=label, color=color, cat=cat, ms=ms))
        print(f"{label:32s}: {ms:8.2f} ms/turn  ({path.name})")

    # sort by latency descending (slowest at top)
    rows = sorted(rows, key=lambda r: -r["ms"])

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    y_positions = np.arange(len(rows))
    labels = [r["label"] for r in rows]
    colors = [r["color"] for r in rows]
    ms_vals = [r["ms"] for r in rows]

    bars = ax.barh(y_positions, ms_vals, color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.92)
    for b, v in zip(bars, ms_vals):
        ax.text(v * 1.08, b.get_y() + b.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=8.5,
                color="#222")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlabel("Latency per turn  (ms, log scale)", fontsize=11.5)
    ax.set_title("Per-turn segmentation latency on Long-MT-Bench+ "
                 "(11 conv, 720 turns, idle CPU)",
                 fontsize=11, pad=8)
    ax.grid(True, axis="x", which="major", alpha=0.25, linestyle="-",
            linewidth=0.5)
    ax.grid(True, axis="x", which="minor", alpha=0.12, linestyle=":")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.invert_yaxis()   # slowest at top

    # Threshold markers
    for thresh, txt in [(10, "10 ms"), (100, "100 ms"), (1000, "1 s")]:
        ax.axvline(thresh, color="#888", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.text(thresh, len(rows) - 0.4, f" {txt}", fontsize=8, color="#888",
                ha="left", va="bottom")

    plt.tight_layout()
    pdf = OUT_DIR / "latency_scatter.pdf"
    png = OUT_DIR / "latency_scatter.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nsaved {pdf}\nsaved {png}")


if __name__ == "__main__":
    main()
