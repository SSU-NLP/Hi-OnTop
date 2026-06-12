#!/usr/bin/env python3
"""Figure V — timeline tape for RealtimeSTT asr-reference.wav (single-topic).

Figure U (lme_timeline_tape) 와 동일 스타일.
단일 토픽 오디오에서 Hi-OnTop / LLM 이 과잉분할하는지 관찰.

출처: KoljaB/RealtimeSTT tests/unit/audio/asr-reference.expected_sentences.json
     (Whisper large-v2 GPU 전사, VAD-split 9 utterances, 51.9초)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
load_dotenv(REPO / ".env")

from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

# ── utterances (RealtimeSTT asr-reference.wav VAD-split) ─────────────────
UTTERANCES = [
    "Hey guys.",
    "Welcome to the new demo of my real-time transcription library designed"
    " to showcase its lightning-fast capabilities.",
    "As you'll see, speech is transcribed almost instantly into text, which I"
    " think is a pretty cool achievement given the challenges of working with Whisper.",
    "I've put a lot of effort into making this tool stable, efficient and easy"
    " to use and I really hope you find it valuable in your projects.",
    "This library is completely open source and it would mean a lot to me if"
    " you could support it by giving a star on GitHub.",
    "Visibility and community feedback are key to making it even better and"
    " every bit of engagement helps improve its future development.",
    "Whether you are working on a small project or something bigger, I hope"
    " this tool becomes a helpful resource and sparks some innovation.",
    "So feel free to download it, use it in your apps and let me know if you"
    " have any feedback.",
    "Thanks for watching and I hope you enjoyed the demo.",
]

# ── δ* (MiniLM-int8, cross-benchmark mean from dts_result.md) ────────────
DSTAR = {"p60": 0.721, "p70": 0.771, "p80": 0.822}

N = len(UTTERANCES)


# ── LLM helpers (재사용: run_share_boundary_comparison.py 패턴) ───────────
def _build_prompt(turns: list[str]) -> str:
    exchanges = "".join(f"[Exchange {i}]: {t}\n\n" for i, t in enumerate(turns))
    return f"""# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn dialogue into topically coherent units. \
Exchanges about the same topic should be grouped together; create a new unit when the topic shifts.
- **Data**: A series of exchanges separated by "\\n\\n".

## Output Format

Output segmentation results as **jsonl lines** inside <segmentation></segmentation> tags. \
Each line is a JSON dict with keys:
  - segment_id          : int (starting from 0)
  - start_exchange_number : int
  - end_exchange_number   : int
  - num_exchanges         : int

## Constraints

- Cover ALL exchanges (no gaps, no overlaps).
- start_exchange_number of the first segment = 0.
- end_exchange_number of the last segment = {len(turns) - 1}.

# Data

{exchanges.strip()}

# Output

<segmentation>
"""


def _parse_segments(response: str, n: int) -> list[int]:
    """Return 0-based turn indices AFTER which a boundary occurs."""
    # try tag-wrapped first, fallback to raw JSONL
    m = re.search(r"<segmentation>([\s\S]*?)(?:</segmentation>|$)", response)
    block = m.group(1).strip() if m else response.strip()

    boundaries: list[int] = []
    cursor = 0
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            cursor += int(d["num_exchanges"])
            if cursor < n:
                boundaries.append(cursor - 1)
        except Exception:
            pass
    return boundaries


def _call_llm(model_slug: str, prompt: str) -> str:
    from openai import OpenAI
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.ssunlp.co.kr/v1")
    client   = OpenAI(api_key=api_key, base_url=base_url)
    is_gpt5 = "gpt-5" in model_slug.lower()
    kw: dict = {}
    if is_gpt5:
        kw["max_completion_tokens"] = 2048
        kw["reasoning_effort"] = "minimal"
    else:
        kw["max_tokens"] = 2048
        kw["temperature"] = 0.0
        kw["extra_body"] = {"thinking": {"type": "disabled"}, "reasoning_effort": "none"}
    resp = client.chat.completions.create(
        model=model_slug,
        messages=[{"role": "user", "content": prompt}],
        **kw,
    )
    return resp.choices[0].message.content or ""


def run_llm(model_slug: str, label: str) -> list[int]:
    print(f"  calling {label} ({model_slug})…", flush=True)
    prompt = _build_prompt(UTTERANCES)
    try:
        resp = _call_llm(model_slug, prompt)
        print(f"    response: {resp[:200]!r}", flush=True)
        bnd = _parse_segments(resp, N)
        print(f"    boundaries: {bnd}", flush=True)
        return bnd
    except Exception as e:
        print(f"    LLM call failed: {e}", flush=True)
        return []


# ── Hi-OnTop ──────────────────────────────────────────────────────────────
def run_hiontop() -> tuple[np.ndarray, dict[str, list[int]]]:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"})
    vecs = enc.encode(UTTERANCES, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False)

    graded_p70 = []
    bnds: dict[str, list[int]] = {}
    for pkey, dstar in DSTAR.items():
        seg = HiOnTop(dim=vecs.shape[1], delta_star=dstar,
                      ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
        g_list = []
        b_list = []
        for i, v in enumerate(vecs):
            seg.assign(v.astype(np.float64))
            h = seg.history()[-1]
            g_list.append(float(h["graded_score"]))
            if h["is_boundary"]:
                b_list.append(i - 1)   # boundary after turn i-1
        bnds[pkey] = b_list
        if pkey == "p70":
            graded_p70 = g_list
        print(f"  Hi-OnTop {pkey}: boundaries={b_list}  "
              f"g_t={[f'{g:.2f}' for g in g_list]}", flush=True)

    return np.array(graded_p70), bnds


# ── plot ──────────────────────────────────────────────────────────────────
def plot(graded: np.ndarray, bnds: dict[str, list[int]]) -> None:
    # ── Figure U 와 동일한 스타일 ─────────────────────────────────────────
    row_labels = [
        ("gpt5",    "GPT-5",                 "#8E44AD"),
        ("qwen122b","Qwen3.5-122B",          "#0D5E1F"),
        ("qwen27b", "Qwen3.5-27B",           "#1F8E3F"),
        ("p60",     "Hi-OnTop\n(int8, p60)", "#5BA3D0"),
        ("p70",     "Hi-OnTop\n(int8, p70)", "#2980B9"),
        ("p80",     "Hi-OnTop\n(int8, p80)", "#1A5276"),
    ]

    n_rows  = len(row_labels) + 1
    heights = [3.2] + [1.0] * len(row_labels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.0, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.15})
    ax_score, row_axes = axes[0], axes[1:]

    # g_t heatmap (Figure U 와 동일)
    cmap = plt.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=0, vmax=2.0)
    ax_score.imshow(graded[None, :], cmap=cmap, norm=norm, aspect="auto",
                    extent=(-0.5, N - 0.5, -0.5, 0.5))
    ax_score.set_yticks([])
    ax_score.set_ylabel("Hi-OnTop\n$g_t$", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_score.axhline(0.5,  color="black", linewidth=0.3)
    ax_score.axhline(-0.5, color="black", linewidth=0.3)
    for b in bnds.get("p70", []):
        ax_score.annotate("", xy=(b, 0.55), xytext=(b, 1.1),
                          arrowprops=dict(arrowstyle="->", color="#1A3A5C", lw=1.3),
                          annotation_clip=False)

    # rows (Figure U 와 동일)
    for ax, (key, label, color) in zip(row_axes, row_labels):
        ax.set_yticks([]); ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.2)
        ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(0, 1)
        for b in bnds.get(key, []):
            ax.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                        color=color, alpha=0.85,
                                        linewidth=0.4, edgecolor="black"))

    # consensus 강조 (gpt5 빈 경우 제외하고 나머지 LLM + p70)
    llm_bnds = [set(bnds.get(k, [])) for k in ("gpt5", "qwen27b", "qwen122b") if bnds.get(k)]
    consensus = set(bnds.get("p70", []))
    for s in llm_bnds:
        consensus &= s
    for b in consensus:
        for ax_ in row_axes:
            ax_.add_patch(plt.Rectangle((b + 0.05, 0.05), 0.9, 0.9,
                                         fill=False, edgecolor="gold", linewidth=1.5))

    # x-axis
    short = ["Hey guys.", "Welcome…", "As you'll see…", "Effort…",
             "Open source…", "Visibility…", "Whether…", "Feel free…", "Thanks…"]
    row_axes[-1].set_xlabel("Utterance (VAD-split, RealtimeSTT asr-reference.wav)", fontsize=8.5)
    tick_step = 1
    row_axes[-1].set_xticks(range(N))
    row_axes[-1].set_xticklabels(short, rotation=28, ha="right", fontsize=7.5)
    row_axes[-1].tick_params(labelsize=8)

    # colorbar (Figure U 와 동일)
    fig.subplots_adjust(right=0.87)
    cax = fig.add_axes([0.89, 0.50, 0.012, 0.38])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cb.set_ticklabels(["0", "0.5", "1.0\n(thr)", "1.5", "2.0"])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("$g_t=\\delta_{eff}/\\delta^*$", fontsize=8)

    # legend (Figure U 와 동일)
    handles = [
        mpatches.Patch(color="#8E44AD", alpha=0.85, label="GPT-5"),
        mpatches.Patch(color="#0D5E1F", alpha=0.85, label="Qwen3.5-122B"),
        mpatches.Patch(color="#1F8E3F", alpha=0.85, label="Qwen3.5-27B"),
        mpatches.Patch(color="#5BA3D0", alpha=0.85, label="Hi-OnTop (int8, p60)"),
        mpatches.Patch(color="#2980B9", alpha=0.85, label="Hi-OnTop (int8, p70)"),
        mpatches.Patch(color="#1A5276", alpha=0.85, label="Hi-OnTop (int8, p80)"),
        mpatches.Patch(facecolor="none", edgecolor="gold", linewidth=1.5, label="LLMs+p70 agree"),
    ]
    ax_score.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.55),
                    ncol=3, fontsize=7.0, frameon=False, handlelength=1.2, columnspacing=0.7)

    fig.suptitle(
        "Timeline tape — RealtimeSTT asr-reference.wav (single-topic, 9 utterances)\n"
        "(ground truth: 0 boundaries — one continuous topic)",
        fontsize=9, y=0.995)
    plt.tight_layout(rect=(0, 0, 0.88, 0.95))

    # save
    out_exp = REPO / "outputs" / "experiments" / "2026-06-03_realtimestt_segmentation"
    out_exp.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO / "outputs" / "figures"

    for path in [
        out_exp / "figure_V_realtimestt_timeline_tape.pdf",
        out_exp / "figure_V_realtimestt_timeline_tape.png",
        fig_dir  / "figure_V_realtimestt_timeline_tape.pdf",
        fig_dir  / "figure_V_realtimestt_timeline_tape.png",
    ]:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved → {path}")
    plt.close(fig)


def main():
    print("=== Hi-OnTop ===", flush=True)
    graded, bnds = run_hiontop()

    print("\n=== LLM segmenters ===", flush=True)
    bnds["gpt5"]     = run_llm("openrouter/openai/gpt-5",              "GPT-5")
    bnds["qwen27b"]  = run_llm("openrouter/qwen/qwen3.5-27b",          "Qwen3.5-27B")
    bnds["qwen122b"] = run_llm("openrouter/qwen/qwen3.5-122b-a10b",    "Qwen3.5-122B")

    print("\n=== plotting ===", flush=True)
    plot(graded, bnds)
    print("done", flush=True)


if __name__ == "__main__":
    main()
