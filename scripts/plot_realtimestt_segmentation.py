#!/usr/bin/env python3
"""Figure U — Hi-OnTop g_t on RealtimeSTT asr-reference.wav transcription.

단일 토픽(RealtimeSTT 소개) 9 utterance 에 Hi-OnTop(MiniLM-int8)을 적용하고
g_t = delta_eff / delta* 분포를 시각화. 다른 DTS 벤치마크 데이터의 g_t 분포와
overlay 하여 단일 토픽에서의 과잉분할 여부를 관찰.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

# ── 1. RealtimeSTT asr-reference.wav 전사 결과 (VAD-split utterances) ──────
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

# ── 2. MiniLM-int8 encoder (ONNX quint8_avx2) ────────────────────────────
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"},
    )

# ── 3. MiniLM-int8 p70 δ* (cross-dataset mean from dts_result.md) ─────────
# TIAGE 0.7763 / DS711 0.7519 / SDS 0.7839 → mean 0.771
DELTA_STAR = 0.771

# ── 4. DTS benchmark g_t 샘플 (비교용 — tiage test 첫 N dialog) ──────────
def load_benchmark_gt(n_dial: int = 30) -> tuple[list[float], list[float]]:
    """tiage test set 에서 g_t 값을 boundary / non-boundary 로 분리."""
    import json, pickle
    SDS = REPO / "benchmarks" / "superdialseg_data" / "tiage"
    raw = json.loads((SDS / "segmentation_file_test.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]][:n_dial]

    cache = REPO / "outputs" / "runs" / "_misc" / "enccmp_tiage_test_minilm-int8.pkl"
    if not cache.exists():
        print("[warn] benchmark embedding cache not found — skipping bench overlay")
        return [], []
    with open(cache, "rb") as f:
        all_embs = pickle.load(f)
    all_embs = all_embs[:n_dial]

    gt_boundary, gt_non = [], []
    for (dial, embs) in zip(arr, all_embs):
        utts = [t["utterance"] for t in dial["turns"]]
        yt   = [int(t.get("segmentation_label", 0)) for t in dial["turns"]]
        if yt: yt[-1] = 0
        seg = HiOnTop(dim=embs.shape[1], delta_star=DELTA_STAR,
                      ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
        for s in embs:
            seg.assign(s.astype(np.float64))
        for i, h in enumerate(seg.history()):
            if i == 0: continue
            g = h["graded_score"]
            if yt[i]:
                gt_boundary.append(g)
            else:
                gt_non.append(g)
    return gt_boundary, gt_non


def main():
    print("Loading encoder…", flush=True)
    enc = load_encoder()

    print("Encoding utterances…", flush=True)
    embs = enc.encode(UTTERANCES, normalize_embeddings=True,
                      show_progress_bar=False)

    print("Running Hi-OnTop…", flush=True)
    seg = HiOnTop(dim=embs.shape[1], delta_star=DELTA_STAR,
                  ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
    for s in embs:
        seg.assign(s.astype(np.float64))

    history = seg.history()
    turns   = list(range(len(history)))
    gt_vals = [h["graded_score"] for h in history]
    de_vals = [h["delta_eff"]    for h in history]

    print("g_t values:", [f"{g:.3f}" for g in gt_vals])

    print("Loading benchmark g_t…", flush=True)
    bench_bnd, bench_non = load_benchmark_gt(n_dial=30)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(
        "Figure U — Hi-OnTop $g_t$ on Single-Topic Audio (RealtimeSTT demo)",
        fontsize=11, fontweight="bold",
    )

    # ── Left: g_t timeline ──────────────────────────────────────────────────
    ax = axes[0]
    bar_colors = ["#c0392b" if g >= 1.0 else "#2980b9" for g in gt_vals]
    ax.bar(turns, gt_vals, color=bar_colors, alpha=0.8, width=0.6)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="boundary threshold ($g_t=1$)")
    ax.axhspan(0.0, 0.7, alpha=0.06, color="green",
               label="very weak ($g_t<0.7$)")
    ax.axhspan(0.7, 1.0, alpha=0.06, color="orange",
               label="weak / within-segment")

    short_labels = [
        "Hey guys.", "Welcome…", "As you'll see…", "I've put effort…",
        "Open source…", "Visibility…", "Whether you…",
        "Feel free…", "Thanks…",
    ]
    ax.set_xticks(turns)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("$g_t = \\delta_{eff} / \\delta^*$", fontsize=10)
    ax.set_xlabel("Utterance (VAD-split, RealtimeSTT)", fontsize=9)
    ax.set_ylim(0, max(1.6, max(gt_vals) * 1.15))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"MiniLM-int8, $\\delta^*={DELTA_STAR}$ (p70 cross-bench avg)",
                 fontsize=9)

    # turn 0 g_t = 0 by definition — annotate
    ax.text(0, 0.05, "turn 0\n(init)", ha="center", va="bottom",
            fontsize=7, color="gray")

    # ── Right: g_t distribution (RealtimeSTT vs benchmark) ──────────────────
    ax2 = axes[1]
    bins = np.linspace(0, 2.5, 26)

    if bench_non:
        ax2.hist(bench_non, bins=bins, alpha=0.45, color="#27ae60",
                 label=f"TIAGE non-boundary\n(n={len(bench_non)})", density=True)
    if bench_bnd:
        ax2.hist(bench_bnd, bins=bins, alpha=0.45, color="#e67e22",
                 label=f"TIAGE boundary\n(n={len(bench_bnd)})", density=True)

    rts_gt = [g for g in gt_vals[1:]]  # skip turn 0
    ax2.hist(rts_gt, bins=bins, alpha=0.75, color="#2980b9",
             label=f"RealtimeSTT audio\n(n={len(rts_gt)}, single topic)", density=True)
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=1.2,
                label="threshold")

    ax2.set_xlabel("$g_t$", fontsize=10)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("$g_t$ distribution comparison", fontsize=9)
    ax2.legend(fontsize=7.5)

    plt.tight_layout()

    # ── Save ────────────────────────────────────────────────────────────────
    out_exp = REPO / "outputs" / "experiments" / "2026-06-03_realtimestt_segmentation"
    out_exp.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        out_exp / "figure_U_realtimestt_gt.pdf",
        out_exp / "figure_U_realtimestt_gt.png",
        fig_dir  / "figure_U_realtimestt_gt.pdf",
        fig_dir  / "figure_U_realtimestt_gt.png",
    ]:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"saved → {path}")

    # ── REPORT ──────────────────────────────────────────────────────────────
    lines = [
        "# Figure U — Hi-OnTop g_t on Single-Topic Audio (2026-06-03)",
        "",
        "## 실험 setup",
        "- 입력: `KoljaB/RealtimeSTT` `tests/unit/audio/asr-reference.wav`",
        "  의 VAD-split 전사 결과 (Whisper large-v2, 51.9초, 9 utterances)",
        "- 인코더: MiniLM-int8 (ONNX quint8_avx2, dim=384)",
        f"- δ* = {DELTA_STAR} (MiniLM-int8 p70 cross-benchmark 평균)",
        "  - TIAGE 0.7763 / DS711 0.7519 / SDS 0.7839",
        "- HP: m=2, ρ=0.7, a=0.5",
        "",
        "## g_t 값 (turn별)",
        "",
        "| turn | utterance (앞 30자) | δ_eff | g_t | boundary? |",
        "|---|---|---:|---:|---|",
    ]
    for i, (h, utt) in enumerate(zip(history, UTTERANCES)):
        bnd = "**YES**" if h["graded_score"] >= 1.0 else "no"
        lines.append(
            f"| {i} | {utt[:40]}… | {h['delta_eff']:.4f} | "
            f"{h['graded_score']:.4f} | {bnd} |"
        )
    lines += [
        "",
        "## 관찰",
        f"- 9개 utterance 중 boundary 판정: "
        f"{sum(1 for h in history if h['graded_score']>=1.0)}개",
        f"- g_t 최대값: {max(gt_vals):.4f}",
        f"- g_t mean (turn 1~8): {np.mean(gt_vals[1:]):.4f}",
        "",
        "## 한계",
        "- δ* = cross-benchmark 평균값 사용 (이 오디오로 calibration 불가 — 레이블 없음)",
        "- 9 utterance 로 통계적으로 의미있는 결론 내리기 어려움 (관찰용)",
    ]
    (out_exp / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"REPORT → {out_exp}/REPORT.md")


if __name__ == "__main__":
    main()
