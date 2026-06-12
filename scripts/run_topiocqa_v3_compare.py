#!/usr/bin/env python3
"""TopiOCQA segmentation: v2 Gaussian vs v3.1 cosine MAP vs v3.2 cosine PE.

Runs all three segmenters on TopiOCQA dev with shared embeddings and
the same sticky-CRP hyperparameters; prints + writes one table.

Usage::

    uv run python scripts/run_topiocqa_v3_compare.py
    uv run python scripts/run_topiocqa_v3_compare.py --limit-convs 50 \\
        --tau-sweep 30 50 100 --cos-thr-sweep 0.5 0.7 0.85 \\
        --rho-sweep 0.3 0.5 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_ontop.embedding import QueryEncoder  # noqa: E402
from hi_ontop.sem_core import HiOnTopSegmenter  # noqa: E402
from hi_ontop.sem_core_optimize import HiOnTopSegmenterV3  # noqa: E402
from hi_ontop.sem_core_optimize_pe import HiOnTopSegmenterV32  # noqa: E402


def load_topiocqa_dev() -> list[list[dict]]:
    path = (
        REPO_ROOT / "benchmarks" / "topiocqa" / "downloads" / "data"
        / "topiocqa_dataset" / "dev.json"
    )
    raw = json.loads(path.read_text())
    buckets: dict[int, list[dict]] = defaultdict(list)
    for t in raw:
        buckets[t["Conversation_no"]].append(t)
    return [
        sorted(buckets[cno], key=lambda x: x["Turn_no"]) for cno in sorted(buckets)
    ]


def ground_truth_shifts(conv: list[dict]) -> list[bool]:
    topics = [t["Topic"] for t in conv]
    return [topics[i] != topics[i - 1] for i in range(1, len(topics))]


def f1_score(gt: list[bool], pred: list[bool]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def run_v2(convs, embeddings, alpha, lmda, sigma0_sq):
    gt, pred = [], []
    n_topics_per_conv = []
    max_share_per_conv = []
    for conv, emb in zip(convs, embeddings):
        seg = HiOnTopSegmenter(dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq)
        gt.extend(ground_truth_shifts(conv))
        ids = [seg.assign(e)[0] for e in emb]
        pred.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        n_topics_per_conv.append(len(seg.topics))
        if ids:
            counts = np.bincount(ids)
            max_share_per_conv.append(counts.max() / counts.sum())
    return gt, pred, n_topics_per_conv, max_share_per_conv


def run_v3_1(convs, embeddings, alpha, lmda, tau, cos_threshold=0.5):
    gt, pred = [], []
    n_topics_per_conv = []
    max_share_per_conv = []
    for conv, emb in zip(convs, embeddings):
        seg = HiOnTopSegmenterV3(dim=emb.shape[1], alpha=alpha, lmda=lmda,
                              tau=tau, cos_threshold=cos_threshold)
        gt.extend(ground_truth_shifts(conv))
        ids = [seg.assign(e)[0] for e in emb]
        pred.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        n_topics_per_conv.append(len(seg.topics))
        if ids:
            counts = np.bincount(ids)
            max_share_per_conv.append(counts.max() / counts.sum())
    return gt, pred, n_topics_per_conv, max_share_per_conv


def run_v3_2(convs, embeddings, alpha, lmda, tau, cos_threshold, rho):
    gt, pred = [], []
    n_topics_per_conv = []
    max_share_per_conv = []
    for conv, emb in zip(convs, embeddings):
        seg = HiOnTopSegmenterV32(
            dim=emb.shape[1], alpha=alpha, lmda=lmda,
            tau=tau, cos_threshold=cos_threshold, rho=rho,
        )
        gt.extend(ground_truth_shifts(conv))
        ids = [seg.assign(e)[0] for e in emb]
        pred.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        n_topics_per_conv.append(len(seg.topics))
        if ids:
            counts = np.bincount(ids)
            max_share_per_conv.append(counts.max() / counts.sum())
    return gt, pred, n_topics_per_conv, max_share_per_conv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lmda", type=float, default=10.0)
    p.add_argument("--sigma0-sq", type=float, default=0.01)
    p.add_argument("--tau", type=float, default=50.0)
    p.add_argument("--cos-threshold", type=float, default=0.7,
                   help="New-cluster cosine baseline for v3.x.")
    p.add_argument("--rho", type=float, default=0.5,
                   help="v3.2 AR(1) weight on previous embedding.")
    p.add_argument("--tau-sweep", type=float, nargs="*", default=None,
                   help="Sweep these tau values for v3.1 + v3.2; pick best F1.")
    p.add_argument("--cos-thr-sweep", type=float, nargs="*", default=None,
                   help="Sweep these cos_threshold values (combined with tau-sweep).")
    p.add_argument("--rho-sweep", type=float, nargs="*", default=None,
                   help="v3.2 only: sweep these rho values (combined with tau/thr).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit-convs", type=int, default=None)
    p.add_argument("--output", type=str,
                   default=str(REPO_ROOT / "outputs" / "topiocqa_v3_compare.md"))
    args = p.parse_args()

    print("[1/3] loading TopiOCQA dev...")
    convs = load_topiocqa_dev()
    if args.limit_convs:
        convs = convs[: args.limit_convs]
    n_convs = len(convs)
    n_turns = sum(len(c) for c in convs)
    n_shifts = sum(sum(ground_truth_shifts(c)) for c in convs)
    print(f"  {n_convs} conv / {n_turns} turns / {n_shifts} GT shifts")

    print("[2/3] encoding...")
    encoder = QueryEncoder(device=args.device)
    all_q = [t["Question"] for c in convs for t in c]
    t0 = time.perf_counter()
    emb_flat = np.asarray(encoder.encode(all_q))
    embed_sec = time.perf_counter() - t0
    print(f"  {embed_sec:.1f}s on {encoder.device}")

    embeddings: list[np.ndarray] = []
    idx = 0
    for c in convs:
        embeddings.append(emb_flat[idx : idx + len(c)])
        idx += len(c)

    print("[3/3] segmenting v2 + v3.1 + v3.2...")
    gt2, pred2, ntopics2, mshare2 = run_v2(
        convs, embeddings, args.alpha, args.lmda, args.sigma0_sq
    )
    p2, r2, f1_2 = f1_score(gt2, pred2)
    print(f"  v2 (Gaussian, sigma0_sq={args.sigma0_sq}): "
          f"F1={f1_2:.3f} P={p2:.3f} R={r2:.3f} "
          f"avg_topics={np.mean(ntopics2):.1f} avg_max_share={np.mean(mshare2):.2f}")

    tau_grid = args.tau_sweep or [args.tau]
    thr_grid = args.cos_thr_sweep or [args.cos_threshold]
    rho_grid = args.rho_sweep or [args.rho]

    # v3.1 sweep
    best31 = None
    for tau in tau_grid:
        for thr in thr_grid:
            gt31, pred31, ntopics31, mshare31 = run_v3_1(
                convs, embeddings, args.alpha, args.lmda, tau, thr,
            )
            p31, r31, f1_31 = f1_score(gt31, pred31)
            print(f"  v3.1 tau={tau:5.1f} thr={thr:.2f}: "
                  f"F1={f1_31:.3f} P={p31:.3f} R={r31:.3f} "
                  f"avg_topics={np.mean(ntopics31):.1f} "
                  f"avg_max_share={np.mean(mshare31):.2f}")
            if best31 is None or f1_31 > best31[0]:
                best31 = (f1_31, p31, r31, tau, thr, ntopics31, mshare31)
    f1_31, p31, r31, tau31, thr31, ntopics31, mshare31 = best31
    print(f"  v3.1 BEST tau={tau31} thr={thr31:.2f} F1={f1_31:.3f}")

    # v3.2 sweep
    best32 = None
    for tau in tau_grid:
        for thr in thr_grid:
            for rho in rho_grid:
                gt32, pred32, ntopics32, mshare32 = run_v3_2(
                    convs, embeddings, args.alpha, args.lmda, tau, thr, rho,
                )
                p32, r32, f1_32 = f1_score(gt32, pred32)
                print(f"  v3.2 tau={tau:5.1f} thr={thr:.2f} rho={rho:.2f}: "
                      f"F1={f1_32:.3f} P={p32:.3f} R={r32:.3f} "
                      f"avg_topics={np.mean(ntopics32):.1f} "
                      f"avg_max_share={np.mean(mshare32):.2f}")
                if best32 is None or f1_32 > best32[0]:
                    best32 = (f1_32, p32, r32, tau, thr, rho,
                              ntopics32, mshare32)
    f1_32, p32, r32, tau32, thr32, rho32, ntopics32, mshare32 = best32
    print(f"  v3.2 BEST tau={tau32} thr={thr32:.2f} rho={rho32:.2f} F1={f1_32:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TopiOCQA — Hi-OnTop 변형 모두 비교 (v2 / v3.1 / v3.2)",
        "",
        f"`alpha={args.alpha}` `lmda={args.lmda}` `sigma0_sq={args.sigma0_sq}` "
        f"device={args.device or 'auto'} limit_convs={args.limit_convs}",
        "",
        f"Data: {n_convs} conv / {n_turns} turns / {n_shifts} GT shifts",
        "",
        "## Topic shift F1",
        "",
        "| Method | Precision | Recall | F1 | avg #topics | avg max-share |",
        "|---|---|---|---|---|---|",
        f"| v2 hi-ontop-full-v1 (Gaussian) | {p2:.3f} | {r2:.3f} | {f1_2:.3f} | "
        f"{np.mean(ntopics2):.1f} | {np.mean(mshare2):.3f} |",
        f"| v3.1 hi-ontop-full-v3.1.1 (Bounded Cosine MAP, τ={tau31}, thr={thr31}) | "
        f"{p31:.3f} | {r31:.3f} | {f1_31:.3f} | "
        f"{np.mean(ntopics31):.1f} | {np.mean(mshare31):.3f} |",
        f"| v3.2 hi-ontop-full-v3.2.1 (Cosine PE, τ={tau32}, thr={thr32}, ρ={rho32}) | "
        f"{p32:.3f} | {r32:.3f} | {f1_32:.3f} | "
        f"{np.mean(ntopics32):.1f} | {np.mean(mshare32):.3f} |",
        "",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"report → {out}")


if __name__ == "__main__":
    main()
