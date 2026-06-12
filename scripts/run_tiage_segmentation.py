#!/usr/bin/env python3
"""Run Hi-OnTop + baselines on TIAGE for topic-shift detection.

Data: benchmarks/tiage/data/personachat/anno/{train,dev,test}/anno_{split}.json
    dialog = list of [utterance, label]
    label: '-1' (first turn), '0' (no shift), '1' (shift at this turn)

Metric: topic-shift F1 (turn-transition binary)
Baselines:
    (a) all-boundary
    (b) cosine-threshold (sweep on dev, fix on test — reported here)
    (c) Hi-OnTop option A

HP: Hi-OnTop persistence default (α=1, λ=10, σ₀²=0.01) vs
    freq-shift (α=10, λ=1, σ₀²=0.1). TopiOCQA-style grid sweep optional.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_ontop.embedding import QueryEncoder  # noqa: E402
from hi_ontop.sem_core import HiOnTopSegmenter  # noqa: E402


DATA_DIR = REPO_ROOT / "benchmarks" / "tiage" / "data" / "personachat" / "anno"


def load_split(split: str) -> dict[str, list[tuple[str, str]]]:
    path = DATA_DIR / split / f"anno_{split}.json"
    raw = json.loads(path.read_text())
    return {cid: [(t[0], t[1]) for t in dialog] for cid, dialog in raw.items()}


def gt_shifts(dialog: list[tuple[str, str]]) -> list[bool]:
    """Return list of length N-1. True if shift at transition i→i-1.

    Label on turn i (1 ≤ i < N) is '0' or '1'. '-1' only on i=0.
    Shift_{i→i+1} = (label[i+1] == '1'). Indices over transitions (skip i=0).
    """
    labels = [t[1] for t in dialog]
    return [labels[i] == '1' for i in range(1, len(labels))]


def f1_score(gt: list[bool], pred: list[bool]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        if g and p: tp += 1
        elif p and not g: fp += 1
        elif g and not p: fn += 1
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def run_hi_ontop(dialogs, embeddings, alpha, lmda, sigma0_sq):
    gt_all, pred_all = [], []
    t0 = time.perf_counter()
    n_turns = 0
    for (cid, dialog), emb in zip(dialogs.items(), embeddings):
        seg = HiOnTopSegmenter(dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq)
        assignments = [seg.assign(s) for s in emb]
        gt_all.extend(gt_shifts(dialog))
        pred_all.extend(a[1] for a in assignments[1:])
        n_turns += len(dialog)
    return f1_score(gt_all, pred_all), time.perf_counter() - t0, n_turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--threshold-sweep", nargs="+", type=float,
                        default=list(np.arange(0.3, 0.95, 0.025)))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lmda", type=float, default=10.0)
    parser.add_argument("--sigma0-sq", type=float, default=0.01)
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "outputs" / "phase-1-tiage.md"))
    args = parser.parse_args()

    print(f"[data] loading TIAGE {args.split}...")
    dialogs = load_split(args.split)
    n_convs = len(dialogs)
    n_turns = sum(len(d) for d in dialogs.values())
    n_shifts = sum(sum(gt_shifts(d)) for d in dialogs.values())
    print(f"  {n_convs} conv / {n_turns} turns / {n_shifts} shifts")

    print("[encode] bge-base-en-v1.5...")
    enc = QueryEncoder()
    all_utts = [t[0] for d in dialogs.values() for t in d]
    t0 = time.perf_counter()
    emb_flat = np.asarray(enc.encode(all_utts))
    enc_sec = time.perf_counter() - t0
    print(f"  encoded {n_turns} turns in {enc_sec:.1f}s ({enc_sec/n_turns*1000:.2f} ms/turn) on {enc.device}")

    embeddings = []
    idx = 0
    for d in dialogs.values():
        embeddings.append(emb_flat[idx:idx + len(d)])
        idx += len(d)

    # (a) all-boundary
    gt, pred = [], []
    for d in dialogs.values():
        gs = gt_shifts(d)
        gt.extend(gs)
        pred.extend([True] * len(gs))
    ab_prf = f1_score(gt, pred)
    print(f"(a) all-boundary : P={ab_prf[0]:.3f} R={ab_prf[1]:.3f} F1={ab_prf[2]:.3f}")

    # (b) cosine threshold
    best = (0.0, 0.0, -1.0, None)
    for thr in args.threshold_sweep:
        gt, pred = [], []
        for d, emb in zip(dialogs.values(), embeddings):
            gt.extend(gt_shifts(d))
            for i in range(1, len(d)):
                pred.append(float(np.dot(emb[i], emb[i-1])) < thr)
        prf = f1_score(gt, pred)
        if prf[2] > best[2]:
            best = (*prf, float(thr))
    print(f"(b) cosine θ={best[3]:.3f}: P={best[0]:.3f} R={best[1]:.3f} F1={best[2]:.3f}")

    # (c) Hi-OnTop — persistence HP
    (p, r, f), hi_sec, _ = run_hi_ontop(dialogs, embeddings, args.alpha, args.lmda, args.sigma0_sq)
    print(f"(c) Hi-OnTop α={args.alpha}, λ={args.lmda}, σ₀²={args.sigma0_sq}: "
          f"P={p:.3f} R={r:.3f} F1={f:.3f}  (assign {hi_sec*1000/n_turns:.3f} ms/turn)")

    # Hi-OnTop freq-shift HP for comparison
    (p2, r2, f2), _, _ = run_hi_ontop(dialogs, embeddings, alpha=10.0, lmda=1.0, sigma0_sq=0.1)
    print(f"(c') Hi-OnTop freq-shift: P={p2:.3f} R={r2:.3f} F1={f2:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overhead_ms = enc_sec / n_turns * 1000 + hi_sec / n_turns * 1000
    md = f"""# Phase 1-3 (augment) — TIAGE topic-shift detection

실행: `python scripts/run_tiage_segmentation.py --split {args.split}` (device={enc.device})

## 데이터 — TIAGE {args.split}
- dialogs: {n_convs}
- total turns: {n_turns}
- topic-shift labels ('1'): {n_shifts} (shift rate {n_shifts/(n_turns-n_convs):.3f} / transition)

## 지표 — topic-shift F1 (turn-transition binary)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| (a) all-boundary | {ab_prf[0]:.3f} | {ab_prf[1]:.3f} | {ab_prf[2]:.3f} |
| (b) cosine-threshold (θ={best[3]:.3f}) | {best[0]:.3f} | {best[1]:.3f} | {best[2]:.3f} |
| (c) Hi-OnTop persistence (α={args.alpha}, λ={args.lmda}, σ₀²={args.sigma0_sq}) | {p:.3f} | {r:.3f} | {f:.3f} |
| (c') Hi-OnTop freq-shift (α=10, λ=1, σ₀²=0.1) | {p2:.3f} | {r2:.3f} | {f2:.3f} |

## Latency
- embed: {enc_sec:.1f}s / {enc_sec/n_turns*1000:.2f} ms/turn
- Hi-OnTop assign: {hi_sec*1000:.1f} ms total / {hi_sec/n_turns*1000:.3f} ms/turn
- 총 overhead: {overhead_ms:.2f} ms/turn

## Gate 판정 (plan.md Phase 1-4 criteria, TIAGE 적용)

- Hi-OnTop F1 > cosine baseline F1: **{f > best[2]}** ({f:.3f} vs {best[2]:.3f})
- Hi-OnTop F1 > 0.4: **{f > 0.4}** ({f:.3f})
- 턴당 overhead ≤ 200ms (LLM 1000ms의 20%): **{overhead_ms <= 200.0}** ({overhead_ms:.2f} ms)

**Gate**: {'PASS' if (f > best[2] and f > 0.4 and overhead_ms <= 200.0) else 'FAIL'}
"""
    out_path.write_text(md)
    print(f"\nreport → {out_path}")


if __name__ == "__main__":
    main()
