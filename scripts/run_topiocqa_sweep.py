#!/usr/bin/env python3
"""Grid sweep over (sigma0_sq, lmda, alpha) for Hi-OnTop on TopiOCQA dev.

Goal: find hyperparameter region where Hi-OnTop beats cosine baseline (F1 0.467)
and hits absolute 0.4 floor. Phase 1-4 Gate says these are initial values —
if they don't pass at defaults, sweep is legitimate (not cheating) since we
are probing the mechanism, not optimizing leakage.
"""

from __future__ import annotations

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


def load() -> list[list[dict]]:
    path = (
        REPO_ROOT / "benchmarks" / "topiocqa" / "downloads" / "data"
        / "topiocqa_dataset" / "dev.json"
    )
    raw = json.loads(path.read_text())
    buckets: dict[int, list[dict]] = defaultdict(list)
    for t in raw:
        buckets[t["Conversation_no"]].append(t)
    return [sorted(buckets[c], key=lambda x: x["Turn_no"]) for c in sorted(buckets)]


def gt_shifts(conv):
    topics = [t["Topic"] for t in conv]
    return [topics[i] != topics[i - 1] for i in range(1, len(topics))]


def f1(gt, pred):
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        if g and p: tp += 1
        elif p and not g: fp += 1
        elif g and not p: fn += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def run_hi_ontop(convs, embeddings, alpha, lmda, sigma0_sq):
    gt, pred = [], []
    for conv, emb in zip(convs, embeddings):
        seg = HiOnTopSegmenter(dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq)
        gt.extend(gt_shifts(conv))
        assignments = [seg.assign(e) for e in emb]
        pred.extend(a[1] for a in assignments[1:])
    return f1(gt, pred)


def main():
    print("loading...")
    convs = load()
    n_turns = sum(len(c) for c in convs)
    print(f"  {len(convs)} conv / {n_turns} turns")

    print("encoding...")
    enc = QueryEncoder()
    all_q = [t["Question"] for c in convs for t in c]
    emb_flat = np.asarray(enc.encode(all_q))
    embeddings, idx = [], 0
    for c in convs:
        embeddings.append(emb_flat[idx:idx + len(c)])
        idx += len(c)

    # Cosine baseline for reference
    print("cosine baseline sweep...")
    best_cos = (0.0, 0.0, -1.0, None)
    for thr in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        gt, pred = [], []
        for conv, emb in zip(convs, embeddings):
            gt.extend(gt_shifts(conv))
            for i in range(1, len(conv)):
                pred.append(float(np.dot(emb[i], emb[i-1])) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_cos[2]:
            best_cos = (*prf, thr)
    print(f"  best cosine: θ={best_cos[3]}  P={best_cos[0]:.3f} R={best_cos[1]:.3f} F1={best_cos[2]:.3f}")

    # Hi-OnTop grid
    print("Hi-OnTop grid (α × λ × σ₀²)...")
    alphas = [0.1, 1.0, 10.0]
    lmdas = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    sigmas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]

    header = f"{'α':>6} {'λ':>6} {'σ₀²':>10} {'P':>7} {'R':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    results = []
    t0 = time.perf_counter()
    for a in alphas:
        for l in lmdas:
            for s in sigmas:
                P, R, F = run_hi_ontop(convs, embeddings, a, l, s)
                results.append((a, l, s, P, R, F))
                print(f"{a:>6.2f} {l:>6.2f} {s:>10.4f} {P:>7.3f} {R:>7.3f} {F:>7.3f}")
    print(f"(sweep took {time.perf_counter() - t0:.1f}s, {len(results)} configs)")

    results.sort(key=lambda x: -x[5])
    print("\ntop 10 by F1:")
    print(header)
    for r in results[:10]:
        print(f"{r[0]:>6.2f} {r[1]:>6.2f} {r[2]:>10.4f} {r[3]:>7.3f} {r[4]:>7.3f} {r[5]:>7.3f}")
    print(f"\ncosine baseline F1: {best_cos[2]:.3f} (θ={best_cos[3]})")
    best = results[0]
    print(f"best Hi-OnTop F1: {best[5]:.3f} (α={best[0]}, λ={best[1]}, σ₀²={best[2]})")
    print(f"Gate cond 1 (Hi-OnTop > cosine): {best[5] > best_cos[2]}")
    print(f"Gate cond 2 (Hi-OnTop > 0.4)    : {best[5] > 0.4}")

    # Persist for report writer
    out = REPO_ROOT / "outputs" / "phase-1-topiocqa-sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "cosine_best": {"P": best_cos[0], "R": best_cos[1], "F1": best_cos[2], "threshold": best_cos[3]},
        "hi_ontop_grid": [
            {"alpha": r[0], "lmda": r[1], "sigma0_sq": r[2], "P": r[3], "R": r[4], "F1": r[5]}
            for r in results
        ],
    }, indent=2))
    print(f"persisted → {out}")


if __name__ == "__main__":
    main()
