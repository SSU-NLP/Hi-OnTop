#!/usr/bin/env python3
"""Grid sweep over (sigma0_sq, lmda, alpha) for Hi-OnTop on TIAGE test.

TopiOCQA-symmetric counterpart of `run_topiocqa_sweep.py`.

Goal: confirm whether **any** HP configuration in the same grid lifts Hi-OnTop
above the cosine baseline (F1 0.421) on TIAGE chit-chat. Two-point evidence
(persistence 0.317, freq-shift 0.377) is insufficient for the Phase 1-6
reframing argument; a full grid sweep removes the asymmetry vs TopiOCQA.
"""

from __future__ import annotations

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


def load_split(split: str = "test") -> dict[str, list[tuple[str, str]]]:
    path = DATA_DIR / split / f"anno_{split}.json"
    raw = json.loads(path.read_text())
    return {cid: [(t[0], t[1]) for t in dialog] for cid, dialog in raw.items()}


def gt_shifts(dialog: list[tuple[str, str]]) -> list[bool]:
    labels = [t[1] for t in dialog]
    return [labels[i] == "1" for i in range(1, len(labels))]


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


def run_hi_ontop(dialogs, embeddings, alpha, lmda, sigma0_sq):
    gt_all, pred_all = [], []
    for (cid, dialog), emb in zip(dialogs.items(), embeddings):
        seg = HiOnTopSegmenter(dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq)
        assignments = [seg.assign(s) for s in emb]
        gt_all.extend(gt_shifts(dialog))
        pred_all.extend(a[1] for a in assignments[1:])
    return f1(gt_all, pred_all)


def main():
    split = "test"
    print(f"loading TIAGE {split}...")
    dialogs = load_split(split)
    n_turns = sum(len(d) for d in dialogs.values())
    print(f"  {len(dialogs)} conv / {n_turns} turns")

    print("encoding...")
    enc = QueryEncoder()
    all_utts = [u for dialog in dialogs.values() for (u, _) in dialog]
    emb_flat = np.asarray(enc.encode(all_utts))
    embeddings, idx = [], 0
    for dialog in dialogs.values():
        embeddings.append(emb_flat[idx:idx + len(dialog)])
        idx += len(dialog)

    # Cosine baseline (same θ grid as run_tiage_segmentation default)
    print("cosine baseline sweep...")
    best_cos = (0.0, 0.0, -1.0, None)
    for thr in list(np.arange(0.3, 0.95, 0.025)):
        gt, pred = [], []
        for (cid, dialog), emb in zip(dialogs.items(), embeddings):
            gt.extend(gt_shifts(dialog))
            for i in range(1, len(dialog)):
                pred.append(float(np.dot(emb[i], emb[i - 1])) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_cos[2]:
            best_cos = (*prf, float(thr))
    print(f"  best cosine: θ={best_cos[3]:.3f}  P={best_cos[0]:.3f} R={best_cos[1]:.3f} F1={best_cos[2]:.3f}")

    # Hi-OnTop grid — identical to TopiOCQA sweep (3×6×6 = 108 configs)
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
                P, R, F = run_hi_ontop(dialogs, embeddings, a, l, s)
                results.append((a, l, s, P, R, F))
                print(f"{a:>6.2f} {l:>6.2f} {s:>10.4f} {P:>7.3f} {R:>7.3f} {F:>7.3f}")
    print(f"(sweep took {time.perf_counter() - t0:.1f}s, {len(results)} configs)")

    results.sort(key=lambda x: -x[5])
    print("\ntop 10 by F1:")
    print(header)
    for r in results[:10]:
        print(f"{r[0]:>6.2f} {r[1]:>6.2f} {r[2]:>10.4f} {r[3]:>7.3f} {r[4]:>7.3f} {r[5]:>7.3f}")
    print(f"\ncosine baseline F1: {best_cos[2]:.3f} (θ={best_cos[3]:.3f})")
    best = results[0]
    print(f"best Hi-OnTop F1: {best[5]:.3f} (α={best[0]}, λ={best[1]}, σ₀²={best[2]})")
    print(f"Gate cond 1 (Hi-OnTop > cosine): {best[5] > best_cos[2]}")
    print(f"Gate cond 2 (Hi-OnTop > 0.4)    : {best[5] > 0.4}")

    out = REPO_ROOT / "outputs" / "phase-1-tiage-sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "split": split,
        "cosine_best": {"P": best_cos[0], "R": best_cos[1], "F1": best_cos[2], "threshold": best_cos[3]},
        "hi_ontop_grid": [
            {"alpha": r[0], "lmda": r[1], "sigma0_sq": r[2], "P": r[3], "R": r[4], "F1": r[5]}
            for r in results
        ],
    }, indent=2))
    print(f"persisted → {out}")


if __name__ == "__main__":
    main()
