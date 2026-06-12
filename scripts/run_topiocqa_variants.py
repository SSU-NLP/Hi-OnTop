#!/usr/bin/env python3
"""Explore structural variants of Hi-OnTop option A on TopiOCQA dev.

Goal: break the F1 ~0.47 ceiling observed under (origin, Gaussian).

Variants:
    V0  Gaussian-origin        (current implementation)
    V1  Gaussian-globalmean    (new-cluster cold start = running corpus mean)
    V2  Gaussian-selfcenter    (new-cluster scene = its own centroid — ablation)
    V3  vMF-origin             (cosine-scaled likelihood, origin new-cluster)
    V4  vMF-constlik           (cosine for existing, fixed constant for new)

For each variant, sweep (alpha, lmda, concentration) over a small grid and
report best F1. We skip sigma0_sq for vMF (no Gaussian variance). Kappa (vMF
concentration) replaces it.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_ontop.embedding import QueryEncoder  # noqa: E402


# --- Data --------------------------------------------------------------

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


# --- Unified segmenter with pluggable likelihood ------------------------

class TopicG:
    """Gaussian-centroid topic."""
    __slots__ = ("mu", "M2", "n", "dim")
    def __init__(self, dim):
        self.dim = dim
        self.mu = np.zeros(dim)
        self.M2 = np.zeros(dim)
        self.n = 0
    def update(self, s):
        self.n += 1
        d = s - self.mu
        self.mu = self.mu + d / self.n
        self.M2 = self.M2 + d * (s - self.mu)
    def variance(self, sigma0_sq, sigma_min_sq=1e-6):
        if self.n < 3:
            return np.full(self.dim, sigma0_sq)
        return np.maximum(self.M2 / self.n, sigma_min_sq)


def log_prior(counts, prev_k, alpha, lmda):
    prior = counts.astype(np.float64, copy=True)
    n_vis = int(np.count_nonzero(counts))
    if n_vis < counts.shape[0]:
        prior[n_vis] += alpha
    if prev_k is not None:
        prior[prev_k] += lmda
    return prior


def run_variant(convs, embeddings, variant, alpha, lmda, param):
    """variant='gauss-origin'|'gauss-global'|'gauss-self'|'vmf-origin'|'vmf-const'
    param = sigma0_sq for gauss-*, kappa for vmf-*.
    """
    gt_all, pred_all = [], []
    for conv, emb in zip(convs, embeddings):
        dim = emb.shape[1]
        topics: list[TopicG] = []
        counts = np.zeros(64, dtype=np.int64)
        prev_k: int | None = None
        mu_global = np.zeros(dim)
        n_global = 0
        assignments = []
        for s in emb:
            n_global += 1
            mu_global = mu_global + (s - mu_global) / n_global
            prior = log_prior(counts, prev_k, alpha, lmda)
            active = np.flatnonzero(prior)
            log_scores = np.empty(active.shape[0])
            for i, k in enumerate(active):
                k_int = int(k)
                if k_int < len(topics):
                    t = topics[k_int]
                    if variant.startswith("gauss"):
                        s2 = t.variance(param)
                        diff = s - t.mu
                        ll = -0.5 * np.sum(diff * diff / s2 + np.log(2 * math.pi * s2))
                    else:  # vmf
                        ll = param * float(np.dot(s, t.mu) / (np.linalg.norm(t.mu) + 1e-9))
                else:
                    if variant == "gauss-origin":
                        s2 = param
                        ll = -0.5 * (float(np.sum(s * s)) / s2 + dim * math.log(2 * math.pi * s2))
                    elif variant == "gauss-global":
                        s2 = param
                        diff = s - mu_global
                        ll = -0.5 * (float(np.sum(diff * diff)) / s2 + dim * math.log(2 * math.pi * s2))
                    elif variant == "gauss-self":
                        s2 = param
                        ll = -0.5 * dim * math.log(2 * math.pi * s2)
                    elif variant == "vmf-origin":
                        # baseline cosine toward zero-vector is 0
                        ll = 0.0
                    elif variant == "vmf-const":
                        ll = -1.0 * param  # constant threshold; kappa-scaled
                    else:
                        raise ValueError(variant)
                log_scores[i] = math.log(prior[k_int]) + ll
            chosen = int(active[int(np.argmax(log_scores))])
            while len(topics) <= chosen:
                topics.append(TopicG(dim))
            topics[chosen].update(s)
            counts[chosen] += 1
            assignments.append((chosen, prev_k is not None and chosen != prev_k))
            prev_k = chosen
        gt_all.extend(gt_shifts(conv))
        pred_all.extend(a[1] for a in assignments[1:])
    return f1(gt_all, pred_all)


def main():
    print("loading...")
    convs = load()
    print(f"  {len(convs)} conv / {sum(len(c) for c in convs)} turns")

    print("encoding...")
    enc = QueryEncoder()
    all_q = [t["Question"] for c in convs for t in c]
    emb_flat = np.asarray(enc.encode(all_q))
    embeddings, idx = [], 0
    for c in convs:
        embeddings.append(emb_flat[idx:idx + len(c)])
        idx += len(c)

    # Cosine ref
    print("cosine reference (θ sweep)...")
    best_cos = (0.0, 0.0, -1.0, None)
    for thr in np.arange(0.3, 0.90, 0.025):
        gt, pred = [], []
        for conv, emb in zip(convs, embeddings):
            gt.extend(gt_shifts(conv))
            for i in range(1, len(conv)):
                pred.append(float(np.dot(emb[i], emb[i-1])) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_cos[2]:
            best_cos = (*prf, float(thr))
    print(f"  best cosine: θ={best_cos[3]:.3f}  F1={best_cos[2]:.3f}")

    grids = {
        "gauss-origin":  dict(alphas=[1, 10], lmdas=[0, 1, 3], params=[0.01, 0.05, 0.1, 0.2]),
        "gauss-global":  dict(alphas=[0.1, 1, 10], lmdas=[0, 0.3, 1, 3], params=[0.001, 0.005, 0.01, 0.05, 0.1]),
        "gauss-self":    dict(alphas=[0.1, 1, 10], lmdas=[0, 0.3, 1, 3], params=[0.01, 0.05, 0.1]),
        "vmf-origin":    dict(alphas=[0.1, 1, 10], lmdas=[0, 0.3, 1, 3], params=[1.0, 3.0, 10.0, 30.0]),
        "vmf-const":     dict(alphas=[0.1, 1, 10], lmdas=[0, 0.3, 1, 3], params=[0.5, 1.0, 2.0, 3.0]),
    }

    summary = {}
    for variant, g in grids.items():
        print(f"\n[variant] {variant}")
        best = (0.0, 0.0, -1.0, None, None, None)
        t0 = time.perf_counter()
        for a in g["alphas"]:
            for l in g["lmdas"]:
                for p in g["params"]:
                    P, R, F = run_variant(convs, embeddings, variant, a, l, p)
                    if F > best[2]:
                        best = (P, R, F, a, l, p)
        took = time.perf_counter() - t0
        print(
            f"  best: α={best[3]} λ={best[4]} param={best[5]}  "
            f"P={best[0]:.3f} R={best[1]:.3f} F1={best[2]:.3f} ({took:.1f}s)"
        )
        summary[variant] = {
            "alpha": best[3], "lmda": best[4], "param": best[5],
            "P": best[0], "R": best[1], "F1": best[2],
        }

    print("\n" + "=" * 58)
    print(f"cosine baseline F1: {best_cos[2]:.3f} (θ={best_cos[3]:.3f})")
    print(f"{'variant':<18}{'α':>6}{'λ':>8}{'param':>10}{'P':>7}{'R':>7}{'F1':>7}")
    for v, r in summary.items():
        print(f"{v:<18}{r['alpha']:>6}{r['lmda']:>8}{r['param']:>10}"
              f"{r['P']:>7.3f}{r['R']:>7.3f}{r['F1']:>7.3f}")

    out = REPO_ROOT / "outputs" / "phase-1-topiocqa-variants.json"
    out.write_text(json.dumps(
        {"cosine": {"threshold": best_cos[3], "F1": best_cos[2]}, "variants": summary},
        indent=2,
    ))
    print(f"\npersisted → {out}")


if __name__ == "__main__":
    main()
