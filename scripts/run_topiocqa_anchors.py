#!/usr/bin/env python3
"""Test alternative anchoring strategies for the Hi-OnTop likelihood on TopiOCQA.

Motivated by: with avg topic length 3.3 turns in TopiOCQA, the running
centroid averages over only a handful of (potentially section-varying)
points. A non-stationary-friendly anchor (EMA or last-turn) may be more
sensitive to true topic boundaries.

Variants scored against existing topic k on the arrival of s:
    centroid     cos(s, mu_k) with mu_k = running mean  (current)
    ema          cos(s, mu_k) with EMA update mu_k = beta*s + (1-beta)*mu_k
    last         cos(s, last_in_k)  (most recent scene assigned to k)
    max_c_l      max(cos(s, centroid), cos(s, last_in_k))  (take stronger)

All wrapped inside the standard sCRP prior. HP alpha, lmda, sigma0_sq
swept around the Phase 1-3 winners.
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


def load():
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


def cos_safe(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run(convs, embeddings, variant, alpha, lmda, beta_ema, sigma0_sq, new_baseline):
    """Variant ∈ {'centroid', 'ema', 'last', 'max_c_l'}.

    All variants score existing topic as `kappa * cos(s, anchor)` with
    kappa=1/sigma0_sq  — units match Gaussian log-likelihood scale.
    """
    kappa = 1.0 / sigma0_sq
    gt_all, pred_all = [], []
    for conv, emb in zip(convs, embeddings):
        dim = emb.shape[1]
        centroids: list[np.ndarray] = []
        last_scene: list[np.ndarray] = []
        n_in_topic: list[int] = []
        counts = np.zeros(64, dtype=np.int64)
        prev_k: int | None = None
        assignments = []
        for s in emb:
            prior = counts.astype(np.float64, copy=True)
            n_vis = int(np.count_nonzero(counts))
            if n_vis < counts.shape[0]:
                prior[n_vis] += alpha
            if prev_k is not None:
                prior[prev_k] += lmda
            active = np.flatnonzero(prior)
            log_scores = np.empty(active.shape[0])
            for i, k in enumerate(active):
                k_int = int(k)
                if k_int < len(centroids):
                    if variant == "centroid":
                        cs = cos_safe(s, centroids[k_int])
                    elif variant == "ema":
                        cs = cos_safe(s, centroids[k_int])  # EMA stored in centroids
                    elif variant == "last":
                        cs = cos_safe(s, last_scene[k_int])
                    elif variant == "max_c_l":
                        cs = max(cos_safe(s, centroids[k_int]), cos_safe(s, last_scene[k_int]))
                    else:
                        raise ValueError(variant)
                    score = kappa * cs
                else:
                    score = kappa * new_baseline
                log_scores[i] = math.log(prior[k_int]) + score
            chosen = int(active[int(np.argmax(log_scores))])
            while len(centroids) <= chosen:
                centroids.append(np.zeros(dim))
                last_scene.append(np.zeros(dim))
                n_in_topic.append(0)
            # update
            n = n_in_topic[chosen] + 1
            if variant == "ema":
                if n_in_topic[chosen] == 0:
                    centroids[chosen] = s.copy()
                else:
                    centroids[chosen] = beta_ema * s + (1.0 - beta_ema) * centroids[chosen]
            else:
                centroids[chosen] = centroids[chosen] + (s - centroids[chosen]) / n
            last_scene[chosen] = s
            n_in_topic[chosen] = n
            counts[chosen] += 1
            assignments.append((chosen, prev_k is not None and chosen != prev_k))
            prev_k = chosen
        gt_all.extend(gt_shifts(conv))
        pred_all.extend(a[1] for a in assignments[1:])
    return f1(gt_all, pred_all)


def main():
    convs = load()
    n_turns = sum(len(c) for c in convs)
    print(f"[data] {len(convs)} conv / {n_turns} turns")

    print("encoding with bge-base-en-v1.5...")
    enc = QueryEncoder()
    all_q = [t["Question"] for c in convs for t in c]
    t0 = time.perf_counter()
    emb_flat = np.asarray(enc.encode(all_q))
    print(f"  encoded in {time.perf_counter()-t0:.1f}s")
    embeddings, idx = [], 0
    for c in convs:
        embeddings.append(emb_flat[idx:idx + len(c)])
        idx += len(c)

    # cosine reference
    best_cos = (0.0, 0.0, -1.0, None)
    for thr in np.arange(0.3, 0.95, 0.025):
        gt, pred = [], []
        for c, e in zip(convs, embeddings):
            gt.extend(gt_shifts(c))
            for i in range(1, len(c)):
                pred.append(float(np.dot(e[i], e[i-1])) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_cos[2]:
            best_cos = (*prf, float(thr))
    print(f"cosine ref θ={best_cos[3]:.3f} F1={best_cos[2]:.3f}")

    # Hi-OnTop Phase 1-3 best (centroid+Gaussian): re-run a reference point
    print("\n[anchor grids]")
    alphas = [1.0, 10.0]
    lmdas = [0.0, 1.0, 3.0]
    sigmas = [0.05, 0.1]  # scale for kappa
    new_bases = [0.3, 0.5, 0.7]
    betas_ema = [0.3, 0.5, 0.7, 0.9]

    results = defaultdict(list)
    for variant in ("centroid", "ema", "last", "max_c_l"):
        best = (0.0, 0.0, -1.0, None, None, None, None, None)
        for a in alphas:
            for l in lmdas:
                for s in sigmas:
                    for nb in new_bases:
                        betas = betas_ema if variant == "ema" else [None]
                        for b in betas:
                            P, R, F = run(convs, embeddings, variant, a, l, b or 0.5, s, nb)
                            if F > best[2]:
                                best = (P, R, F, a, l, s, nb, b)
        print(
            f"  {variant:<10} best: α={best[3]} λ={best[4]} σ={best[5]} "
            f"nb={best[6]} β={best[7]}  P={best[0]:.3f} R={best[1]:.3f} F1={best[2]:.3f}"
        )
        results[variant] = {
            "P": best[0], "R": best[1], "F1": best[2],
            "alpha": best[3], "lmda": best[4], "sigma0_sq": best[5],
            "new_baseline": best[6], "beta_ema": best[7],
        }

    out = REPO_ROOT / "outputs" / "phase-1-topiocqa-anchors.json"
    out.write_text(json.dumps({
        "cosine_ref": {"threshold": best_cos[3], "F1": best_cos[2]},
        "variants": dict(results),
    }, indent=2))
    print(f"\npersisted → {out}")


if __name__ == "__main__":
    main()
