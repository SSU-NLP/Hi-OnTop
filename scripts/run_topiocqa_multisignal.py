#!/usr/bin/env python3
"""Multi-signal (option D) exploration on TopiOCQA dev.

Signals:
    cos   — bge cosine between s_t and topic centroid
    jac   — token jaccard between question_t and topic's accumulated tokens
    ent   — subset of jac restricted to proper-noun-ish tokens (title-case in raw
            Question) — cheap proxy for entity overlap

Scoring (for existing topic k):
    score(s, k) = log(prior_k) + β_cos * cos(s, μ_k) + β_jac * jac(s, k)
                  + β_ent * ent(s, k)

New-cluster score = log(α) + constant baseline.

Sweeps: β_cos, β_jac, β_ent, new-cluster baseline C.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_ontop.embedding import QueryEncoder  # noqa: E402


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "to", "and", "or", "but", "at", "by", "for", "from",
    "with", "about", "as", "this", "that", "these", "those", "it", "its",
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "do", "does", "did", "have", "has", "had", "can", "could", "would",
    "should", "may", "might", "will", "shall",
}


def tokens(q: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", q.lower())
    return {w for w in words if w not in STOPWORDS}


def entities(q: str) -> set[str]:
    """Cheap entity proxy: title-cased runs from the raw question (sans start)."""
    s = q.strip()
    # Skip first word (likely "what/when/how" etc.)
    rest = s[s.find(" "):] if " " in s else ""
    ents = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", rest)
    return {e.lower() for e in ents}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- Data --------------------------------------------------------------

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


# --- Multi-signal segmenter --------------------------------------------

def run_ms(
    convs, embeddings, question_tokens, question_ents,
    alpha, lmda, b_cos, b_jac, b_ent, new_baseline,
):
    gt_all, pred_all = [], []
    for conv, emb, qtok, qent in zip(convs, embeddings, question_tokens, question_ents):
        dim = emb.shape[1]
        topic_mu: list[np.ndarray] = []
        topic_n: list[int] = []
        topic_tok: list[set[str]] = []
        topic_ent: list[set[str]] = []
        counts = np.zeros(64, dtype=np.int64)
        prev_k: int | None = None
        assignments = []
        for s, tok, ent in zip(emb, qtok, qent):
            # prior
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
                if k_int < len(topic_mu):
                    mu = topic_mu[k_int]
                    cs = float(np.dot(s, mu) / (np.linalg.norm(mu) + 1e-9))
                    jc = jaccard(tok, topic_tok[k_int])
                    en = jaccard(ent, topic_ent[k_int])
                    sc = b_cos * cs + b_jac * jc + b_ent * en
                else:
                    sc = new_baseline
                log_scores[i] = np.log(prior[k_int]) + sc
            chosen = int(active[int(np.argmax(log_scores))])
            while len(topic_mu) <= chosen:
                topic_mu.append(np.zeros(dim))
                topic_n.append(0)
                topic_tok.append(set())
                topic_ent.append(set())
            n = topic_n[chosen] + 1
            topic_mu[chosen] = topic_mu[chosen] + (s - topic_mu[chosen]) / n
            topic_n[chosen] = n
            topic_tok[chosen] |= tok
            topic_ent[chosen] |= ent
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

    print("tokenizing / entity-extracting...")
    question_tokens = [[tokens(t["Question"]) for t in c] for c in convs]
    question_ents = [[entities(t["Question"]) for t in c] for c in convs]
    # quick stats
    n_tok = sum(len(x) for c in question_tokens for x in c)
    n_ent = sum(len(x) for c in question_ents for x in c)
    n_turns = sum(len(c) for c in convs)
    print(f"  avg tokens/turn: {n_tok/n_turns:.2f}, avg entities/turn: {n_ent/n_turns:.2f}")

    print("\n[baselines — single signal]")
    # Jaccard alone: shift if jaccard(tok_t, tok_{t-1}) < θ
    best_jac = (0.0, 0.0, -1.0, None)
    for thr in np.arange(0.0, 0.6, 0.02):
        gt, pred = [], []
        for c, qtok in zip(convs, question_tokens):
            gt.extend(gt_shifts(c))
            for i in range(1, len(c)):
                pred.append(jaccard(qtok[i], qtok[i-1]) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_jac[2]:
            best_jac = (*prf, float(thr))
    print(f"  jaccard-only (θ={best_jac[3]:.2f}): P={best_jac[0]:.3f} R={best_jac[1]:.3f} F1={best_jac[2]:.3f}")

    # Entity alone
    best_ent = (0.0, 0.0, -1.0, None)
    for thr in np.arange(0.0, 0.6, 0.02):
        gt, pred = [], []
        for c, qent in zip(convs, question_ents):
            gt.extend(gt_shifts(c))
            for i in range(1, len(c)):
                pred.append(jaccard(qent[i], qent[i-1]) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_ent[2]:
            best_ent = (*prf, float(thr))
    print(f"  entity-only  (θ={best_ent[3]:.2f}): P={best_ent[0]:.3f} R={best_ent[1]:.3f} F1={best_ent[2]:.3f}")

    # Cosine alone (for ref, against prev turn)
    best_cos = (0.0, 0.0, -1.0, None)
    for thr in np.arange(0.3, 0.90, 0.025):
        gt, pred = [], []
        for c, emb in zip(convs, embeddings):
            gt.extend(gt_shifts(c))
            for i in range(1, len(c)):
                pred.append(float(np.dot(emb[i], emb[i-1])) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_cos[2]:
            best_cos = (*prf, float(thr))
    print(f"  cosine-only  (θ={best_cos[3]:.3f}): P={best_cos[0]:.3f} R={best_cos[1]:.3f} F1={best_cos[2]:.3f}")

    print("\n[multi-signal Hi-OnTop grid]")
    # Coarse grid
    alphas = [1.0]
    lmdas = [0.0, 0.3, 1.0]
    b_cos_grid = [0.0, 1.0, 3.0]
    b_jac_grid = [0.0, 3.0, 10.0, 30.0]
    b_ent_grid = [0.0, 3.0, 10.0, 30.0]
    new_baseline_grid = [0.3, 0.5, 0.7, 1.0]

    results = []
    t0 = time.perf_counter()
    for a in alphas:
        for l in lmdas:
            for bc in b_cos_grid:
                for bj in b_jac_grid:
                    for be in b_ent_grid:
                        for nb in new_baseline_grid:
                            if bc == 0 and bj == 0 and be == 0:
                                continue
                            P, R, F = run_ms(
                                convs, embeddings, question_tokens, question_ents,
                                a, l, bc, bj, be, nb,
                            )
                            results.append((a, l, bc, bj, be, nb, P, R, F))
    took = time.perf_counter() - t0
    results.sort(key=lambda x: -x[8])
    print(f"  {len(results)} configs, took {took:.1f}s")

    print(f"\n{'α':>4}{'λ':>5}{'bcos':>6}{'bjac':>6}{'bent':>6}{'nb':>5}{'P':>7}{'R':>7}{'F1':>7}")
    for r in results[:15]:
        print(f"{r[0]:>4.1f}{r[1]:>5.2f}{r[2]:>6.1f}{r[3]:>6.1f}{r[4]:>6.1f}{r[5]:>5.2f}"
              f"{r[6]:>7.3f}{r[7]:>7.3f}{r[8]:>7.3f}")

    best = results[0]
    print("\n" + "=" * 60)
    print(f"cosine ref F1   : {best_cos[2]:.3f}")
    print(f"jaccard ref F1  : {best_jac[2]:.3f}")
    print(f"entity ref F1   : {best_ent[2]:.3f}")
    print(f"best multi-sig  : F1={best[8]:.3f} "
          f"(α={best[0]}, λ={best[1]}, bcos={best[2]}, bjac={best[3]}, bent={best[4]}, nb={best[5]})")
    print(f"Gate cond 1 (> cosine): {best[8] > best_cos[2]}")
    print(f"Gate cond 2 (> 0.4)   : {best[8] > 0.4}")

    out = REPO_ROOT / "outputs" / "phase-1-topiocqa-multisignal.json"
    out.write_text(json.dumps({
        "single": {
            "cosine": {"threshold": best_cos[3], "F1": best_cos[2]},
            "jaccard": {"threshold": best_jac[3], "F1": best_jac[2]},
            "entity":  {"threshold": best_ent[3], "F1": best_ent[2]},
        },
        "best_multi": {
            "alpha": best[0], "lmda": best[1],
            "b_cos": best[2], "b_jac": best[3], "b_ent": best[4], "new_baseline": best[5],
            "P": best[6], "R": best[7], "F1": best[8],
        },
        "top10": [
            {
                "alpha": r[0], "lmda": r[1],
                "b_cos": r[2], "b_jac": r[3], "b_ent": r[4], "new_baseline": r[5],
                "P": r[6], "R": r[7], "F1": r[8],
            } for r in results[:10]
        ],
    }, indent=2))
    print(f"persisted → {out}")


if __name__ == "__main__":
    main()
