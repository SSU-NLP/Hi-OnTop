#!/usr/bin/env python3
"""Contextualized scene embedding experiment on TopiOCQA dev.

Rationale
    TopiOCQA is a conversational QA benchmark: turns are written with
    coreference and ellipsis resolved *against prior Context*. Using the
    raw Question alone creates artificial ambiguity ("what is it about?",
    "the latter place" — untethered without Context). Hi-OnTop in its real
    use case (Claude-style long chat) would *naturally* see the running
    conversation state, so embedding decontextualized Questions is a
    **design misuse**, not a fair ceiling.

    This experiment measures Hi-OnTop + baselines when the scene vector is
    the embedding of ``"[last K (Q, A) pairs] [current Q]"`` — the
    standard conversational-QA convention, not a TopiOCQA-specific trick.

We sweep window size K ∈ {0, 1, 2, 3} and also "all" (entire prior
Context). The best K is expected to be small (1~2) because bge-base has
512-token cap and longer context may dilute the topic signal.

Hi-OnTop HP fixed at the Phase 1-3 winner: α=10, λ=1, σ₀²=0.1 (SEM2 defaults
— TopiOCQA's frequent-shift regime).
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


# --- Contextualized scene text ----------------------------------------

def contextualized_text(turn: dict, k: int | str) -> str:
    """Return ``"Q: ... A: ... ... Q: current"`` string.

    ``k`` = number of prior (Q, A) pairs to prepend. ``"all"`` uses the
    entire Context field.
    """
    ctx_pairs: list[tuple[str, str]] = []
    raw = turn["Context"]
    # Context is a flat list [Q1, A1, Q2, A2, ...]
    for i in range(0, len(raw) - 1, 2):
        ctx_pairs.append((raw[i], raw[i + 1]))
    if k == "all":
        pairs = ctx_pairs
    else:
        pairs = ctx_pairs[-int(k):] if int(k) > 0 else []
    parts = []
    for q, a in pairs:
        parts.append(f"Q: {q} A: {a}")
    parts.append(f"Q: {turn['Question']}")
    return " ".join(parts)


def encode_all(enc: QueryEncoder, convs, k) -> list[np.ndarray]:
    texts = []
    index = []
    for ci, c in enumerate(convs):
        for ti, t in enumerate(c):
            texts.append(contextualized_text(t, k))
            index.append((ci, ti))
    arr = np.asarray(enc.encode(texts))
    embeddings: list[np.ndarray] = [np.zeros((len(c), arr.shape[1])) for c in convs]
    for (ci, ti), v in zip(index, arr):
        embeddings[ci][ti] = v
    return embeddings


# --- Metrics pipelines ------------------------------------------------

def eval_cosine_threshold(convs, embeddings):
    best = (0.0, 0.0, -1.0, None)
    for thr in np.arange(0.3, 0.95, 0.025):
        gt, pred = [], []
        for c, emb in zip(convs, embeddings):
            gt.extend(gt_shifts(c))
            for i in range(1, len(c)):
                pred.append(float(np.dot(emb[i], emb[i-1])) < thr)
        prf = f1(gt, pred)
        if prf[2] > best[2]:
            best = (*prf, float(thr))
    return best  # (P, R, F1, thr)


def eval_hi_ontop(convs, embeddings, alpha, lmda, sigma0_sq):
    gt_all, pred_all = [], []
    for c, emb in zip(convs, embeddings):
        seg = HiOnTopSegmenter(dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq)
        gt_all.extend(gt_shifts(c))
        assignments = [seg.assign(e) for e in emb]
        pred_all.extend(a[1] for a in assignments[1:])
    return f1(gt_all, pred_all)


def eval_all_boundary(convs):
    gt, pred = [], []
    for c in convs:
        gs = gt_shifts(c)
        gt.extend(gs)
        pred.extend([True] * len(gs))
    return f1(gt, pred)


# --- Main -------------------------------------------------------------

def main():
    convs = load()
    n_turns = sum(len(c) for c in convs)
    print(f"[data] {len(convs)} conv / {n_turns} turns")

    enc = QueryEncoder()
    ab_prf = eval_all_boundary(convs)
    print(f"(a) all-boundary  : F1={ab_prf[2]:.3f}")

    windows = [0, 1, 2, 3, "all"]
    summary = []
    for K in windows:
        print(f"\n[K={K}] encoding contextualized text...")
        t0 = time.perf_counter()
        embeddings = encode_all(enc, convs, K)
        enc_sec = time.perf_counter() - t0
        # average token-ish length
        avg_chars = np.mean([len(contextualized_text(t, K))
                             for c in convs for t in c])
        print(f"   encoded in {enc_sec:.1f}s ({enc_sec/n_turns*1000:.1f} ms/turn), avg {avg_chars:.0f} chars/turn")

        # cosine baseline
        best_cos = eval_cosine_threshold(convs, embeddings)
        print(f"   cosine θ={best_cos[3]:.3f}  "
              f"P={best_cos[0]:.3f} R={best_cos[1]:.3f} F1={best_cos[2]:.3f}")

        # Hi-OnTop (SEM2 defaults = Phase 1-3 winner for TopiOCQA regime)
        hi_prf = eval_hi_ontop(convs, embeddings, alpha=10.0, lmda=1.0, sigma0_sq=0.1)
        print(f"   Hi-OnTop            "
              f"P={hi_prf[0]:.3f} R={hi_prf[1]:.3f} F1={hi_prf[2]:.3f}")

        # Hi-OnTop with persistence HP (Hi-OnTop-original) also, for reference
        hi_pers = eval_hi_ontop(convs, embeddings, alpha=1.0, lmda=10.0, sigma0_sq=0.01)
        print(f"   Hi-OnTop(α=1,λ=10)  "
              f"P={hi_pers[0]:.3f} R={hi_pers[1]:.3f} F1={hi_pers[2]:.3f}")

        summary.append({
            "K": K,
            "avg_chars": float(avg_chars),
            "encode_ms_per_turn": enc_sec / n_turns * 1000,
            "cosine": {"threshold": best_cos[3], "P": best_cos[0], "R": best_cos[1], "F1": best_cos[2]},
            "hi_ontop_freqshift": {"alpha": 10.0, "lmda": 1.0, "sigma0_sq": 0.1,
                                "P": hi_prf[0], "R": hi_prf[1], "F1": hi_prf[2]},
            "hi_ontop_persistence": {"alpha": 1.0, "lmda": 10.0, "sigma0_sq": 0.01,
                                  "P": hi_pers[0], "R": hi_pers[1], "F1": hi_pers[2]},
        })

    out = REPO_ROOT / "outputs" / "phase-1-topiocqa-contextualized.json"
    out.write_text(json.dumps({"all_boundary_F1": ab_prf[2], "windows": summary}, indent=2))
    print(f"\npersisted → {out}")

    print("\n" + "=" * 72)
    print(f"{'K':>5}{'chars':>10}{'enc_ms':>10}{'cos_F1':>10}{'hi_F1':>10}{'hi_F1 pers':>12}")
    for s in summary:
        print(
            f"{str(s['K']):>5}"
            f"{int(s['avg_chars']):>10}"
            f"{s['encode_ms_per_turn']:>10.1f}"
            f"{s['cosine']['F1']:>10.3f}"
            f"{s['hi_ontop_freqshift']['F1']:>10.3f}"
            f"{s['hi_ontop_persistence']['F1']:>12.3f}"
        )
    print(f"all-boundary F1: {ab_prf[2]:.3f} (constant, K-invariant)")


if __name__ == "__main__":
    main()
