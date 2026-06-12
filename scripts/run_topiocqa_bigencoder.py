#!/usr/bin/env python3
"""Test stronger embedding models for TopiOCQA topic shift detection.

Rationale: bge-base-en-v1.5 is compact (110M params, 768d). If the F1
ceiling ~0.47 across all our prior experiments reflects *that* encoder's
limit, a larger model in the same family (bge-large-en-v1.5, 335M params,
1024d) should close the gap. If even bge-large tops out similarly, the
bottleneck is the data (short decontextualized factoid queries), not the
encoder.

We compare two encoders on the same segmentation pipeline:
    bge-base-en-v1.5   (current Hi-OnTop default)
    bge-large-en-v1.5  (larger, same family)

Metric: best-of-three = max F1 across (cosine-θ, Hi-OnTop α=10/λ=1/σ=0.1,
Hi-OnTop α=1/λ=10/σ=0.01).
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_ontop.sem_core import HiOnTopSegmenter  # noqa: E402


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


def eval_model(model_name, convs):
    print(f"\n[encoder] {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    enc = SentenceTransformer(model_name, device=device)
    load_sec = time.perf_counter() - t0

    all_q = [t["Question"] for c in convs for t in c]
    n_turns = len(all_q)
    t1 = time.perf_counter()
    emb_flat = np.asarray(enc.encode(all_q, normalize_embeddings=True, show_progress_bar=False))
    enc_sec = time.perf_counter() - t1
    print(f"  load {load_sec:.1f}s, encode {enc_sec:.1f}s ({enc_sec/n_turns*1000:.1f} ms/turn), dim={emb_flat.shape[1]}")

    embeddings, idx = [], 0
    for c in convs:
        embeddings.append(emb_flat[idx:idx + len(c)])
        idx += len(c)

    # (a) all-boundary
    gt, pred = [], []
    for c in convs:
        gs = gt_shifts(c)
        gt.extend(gs)
        pred.extend([True] * len(gs))
    all_b = f1(gt, pred)

    # (b) cosine-θ sweep
    best_cos = (0.0, 0.0, -1.0, None)
    for thr in np.arange(0.3, 0.95, 0.02):
        gt, pred = [], []
        for c, e in zip(convs, embeddings):
            gt.extend(gt_shifts(c))
            for i in range(1, len(c)):
                pred.append(float(np.dot(e[i], e[i-1])) < thr)
        prf = f1(gt, pred)
        if prf[2] > best_cos[2]:
            best_cos = (*prf, float(thr))

    # (c) Hi-OnTop (frequent-shift HP)
    def hi_ontop(a, l, s):
        gt, pred = [], []
        for c, e in zip(convs, embeddings):
            seg = HiOnTopSegmenter(dim=e.shape[1], alpha=a, lmda=l, sigma0_sq=s)
            gt.extend(gt_shifts(c))
            ass = [seg.assign(x) for x in e]
            pred.extend(ax[1] for ax in ass[1:])
        return f1(gt, pred)

    hi_freq = hi_ontop(10.0, 1.0, 0.1)
    hi_pers = hi_ontop(1.0, 10.0, 0.01)

    print(f"  all-boundary     : F1={all_b[2]:.3f}")
    print(f"  cosine θ={best_cos[3]:.3f}  : F1={best_cos[2]:.3f} (P={best_cos[0]:.3f} R={best_cos[1]:.3f})")
    print(f"  Hi-OnTop freq-shift : F1={hi_freq[2]:.3f} (P={hi_freq[0]:.3f} R={hi_freq[1]:.3f})")
    print(f"  Hi-OnTop persistence: F1={hi_pers[2]:.3f} (P={hi_pers[0]:.3f} R={hi_pers[1]:.3f})")

    # cleanup
    del enc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "model": model_name,
        "dim": int(emb_flat.shape[1]),
        "encode_ms_per_turn": enc_sec / n_turns * 1000,
        "all_boundary_F1": all_b[2],
        "cosine":    {"threshold": best_cos[3], "P": best_cos[0], "R": best_cos[1], "F1": best_cos[2]},
        "hi_ontop_freqshift":  {"P": hi_freq[0], "R": hi_freq[1], "F1": hi_freq[2]},
        "hi_ontop_persistence": {"P": hi_pers[0], "R": hi_pers[1], "F1": hi_pers[2]},
    }


def main():
    convs = load()
    print(f"[data] {len(convs)} conv / {sum(len(c) for c in convs)} turns")

    models = [
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]
    results = [eval_model(m, convs) for m in models]

    print("\n" + "=" * 78)
    print(f"{'model':<30}{'dim':>5}{'enc ms':>10}{'cos_F1':>9}{'hi_fq':>8}{'hi_ps':>8}")
    for r in results:
        print(
            f"{r['model']:<30}{r['dim']:>5}"
            f"{r['encode_ms_per_turn']:>10.1f}"
            f"{r['cosine']['F1']:>9.3f}"
            f"{r['hi_ontop_freqshift']['F1']:>8.3f}"
            f"{r['hi_ontop_persistence']['F1']:>8.3f}"
        )

    out = REPO_ROOT / "outputs" / "phase-1-topiocqa-encoder.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"persisted → {out}")


if __name__ == "__main__":
    main()
