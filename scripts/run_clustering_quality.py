#!/usr/bin/env python3
"""Phase 1-6 옵션 5: clustering quality (V-measure / NMI / ARI) 측정.

Boundary F1만으로는 Hi-OnTop의 차별점(토픽 ID 부여 + centroid 누적)을 측정할 수 없다.
Cosine baseline은 "boundary 지나면 다른 cluster"라 **돌아온 토픽을 같은 cluster로
묶지 못함**. Hi-OnTop은 centroid 비교라 같은 토픽으로 복귀 시 같은 ID 부여 가능.

이 스크립트는 두 벤치마크에서 그 가설을 직접 측정한다.

Method:
    - GT cluster ID
        TopiOCQA: 각 turn의 ``Topic`` 필드 그대로 (string label)
        TIAGE: binary shift label로부터 derive — shift=1 만나면 ID 증가
    - Predicted cluster ID
        cosine(θ*): boundary 발생 위치마다 ID 증가 (sequential)
        Hi-OnTop:      ``HiOnTopSegmenter.assign()`` 의 topic_id 그대로
    - Metrics: V-measure, NMI, ARI (sklearn)

Best HP per benchmark (Phase 1-4/1-6 sweep 결과):
    TopiOCQA: α=10, λ=1.0, σ₀²=0.1   (frequent-shift regime)
    TIAGE:    α=10, λ=3.0, σ₀²=0.1   (sweep best from run_tiage_sweep.py)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_ontop.embedding import QueryEncoder  # noqa: E402
from hi_ontop.sem_core import HiOnTopSegmenter  # noqa: E402


# --- Data loaders --------------------------------------------------------

def load_topiocqa() -> list[list[dict]]:
    path = (
        REPO_ROOT / "benchmarks" / "topiocqa" / "downloads" / "data"
        / "topiocqa_dataset" / "dev.json"
    )
    raw = json.loads(path.read_text())
    buckets: dict[int, list[dict]] = defaultdict(list)
    for t in raw:
        buckets[t["Conversation_no"]].append(t)
    return [sorted(buckets[c], key=lambda x: x["Turn_no"]) for c in sorted(buckets)]


def load_tiage(split: str = "test") -> dict[str, list[tuple[str, str]]]:
    path = REPO_ROOT / "benchmarks" / "tiage" / "data" / "personachat" / "anno" / split / f"anno_{split}.json"
    raw = json.loads(path.read_text())
    return {cid: [(t[0], t[1]) for t in dialog] for cid, dialog in raw.items()}


# --- GT cluster ID extraction --------------------------------------------

def topiocqa_gt_ids(conv: list[dict]) -> list[int]:
    """Map each turn's Topic string → integer ID (per-conversation reset)."""
    ids: dict[str, int] = {}
    out = []
    for t in conv:
        topic = t["Topic"]
        if topic not in ids:
            ids[topic] = len(ids)
        out.append(ids[topic])
    return out


def tiage_gt_ids(dialog: list[tuple[str, str]]) -> list[int]:
    """Derive cluster ID from binary shift labels: shift=1 → new ID.

    label[0] = '-1' (first turn, ID=0)
    label[i] = '1' → ID 증가
    label[i] = '0' → ID 유지
    """
    ids = [0]
    cur = 0
    for i in range(1, len(dialog)):
        if dialog[i][1] == "1":
            cur += 1
        ids.append(cur)
    return ids


# --- Predicted cluster ID ------------------------------------------------

def cosine_pred_ids(emb: np.ndarray, threshold: float) -> list[int]:
    """Sequential cluster: cos(s_i, s_{i-1}) < θ → boundary → new ID."""
    if len(emb) == 0:
        return []
    ids = [0]
    cur = 0
    for i in range(1, len(emb)):
        if float(np.dot(emb[i], emb[i - 1])) < threshold:
            cur += 1
        ids.append(cur)
    return ids


def hi_ontop_pred_ids(
    emb: np.ndarray, alpha: float, lmda: float, sigma0_sq: float
) -> list[int]:
    seg = HiOnTopSegmenter(dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq)
    return [seg.assign(s)[0] for s in emb]


# --- Aggregation across conversations ------------------------------------

def aggregate(gt_per_conv: list[list[int]], pred_per_conv: list[list[int]]) -> dict:
    """Concat with per-conv offset (so cluster IDs across conversations don't collide)."""
    gt_flat: list[int] = []
    pred_flat: list[int] = []
    gt_off = 0
    pred_off = 0
    for g, p in zip(gt_per_conv, pred_per_conv):
        gt_flat.extend([x + gt_off for x in g])
        pred_flat.extend([x + pred_off for x in p])
        gt_off += max(g) + 1 if g else 0
        pred_off += max(p) + 1 if p else 0
    gt_arr = np.asarray(gt_flat)
    pred_arr = np.asarray(pred_flat)
    return {
        "n_turns": len(gt_arr),
        "n_gt_clusters": int(gt_arr.max() + 1) if len(gt_arr) else 0,
        "n_pred_clusters": int(pred_arr.max() + 1) if len(pred_arr) else 0,
        "v_measure": float(v_measure_score(gt_arr, pred_arr)),
        "nmi": float(normalized_mutual_info_score(gt_arr, pred_arr)),
        "ari": float(adjusted_rand_score(gt_arr, pred_arr)),
        "homogeneity": float(homogeneity_score(gt_arr, pred_arr)),
        "completeness": float(completeness_score(gt_arr, pred_arr)),
    }


def find_best_cosine_threshold(
    embeddings: list[np.ndarray], gt_per_conv: list[list[int]], grid: list[float]
) -> tuple[float, dict]:
    """Pick θ that maximizes V-measure (parallel to F1-best in segmentation script)."""
    best_thr = grid[0]
    best_metrics = aggregate(gt_per_conv, [cosine_pred_ids(e, grid[0]) for e in embeddings])
    for thr in grid[1:]:
        m = aggregate(gt_per_conv, [cosine_pred_ids(e, thr) for e in embeddings])
        if m["v_measure"] > best_metrics["v_measure"]:
            best_thr = thr
            best_metrics = m
    return best_thr, best_metrics


# --- Per-benchmark runner ------------------------------------------------

def run_topiocqa(encoder: QueryEncoder) -> dict:
    print("\n[TopiOCQA dev]")
    convs = load_topiocqa()
    n_turns = sum(len(c) for c in convs)
    print(f"  {len(convs)} conv / {n_turns} turns")

    print("  encoding queries...")
    all_q = [t["Question"] for c in convs for t in c]
    emb_flat = np.asarray(encoder.encode(all_q))
    embeddings, idx = [], 0
    for c in convs:
        embeddings.append(emb_flat[idx:idx + len(c)])
        idx += len(c)

    gt_per_conv = [topiocqa_gt_ids(c) for c in convs]

    cos_grid = [round(0.3 + 0.025 * i, 4) for i in range(int((0.95 - 0.3) / 0.025) + 1)]
    print("  cosine θ sweep (V-measure best)...")
    best_thr, cos_metrics = find_best_cosine_threshold(embeddings, gt_per_conv, cos_grid)
    print(f"    best θ={best_thr}: V={cos_metrics['v_measure']:.3f} NMI={cos_metrics['nmi']:.3f} ARI={cos_metrics['ari']:.3f}")

    hi_ontop_configs = [
        ("freq-shift", 10.0, 1.0, 0.1),
        ("persistence", 1.0, 10.0, 0.01),
    ]
    hi_ontop_results = {}
    for name, a, l, s in hi_ontop_configs:
        print(f"  Hi-OnTop {name} (α={a}, λ={l}, σ₀²={s})...")
        pred = [hi_ontop_pred_ids(e, a, l, s) for e in embeddings]
        m = aggregate(gt_per_conv, pred)
        print(f"    V={m['v_measure']:.3f} NMI={m['nmi']:.3f} ARI={m['ari']:.3f}")
        hi_ontop_results[name] = {"hp": {"alpha": a, "lmda": l, "sigma0_sq": s}, **m}

    return {
        "benchmark": "topiocqa_dev",
        "cosine_best_threshold": best_thr,
        "cosine": cos_metrics,
        "hi_ontop": hi_ontop_results,
    }


def run_tiage(encoder: QueryEncoder) -> dict:
    print("\n[TIAGE test]")
    dialogs = load_tiage("test")
    n_turns = sum(len(d) for d in dialogs.values())
    print(f"  {len(dialogs)} conv / {n_turns} turns")

    print("  encoding utterances...")
    all_u = [u for dialog in dialogs.values() for (u, _) in dialog]
    emb_flat = np.asarray(encoder.encode(all_u))
    embeddings, idx = [], 0
    for dialog in dialogs.values():
        embeddings.append(emb_flat[idx:idx + len(dialog)])
        idx += len(dialog)

    gt_per_conv = [tiage_gt_ids(d) for d in dialogs.values()]

    cos_grid = [round(0.3 + 0.025 * i, 4) for i in range(int((0.95 - 0.3) / 0.025) + 1)]
    print("  cosine θ sweep (V-measure best)...")
    best_thr, cos_metrics = find_best_cosine_threshold(embeddings, gt_per_conv, cos_grid)
    print(f"    best θ={best_thr}: V={cos_metrics['v_measure']:.3f} NMI={cos_metrics['nmi']:.3f} ARI={cos_metrics['ari']:.3f}")

    hi_ontop_configs = [
        ("sweep-best", 10.0, 3.0, 0.1),
        ("persistence", 1.0, 10.0, 0.01),
    ]
    hi_ontop_results = {}
    for name, a, l, s in hi_ontop_configs:
        print(f"  Hi-OnTop {name} (α={a}, λ={l}, σ₀²={s})...")
        pred = [hi_ontop_pred_ids(e, a, l, s) for e in embeddings]
        m = aggregate(gt_per_conv, pred)
        print(f"    V={m['v_measure']:.3f} NMI={m['nmi']:.3f} ARI={m['ari']:.3f}")
        hi_ontop_results[name] = {"hp": {"alpha": a, "lmda": l, "sigma0_sq": s}, **m}

    return {
        "benchmark": "tiage_test",
        "cosine_best_threshold": best_thr,
        "cosine": cos_metrics,
        "hi_ontop": hi_ontop_results,
    }


def main() -> None:
    enc = QueryEncoder()
    print(f"encoder on {enc.device}")

    results = {
        "topiocqa": run_topiocqa(enc),
        "tiage": run_tiage(enc),
    }

    print("\n=== SUMMARY ===")
    print(f"{'benchmark':<14} {'method':<22} {'V-meas':>7} {'NMI':>7} {'ARI':>7} {'Hom':>6} {'Comp':>6}")
    for bench_key, r in results.items():
        rows = [(f"cosine(θ={r['cosine_best_threshold']})", r["cosine"])]
        for name, m in r["hi_ontop"].items():
            rows.append((f"Hi-OnTop {name}", m))
        for label, m in rows:
            print(f"{r['benchmark']:<14} {label:<22} "
                  f"{m['v_measure']:>7.3f} {m['nmi']:>7.3f} {m['ari']:>7.3f} "
                  f"{m['homogeneity']:>6.3f} {m['completeness']:>6.3f}")

    out_path = REPO_ROOT / "outputs" / "phase-1-clustering-quality.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\npersisted → {out_path}")


if __name__ == "__main__":
    main()
