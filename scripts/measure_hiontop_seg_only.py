"""Clean per-variant Seg-only latency (HiOnTop.assign timing).

For each (encoder × variant), measure HiOnTop.assign() timing using
the bench-specific δ* on cached embeddings of that bench, then average
across 3 benches. This avoids the unprincipled "mean δ*" approach —
each measurement corresponds to a real (variant, bench) cell.

Output: outputs/experiments/2026-05-24_hiontop_seg_only_latency/
  per_variant_seg_ms.json
  REPORT.md
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

OUT_DIR = REPO / "outputs" / "experiments" / "2026-05-24_hiontop_seg_only_latency"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = REPO / "outputs" / "runs" / "_misc"

# Cached embeddings per (encoder, bench)
EMB_PKLS = {
    ("mpnet", "tiage"):      "sds_emb_tiage_test.pkl",
    ("mpnet", "dialseg711"): "sds_emb_dialseg711_test.pkl",
    ("mpnet", "superseg"):   "sds_emb_superseg_test.pkl",
    ("int8",  "tiage"):      "enccmp_tiage_test_minilm-int8.pkl",
    ("int8",  "dialseg711"): "enccmp_dialseg711_test_minilm-int8.pkl",
    ("int8",  "superseg"):   "enccmp_superseg_test_minilm-int8.pkl",
}

# δ* values per (encoder, variant, bench) — exactly as used in dts_result.md
DSTAR = {
    ("mpnet", "p60"): {"tiage": 0.5296, "dialseg711": 0.5145, "superseg": 0.5304},
    ("mpnet", "p70"): {"tiage": 0.5618, "dialseg711": 0.5514, "superseg": 0.5751},
    ("mpnet", "p80"): {"tiage": 0.6016, "dialseg711": 0.5939, "superseg": 0.6194},
    ("mpnet", "sup"): {"tiage": 0.5737, "dialseg711": 0.6144, "superseg": 0.5534},
    ("mpnet", "oracle"): {"tiage": 0.5432, "dialseg711": 0.6144, "superseg": 0.5432},
    ("int8",  "p60"): {"tiage": 0.7334, "dialseg711": 0.7033, "superseg": 0.7255},
    ("int8",  "p70"): {"tiage": 0.7763, "dialseg711": 0.7519, "superseg": 0.7839},
    ("int8",  "p80"): {"tiage": 0.8223, "dialseg711": 0.8029, "superseg": 0.8409},
    ("int8",  "sup"): {"tiage": 0.7873, "dialseg711": 0.8280, "superseg": 0.6042},
    ("int8",  "oracle"): {"tiage": 0.7771, "dialseg711": 0.8178, "superseg": 0.6246},
}
HP_M, HP_RHO, HP_A = 2, 0.7, 0.5


def load_embs(enc: str, bench: str):
    with open(CACHE / EMB_PKLS[(enc, bench)], "rb") as fh:
        embs = pickle.load(fh)
    return [np.asarray(e) for e in embs]


def time_one_bench(embs, dstar, n_warmup=200, n_seeds=3, sample_size=1500):
    """Time HiOnTop.assign() on samples from one bench's embeddings."""
    flat = [v for emb in embs for v in emb]
    n_total = len(flat)
    if n_total < sample_size + n_warmup:
        sample_size = max(100, n_total - n_warmup - 100)
    all_ms = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_total)[:sample_size]
        sample = [flat[i] for i in idx]
        # warmup
        seg = HiOnTop(dim=sample[0].shape[0], delta_star=dstar,
                     ctx_window=HP_M, ctx_decay=HP_RHO, ctx_blend_a=HP_A)
        for v in sample[:n_warmup]:
            seg.assign(np.asarray(v, dtype=np.float64))
        # timed
        seg = HiOnTop(dim=sample[0].shape[0], delta_star=dstar,
                     ctx_window=HP_M, ctx_decay=HP_RHO, ctx_blend_a=HP_A)
        ms_list = []
        for v in sample:
            v64 = np.asarray(v, dtype=np.float64)
            t0 = time.perf_counter()
            seg.assign(v64)
            ms_list.append((time.perf_counter() - t0) * 1000.0)
        all_ms.extend(ms_list[1:])
    return np.asarray(all_ms)


def main():
    cached_embs = {}
    results = {}
    for (enc, variant), bench_dstar in DSTAR.items():
        per_bench_ms = {}
        for bench, dstar in bench_dstar.items():
            key_embs = (enc, bench)
            if key_embs not in cached_embs:
                cached_embs[key_embs] = load_embs(enc, bench)
            embs = cached_embs[key_embs]
            ms_arr = time_one_bench(embs, dstar)
            per_bench_ms[bench] = ms_arr
        # cross-bench mean & per-bench reporting
        all_ms = np.concatenate(list(per_bench_ms.values()))
        bench_means = {b: float(m.mean()) for b, m in per_bench_ms.items()}
        results[f"{enc}_{variant}"] = dict(
            encoder=enc, variant=variant,
            delta_star_per_bench=bench_dstar,
            seg_ms_per_bench_mean=bench_means,
            seg_ms_cross_bench_mean=float(all_ms.mean()),
            seg_ms_cross_bench_p50=float(np.percentile(all_ms, 50)),
            seg_ms_cross_bench_std=float(all_ms.std()),
            n_total=int(len(all_ms)),
        )
        print(f"{enc}_{variant}: cross-bench mean={all_ms.mean():.4f} "
              f"(tiage={bench_means.get('tiage',0):.4f} "
              f"ds711={bench_means.get('dialseg711',0):.4f} "
              f"sds={bench_means.get('superseg',0):.4f}) ms",
              flush=True)

    (OUT_DIR / "per_variant_seg_ms.json").write_text(json.dumps(results, indent=2))

    L = ["# Hi-OnTop per-variant Seg latency (HiOnTop.assign timing, per-bench δ*)",
         "", "## 정의",
         "- 각 (variant × bench) 의 *실제 δ\\** 로 `HiOnTop.assign()` 만 perf_counter.",
         "- 3-seed × 1500-turn sample per bench → 3 bench → 1 variant 당 ~13500 sample.",
         "- 보고값 = cross-bench mean (3 bench 평균).",
         "- 200-turn warmup, first turn 제외. cached embeddings.",
         "", "## 결과",
         "| variant | tiage Seg | ds711 Seg | sds Seg | cross-bench mean |",
         "|---|---:|---:|---:|---:|"]
    for key, r in results.items():
        bm = r["seg_ms_per_bench_mean"]
        L.append(f"| {key} | {bm.get('tiage',0):.4f} | {bm.get('dialseg711',0):.4f} | "
                 f"{bm.get('superseg',0):.4f} | **{r['seg_ms_cross_bench_mean']:.4f}** |")
    (OUT_DIR / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
