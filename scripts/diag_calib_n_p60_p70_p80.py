#!/usr/bin/env python3
"""Calib N convergence — 3 encoders × 3 datasets × {p60, p70, p80}.

기존 `diag_calib_n_convergence.py` 의 setup 그대로 (70:30 calib/test for
dialseg711, Score-based oracle, Score-based best_score_dstar) 사용. 차이:
- 인코더 = 3종 모두 (mpnet, minilm, minilm-int8)
- percentile = p60, p70, p80 (3종)
- N grid = [3, 5, 10, 25, 50, 100, 200, 400]
- best (continuous sweep on calib) 도 같이

cached embeddings 만 사용 → CSM full run 방해 없음.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402
from run_encoder_comparison import (  # noqa: E402
    DSTAR_GRID, M, RHO, A, ENCODERS, MPNET_REUSE,
    load_dialogs, score_set, best_score_dstar,
)
from diag_calib_n_convergence import load_calib, delta_eff_seq  # noqa: E402

ENCS = ("mpnet", "minilm", "minilm-int8")
PERCENTILES = (60, 70, 80)
N_GRID = [3, 5, 10, 25, 50, 100, 200, 400]
SEEDS = [0, 1, 2]


def test_oracle(te_dia, te_deff):
    best = (-1.0, 0.0)
    for d in DSTAR_GRID:
        sc = score_set(te_dia, te_deff, d)["score"]
        if sc > best[0]:
            best = (sc, float(d))
    return best


def main() -> None:
    out = REPO / "outputs" / "experiments" / "2026-05-24_calib_n_p60_p70_p80"
    out.mkdir(parents=True, exist_ok=True)

    results = {}  # (enc, ds) -> {oracle, cap, rows: [(N, p60_mean, p60_std, p70_..., p80_..., best_mean, best_std)]}
    for enc in ENCS:
        for ds in ("tiage", "dialseg711", "superseg"):
            print(f"\n========== {enc} / {ds} ==========", flush=True)
            cal_dia, cal_emb, te_dia, te_emb = load_calib(enc, ds)
            cap = len(cal_dia)
            ns = [n for n in N_GRID if n <= cap]
            if ns[-1] != cap:
                ns.append(cap)
            print(f"  calib cap={cap}  N grid={ns}", flush=True)
            cal_deff = [delta_eff_seq(e) for e in cal_emb]
            te_deff = [delta_eff_seq(e) for e in te_emb]
            oracle_score, oracle_dstar = test_oracle(te_dia, te_deff)
            print(f"  test-side oracle: Score={oracle_score:.4f}",
                  flush=True)

            rows = []
            for N in ns:
                # per-seed accumulators
                p_acc = {p: [] for p in PERCENTILES}
                best_acc = []
                seed_iter = [None] if N == cap else SEEDS
                for seed in seed_iter:
                    if seed is None:
                        idx = list(range(cap))
                    else:
                        rng = np.random.default_rng(seed)
                        idx = sorted(rng.permutation(cap)[:N])
                    cd = [cal_dia[i] for i in idx]
                    cde = [cal_deff[i] for i in idx]
                    allv = np.array([d for s in cde for d in s[1:]])
                    for p in PERCENTILES:
                        dstar_p = float(np.percentile(allv, p))
                        sc = score_set(te_dia, te_deff, dstar_p)["score"]
                        p_acc[p].append(sc)
                    dstar_bs = best_score_dstar(cd, cde)
                    best_acc.append(score_set(te_dia, te_deff, dstar_bs)["score"])
                row = dict(N=N)
                for p in PERCENTILES:
                    row[f"p{p}_mean"] = float(np.mean(p_acc[p]))
                    row[f"p{p}_std"] = float(np.std(p_acc[p]))
                row["best_mean"] = float(np.mean(best_acc))
                row["best_std"] = float(np.std(best_acc))
                rows.append(row)
                print(f"    N={N:5d}  "
                      f"p60={row['p60_mean']:.4f}±{row['p60_std']:.4f}  "
                      f"p70={row['p70_mean']:.4f}±{row['p70_std']:.4f}  "
                      f"p80={row['p80_mean']:.4f}±{row['p80_std']:.4f}  "
                      f"best={row['best_mean']:.4f}±{row['best_std']:.4f}",
                      flush=True)

            results[f"{enc}/{ds}"] = dict(
                oracle=oracle_score, oracle_dstar=oracle_dstar,
                cap=cap, rows=rows)

    (out / "results.json").write_text(json.dumps(results, indent=2))

    # REPORT
    L = ["# Calib N convergence — p60 / p70 / p80 × 3 encoders × 3 benches",
         "",
         "Setup (기존 `diag_calib_n_convergence.py` 와 동일):",
         "- harness = `run_encoder_comparison.py` 함수 import + cached embeddings",
         "- split: tiage = train(300)/test(100), superseg = train(400)/test(1322), "
         "dialseg711 = test 70:30 (calib 498 / test 213; 70% as calib)",
         "- **best** = continuous Score sweep on calib → test 평가",
         "- **oracle** = continuous Score sweep on **test** itself (절대 천장, label leakage 인정)",
         "- N grid = [3, 5, 10, 25, 50, 100, 200, 400] (cap 까지)",
         "- 3-seed bootstrap (seed ∈ {0,1,2}), N=cap 은 single point",
         "- Hi-OnTop HP: m=2, ρ=0.7, a=0.5",
         ""]
    for enc in ENCS:
        for ds in ("tiage", "dialseg711", "superseg"):
            key = f"{enc}/{ds}"
            r = results[key]
            L += [f"## {enc} / {ds}  (oracle {r['oracle']:.4f}, cap {r['cap']})",
                  "",
                  "| N | Score (p60) | Score (p70) | Score (p80) | Score (best) | gap (best→oracle) |",
                  "|---:|---:|---:|---:|---:|---:|"]
            for row in r["rows"]:
                gap = r["oracle"] - row["best_mean"]
                L.append(f"| {row['N']} | "
                         f"{row['p60_mean']:.4f} ± {row['p60_std']:.4f} | "
                         f"{row['p70_mean']:.4f} ± {row['p70_std']:.4f} | "
                         f"{row['p80_mean']:.4f} ± {row['p80_std']:.4f} | "
                         f"{row['best_mean']:.4f} ± {row['best_std']:.4f} | "
                         f"{gap:+.4f} |")
            L.append("")

    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
