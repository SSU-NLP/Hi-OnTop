#!/usr/bin/env python3
"""Calib N convergence — Option A grid (dense in convergence region).

N grid = [3, 5, 7, 10, 13, 17, 22, 30, 40, 50, 75, 100, 150, 200, 300, 400]
seeds = 5
percentiles = p60, p70, p80
+ best (continuous Score sweep on calib)

cached embeddings → CSM full run 방해 없음 (numpy + NLTK only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_encoder_comparison import (  # noqa: E402
    DSTAR_GRID, score_set, best_score_dstar)
from diag_calib_n_convergence import load_calib, delta_eff_seq  # noqa: E402

ENCS = ("mpnet", "minilm", "minilm-int8")
PERCENTILES = (60, 70, 80)
N_GRID = [3, 5, 7, 10, 13, 17, 22, 30, 40, 50, 75, 100, 150,
          175, 200, 225, 250, 275, 300]
SEEDS = [0, 1, 2, 3, 4]


def test_oracle(te_dia, te_deff):
    best = (-1.0, 0.0)
    for d in DSTAR_GRID:
        sc = score_set(te_dia, te_deff, d)["score"]
        if sc > best[0]:
            best = (sc, float(d))
    return best


def main() -> None:
    out = REPO / "outputs" / "experiments" / "2026-05-24_calib_n_option_a"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for enc in ENCS:
        for ds in ("tiage", "dialseg711", "superseg"):
            print(f"\n========== {enc} / {ds} ==========", flush=True)
            cal_dia, cal_emb, te_dia, te_emb = load_calib(enc, ds)
            cap = len(cal_dia)
            ns = [n for n in N_GRID if n <= cap]
            # do NOT auto-append cap — Figure H is fixed to N≤300 across all benches
            cal_deff = [delta_eff_seq(e) for e in cal_emb]
            te_deff = [delta_eff_seq(e) for e in te_emb]
            oracle_score, oracle_dstar = test_oracle(te_dia, te_deff)
            print(f"  cap={cap}  N grid={ns}  oracle={oracle_score:.4f}",
                  flush=True)
            rows = []
            for N in ns:
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
                print(f"    N={N:4d}  "
                      f"p60={row['p60_mean']:.4f}±{row['p60_std']:.4f}  "
                      f"p70={row['p70_mean']:.4f}±{row['p70_std']:.4f}  "
                      f"p80={row['p80_mean']:.4f}±{row['p80_std']:.4f}  "
                      f"best={row['best_mean']:.4f}±{row['best_std']:.4f}",
                      flush=True)
            results[f"{enc}/{ds}"] = dict(
                oracle=oracle_score, oracle_dstar=oracle_dstar,
                cap=cap, rows=rows)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDONE → {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
