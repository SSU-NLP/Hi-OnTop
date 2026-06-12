#!/usr/bin/env python3
"""3D sweep of (m, rho, a) with per-bench oracle delta_star.

For each (m, rho, a) on the grid:
  - For each bench (tiage/dialseg711/superseg test sets):
      * compute delta_eff sequence per dialog (depends on m, rho, a)
      * oracle delta_star = argmax Score over a dense delta_star grid
      * record per-bench Score under that oracle
  - mean3 = mean over 3 benches
  - additionally compute best-p_x in {p50..p95} variant
    (semi-supervised, paper-deployable family)

Output: outputs/experiments/2026-05-25_hiontop_mra_sweep/sweep.json + REPORT.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from run_hiontop_hp_sweep import (  # noqa: E402
    load_dialogs, load_embs, split_frac, official_pk_wd,
)
from sklearn.metrics import f1_score
from hi_ontop.hi_ontop import HiOnTop

EXP_NAME = "2026-05-25_hiontop_mra_sweep"

GRID_M     = [2, 3, 4, 5, 6, 8]
GRID_RHO   = [0.5, 0.7, 0.9]
GRID_A     = [0.0, 0.3, 0.5, 0.7, 1.0]
DSTAR_GRID = np.linspace(0.35, 0.95, 121)   # continuous oracle sweep
PERCENTILES = list(range(50, 100, 5))       # {p50, p55, ..., p95}

DEFAULT_MRA = (2, 0.7, 0.5)


def delta_eff_seq(emb: np.ndarray, m: int, rho: float, a: float) -> list[float]:
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                  ctx_window=m, ctx_decay=rho, ctx_blend_a=a)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def boundaries(seq, dstar):
    return [1 if seq[i] >= dstar else 0 for i in range(1, len(seq))] + [0]


def score_set(dialogs, deffs, dstar):
    pks, wds, g, p = [], [], [], []
    for (utts, yt), seq in zip(dialogs, deffs):
        yp = boundaries(seq, dstar)
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    if not g:
        return dict(pk=1.0, wd=1.0, f1=0.0, score=0.0)
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    return dict(pk=pk_m, wd=wd_m, f1=f1,
                 score=0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m))


def main() -> None:
    exp_dir = REPO / "outputs" / "experiments" / EXP_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_json = exp_dir / "sweep.json"

    # test sets — paper convention: dialseg711 uses 30% eval (70% reserved for calib)
    # tiage / superseg: full test split.
    print("[load] test sets...")
    ds711_dia = load_dialogs("dialseg711", "test")
    ds711_emb = load_embs("dialseg711", "test")
    # split_frac(0.70, 0) → first 70% = calib (unused here), last 30% = eval.
    _, ds711_eval = split_frac(ds711_dia, ds711_emb, 0.70, 0)
    test_sets = {
        "tiage":      (load_dialogs("tiage", "test"),
                        load_embs("tiage", "test")),
        "dialseg711": ds711_eval,    # 30% eval (paper convention)
        "superseg":   (load_dialogs("superseg", "test"),
                        load_embs("superseg", "test")),
    }
    for n, (d, _) in test_sets.items():
        print(f"  test/{n}: {len(d)} dialogs")

    results = []
    if out_json.exists():
        results = json.loads(out_json.read_text()).get("grid", [])
        print(f"[resume] {len(results)} rows loaded")
    done = {(r["m"], r["rho"], r["a"]) for r in results}

    grid = [(m, rho, a) for m in GRID_M for rho in GRID_RHO for a in GRID_A]
    print(f"\n[sweep] {len(grid)} configs ({len(done)} done)")
    t0 = time.perf_counter()
    for i, (m, rho, a) in enumerate(grid):
        if (m, rho, a) in done:
            continue
        row = dict(m=m, rho=rho, a=a)
        oracle_scores = {}
        bestpx_scores = {}
        oracle_dstars = {}
        bestpx_choices = {}
        for name, (dia, emb) in test_sets.items():
            deffs = [delta_eff_seq(e, m, rho, a) for e in emb]
            # oracle: continuous sweep on test
            best_d, best_s = None, -1.0
            for d in DSTAR_GRID:
                s = score_set(dia, deffs, float(d))["score"]
                if s > best_s:
                    best_s = s; best_d = float(d)
            oracle_scores[name] = best_s
            oracle_dstars[name] = best_d
            # best-p_x family
            allv = np.array([v for s in deffs for v in s[1:]])
            best_p, best_ps = None, -1.0
            for p in PERCENTILES:
                d = float(np.percentile(allv, p))
                s = score_set(dia, deffs, d)["score"]
                if s > best_ps:
                    best_ps = s; best_p = p
            bestpx_scores[name] = best_ps
            bestpx_choices[name] = best_p
        row["oracle_scores"] = oracle_scores
        row["oracle_dstars"] = oracle_dstars
        row["oracle_mean3"]  = float(np.mean(list(oracle_scores.values())))
        row["bestpx_scores"] = bestpx_scores
        row["bestpx_choices"] = bestpx_choices
        row["bestpx_mean3"]  = float(np.mean(list(bestpx_scores.values())))
        results.append(row)
        if (i + 1) % 5 == 0 or i == len(grid) - 1:
            out_json.write_text(json.dumps({"grid": results}, indent=2))
            dt = time.perf_counter() - t0
            eta = dt / (len(results)) * (len(grid) - len(results))
            print(f"  [{i+1}/{len(grid)}] m={m} ρ={rho} a={a}  "
                  f"oracle_mean3={row['oracle_mean3']:.4f}  "
                  f"bestpx_mean3={row['bestpx_mean3']:.4f}  "
                  f"({dt:.0f}s elapsed, ~{eta:.0f}s eta)", flush=True)
    out_json.write_text(json.dumps({"grid": results}, indent=2))

    # rankings
    print("\n=== Top-10 by ORACLE mean-3 ===")
    by_o = sorted(results, key=lambda r: -r["oracle_mean3"])
    print(f"{'rank':>4}  {'m':>2} {'ρ':>4} {'a':>4}  "
           f"{'tiage':>7} {'ds711':>7} {'sds':>7} {'mean3':>7}")
    for i, r in enumerate(by_o[:10], 1):
        mark = " ← default" if (r["m"], r["rho"], r["a"]) == DEFAULT_MRA else ""
        print(f"{i:>4}  {r['m']:>2} {r['rho']:>4} {r['a']:>4}  "
              f"{r['oracle_scores']['tiage']:>7.4f} "
              f"{r['oracle_scores']['dialseg711']:>7.4f} "
              f"{r['oracle_scores']['superseg']:>7.4f} "
              f"{r['oracle_mean3']:>7.4f}{mark}")

    print("\n=== Top-10 by BEST-p_x mean-3 ===")
    by_p = sorted(results, key=lambda r: -r["bestpx_mean3"])
    print(f"{'rank':>4}  {'m':>2} {'ρ':>4} {'a':>4}  "
           f"{'tiage':>7} {'ds711':>7} {'sds':>7} {'mean3':>7}")
    for i, r in enumerate(by_p[:10], 1):
        mark = " ← default" if (r["m"], r["rho"], r["a"]) == DEFAULT_MRA else ""
        print(f"{i:>4}  {r['m']:>2} {r['rho']:>4} {r['a']:>4}  "
              f"{r['bestpx_scores']['tiage']:>7.4f} "
              f"{r['bestpx_scores']['dialseg711']:>7.4f} "
              f"{r['bestpx_scores']['superseg']:>7.4f} "
              f"{r['bestpx_mean3']:>7.4f}{mark}")

    # default ranks
    def find_rank(by_metric):
        for idx, row in enumerate(by_metric, 1):
            if (row["m"], row["rho"], row["a"]) == DEFAULT_MRA:
                return idx, row
        return None, None
    o_rank, o_row = find_rank(by_o)
    p_rank, p_row = find_rank(by_p)
    print(f"\n=== Default (m=2, ρ=0.7, a=0.5) ===")
    if o_row:
        print(f"  ORACLE  rank: {o_rank} / {len(by_o)}  mean3={o_row['oracle_mean3']:.4f}  "
              f"(Δ from rank-1: {o_row['oracle_mean3'] - by_o[0]['oracle_mean3']:+.4f})")
    if p_row:
        print(f"  BEST-px rank: {p_rank} / {len(by_p)}  mean3={p_row['bestpx_mean3']:.4f}  "
              f"(Δ from rank-1: {p_row['bestpx_mean3'] - by_p[0]['bestpx_mean3']:+.4f})")


if __name__ == "__main__":
    main()
