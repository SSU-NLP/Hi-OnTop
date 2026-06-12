#!/usr/bin/env python3
"""Evaluate all 630 HP sweep configs on the *test* set.

The original sweep (`run_hiontop_hp_sweep.py`) computed Score on the
*tuning* split only. This script reuses the saved grid + cached embeddings
and computes test-set scores (tiage/test, dialseg711 70% holdout,
superseg/test) for every config, so we can verify whether the global
default is actually best on held-out test, not just on tuning.
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
    load_dialogs, load_embs, split_frac, eval_on,
)

DEFAULT_HP = dict(delta_star=0.5594, ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)

SRC_NAME = "2026-05-22_hiontop_hp_sweep"
DST_NAME = "2026-05-25_hiontop_hp_sweep_test_eval"


def main() -> None:
    src_dir = REPO / "outputs" / "experiments" / SRC_NAME
    dst_dir = REPO / "outputs" / "experiments" / DST_NAME
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_json = src_dir / "sweep_results.json"
    grid = json.loads(src_json.read_text())["grid"]
    print(f"[load] {len(grid)} configs from {src_json}")

    # build test sets (same as run_hiontop_hp_sweep main)
    print("[load] test sets...")
    ds711_dia = load_dialogs("dialseg711", "test")
    ds711_emb = load_embs("dialseg711", "test")
    _, ds711_test = split_frac(ds711_dia, ds711_emb, 0.30, 0)
    test_sets = {
        "tiage": (load_dialogs("tiage", "test"), load_embs("tiage", "test")),
        "dialseg711": ds711_test,
        "superseg": (load_dialogs("superseg", "test"), load_embs("superseg", "test")),
    }
    for n, (d, _) in test_sets.items():
        print(f"  test/{n}: {len(d)} dialogs")

    # evaluate every config on every test set
    out_json = dst_dir / "sweep_test_scores.json"
    results = []
    if out_json.exists():
        results = json.loads(out_json.read_text()).get("grid", [])
        print(f"[resume] {len(results)} rows loaded")
    done = {(r["delta_star"], r["ctx_window"], r["ctx_decay"], r["ctx_blend_a"])
            for r in results}

    n = len(grid)
    t0 = time.perf_counter()
    for i, cfg in enumerate(grid):
        key = (cfg["delta_star"], cfg["ctx_window"], cfg["ctx_decay"],
                cfg["ctx_blend_a"])
        if key in done:
            continue
        hp = dict(delta_star=cfg["delta_star"], ctx_window=cfg["ctx_window"],
                   ctx_decay=cfg["ctx_decay"], ctx_blend_a=cfg["ctx_blend_a"])
        per = {ds: eval_on(d, e, hp) for ds, (d, e) in test_sets.items()}
        mean3 = float(np.mean([per[ds]["score"] for ds in test_sets]))
        row = dict(
            delta_star=hp["delta_star"],
            ctx_window=hp["ctx_window"],
            ctx_decay=hp["ctx_decay"],
            ctx_blend_a=hp["ctx_blend_a"],
            tuning_score=cfg["tuning_score"],
            test_tiage=per["tiage"]["score"],
            test_dialseg711=per["dialseg711"]["score"],
            test_superseg=per["superseg"]["score"],
            test_mean3=mean3,
        )
        results.append(row)
        if (i + 1) % 20 == 0 or i == n - 1:
            out_json.write_text(json.dumps({"grid": results}, indent=2))
            dt = time.perf_counter() - t0
            eta = dt / (len(results)) * (n - len(results))
            print(f"  [{i+1}/{n}] mean3={mean3:.4f}  ({dt:.0f}s elapsed, ~{eta:.0f}s eta)",
                   flush=True)
    out_json.write_text(json.dumps({"grid": results}, indent=2))

    # ranking
    by_mean3 = sorted(results, key=lambda r: -r["test_mean3"])
    print("\n=== Top-15 by test mean-3 ===")
    print(f"{'rank':>4}  {'δ*':>6}  {'m':>2} {'ρ':>4} {'a':>4}  "
          f"{'tune':>7} {'tiage':>7} {'ds711':>7} {'sds':>7} {'mean3':>7}")
    for r, row in enumerate(by_mean3[:15], 1):
        print(f"{r:>4}  {row['delta_star']:>6}  {row['ctx_window']:>2} "
              f"{row['ctx_decay']:>4} {row['ctx_blend_a']:>4}  "
              f"{row['tuning_score']:>7.4f} "
              f"{row['test_tiage']:>7.4f} {row['test_dialseg711']:>7.4f} "
              f"{row['test_superseg']:>7.4f} {row['test_mean3']:>7.4f}")

    # find default's rank
    default_key = (DEFAULT_HP["delta_star"], DEFAULT_HP["ctx_window"],
                    DEFAULT_HP["ctx_decay"], DEFAULT_HP["ctx_blend_a"])
    default_idx = None
    for idx, row in enumerate(by_mean3, 1):
        k = (row["delta_star"], row["ctx_window"], row["ctx_decay"], row["ctx_blend_a"])
        if k == default_key:
            default_idx = idx
            default_row = row
            break

    print(f"\n=== Default ({DEFAULT_HP}) ===")
    if default_idx is not None:
        print(f"  rank: {default_idx} / {len(by_mean3)}  test mean3 = {default_row['test_mean3']:.4f}")
        print(f"  Δ from rank-1: {default_row['test_mean3'] - by_mean3[0]['test_mean3']:+.4f}")
        within_005 = sum(1 for r in by_mean3 if r["test_mean3"] >= default_row["test_mean3"] - 0.005)
        print(f"  configs within 0.005 of default test mean3: {within_005}")
    else:
        print("  default not found in grid")


if __name__ == "__main__":
    main()
