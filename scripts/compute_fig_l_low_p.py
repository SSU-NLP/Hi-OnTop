#!/usr/bin/env python3
"""Extend Fig L data to p ∈ [5, 45] (the low-p tail not yet measured).

Uses cached DTS embeddings via run_encoder_comparison helpers — no encoder
forward, pure numpy + Pk/WD/F1 computation. Designed to run alongside
SeCom compress (CPU-light, ~10-15 min).

Output: outputs/experiments/2026-05-26_fig_l_low_p_extension/per_metric.json
        + REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import (  # noqa: E402
    TRAIN_CAP, delta_eff_seq, encode, load_dialogs, score_set)


PERCENTILES_NEW = (5, 10, 15, 20, 25, 30, 35, 40, 45)
ENC_ORDER = ("mpnet", "minilm", "minilm-int8")
DATASETS = ("tiage", "dialseg711", "superseg")


def main() -> None:
    out_dir = REPO / "outputs/experiments/2026-05-26_fig_l_low_p_extension"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    results = {enc: {} for enc in ENC_ORDER}

    for enc in ENC_ORDER:
        for ds in DATASETS:
            if ds == "dialseg711":
                dia = load_dialogs(ds, "test")
                emb = encode(enc, ds, "test", dia)
                idx = rng.permutation(len(dia))
                cut = int(round(len(idx) * 0.70))
                ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
                calib = ([dia[i] for i in ci], [emb[i] for i in ci])
                test = ([dia[i] for i in ti], [emb[i] for i in ti])
                note = f"test 70:30 (calib {len(ci)} / test {len(ti)})"
            else:
                tr = load_dialogs(ds, "train")
                if len(tr) > TRAIN_CAP:
                    sub = sorted(rng.permutation(len(tr))[:TRAIN_CAP])
                    tr = [tr[i] for i in sub]
                tr_emb = encode(enc, ds, "train", tr)
                te = load_dialogs(ds, "test")
                te_emb = encode(enc, ds, "test", te)
                calib = (tr, tr_emb)
                test = (te, te_emb)
                note = f"train split (calib {len(tr)} / test {len(te)})"

            calib_deff = [delta_eff_seq(e) for e in calib[1]]
            test_deff = [delta_eff_seq(e) for e in test[1]]
            allv = np.array([d for s in calib_deff for d in s[1:]])

            per_p = {}
            for p in PERCENTILES_NEW:
                dstar = float(np.percentile(allv, p))
                m = score_set(test[0], test_deff, dstar)
                per_p[f"p{p}"] = dict(dstar=dstar, **m)
                print(f"[{enc}/{ds}/p{p:02d}]  δ*={dstar:.4f}  "
                      f"Pk={m['pk']:.4f} WD={m['wd']:.4f} "
                      f"F1={m['f1']:.4f} Score={m['score']:.4f}", flush=True)

            results[enc][ds] = dict(note=note, **per_p)

    (out_dir / "per_metric.json").write_text(json.dumps(results, indent=2))
    print(f"\nDONE → {out_dir / 'per_metric.json'}", flush=True)


if __name__ == "__main__":
    main()
