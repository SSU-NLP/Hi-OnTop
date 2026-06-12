#!/usr/bin/env python3
"""Per-metric (Pk/WD/F1/Score) for MPNet × 3 benches × {p70, p75, p80}.

Calibration set size: N=200 per benchmark (consistent with paper Claim 1).
Output JSON consumed by outputs/reports/dts_result.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from diag_percentile_generality import (  # noqa: E402
    delta_eff_seq,
    load_calib_full,
)
from run_encoder_comparison import score_set  # noqa: E402

ENC = "mpnet"
BENCHES = ("tiage", "dialseg711", "superseg")
PERCENTILES = (70, 75, 80)
N_CALIB = 200
SEED = 0


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {"encoder": ENC, "n_calib": N_CALIB, "seed": SEED, "cells": {}}
    for ds in BENCHES:
        print(f"\n========== {ENC} / {ds} ==========", flush=True)
        cal_dia, cal_emb, te_dia, te_emb = load_calib_full(ENC, ds)
        if len(cal_dia) > N_CALIB:
            idx = sorted(rng.permutation(len(cal_dia))[:N_CALIB].tolist())
            cal_dia = [cal_dia[i] for i in idx]
            cal_emb = [cal_emb[i] for i in idx]
            note = f"subsample {N_CALIB} of {len(idx)} (seed={SEED})"
        else:
            note = f"full {len(cal_dia)} (< N_CALIB={N_CALIB})"
        print(f"  calib {note} · test N={len(te_dia)}", flush=True)
        cal_deff = [delta_eff_seq(e) for e in cal_emb]
        te_deff = [delta_eff_seq(e) for e in te_emb]
        allv = np.array([d for s in cal_deff for d in s[1:]])
        print(f"  calib δ_eff samples = {len(allv)}", flush=True)
        cell = {"calib_note": note, "n_delta": int(allv.size)}
        for p in PERCENTILES:
            dstar = float(np.percentile(allv, p))
            res = score_set(te_dia, te_deff, dstar)
            cell[f"p{p}"] = {
                "dstar": dstar,
                "pk": res["pk"],
                "wd": res["wd"],
                "f1": res["f1"],
                "score": res["score"],
            }
            print(
                f"  p{p}  δ*={dstar:.4f}  "
                f"Pk={res['pk']:.4f}  WD={res['wd']:.4f}  "
                f"F1={res['f1']:.4f}  Score={res['score']:.4f}",
                flush=True,
            )
        out["cells"][ds] = cell

    out_path = (
        REPO / "outputs" / "experiments" / "2026-05-23_percentile_generality"
        / "per_metric_mpnet_n200.json"
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
