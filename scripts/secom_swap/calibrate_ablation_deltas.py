"""Compute δ* (p70) for blend ablation: a=0.0 (ctx only) and a=1.0 (prev only).

Uses cached MTB+ embeddings from
``outputs/experiments/2026-05-25_llm_distillation_calib/emb_*.pkl``.

Output: ``outputs/experiments/2026-05-26_ablation_blend/calib.json`` containing
per-encoder δ* at p70 for each (a, encoder).

Numpy-only, no encoder forward, CPU-light (~5 sec). Safe to run during
SeCom compress.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

CACHE = REPO / "outputs/experiments/2026-05-25_llm_distillation_calib"
OUT_DIR = REPO / "outputs/experiments/2026-05-26_ablation_blend"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = ("MPNet", "MiniLM", "MiniLM-int8")
M, RHO = 2, 0.7
PERCENTILE = 70


def compute_deltas(per_conv_emb, ctx_blend_a):
    out = []
    for cid, sess_emb in per_conv_emb.items():
        for si, vecs in sess_emb:
            seg = HiOnTop(dim=vecs.shape[1], delta_star=1.0,
                          ctx_window=M, ctx_decay=RHO,
                          ctx_blend_a=ctx_blend_a)
            for v in vecs:
                seg.assign(v.astype(np.float64))
            deffs = [float(h["delta_eff"]) for h in seg.history()]
            out.extend(deffs[1:])
    return np.array(out)


def main():
    result = {}
    for enc in ENCODERS:
        emb = pickle.loads((CACHE / f"emb_{enc}.pkl").read_bytes())
        for a in (0.0, 1.0):
            deffs = compute_deltas(emb, a)
            dstar = float(np.percentile(deffs, PERCENTILE))
            key = f"{enc}/a{a:.1f}"
            result[key] = dict(
                ctx_blend_a=a, n=int(deffs.size),
                mean=float(deffs.mean()), std=float(deffs.std()),
                p50=float(np.percentile(deffs, 50)),
                p70=dstar,
                p80=float(np.percentile(deffs, 80)),
            )
            print(f"{key:25s}: n={deffs.size}  μ={deffs.mean():.4f}  "
                   f"σ={deffs.std():.4f}  p70={dstar:.4f}", flush=True)
    out = OUT_DIR / "calib.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
