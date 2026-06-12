#!/usr/bin/env python3
"""v4.2.2 δ* train-calibration with ``aws-ai/dse-bert-base`` encoder.

Thin variant of ``calibrate_v411_delta_star.py``: identical (m, ρ, a)
sweep / F1-optimal threshold logic — only the scene-vector encoder is
swapped to DSE-BERT to match the v4.2.2 spec
(``context/methodology/v4.2.2.md``).

Why a separate script: ``delta_star`` is corpus/encoder-dependent.
v4.1.1's TIAGE-train δ* (0.5594 @ m=2,ρ=0.7,a=0.5) was calibrated with
``sentence-transformers/multi-qa-mpnet-base-dot-v1`` (Hi-OnTop local
default). DSE-BERT has a different cosine geometry on dialogue
turns → must re-calibrate before reporting v4.2.2 numbers.

Output: ``outputs/runs/_misc/v422_delta_star_train.md`` (mirrors the
v4.1.1 output path convention).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.embedding import QueryEncoder  # noqa: E402

ENCODER = "aws-ai/dse-bert-base"
DATA = REPO / "benchmarks" / "tiage" / "data" / "personachat" / "anno"
OUT = REPO / "outputs" / "runs" / "_misc" / "v422_delta_star_train.md"

GRID_M = [2, 3, 4]
GRID_RHO = [0.5, 0.7, 0.9]
GRID_A = [0.0, 0.5, 1.0]  # a=1 → pure δ_prev sanity


def load(split):
    raw = json.loads((DATA / split / f"anno_{split}.json").read_text())
    return [[(t[0], t[1]) for t in d] for d in raw.values()]


def gt_shifts(dialog):
    lab = [t[1] for t in dialog]
    return [lab[i] == "1" for i in range(1, len(lab))]


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def delta_eff_seq(emb, m, rho, a):
    out = []
    for t in range(1, len(emb)):
        cp = _cos(emb[t - 1], emb[t])
        d_prev = None if cp is None else 1.0 - cp
        win = emb[max(0, t - m):t][::-1]
        c = np.zeros_like(emb[t], dtype=np.float64)
        for i, v in enumerate(win):
            c += (rho ** i) * v
        cc = _cos(c, emb[t])
        d_ctx = None if cc is None else 1.0 - cc
        if d_prev is None:
            out.append(None)
        elif d_ctx is None:
            out.append(d_prev)
        else:
            out.append(a * d_prev + (1.0 - a) * d_ctx)
    return out


def main() -> None:
    dialogs = load("train")
    print(f"[1/2] TIAGE train: {len(dialogs)} conv | encoder={ENCODER}")
    enc = QueryEncoder(model_name=ENCODER)
    print(f"      QueryEncoder ready: dim={enc.dim} device={enc.device}")
    embs = [np.asarray(enc.encode([t[0] for t in d])) for d in dialogs]
    print("[2/2] encoded — sweeping (m,ρ,a) ...")

    gt_all = []
    for d in dialogs:
        gt_all += gt_shifts(d)
    gt_all = np.array(gt_all)

    rows = []
    for m, rho, a in itertools.product(GRID_M, GRID_RHO, GRID_A):
        sig = []
        for e in embs:
            sig += delta_eff_seq(e, m, rho, a)
        sig = np.array([np.nan if x is None else x for x in sig], float)
        ok = ~np.isnan(sig)
        s, g = sig[ok], gt_all[ok]
        best = (0.0, 0.0)
        for th in np.linspace(s.min(), s.max(), 100):
            f = f1_score(g, s >= th)
            if f > best[0]:
                best = (f, float(th))
        rows.append((m, rho, a, best[1], best[0]))
        print(f"  m={m} ρ={rho} a={a}: δ*={best[1]:.4f}  trainF1={best[0]:.3f}")

    rows.sort(key=lambda r: -r[4])
    lines = [
        f"# v4.2.2 δ* train-calibration (TIAGE train, encoder={ENCODER})",
        f"n_conv={len(dialogs)} n_trans={int(ok.sum())} "
        f"GT_shifts={int(g.sum())} | encoder=aws-ai/dse-bert-base (BertModel+mean-pool)",
        "δ_eff = a·δ_prev + (1-a)·δ_ctx ; boundary when δ_eff ≥ δ*. "
        "a=1.0 row = pure δ_prev (v3.3.9 equivalent, sanity).",
        "",
        "| rank | m | ρ | a | δ* (train) | train F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for i, (m, rho, a, ds, f) in enumerate(rows, 1):
        lines.append(
            f"| {i} | {m} | {rho:g} | {a:g} | {ds:.4f} | {f:.3f} |"
        )
    lines += [
        "",
        f"**train-best**: m={rows[0][0]} ρ={rows[0][1]} a={rows[0][2]} "
        f"δ*={rows[0][3]:.4f} (train F1 {rows[0][4]:.3f})",
        "",
        "→ Evaluate the top configs ONCE on TIAGE test with these fixed "
        "δ* (Hi-OnTop `seg_pred_v411(..., encoder='aws-ai/dse-bert-base')` "
        "via HIONTOP_EMBEDDING_BACKEND=local). Compare against v4.1.1 "
        "(mpnet) calibration table for the v4.2.2 promotion decision.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[4:]))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
