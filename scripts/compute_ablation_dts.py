#!/usr/bin/env python3
"""DTS-3 ablation: Hi-OnTop with a=0.0 (ctx only) / a=1.0 (prev only) vs
baseline a=0.5. Uses cached embeddings via run_encoder_comparison helpers —
no encoder forward, pure numpy. CPU-light (~15 min).

Outputs:
- outputs/experiments/2026-05-26_ablation_blend/dts_metrics.json (per a × enc × ds)
- outputs/experiments/2026-05-26_ablation_blend/REPORT.md
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
    TRAIN_CAP, encode, load_dialogs, score_set)
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

OUT_DIR = REPO / "outputs/experiments/2026-05-26_ablation_blend"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENC_ORDER = ("mpnet", "minilm", "minilm-int8")
DATASETS = ("tiage", "dialseg711", "superseg")
BLENDS = (0.0, 0.5, 1.0)  # 0.5 = baseline
M, RHO = 2, 0.7
PERCENTILE = 70  # default p70 (paper layer 1)


def delta_eff_seq_a(emb, a):
    """δ_eff sequence using blend weight a."""
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                  ctx_window=M, ctx_decay=RHO, ctx_blend_a=a)
    for v in emb:
        seg.assign(v.astype(np.float64))
    return np.array([float(h["delta_eff"]) for h in seg.history()])


def main() -> None:
    rng = np.random.default_rng(0)
    results = {}

    for enc in ENC_ORDER:
        results[enc] = {}
        for ds in DATASETS:
            if ds == "dialseg711":
                dia = load_dialogs(ds, "test")
                emb = encode(enc, ds, "test", dia)
                idx = rng.permutation(len(dia))
                cut = int(round(len(idx) * 0.70))
                ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
                calib = ([dia[i] for i in ci], [emb[i] for i in ci])
                test = ([dia[i] for i in ti], [emb[i] for i in ti])
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

            per_a = {}
            for a in BLENDS:
                # Recompute δ_eff with this a value
                calib_deff = [delta_eff_seq_a(e, a) for e in calib[1]]
                test_deff = [delta_eff_seq_a(e, a) for e in test[1]]
                allv = np.array([d for s in calib_deff for d in s[1:]])
                dstar = float(np.percentile(allv, PERCENTILE))
                m = score_set(test[0], test_deff, dstar)
                per_a[f"a{a:.1f}"] = dict(dstar=dstar, **m)
                print(f"[{enc}/{ds}/a={a:.1f}/p70]  δ*={dstar:.4f}  "
                       f"Pk={m['pk']:.4f} WD={m['wd']:.4f} "
                       f"F1={m['f1']:.4f} Score={m['score']:.4f}", flush=True)
            results[enc][ds] = per_a

    (OUT_DIR / "dts_metrics.json").write_text(json.dumps(results, indent=2))

    # Markdown report
    L = ["# DTS-3 Ablation — Hi-OnTop blend weight a", "",
         "δ_eff = a · δ_prev + (1−a) · δ_ctx  (m=2, ρ=0.7)",
         "p70 percentile threshold on calib pool per (encoder × dataset × a).",
         "", "| Enc | Dataset | a | δ\\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for enc in ENC_ORDER:
        for ds in DATASETS:
            for a in BLENDS:
                x = results[enc][ds][f"a{a:.1f}"]
                marker = " ⭐" if a == 0.5 else ""
                L.append(f"| {enc} | {ds} | {a:.1f}{marker} | {x['dstar']:.4f} | "
                         f"{x['pk']:.4f} | {x['wd']:.4f} | {x['f1']:.4f} | "
                         f"**{x['score']:.4f}** |")
    L.append("")
    L.append("⭐ = baseline (paper default).")
    (OUT_DIR / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {OUT_DIR/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
