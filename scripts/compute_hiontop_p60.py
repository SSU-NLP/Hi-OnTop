#!/usr/bin/env python3
"""Hi-OnTop p60 per-metric for MPNet + MiniLM-int8 (3 benches).

dts_result.md 에서 p75 빼고 p60 추가하는 사용자 요청. harness =
run_encoder_comparison.py 함수 import + cached embeddings. CSM 방해 없음.
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


def main() -> None:
    out_dir = REPO / "outputs" / "experiments" / "2026-05-24_hiontop_p60"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    enc_order = ["mpnet", "minilm", "minilm-int8"]
    target_encs = ("mpnet", "minilm", "minilm-int8")

    results = {}
    for enc in enc_order:
        if enc in target_encs:
            results[enc] = {}
        for ds in ("tiage", "dialseg711", "superseg"):
            if ds == "dialseg711":
                dia = load_dialogs(ds, "test")
                emb = encode(enc, ds, "test", dia)
                idx = rng.permutation(len(dia))
                cut = int(round(len(idx) * 0.70))
                ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
                calib = ([dia[i] for i in ci], [emb[i] for i in ci])
                test = ([dia[i] for i in ti], [emb[i] for i in ti])
                note = f"test 70:30 split (calib {len(ci)} / test {len(ti)})"
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

            if enc not in target_encs:
                continue

            calib_deff = [delta_eff_seq(e) for e in calib[1]]
            test_deff = [delta_eff_seq(e) for e in test[1]]
            allv = np.array([d for s in calib_deff for d in s[1:]])
            dstar = float(np.percentile(allv, 60))
            m = score_set(test[0], test_deff, dstar)
            results[enc][ds] = dict(note=note, dstar=dstar, **m)
            print(f"[{enc}/{ds}/p60]  δ*={dstar:.4f}  Pk={m['pk']:.4f} "
                  f"WD={m['wd']:.4f} F1={m['f1']:.4f} Score={m['score']:.4f}",
                  flush=True)

    (out_dir / "per_metric.json").write_text(json.dumps(results, indent=2))
    print(f"\nDONE → {out_dir / 'per_metric.json'}", flush=True)


if __name__ == "__main__":
    main()
