#!/usr/bin/env python3
"""Hi-OnTop p70/p75/p80 per-metric for MiniLM (3 benches).

dts_result.md 의 Online Ours 행 보완 — MPNet 의 p70/p75/p80 와 같은 자리에
MiniLM percentile family 행도 추가하기 위함.

harness: `run_encoder_comparison.py` 함수 직접 import. cached embeddings 만
사용 → 인코더 forward 없음, CSM 동시 실행 방해 없음.

데이터: superdialseg_data 공식 splits, NLTK Pk/WD k=auto + sklearn binary F1,
Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).

δ\\*_p = calib δ_eff 분포의 p-percentile. calib = (tiage/superseg) train,
(dialseg711) test 70:30 split 의 70%.
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
    out_dir = REPO / "outputs" / "experiments" / "2026-05-26_hiontop_minilm_percentile"
    out_dir.mkdir(parents=True, exist_ok=True)

    # run_encoder_comparison 의 rng path 그대로 재현 (cache 정렬 보장).
    rng = np.random.default_rng(0)
    enc_order = ["mpnet", "minilm", "minilm-int8"]
    percentiles = (70, 75, 80)
    target_enc = "minilm"

    results = {}
    for enc in enc_order:
        results[enc] = {}
        for ds in ("tiage", "dialseg711", "superseg"):
            # split (run_encoder_comparison 와 동일)
            if ds == "dialseg711":
                dia = load_dialogs(ds, "test")
                emb = encode(enc, ds, "test", dia)
                idx = rng.permutation(len(dia))
                cut = int(round(len(idx) * 0.70))
                ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
                calib = ([dia[i] for i in ci], [emb[i] for i in ci])
                test = ([dia[i] for i in ti], [emb[i] for i in ti])
                calib_note = f"test 70:30 split (calib {len(ci)} / test {len(ti)})"
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
                calib_note = f"train split (calib {len(tr)} / test {len(te)})"

            if enc != target_enc:
                continue  # rng path 만 소비

            calib_deff = [delta_eff_seq(e) for e in calib[1]]
            test_deff = [delta_eff_seq(e) for e in test[1]]
            allv = np.array([d for s in calib_deff for d in s[1:]])

            per_p = {}
            for p in percentiles:
                dstar = float(np.percentile(allv, p))
                m = score_set(test[0], test_deff, dstar)
                per_p[f"p{p}"] = dict(dstar=dstar, **m)
                print(f"[{enc}/{ds}/p{p}]  δ*={dstar:.4f}  "
                      f"Pk={m['pk']:.4f} WD={m['wd']:.4f} "
                      f"F1={m['f1']:.4f} Score={m['score']:.4f}", flush=True)

            results[enc][ds] = dict(note=calib_note, **per_p)

    (out_dir / "per_metric.json").write_text(json.dumps(results, indent=2))

    L = ["# Hi-OnTop p70/p75/p80 per-metric (MiniLM, superdialseg_data)",
         "",
         "δ\\*_p = calib δ_eff p-percentile. harness = `run_encoder_comparison.py` "
         "함수 import + cached embeddings (인코더 forward 없음).",
         "",
         "| 벤치 | p | δ\\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | calib note |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
    for ds in ("tiage", "dialseg711", "superseg"):
        r = results[target_enc].get(ds)
        if not r:
            continue
        for p in (70, 75, 80):
            x = r[f"p{p}"]
            L.append(f"| {ds} | p{p} | {x['dstar']:.4f} | "
                     f"{x['pk']:.4f} | {x['wd']:.4f} | {x['f1']:.4f} | "
                     f"**{x['score']:.4f}** | {r['note']} |")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
