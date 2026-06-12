#!/usr/bin/env python3
"""Hi-OnTop p60/p70/p80 for sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.

harness = run_encoder_comparison.py 함수 import.
embedding 은 신규 인코더라 캐시 없음 → 첫 실행 시 인코딩 후 pkl 저장.
결과 → outputs/experiments/2026-06-03_paraphrase_multilingual_percentile/REPORT.md
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

import run_encoder_comparison as _renc  # noqa: E402

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_KEY = "paraphrase-multilingual-int8"

# MiniLM-int8 과 동일 방식: ONNX quint8_avx2 양자화
_renc.ENCODERS[ENCODER_KEY] = {
    "model": MODEL,
    "backend": "onnx",
    "file_name": "onnx/model_quint8_avx2.onnx",
}


from run_encoder_comparison import (  # noqa: E402
    TRAIN_CAP, delta_eff_seq, encode, load_dialogs, score_set)

PERCENTILES = (60, 70, 80)
EXP_NAME = "2026-06-03_paraphrase_multilingual_percentile"


def main() -> None:
    out_dir = REPO / "outputs" / "experiments" / EXP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    results: dict = {p: {} for p in PERCENTILES}

    for ds in ("tiage", "dialseg711", "superseg"):
        print(f"\n=== {ds} ===", flush=True)
        t0 = time.perf_counter()

        if ds == "dialseg711":
            dia = load_dialogs(ds, "test")
            emb = encode(ENCODER_KEY, ds, "test", dia)
            idx = rng.permutation(len(dia))
            cut = int(round(len(idx) * 0.70))
            ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
            calib = ([dia[i] for i in ci], [emb[i] for i in ci])
            test  = ([dia[i] for i in ti], [emb[i] for i in ti])
            note  = f"test 70:30 split (calib {len(ci)} / test {len(ti)})"
        else:
            tr = load_dialogs(ds, "train")
            if len(tr) > TRAIN_CAP:
                sub = sorted(rng.permutation(len(tr))[:TRAIN_CAP])
                tr = [tr[i] for i in sub]
            tr_emb = encode(ENCODER_KEY, ds, "train", tr)
            te     = load_dialogs(ds, "test")
            te_emb = encode(ENCODER_KEY, ds, "test", te)
            calib  = (tr, tr_emb)
            test   = (te, te_emb)
            note   = f"train split (calib {len(tr)} / test {len(te)})"

        calib_deff = [delta_eff_seq(e) for e in calib[1]]
        test_deff  = [delta_eff_seq(e) for e in test[1]]
        allv = np.array([d for s in calib_deff for d in s[1:]])

        for p in PERCENTILES:
            dstar = float(np.percentile(allv, p))
            m = score_set(test[0], test_deff, dstar)
            results[p][ds] = dict(note=note, dstar=dstar, **m)
            print(f"  [p{p}]  δ*={dstar:.4f}  Pk={m['pk']:.4f}  "
                  f"WD={m['wd']:.4f}  F1={m['f1']:.4f}  Score={m['score']:.4f}",
                  flush=True)

        print(f"  wall: {time.perf_counter()-t0:.1f}s", flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # ---- REPORT.md ----
    lines = [
        f"# Hi-OnTop Paraphrase-Multilingual-MiniLM-L12-v2 인코더 비교 ({EXP_NAME})",
        "",
        "## 실험 setup",
        f"- 인코더: `{MODEL}` (dim=384, multilingual, ONNX quint8_avx2)",
        "- 비교 기준: 기존 MiniLM-int8 (dim=384, 11.7 ms/turn Pre.)",
        "- 데이터: superdialseg_data (TIAGE / Dialseg711 / SuperDialseg)",
        "- 메트릭: Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)",
        "- HP: m=2, ρ=0.7, a=0.5 (Hi-OnTop canonical)",
        "- δ* calibration: calib split δ_eff 의 percentile (p60/p70/p80)",
        "- seed: 0 (rng), dialseg711 = test 70:30 split",
        "",
        "## 결과",
        "",
        "| percentile | TIAGE Score | DS711 Score | SDS Score | **mean** |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in PERCENTILES:
        r = results[p]
        sc = [r[d]["score"] for d in ("tiage", "dialseg711", "superseg")]
        mean = np.mean(sc)
        lines.append(
            f"| p{p} | {sc[0]:.4f} | {sc[1]:.4f} | {sc[2]:.4f} | **{mean:.4f}** |"
        )

    lines += [
        "",
        "### 상세 (Pk / WD / F1 / Score, 각 dataset)",
        "",
        "| p | dataset | δ* | Pk | WD | F1 | Score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in PERCENTILES:
        for ds in ("tiage", "dialseg711", "superseg"):
            r = results[p][ds]
            lines.append(
                f"| p{p} | {ds} | {r['dstar']:.4f} | {r['pk']:.4f} | "
                f"{r['wd']:.4f} | {r['f1']:.4f} | {r['score']:.4f} |"
            )

    lines += [
        "",
        "## MiniLM-int8 비교 (dts_result.md 기준)",
        "",
        "| percentile | TIAGE | DS711 | SDS | mean |",
        "|---|---:|---:|---:|---:|",
        "| p60 (int8) | 0.470 | 0.503 | 0.423 | 0.465 |",
        "| p70 (int8) | 0.489 | 0.566 | 0.401 | 0.485 |",
        "| p80 (int8) | 0.493 | 0.607 | 0.365 | 0.488 |",
        "",
        "## 한계",
        "- encoder latency 미측정 (별도 measure_hiontop_latency 실행 필요)",
        "- 버전 경고: sentence-transformers 3.4.1 vs 모델 생성 5.1.1",
    ]

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nDONE → {out_dir}/REPORT.md", flush=True)


if __name__ == "__main__":
    main()
