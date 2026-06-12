#!/usr/bin/env python3
"""Hi-OnTop per-metric (Pk/WD/F1/Score) — oracle + best per (encoder, bench).

dts_result.md 의 Online Ours 행 보완. p70/p75/p80 외에:
- **best**: labeled train split sweep → test 적용 (`run_encoder_comparison.py`
  와 동일 정의: tiage/superseg = train, dialseg711 = test 70:30 split 70%).
- **oracle**: test 자체에서 δ\* sweep (test-side ceiling, label leakage 인정).

harness: 정확히 `run_encoder_comparison.py` 와 동일 — superdialseg_data +
공식 NLTK Pk/WD + 동일 cached pkls. cached embeddings 만 사용, 인코더
forward 없음 → CSM 동시 실행 방해 없음.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import (  # noqa: E402
    DSTAR_GRID, TRAIN_CAP,
    best_score_dstar, delta_eff_seq, encode, load_dialogs, score_set)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+",
                    default=["mpnet", "minilm-int8"])
    ap.add_argument("--name", default="2026-05-24_hiontop_oracle_best")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    # run_encoder_comparison.py 는 rng=default_rng(0) 하나를 [mpnet, minilm,
    # minilm-int8] × [tiage, dialseg711, superseg] 9 step 에 걸쳐 소비.
    # cached pkl 들이 그 rng path 의 특정 subsample/split 으로 생성됐으므로
    # 우리도 같은 path 를 재현해야 cache 와 정렬됨. minilm 을 출력에서 빼더라도
    # rng 호출은 동일 순서로 진행한다.
    rng = np.random.default_rng(0)
    enc_order = ["mpnet", "minilm", "minilm-int8"]
    for enc in enc_order:
        print(f"\n########## encoder = {enc} ##########", flush=True)
        skip_output = enc not in args.encoders
        if not skip_output:
            results[enc] = {}
        for ds in ("tiage", "dialseg711", "superseg"):
            # calib / test split (run_encoder_comparison 와 동일)
            if ds == "dialseg711":
                dia = load_dialogs(ds, "test")
                emb = encode(enc, ds, "test", dia)
                idx = rng.permutation(len(dia))
                cut = int(round(len(idx) * 0.70))
                ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
                calib = ([dia[i] for i in ci], [emb[i] for i in ci])
                test = ([dia[i] for i in ti], [emb[i] for i in ti])
                calib_note = (f"test 70:30 split "
                              f"(calib {len(ci)} / test {len(ti)})")
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
                calib_note = (f"train split "
                              f"(calib {len(tr)} / test {len(te)})")

            calib_deff = [delta_eff_seq(e) for e in calib[1]]
            test_deff = [delta_eff_seq(e) for e in test[1]]

            # best: train sweep → test eval
            dstar_best = best_score_dstar(calib[0], calib_deff)
            r_best = score_set(test[0], test_deff, dstar_best)

            # oracle: test sweep (supervised upper bound)
            dstar_oracle = best_score_dstar(test[0], test_deff)
            r_oracle = score_set(test[0], test_deff, dstar_oracle)

            if not skip_output:
                results[enc][ds] = dict(
                    dstar_best=dstar_best, best=r_best,
                    dstar_oracle=dstar_oracle, oracle=r_oracle,
                    note=calib_note,
                )
            print(f"  [{ds}] {calib_note}", flush=True)
            print(f"    best   δ*={dstar_best:.4f}  "
                  f"Pk={r_best['pk']:.4f} WD={r_best['wd']:.4f} "
                  f"F1={r_best['f1']:.4f} Score={r_best['score']:.4f}",
                  flush=True)
            print(f"    oracle δ*={dstar_oracle:.4f}  "
                  f"Pk={r_oracle['pk']:.4f} WD={r_oracle['wd']:.4f} "
                  f"F1={r_oracle['f1']:.4f} Score={r_oracle['score']:.4f}",
                  flush=True)

    (out_dir / "per_metric.json").write_text(json.dumps(results, indent=2))

    L = ["# Hi-OnTop oracle / best per-metric (superdialseg_data, NLTK Pk/WD)",
         "",
         "δ\\* 두 가지 + per-metric Pk/WD/F1/Score:",
         "- **best** = labeled train split sweep → test 적용 "
         "(run_encoder_comparison.py 와 동일 정의).",
         "  tiage/superseg = train split, dialseg711 = test 70:30 split.",
         "- **oracle** = test 에서 δ\\* sweep (supervised upper bound, label "
         "leakage 인정 — not deployable).",
         "",
         f"harness: `run_encoder_comparison.py` 함수 직접 import. δ\\* grid = "
         f"linspace({DSTAR_GRID[0]:.2f}, {DSTAR_GRID[-1]:.2f}, {len(DSTAR_GRID)}).",
         "HP: m=2, ρ=0.7, a=0.5. cached embeddings 만 사용 (인코더 forward 없음).",
         ""]
    for enc, by_ds in results.items():
        L += [f"## encoder = {enc}",
              "",
              "| 벤치 | type | δ\\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | calib note |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
        for ds in ("tiage", "dialseg711", "superseg"):
            r = by_ds.get(ds)
            if not r:
                continue
            b, o = r["best"], r["oracle"]
            L.append(f"| {ds} | best | {r['dstar_best']:.4f} | "
                     f"{b['pk']:.4f} | {b['wd']:.4f} | {b['f1']:.4f} | "
                     f"**{b['score']:.4f}** | {r['note']} |")
            L.append(f"| {ds} | oracle | {r['dstar_oracle']:.4f} | "
                     f"{o['pk']:.4f} | {o['wd']:.4f} | {o['f1']:.4f} | "
                     f"**{o['score']:.4f}** | test sweep |")
        L.append("")
    L += ["## 한계",
          "- oracle: test 에 직접 fit → not deployable. supervised upper bound 로만 의미.",
          "- best (dialseg711): train 부재로 test 70:30 분할 → 30% test 평가. "
          "‡ supervised 라 베이스라인과 비대칭 — 표에서 참고용 행으로 표기.",
          "- HP m/ρ/a fixed (Hi-OnTop default), encoder 만 교체."]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
