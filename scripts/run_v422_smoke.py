#!/usr/bin/env python3
"""v4.2.2 smoke: v4.1.1 (mpnet) vs v4.2.2 (DSE-BERT) × {tiage, dialseg711,
superseg} test.

For each (encoder, dataset, ctx_window m):
  - load test split (SuperDialseg-format)
  - encode dialogue turns (cache per encoder)
  - run causal-window δ_eff + HiOnTopSegmenterV411 (algorithm identical for
    both — only embedding source differs)
  - Pk / WD / F1 / Score(official SuperDialseg formula)

δ* convention (no test leak):
  - tiage      : passed --delta-star-* args (train calibration)
  - superseg   : passed --delta-star-* args (train calibration)
  - dialseg711 : no train split available → use TIAGE-train δ*
                 (cross-corpus; both encoders use their own TIAGE δ*).

Output: outputs/experiments/<name>/REPORT.md with both encoders.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.embedding import QueryEncoder  # noqa: E402
from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

ENCODER_MPNET = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ENCODER_DSE = "aws-ai/dse-bert-base"

# v4.1.1 best on TIAGE-train w/ mpnet (v411_delta_star_train.md):
V411_M, V411_RHO, V411_A, V411_DSTAR = 2, 0.7, 0.5, 0.5594


def load_dialogs(dataset: str, split: str):
    raw = json.loads((SDS / dataset / f"segmentation_file_{split}.json").read_text())
    dd = raw["dial_data"]
    arr = dd[list(dd)[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if yt:
            yt[-1] = 0
        if len(utts) >= 2:
            out.append((utts, yt))
    return out


def official_pk_wd(yt, yp):
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


def encode_cached(dataset: str, split: str, dialogs, encoder_name: str):
    """Cache key includes encoder slug to keep mpnet/DSE caches separate."""
    safe = encoder_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{dataset}_{split}_{safe}.pkl"
    if cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    print(f"  [encode] {encoder_name} ({len(dialogs)} dialogs) — first run, slow on CPU")
    enc = QueryEncoder(model_name=encoder_name)
    embs = [np.asarray(enc.encode(u)) for u, _ in dialogs]
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump(embs, fh)
    return embs


def seg_pred(emb, m, rho, a, dstar):
    seg = HiOnTopSegmenterV411(
        dim=emb.shape[1], alpha=1.0, lmda=10.0, delta_star=dstar,
        ctx_window=m, ctx_decay=rho, ctx_blend_a=a)
    torch.manual_seed(0)
    ids = [seg.assign(s.astype(np.float64))[0] for s in emb]
    return [1 if ids[i] != ids[i + 1] else 0
            for i in range(len(ids) - 1)] + [0]


def eval_run(dialogs, embs, m, rho, a, dstar):
    pks, wds, g, p = [], [], [], []
    for (_, yt), e in zip(dialogs, embs):
        yp = seg_pred(e, m, rho, a, dstar)
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    return dict(pk=pk_m, wd=wd_m, f1=f1, score=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_v422_smoke")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--split", default="test")
    # v4.2.2 TIAGE-train δ* (from scripts/calibrate_v422_delta_star.py)
    ap.add_argument("--v422-m", type=int, default=V411_M)
    ap.add_argument("--v422-rho", type=float, default=V411_RHO)
    ap.add_argument("--v422-a", type=float, default=V411_A)
    ap.add_argument("--v422-dstar", type=float, required=True,
                    help="δ* from calibrate_v422_delta_star.py (TIAGE train, DSE-BERT)")
    args = ap.parse_args()

    arms = [
        ("v4.1.1 (mpnet)", ENCODER_MPNET, V411_M, V411_RHO, V411_A, V411_DSTAR),
        ("v4.2.2 (DSE-BERT)", ENCODER_DSE, args.v422_m, args.v422_rho, args.v422_a, args.v422_dstar),
    ]

    rows = []  # (arm, ds, metrics)
    for ds in args.datasets:
        print(f"\n[load] {ds}/{args.split}")
        dia = load_dialogs(ds, args.split)
        print(f"  n_dial={len(dia)}")
        for label, enc, m, rho, a, dstar in arms:
            embs = encode_cached(ds, args.split, dia, enc)
            t0 = time.perf_counter()
            r = eval_run(dia, embs, m, rho, a, dstar)
            wall = time.perf_counter() - t0
            rows.append((label, ds, m, rho, a, dstar, r, wall))
            print(f"  {label:22s} m={m} ρ={rho} a={a} δ*={dstar:.4f} → "
                  f"Score={r['score']:.4f} F1={r['f1']:.4f} "
                  f"Pk={r['pk']:.4f} WD={r['wd']:.4f}  ({wall:.0f}s)")

    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        f"# v4.2.2 smoke — encoder swap (mpnet → DSE-BERT) on {args.datasets}",
        "",
        "**Setup**:",
        "- Algorithm = v4.1.1 (causal-window δ identity segmenter), encoder only swapped.",
        f"- v4.1.1 baseline: encoder=`{ENCODER_MPNET}`, m={V411_M}, ρ={V411_RHO}, "
        f"a={V411_A}, δ*={V411_DSTAR} (TIAGE-train calibration).",
        f"- v4.2.2 variant: encoder=`{ENCODER_DSE}`, m={args.v422_m}, ρ={args.v422_rho}, "
        f"a={args.v422_a}, δ*={args.v422_dstar} (TIAGE-train, DSE-BERT calibration).",
        "- Dialseg711 has no train split → uses TIAGE-train δ\\* per encoder (cross-corpus).",
        "- Metric: official SuperDialseg (Pk/WD k=auto, F1 binary), "
        "Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        "",
        "## 결과",
        "",
        "| arm | dataset | m | ρ | a | δ\\* | **Score ↑** | F1 ↑ | Pk ↓ | WD ↓ | wall(s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, ds, m, rho, a, dstar, r, wall in rows:
        L.append(
            f"| {arm} | {ds} | {m} | {rho:g} | {a:g} | {dstar:.4f} | "
            f"**{r['score']:.4f}** | {r['f1']:.4f} | {r['pk']:.4f} | {r['wd']:.4f} | {wall:.0f} |"
        )
    L += [
        "",
        "## 해석",
        "",
        "(빈 칸 — 숫자 보고 채워야 함. 사용자 / Claude 가 작성)",
        "",
        "## 판정",
        "",
        "- v4.2.2 Score 가 v4.1.1 대비 모든 데이터셋에서 노이즈 밖 우위면 → 승격, "
        "  methodology v4.2.2.md '결과' 섹션 + decision-log entry append.",
        "- 일부 우위 / 일부 회귀면 → 부분 채택, v4.1.1 default 유지 + v4.2.2 옵션 보존.",
        "- 우위 없으면 → negative result, v4.1.1.md '고려 중인 변형' 한 줄 기록.",
        "",
        "## 한계 / 검증 미해결",
        "",
        "- seed=0 (v4.1.1 default) 단일 run — variance 미측정.",
        "- δ\\* 는 TIAGE-train 만 calibration. superseg-train 도 별도 δ\\* 산출하면 "
        "  더 공정 (현재는 cross-corpus TIAGE δ\\* 그대로 사용).",
        "- DSE-BERT 의 max_seq_length=512 default — 긴 발화 truncation 검증 필요.",
        "- 단일 (m, ρ, a) 만 — encoder 별 ctx_window 최적값 다를 수 있음 (sweep 미수행).",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
