#!/usr/bin/env python3
"""v4.3.2-exp η sweep — frozen ST5 + learned NextEmbedHead 가 δ_model 자리.

설정:
- mpnet channel = v4.1.1 default HP (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- ST5-head channel = precomputed δ_model = 1−cos(\\hat{s}_t, s_st5[t])
  (precompute_v432_delta.py 산물 재사용; head 는 DailyDialog 로 학습)
- η = mpnet weight, (1-η) = ST5-head weight
- δ* = 0.5594 (mpnet 그대로; ST5-head δ scale ≈ 0.3~0.7 로 mpnet 과 자연 양립)

Sanity:
- η=1.0 → v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상)
- η<1.0 → ST5-head δ_model 활성

Datasets: tiage, dialseg711, superseg test.
- mpnet embedding cache: outputs/runs/_misc/sds_emb_{ds}_test_{mpnet}.pkl
- ST5-head δ cache    : outputs/runs/_misc/sds_v432delta_{ds}_test_{tag}.pkl
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
from hi_ontop.sem_core_v432_exp import HiOnTopSegmenterV432Exp  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
ENC_TOPIC = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ENC_ST5 = "sentence-transformers/sentence-t5-base"
DEFAULT_TAG = "st5_m5_mlp1024"

MPNET_M, MPNET_RHO, MPNET_A, MPNET_DSTAR = 2, 0.7, 0.5, 0.5594
V411_BASELINE = {"tiage": 0.4675, "dialseg711": 0.5897, "superseg": 0.4631}


def load_dialogs(dataset, split):
    raw = json.loads((SDS / dataset / f"segmentation_file_{split}.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
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


def load_cache_emb(ds, split, encoder_name):
    safe = encoder_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{ds}_{split}_{safe}.pkl"
    if not cp.exists():
        raise SystemExit(f"emb cache missing: {cp}")
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def load_cache_v432_delta(ds, split, tag):
    cp = CACHE / f"sds_v432delta_{ds}_{split}_{tag}.pkl"
    if not cp.exists():
        raise SystemExit(
            f"v432 δ cache missing: {cp}\n"
            f"  run: uv run python scripts/precompute_v432_delta.py --tag {tag}"
        )
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def run_v432(embs_topic, deltas_st5, eta, head_tag,
             calibrated: bool = False, delta_star_model: float | None = None):
    out = []
    seg_kwargs = dict(
        alpha=1.0, lmda=10.0,
        ctx_window=MPNET_M, ctx_decay=MPNET_RHO, ctx_blend_a=MPNET_A,
        eta_prev=eta, encoder_st5=ENC_ST5,
    )
    if calibrated:
        seg_kwargs.update(calibrated=True,
                          delta_star_adj=MPNET_DSTAR,
                          delta_star_model=delta_star_model)
        # parent's delta_star forced to 1.0 internally; do not pass
    else:
        seg_kwargs["delta_star"] = MPNET_DSTAR
    for et, dm in zip(embs_topic, deltas_st5):
        seg = HiOnTopSegmenterV432Exp(dim=et.shape[1], **seg_kwargs)
        torch.manual_seed(0)
        ids = []
        for i, st in enumerate(et):
            d_model = float(dm[i]) if i < len(dm) else 0.0
            k, _ = seg.assign_pair(st.astype(np.float64), d_model)
            ids.append(k)
        yp = [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]
        out.append(yp)
    return out


def eval_metrics(dialogs, preds):
    pks, wds, g, p = [], [], [], []
    for (_, yt), yp in zip(dialogs, preds):
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    return dict(pk=pk_m, wd=wd_m, f1=f1, score=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-21_v432_eta_sweep")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--etas", nargs="+", type=float,
                    default=[1.0, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="NextEmbedHead checkpoint tag")
    ap.add_argument("--calibrated", action="store_true",
                    help="z-score blend: z_adj=δ_adj/δ*_adj, z_model=δ_model/δ*_model. "
                         "v4.2.3 패턴, scale mismatch (δ_model<<δ_adj) 회피.")
    ap.add_argument("--delta-star-model", default="auto",
                    help="δ*_model: float 또는 'auto' (= test δ_model mean, sanity). "
                         "정식 calib 은 별도 TIAGE-train script.")
    args = ap.parse_args()

    print(f"[load] head_tag={args.tag}, datasets + caches  "
          f"(calibrated={args.calibrated})")
    data = {}
    delta_star_models = {}
    for ds in args.datasets:
        dia = load_dialogs(ds, "test")
        et = load_cache_emb(ds, "test", ENC_TOPIC)
        dm = load_cache_v432_delta(ds, "test", args.tag)
        all_vals = np.concatenate([d[1:] for d in dm if len(d) > 1])
        data[ds] = (dia, et, dm)
        # auto δ*_model = per-dataset test mean (sanity; train calib 별도)
        ds_mean = float(all_vals.mean())
        delta_star_models[ds] = ds_mean
        print(f"  {ds}: n_dial={len(dia)}, δ_model mean={ds_mean:.4f}, "
              f"std={all_vals.std():.4f}")
    # resolve δ*_model
    if args.calibrated:
        if args.delta_star_model == "auto":
            print(f"[calib] δ*_model = per-dataset test mean (sanity, NOT train-calibrated)")
        else:
            v = float(args.delta_star_model)
            delta_star_models = {ds: v for ds in args.datasets}
            print(f"[calib] δ*_model = {v:.4f} (fixed, all datasets)")

    print(f"\n[sweep] η ∈ {args.etas}")
    grid = {}
    for eta in args.etas:
        for ds in args.datasets:
            dia, et, dm = data[ds]
            t0 = time.perf_counter()
            preds = run_v432(et, dm, eta, args.tag,
                             calibrated=args.calibrated,
                             delta_star_model=delta_star_models[ds] if args.calibrated else None)
            m = eval_metrics(dia, preds)
            wall = time.perf_counter() - t0
            grid[(eta, ds)] = (m, wall)
            print(f"  η={eta:.2f}  {ds:12s}  Score={m['score']:.4f}  "
                  f"F1={m['f1']:.4f}  Pk={m['pk']:.4f}  WD={m['wd']:.4f}  ({wall:.0f}s)")

    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        f"# v4.3.2-exp η sweep — frozen ST5 + learned NextEmbedHead 가 δ_model 자리",
        "",
        "**Setup**:",
        f"- mpnet channel (parent): `{ENC_TOPIC}`, "
        f"m={MPNET_M}, ρ={MPNET_RHO}, a={MPNET_A}, δ*={MPNET_DSTAR} (v4.1.1 TIAGE-train).",
        f"- ST5 channel (δ_model slot): frozen `{ENC_ST5}`, "
        f"NextEmbedHead tag=`{args.tag}` (trained on DailyDialog), "
        f"δ_model = 1−cos(\\hat{{s}}_t, s_st5[t]).",
        "- δ_eff² = η·δ_adj² + (1−η)·δ_model² (v4.1.1 식 그대로).",
        "- δ* = 0.5594 (mpnet 그대로; ST5 cosine-distance scale ≈ 0.3~0.7).",
        "- Sanity: η=1.0 == v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상).",
        "",
        "## Score matrix (행=η, 열=dataset, mean = row mean)",
        "",
        "| η | " + " | ".join(args.datasets) + " | mean |",
        "|---:|" + "---:|" * (len(args.datasets) + 1),
    ]
    for eta in args.etas:
        scores = [grid[(eta, ds)][0]["score"] for ds in args.datasets]
        mean = sum(scores) / len(scores)
        row = [f"{eta:.2f}"] + [f"{s:.4f}" for s in scores] + [f"{mean:.4f}"]
        L.append("| " + " | ".join(row) + " |")

    L += [
        "",
        "## Detailed metrics",
        "",
        "| η | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for eta in args.etas:
        for ds in args.datasets:
            m, _ = grid[(eta, ds)]
            L.append(
                f"| {eta:.2f} | {ds} | {m['score']:.4f} | {m['f1']:.4f} | "
                f"{m['pk']:.4f} | {m['wd']:.4f} |"
            )

    L += [
        "",
        "## Per-dataset best",
        "",
        "| dataset | best η | Score | vs v4.1.1 (η=1.0) |",
        "|---|---:|---:|---:|",
    ]
    for ds in args.datasets:
        best_eta, best_score = max(
            ((eta, grid[(eta, ds)][0]["score"]) for eta in args.etas),
            key=lambda x: x[1],
        )
        delta = best_score - V411_BASELINE.get(ds, 0.0)
        L.append(f"| {ds} | {best_eta:.2f} | {best_score:.4f} | {delta:+.4f} |")

    L += [
        "",
        "## 해석 / 판정",
        "",
        "(채울 것 — η sweep 결과 본 후 작성)",
        "",
        "## 한계 / 검증 미해결",
        "- **Domain shift**: DailyDialog (open-domain conversation) → "
        "TIAGE (자연 대화) / Dialseg711 (정형) / SuperDialseg (gov FAQ).",
        "  NextEmbedHead 가 in-domain 으로 학습된 dynamics 가 일반화하는지 미검증.",
        "- **δ\\* re-calibration 미수행** — ST5 cosine 분포가 mpnet 과 다르면 후속 필요.",
        "- **Head architecture**: small MLP only. Transformer / GRU 비교 미수행.",
        "- **m=5 causal window** — m sweep 미수행.",
        "- **DailyDialog 학습 hyperparameter** (lr/epoch/batch) sweep 미수행.",
        "- **Single seed, single ckpt** — head variance 미측정.",
        "- **CSM 과 직접 baseline 비교 미수행** — train_csm_hf.py 산물과 별도 표 필요.",
        "- **TIAGE-test 는 in-domain 가능성** (mpnet δ\\* 가 TIAGE-train 기반).",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
