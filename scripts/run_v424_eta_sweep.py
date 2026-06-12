#!/usr/bin/env python3
"""v4.2.4-exp η sweep — DSE 가 δ_model 자리, η ∈ {1.0, 0.75, 0.5, 0.25, 0.0}.

설정:
- mpnet channel = v4.1.1 default HP (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- DSE channel = v4.2.2 train-best (m_dse=2, ρ_dse=0.5, a_dse=0.0)
- η = mpnet weight, (1-η) = DSE weight
- δ* = 0.5594 (mpnet 그대로)

Sanity:
- η=1.0 → v4.1.1 (mpnet only) → 0.4675 / 0.5897 / 0.4631
- η<1.0 → DSE δ_model 활성화

Datasets: tiage, dialseg711, superseg test. 인코딩 캐시는 v4.2.2 smoke
의 산물 (`sds_emb_<ds>_test_<encoder>.pkl`) 재사용.
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
from hi_ontop.sem_core_v424_exp import HiOnTopSegmenterV424Exp  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
ENC_TOPIC = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ENC_DSE = "aws-ai/dse-bert-base"

# v4.1.1 mpnet TIAGE-train best
MPNET_M, MPNET_RHO, MPNET_A, MPNET_DSTAR = 2, 0.7, 0.5, 0.5594
# v4.2.2 DSE-BERT TIAGE-train best
DSE_M, DSE_RHO, DSE_A = 2, 0.5, 0.0


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


def load_cache(ds, split, encoder_name):
    safe = encoder_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{ds}_{split}_{safe}.pkl"
    if not cp.exists():
        raise SystemExit(f"cache missing: {cp}")
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def run_v424(embs_topic, embs_dse, eta):
    """V424 segmenter, single eta. Returns predictions list."""
    out = []
    for et, ed in zip(embs_topic, embs_dse):
        seg = HiOnTopSegmenterV424Exp(
            dim=et.shape[1], alpha=1.0, lmda=10.0,
            delta_star=MPNET_DSTAR,
            ctx_window=MPNET_M, ctx_decay=MPNET_RHO, ctx_blend_a=MPNET_A,
            eta_prev=eta,
            m_dse=DSE_M, rho_dse=DSE_RHO, a_dse=DSE_A,
        )
        torch.manual_seed(0)
        ids = []
        for st, sd in zip(et, ed):
            k, _ = seg.assign_pair(st.astype(np.float64), sd.astype(np.float64))
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
    ap.add_argument("--name", default="2026-05-21_v424_eta_sweep")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--etas", nargs="+", type=float,
                    default=[1.0, 0.75, 0.5, 0.25, 0.0])
    args = ap.parse_args()

    # preload data + caches
    print("[load] datasets + caches")
    data = {}
    for ds in args.datasets:
        dia = load_dialogs(ds, "test")
        et = load_cache(ds, "test", ENC_TOPIC)
        ed = load_cache(ds, "test", ENC_DSE)
        data[ds] = (dia, et, ed)
        print(f"  {ds}: n_dial={len(dia)}")

    # sweep
    print(f"\n[sweep] η ∈ {args.etas}")
    grid = {}
    for eta in args.etas:
        for ds in args.datasets:
            dia, et, ed = data[ds]
            t0 = time.perf_counter()
            preds = run_v424(et, ed, eta)
            m = eval_metrics(dia, preds)
            wall = time.perf_counter() - t0
            grid[(eta, ds)] = (m, wall)
            print(f"  η={eta:.2f}  {ds:12s}  Score={m['score']:.4f}  "
                  f"F1={m['f1']:.4f}  Pk={m['pk']:.4f}  WD={m['wd']:.4f}  ({wall:.0f}s)")

    # REPORT
    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        f"# v4.2.4-exp η sweep — DSE 가 δ_model 자리, η ∈ {args.etas}",
        "",
        "**Setup**:",
        f"- mpnet channel (parent): encoder=`{ENC_TOPIC}`, "
        f"m={MPNET_M}, ρ={MPNET_RHO}, a={MPNET_A}, δ*={MPNET_DSTAR} "
        f"(v4.1.1 TIAGE-train best).",
        f"- DSE channel (δ_model slot): encoder=`{ENC_DSE}`, "
        f"m_dse={DSE_M}, ρ_dse={DSE_RHO}, a_dse={DSE_A} "
        f"(v4.2.2 TIAGE-train best for DSE).",
        "- δ_eff² = η·δ_adj² + (1−η)·δ_model² (v4.1.1 식 그대로, "
        "RNN PE 자리에 DSE causal-window PE 주입).",
        "- δ* = 0.5594 (mpnet 그대로 — DSE δ_model 의 raw scale 이 약간 다름, "
        "η sweep 으로 effective blend 탐색).",
        "- Sanity: η=1.0 == v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상).",
        "",
        "## Score matrix (행=η, 열=dataset, **bold**=row best)",
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

    # Per-dataset best
    L += [
        "",
        "## Per-dataset best",
        "",
        "| dataset | best η | Score | vs v4.1.1 (η=1.0) |",
        "|---|---:|---:|---:|",
    ]
    V411_BASELINE = {"tiage": 0.4675, "dialseg711": 0.5897, "superseg": 0.4631}
    for ds in args.datasets:
        best_eta, best_score = max(
            ((eta, grid[(eta, ds)][0]["score"]) for eta in args.etas),
            key=lambda x: x[1],
        )
        delta = best_score - V411_BASELINE[ds]
        sign = "+" if delta >= 0 else ""
        L.append(f"| {ds} | {best_eta:.2f} | {best_score:.4f} | {sign}{delta:+.4f} |")

    L += [
        "",
        "## 해석 / 판정",
        "",
        "(채울 것)",
        "",
        "## 한계",
        "- Raw scale mismatch: mpnet δ ≈ 0.4~0.6, DSE δ ≈ 0.3~0.5. η blend 가 raw 로 일어남.",
        "- δ\\* 은 mpnet 기준 (0.5594). η<1 일 때 effective δ_eff scale 이 변하므로 δ\\* 재calibration 가능.",
        "- Single trial per η, no seed variance.",
        "- DSE channel HP 는 v4.2.2 best 고정. 재calibration 시 best η 다를 가능성.",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
