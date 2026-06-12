#!/usr/bin/env python3
"""v4.1.1 ctx_window sweep × 3 SuperDialseg-family benches.

Fixed TIAGE-cfg (ρ=0.7, a=0.5, δ*=0.5594, α=1, λ=10) — sweep only
ctx_window m. Metric = official SuperDialseg Pk/WD + sklearn F1 +
Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD). Embeddings cached per
dataset (reuses pkl from run_superdialseg_eval.py).
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


def load_dialogs(ds: str, split: str):
    raw = json.loads((SDS / ds / f"segmentation_file_{split}.json").read_text())
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
    return (float(_nltk_pk(ts, ps, k=k)),
            float(_nltk_wd(ts, ps, k=k)))


def encode_cached(ds, split, dialogs):
    cp = CACHE / f"sds_emb_{ds}_{split}.pkl"
    if cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    enc = QueryEncoder()
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
    for (u, yt), e in zip(dialogs, embs):
        yp = seg_pred(e, m, rho, a, dstar)
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m = float(np.mean(pks)); wd_m = float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    return dict(pk=pk_m, wd=wd_m, f1=f1, score=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_v411_ctx_sweep")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--windows", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--split", default="test")
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--delta-star", type=float, default=0.5594)
    args = ap.parse_args()

    rows = {}  # (ds, m) -> metrics
    for ds in args.datasets:
        print(f"\n[load] {ds}/{args.split}")
        dia = load_dialogs(ds, args.split)
        embs = encode_cached(ds, args.split, dia)
        print(f"  n_dial={len(dia)} dim={embs[0].shape[1]}")
        for m in args.windows:
            t0 = time.perf_counter()
            r = eval_run(dia, embs, m, args.rho, args.a, args.delta_star)
            wall = time.perf_counter() - t0
            rows[(ds, m)] = r
            print(f"  m={m:2d}  Score={r['score']:.4f}  F1={r['f1']:.4f}  "
                  f"Pk={r['pk']:.4f}  WD={r['wd']:.4f}  ({wall:.0f}s)")

    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# v4.1.1 ctx_window sweep × 3 SuperDialseg-family benches",
        "",
        f"fixed: ρ={args.rho} a={args.a} δ*={args.delta_star} α=1 λ=10 "
        "(TIAGE-cfg) · sweep ctx_window only.",
        "metric = official SuperDialseg (Pk/WD k=auto, F1 binary), "
        "Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        f"split={args.split} · datasets={args.datasets} · "
        f"windows={args.windows}",
        "",
        "## Score 매트릭스 (행=ctx_window, 열=dataset, **bold**=ds best)",
        "",
        "| m | " + " | ".join(args.datasets) + " | mean |",
        "|---:|" + ":---:|" * len(args.datasets) + "---:|",
    ]
    best_per_ds = {ds: max(args.windows, key=lambda mm: rows[(ds, mm)]["score"])
                   for ds in args.datasets}
    for m in args.windows:
        cells = []
        for ds in args.datasets:
            s = rows[(ds, m)]["score"]
            cells.append(f"**{s:.4f}**" if best_per_ds[ds] == m
                         else f"{s:.4f}")
        mean = np.mean([rows[(ds, m)]["score"] for ds in args.datasets])
        L.append(f"| {m} | " + " | ".join(cells) + f" | {mean:.4f} |")
    L += ["",
          "## 세부 (F1/Pk/WD)",
          "",
          "| dataset | m | F1 | Pk | WD | Score |",
          "|---|---:|---:|---:|---:|---:|"]
    for ds in args.datasets:
        for m in args.windows:
            r = rows[(ds, m)]
            L.append(f"| {ds} | {m} | {r['f1']:.4f} | {r['pk']:.4f} | "
                     f"{r['wd']:.4f} | {r['score']:.4f} |")
    L += ["",
          "## 한계",
          "- δ* 는 TIAGE-train calibration(0.5594) 고정 — ctx_window 만 "
          "sweep. 다른 m 에서 δ* 재calibration 시 결과 달라질 수 있음.",
          "- 인코더 = `multi-qa-mpnet`(Hi-OnTop 기본). split=test 직접 사용 "
          "(δ* 도 TIAGE-train 전이값 → leakage 없음).",
          "- metric/Score 정의는 공식 SuperDialseg (literature-comparable)."]
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
