#!/usr/bin/env python3
"""v4.1.2-topicctx ABLATION × 3 SuperDialseg-family benches (smoke).

policies: ('window', m=2 baseline) / topic_cur / topic_prev
fixed: ρ=0.7 a=0.5 δ*=0.5594 α=1 λ=10 (TIAGE-cfg). 임베딩 캐시 재사용.
δ* 재calibration 미수행 — codex 권고대로 1차 smoke 효과 방향만 본 후
필요시 재calib 본평가 별도. ctx_max_len=64 안전캡.
"""

from __future__ import annotations

import argparse
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
from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411  # noqa: E402
from hi_ontop.sem_core_v412_topicctx import HiOnTopSegmenterV412TopicCtx  # noqa: E402

CACHE = REPO / "outputs" / "runs" / "_misc"
SDS = REPO / "benchmarks" / "superdialseg_data"


def load_dialogs(ds, split="test"):
    import json
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


def seg_pred(emb, build_seg):
    seg = build_seg(emb.shape[1])
    torch.manual_seed(0)
    ids = [seg.assign(s.astype(np.float64))[0] for s in emb]
    return [1 if ids[i] != ids[i + 1] else 0
            for i in range(len(ids) - 1)] + [0]


def eval_run(dialogs, embs, build_seg):
    pks, wds, g, p = [], [], [], []
    eff_ms = []  # TopicCur/Prev 의 유효 m 진단(turn 당 ctx 필터 후 길이)
    for (u, yt), e in zip(dialogs, embs):
        yp = seg_pred(e, build_seg)
        pk, wd = official_pk_wd(yt, yp); pks.append(pk); wds.append(wd)
        g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m = float(np.mean(pks)); wd_m = float(np.mean(wds))
    return dict(pk=pk_m, wd=wd_m, f1=f1,
                score=0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_v412_topicctx_smoke")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--a", type=float, default=0.5)
    ap.add_argument("--delta-star", type=float, default=0.5594)
    args = ap.parse_args()

    common = dict(alpha=1.0, lmda=10.0, delta_star=args.delta_star,
                  ctx_window=2, ctx_decay=args.rho, ctx_blend_a=args.a)

    def build_window(dim):
        return HiOnTopSegmenterV411(dim=dim, **common)

    def build_topic_cur(dim):
        return HiOnTopSegmenterV412TopicCtx(
            dim=dim, ctx_policy="topic_cur", ctx_max_len=64, **common)

    def build_topic_prev(dim):
        return HiOnTopSegmenterV412TopicCtx(
            dim=dim, ctx_policy="topic_prev", ctx_max_len=64, **common)

    policies = [
        ("v4.1.1 window m=2 (baseline)", build_window),
        ("v4.1.2-topic_cur", build_topic_cur),
        ("v4.1.2-topic_prev", build_topic_prev),
    ]
    rows = {}  # (ds, label) -> metrics
    for ds in args.datasets:
        print(f"\n[load] {ds}/test")
        dia = load_dialogs(ds)
        with open(CACHE / f"sds_emb_{ds}_test.pkl", "rb") as fh:
            embs = pickle.load(fh)
        print(f"  n_dial={len(dia)} dim={embs[0].shape[1]}")
        for label, build in policies:
            t0 = time.perf_counter()
            r = eval_run(dia, embs, build)
            wall = time.perf_counter() - t0
            rows[(ds, label)] = r
            print(f"  {label:35s}  Score={r['score']:.4f}  F1={r['f1']:.4f}  "
                  f"Pk={r['pk']:.4f}  WD={r['wd']:.4f}  ({wall:.0f}s)")

    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# v4.1.2-topicctx ABLATION × 3 benches (smoke, fixed δ*)",
        "",
        f"fixed: ρ={args.rho} a={args.a} δ*={args.delta_star} α=1 λ=10 "
        "(TIAGE-cfg) · ctx_max_len=64. **δ* 재calib 미수행** — codex 권고: "
        "1차 효과 방향 확인만. 본평가는 재calib 후 별도.",
        "",
        "## Score 매트릭스",
        "",
        "| policy | " + " | ".join(args.datasets) + " | mean | vs window m=2 |",
        "|---|" + ":---:|" * len(args.datasets) + "---:|---:|",
    ]
    win_means = {ds: rows[(ds, policies[0][0])]["score"]
                 for ds in args.datasets}
    for label, _ in policies:
        cells = [f"{rows[(ds, label)]['score']:.4f}" for ds in args.datasets]
        mean = np.mean([rows[(ds, label)]["score"] for ds in args.datasets])
        delta = mean - np.mean(list(win_means.values()))
        L.append(f"| {label} | " + " | ".join(cells) +
                 f" | {mean:.4f} | {delta:+.4f} |")

    L += ["",
          "## 세부 (F1/Pk/WD)",
          "",
          "| dataset | policy | F1 | Pk | WD | Score |",
          "|---|---|---:|---:|---:|---:|"]
    for ds in args.datasets:
        for label, _ in policies:
            r = rows[(ds, label)]
            L.append(f"| {ds} | {label} | {r['f1']:.4f} | {r['pk']:.4f} | "
                     f"{r['wd']:.4f} | {r['score']:.4f} |")
    L += ["",
          "## 한계 / 정직성",
          f"- **δ* 고정 (TIAGE-train {args.delta_star})** — topic-aware ctx 는 "
          "δ_ctx 분포를 바꾸므로 본평가는 δ* 재calib 필요(codex 권고).",
          "- ctx_max_len=64 안전캡 (실 대화는 거의 안 닿음).",
          "- baseline = v4.1.1 fixed-window m=2 (현 TIAGE-cfg). m=8 (dialseg711 "
          "best) 와의 비교는 정책 선택 trade-off 평가 시 별도 필요.",
          "- 판단 기준 (codex): tiage/superseg long-ctx 손실 회복 + dialseg711 "
          "long-ctx 이득 보존을 각각 확인 (mean 만 보지 말 것).",
          "- ablation 위치: `v4.1.2-topicctx` 후보, default 승격은 별건."]
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
