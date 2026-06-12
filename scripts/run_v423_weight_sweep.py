#!/usr/bin/env python3
"""v4.2.3-exp weight sweep — w_topic ∈ {1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.5}.

Single-trial smoke (2026-05-20_v423_smoke) showed w_topic=0.75 wins
TIAGE (+2.3pp) but loses dialseg711/superseg (-2 to -3pp) vs mpnet-only
(v4.1.1). Sweep tests whether a more conservative w_topic recovers the
formal-benchmark performance while keeping TIAGE gain.

Constraint: w_topic + w_flow = 1 (so the energy r preserves boundary
point at z_topic=z_flow=1 → r=1). Lower w_flow == less DSE influence.

Sanity: w_topic=1.0, w_flow=0.0 should match v4.1.1 (mpnet only) numbers
from 2026-05-20_v422_smoke (Score 0.4675 / 0.5897 / 0.4631).
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
from hi_ontop.sem_core_v423_exp import HiOnTopSegmenterV423Exp  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
ENC_TOPIC = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ENC_FLOW = "aws-ai/dse-bert-base"

# v4.2.2 smoke calibration defaults
TOPIC_PARAMS = dict(m=2, rho=0.7, a=0.5, dstar=0.5594)
FLOW_PARAMS = dict(m=2, rho=0.5, a=0.0, dstar=0.4569)


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


def load_cache(dataset, split, encoder_name):
    safe = encoder_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{dataset}_{split}_{safe}.pkl"
    if not cp.exists():
        raise SystemExit(f"cache missing: {cp}")
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def run_v423(embs_t, embs_f, w_topic):
    w_flow = 1.0 - w_topic
    out = []
    for et, ef in zip(embs_t, embs_f):
        seg = HiOnTopSegmenterV423Exp(
            dim=et.shape[1], alpha=1.0, lmda=10.0,
            m_topic=TOPIC_PARAMS["m"], rho_topic=TOPIC_PARAMS["rho"],
            a_topic=TOPIC_PARAMS["a"], delta_star_topic=TOPIC_PARAMS["dstar"],
            m_flow=FLOW_PARAMS["m"], rho_flow=FLOW_PARAMS["rho"],
            a_flow=FLOW_PARAMS["a"], delta_star_flow=FLOW_PARAMS["dstar"],
            w_topic=w_topic, w_flow=w_flow,
        )
        torch.manual_seed(0)
        ids = []
        for st, sf in zip(et, ef):
            k, _ = seg.assign_pair(st.astype(np.float64), sf.astype(np.float64))
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
    ap.add_argument("--name", default="2026-05-20_v423_weight_sweep")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.5])
    args = ap.parse_args()

    # 캐시 + dialogs pre-load (sweep 동안 재사용)
    print(f"[load] datasets + caches")
    data = {}
    for ds in args.datasets:
        dia = load_dialogs(ds, "test")
        et = load_cache(ds, "test", ENC_TOPIC)
        ef = load_cache(ds, "test", ENC_FLOW)
        data[ds] = (dia, et, ef)
        print(f"  {ds}: n_dial={len(dia)}")

    # sweep
    print(f"\n[sweep] weights={args.weights}")
    grid = {}  # (w, ds) → metrics
    for w in args.weights:
        for ds in args.datasets:
            dia, et, ef = data[ds]
            t0 = time.perf_counter()
            preds = run_v423(et, ef, w)
            metrics = eval_metrics(dia, preds)
            wall = time.perf_counter() - t0
            grid[(w, ds)] = (metrics, wall)
            print(f"  w_topic={w:.2f}  {ds:12s}  Score={metrics['score']:.4f}  "
                  f"F1={metrics['f1']:.4f}  Pk={metrics['pk']:.4f}  "
                  f"WD={metrics['wd']:.4f}  ({wall:.0f}s)")

    # REPORT
    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# v4.2.3-exp weight sweep — w_topic ∈ " + str(args.weights),
        "",
        "**Setup**:",
        f"- topic encoder: `{ENC_TOPIC}` (m={TOPIC_PARAMS['m']}, ρ={TOPIC_PARAMS['rho']}, "
        f"a={TOPIC_PARAMS['a']}, δ*={TOPIC_PARAMS['dstar']}). TIAGE-train calibrated.",
        f"- flow encoder:  `{ENC_FLOW}` (m={FLOW_PARAMS['m']}, ρ={FLOW_PARAMS['rho']}, "
        f"a={FLOW_PARAMS['a']}, δ*={FLOW_PARAMS['dstar']}). TIAGE-train (DSE-BERT) calibrated.",
        "- constraint: w_topic + w_flow = 1.",
        "- combine: r = √(w_topic·z_topic² + (1-w_topic)·z_flow²), boundary ⇔ r ≥ 1.",
        "- Sanity check: w_topic=1.0 == v4.1.1 (mpnet-only) (must match 2026-05-20_v422_smoke).",
        "",
        "## Score matrix (행=w_topic, 열=dataset, **bold**=row best)",
        "",
        "| w_topic | w_flow | " + " | ".join(args.datasets) + " | mean |",
        "|---:|---:|" + "---:|" * (len(args.datasets) + 1),
    ]
    for w in args.weights:
        row_scores = [grid[(w, ds)][0]["score"] for ds in args.datasets]
        mean = sum(row_scores) / len(row_scores)
        row = [f"{w:.2f}", f"{1.0 - w:.2f}"] + [f"{s:.4f}" for s in row_scores] + [f"{mean:.4f}"]
        L.append("| " + " | ".join(row) + " |")

    L += [
        "",
        "## Detailed metrics",
        "",
        "| w_topic | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for w in args.weights:
        for ds in args.datasets:
            m, _ = grid[(w, ds)]
            L.append(f"| {w:.2f} | {ds} | {m['score']:.4f} | {m['f1']:.4f} | "
                     f"{m['pk']:.4f} | {m['wd']:.4f} |")

    # Best per dataset
    L += [
        "",
        "## Per-dataset best",
        "",
        "| dataset | best w_topic | Score | vs v4.1.1 baseline |",
        "|---|---:|---:|---:|",
    ]
    V411_BASELINE = {"tiage": 0.4675, "dialseg711": 0.5897, "superseg": 0.4631}
    for ds in args.datasets:
        best_w, best_score = max(
            ((w, grid[(w, ds)][0]["score"]) for w in args.weights),
            key=lambda x: x[1],
        )
        delta = best_score - V411_BASELINE[ds]
        L.append(f"| {ds} | {best_w:.2f} | {best_score:.4f} | "
                 f"{'+' if delta >= 0 else ''}{delta:+.4f} |")

    L += [
        "",
        "## 해석 / 판정",
        "",
        "(채울 것)",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
