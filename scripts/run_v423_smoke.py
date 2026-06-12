#!/usr/bin/env python3
"""v4.2.3-exp smoke: dual-channel (mpnet topic + DSE-BERT flow) on
{tiage, dialseg711, superseg} test, side-by-side with v4.1.1 (mpnet
only) and v4.2.2 (DSE-BERT only) results from 2026-05-20_v422_smoke.

Single trial only (codex 2026-05-20 권고):
  topic (mpnet) @ m=2, ρ=0.7, a=0.5, δ*=0.5594
  flow  (DSE)   @ m=2, ρ=0.5, a=0.0, δ*=0.4569
  r = sqrt(0.75·z_topic² + 0.25·z_flow²),  boundary ⇔ r ≥ 1

Also logs diagnostic monitoring metrics (codex risk list):
  - corr(z_topic, z_flow) per dataset
  - mean / std of z_topic, z_flow, r
  - both-high (z_t>1 and z_f>1) rate
  - both-low rate
  - boundary rate (Pk/WD comparison)
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
from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411  # noqa: E402
from hi_ontop.sem_core_v423_exp import HiOnTopSegmenterV423Exp  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

ENC_TOPIC = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ENC_FLOW = "aws-ai/dse-bert-base"


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


def load_emb_cache(dataset: str, split: str, encoder_name: str):
    safe = encoder_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{dataset}_{split}_{safe}.pkl"
    if not cp.exists():
        raise SystemExit(
            f"인코딩 캐시 없음: {cp}\n"
            "→ scripts/run_v422_smoke.py 먼저 돌려서 두 encoder 캐시 생성 필수."
        )
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def seg_pred_v423(
    embs_topic, embs_flow,
    m_topic, rho_topic, a_topic, dstar_topic,
    m_flow, rho_flow, a_flow, dstar_flow,
    w_topic, w_flow,
):
    """Run v4.2.3-exp on a list of (topic_embs, flow_embs) per dialog.

    Returns: list per dialog of (y_pred per turn, [z_topic_per_trans,
    z_flow_per_trans, r_per_trans]).
    """
    per_dialog = []
    for et, ef in zip(embs_topic, embs_flow):
        seg = HiOnTopSegmenterV423Exp(
            dim=et.shape[1], alpha=1.0, lmda=10.0,
            m_topic=m_topic, rho_topic=rho_topic, a_topic=a_topic,
            delta_star_topic=dstar_topic,
            m_flow=m_flow, rho_flow=rho_flow, a_flow=a_flow,
            delta_star_flow=dstar_flow,
            w_topic=w_topic, w_flow=w_flow,
        )
        torch.manual_seed(0)
        ids, z_t_list, z_f_list, r_list = [], [], [], []
        for st, sf in zip(et, ef):
            k, _ = seg.assign_pair(st.astype(np.float64), sf.astype(np.float64))
            ids.append(k)
            z_t_list.append(seg.last_z_topic)
            z_f_list.append(seg.last_z_flow)
            r_list.append(seg.last_r)
        yp = [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]
        per_dialog.append((yp, z_t_list, z_f_list, r_list))
    return per_dialog


def eval_run(dialogs, predictions):
    pks, wds, g, p = [], [], [], []
    for (_, yt), (yp, _, _, _) in zip(dialogs, predictions):
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    return dict(pk=pk_m, wd=wd_m, f1=f1, score=score)


def diagnostic_stats(predictions):
    """corr(z_topic, z_flow), both-high / both-low rates, etc. (codex risk list)."""
    all_zt, all_zf, all_r = [], [], []
    for _, zt, zf, r in predictions:
        # Skip the first None entries (no _prev_s yet)
        for a, b, c in zip(zt, zf, r):
            if a is None or b is None:
                continue
            all_zt.append(a); all_zf.append(b); all_r.append(c)
    if not all_zt:
        return dict(n=0)
    zt = np.array(all_zt); zf = np.array(all_zf); rr = np.array(all_r)
    corr = float(np.corrcoef(zt, zf)[0, 1]) if len(zt) > 1 else 0.0
    return dict(
        n=len(zt),
        z_topic_mean=float(zt.mean()), z_topic_std=float(zt.std()),
        z_flow_mean=float(zf.mean()), z_flow_std=float(zf.std()),
        r_mean=float(rr.mean()), r_std=float(rr.std()),
        corr_zt_zf=corr,
        both_high_rate=float(((zt > 1) & (zf > 1)).mean()),
        both_low_rate=float(((zt < 1) & (zf < 1)).mean()),
        only_topic_high=float(((zt > 1) & (zf <= 1)).mean()),
        only_flow_high=float(((zt <= 1) & (zf > 1)).mean()),
        boundary_rate=float((rr >= 1).mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_v423_smoke")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--split", default="test")
    # codex single-trial defaults (no sweep)
    ap.add_argument("--m-topic", type=int, default=2)
    ap.add_argument("--rho-topic", type=float, default=0.7)
    ap.add_argument("--a-topic", type=float, default=0.5)
    ap.add_argument("--dstar-topic", type=float, default=0.5594)
    ap.add_argument("--m-flow", type=int, default=2)
    ap.add_argument("--rho-flow", type=float, default=0.5)
    ap.add_argument("--a-flow", type=float, default=0.0)
    ap.add_argument("--dstar-flow", type=float, default=0.4569)
    ap.add_argument("--w-topic", type=float, default=0.75)
    ap.add_argument("--w-flow", type=float, default=0.25)
    args = ap.parse_args()

    rows = []
    diag = {}
    for ds in args.datasets:
        print(f"\n[load] {ds}/{args.split}")
        dia = load_dialogs(ds, args.split)
        et = load_emb_cache(ds, args.split, ENC_TOPIC)
        ef = load_emb_cache(ds, args.split, ENC_FLOW)
        print(f"  n_dial={len(dia)} topic_dim={et[0].shape[1]} flow_dim={ef[0].shape[1]}")
        t0 = time.perf_counter()
        preds = seg_pred_v423(
            et, ef,
            args.m_topic, args.rho_topic, args.a_topic, args.dstar_topic,
            args.m_flow, args.rho_flow, args.a_flow, args.dstar_flow,
            args.w_topic, args.w_flow,
        )
        wall = time.perf_counter() - t0
        r = eval_run(dia, preds)
        d = diagnostic_stats(preds)
        rows.append((ds, r, wall))
        diag[ds] = d
        print(f"  v4.2.3-exp  Score={r['score']:.4f} F1={r['f1']:.4f} "
              f"Pk={r['pk']:.4f} WD={r['wd']:.4f}  ({wall:.0f}s)")
        print(f"  diag  corr(z_t,z_f)={d['corr_zt_zf']:.3f}  "
              f"both_high={d['both_high_rate']:.3f}  "
              f"only_topic={d['only_topic_high']:.3f}  "
              f"only_flow={d['only_flow_high']:.3f}  "
              f"boundary_rate={d['boundary_rate']:.3f}")

    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    # v4.2.2 baseline numbers (from 2026-05-20_v422_smoke REPORT) for direct comparison
    BASELINE = {
        "tiage":      {"v4.1.1 (mpnet)":     {"score": 0.4675, "f1": 0.4102, "pk": 0.4421, "wd": 0.5082},
                       "v4.2.2 (DSE-BERT)":  {"score": 0.4747, "f1": 0.4420, "pk": 0.4366, "wd": 0.5488}},
        "dialseg711": {"v4.1.1 (mpnet)":     {"score": 0.5897, "f1": 0.5493, "pk": 0.3248, "wd": 0.4151},
                       "v4.2.2 (DSE-BERT)":  {"score": 0.4547, "f1": 0.4587, "pk": 0.4472, "wd": 0.6514}},
        "superseg":   {"v4.1.1 (mpnet)":     {"score": 0.4631, "f1": 0.4323, "pk": 0.4711, "wd": 0.5410},
                       "v4.2.2 (DSE-BERT)":  {"score": 0.3886, "f1": 0.3488, "pk": 0.5332, "wd": 0.6098}},
    }

    L = [
        f"# v4.2.3-exp smoke — dual-channel (mpnet topic + DSE-BERT flow)",
        "",
        "**Setup**:",
        "- v4.1.1 algorithm + v4.2.3 dual-channel PE override.",
        f"- topic: encoder=`{ENC_TOPIC}`, m={args.m_topic}, ρ={args.rho_topic}, "
        f"a={args.a_topic}, δ*={args.dstar_topic} (TIAGE-train).",
        f"- flow:  encoder=`{ENC_FLOW}`, m={args.m_flow}, ρ={args.rho_flow}, "
        f"a={args.a_flow}, δ*={args.dstar_flow} (TIAGE-train, DSE-BERT).",
        f"- weights: w_topic={args.w_topic}, w_flow={args.w_flow}.",
        f"- combine: r = √(w_topic·z_topic² + w_flow·z_flow²),  boundary ⇔ r ≥ 1.",
        "- Dialseg711 uses TIAGE-train δ\\* per encoder (no train split available).",
        "- Metric: official SuperDialseg (Pk/WD k=auto, F1 binary), "
        "Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        "",
        "## 결과 vs v4.1.1 (mpnet) / v4.2.2 (DSE-only)",
        "",
        "| arm | dataset | **Score ↑** | F1 ↑ | Pk ↓ | WD ↓ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ds, r, _ in rows:
        L.append(f"| **v4.2.3-exp dual** | {ds} | **{r['score']:.4f}** | {r['f1']:.4f} | {r['pk']:.4f} | {r['wd']:.4f} |")
        for label, b in BASELINE[ds].items():
            L.append(f"| {label} | {ds} | {b['score']:.4f} | {b['f1']:.4f} | {b['pk']:.4f} | {b['wd']:.4f} |")
        L.append("|---|---|---|---|---|---|")

    L += [
        "",
        "## 진단 metric (codex risk list)",
        "",
        "| dataset | n_trans | corr(z_t,z_f) | both_high | both_low | only_t | only_f | boundary_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in args.datasets:
        d = diag[ds]
        L.append(
            f"| {ds} | {d['n']} | {d['corr_zt_zf']:.3f} | "
            f"{d['both_high_rate']:.3f} | {d['both_low_rate']:.3f} | "
            f"{d['only_topic_high']:.3f} | {d['only_flow_high']:.3f} | "
            f"{d['boundary_rate']:.3f} |"
        )

    L += [
        "",
        "**해석 / 판정**: (분석 후 채울 것)",
        "",
        "## 한계 / 검증 미해결",
        "",
        "- 단일 trial (w_topic=0.75, w_flow=0.25 고정). sweep 없음.",
        "- (m, ρ, a) 채널별 분리는 했지만 train calibration 은 각 채널 단독 best 그대로 — "
        "  dual 결합 시 joint optimal 일 가능성 있음.",
        "- seed=0 단일 run, variance 미측정.",
        "- generic opener 등 발화 유형별 bucket 분석은 본 smoke 범위 밖.",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
