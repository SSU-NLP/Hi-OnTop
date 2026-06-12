#!/usr/bin/env python3
"""v4.2.4-exp δ* re-calibration (A: tiage-train, B: superseg-validation).

η ∈ {0.5, 0.75} 의 mixed δ_eff = sqrt(η·δ_adj² + (1-η)·δ_model²) 분포가
mpnet 단독 (v4.1.1) 과 다르기 때문에 δ*=0.5594 (v4.1.1 TIAGE-cfg) 가
부적합할 수 있음. 이 script 는:

  Phase 1 (calibration):
    - tiage train / superseg validation 각각에 대해
    - 두 encoder (mpnet + DSE-BERT) 로 dialog 인코딩 (캐시 사용)
    - η ∈ {0.5, 0.75} 별 mixed δ_eff 시퀀스 산출
    - F1-best δ* 탐색 (200-point grid over [min, max])

  Phase 2 (test eval):
    - 각 (η, calibration-source) combo 의 새 δ* 로 v4.2.4 평가
    - dialseg711-test 는 tiage-train δ* 사용 (cross-corpus, no train split)
    - tiage-test 는 tiage-train δ* (in-domain)
    - superseg-test 는 superseg-val δ* (in-domain)
    - 비교: 원래 v4.2.4 @ δ*=0.5594 (TIAGE-cfg) vs 새 δ*

  Output: outputs/experiments/2026-05-21_v424_calib/REPORT.md
"""

from __future__ import annotations

import json
import math
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
from hi_ontop.sem_core_v424_exp import HiOnTopSegmenterV424Exp  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

ENC_TOPIC = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
ENC_FLOW = "aws-ai/dse-bert-base"

# v4.1.1 mpnet (TIAGE-train best)
MPNET_M, MPNET_RHO, MPNET_A = 2, 0.7, 0.5
# v4.2.2 DSE (TIAGE-train best)
DSE_M, DSE_RHO, DSE_A = 2, 0.5, 0.0

ETAS = [0.5, 0.75]
CALIB_SOURCES = [("tiage", "train"), ("superseg", "validation")]
TEST_DATASETS = ["tiage", "dialseg711", "superseg"]

# v4.2.4 prior sweep results (TIAGE-cfg δ*=0.5594) — for comparison
PRIOR_V424 = {
    0.5:  {"tiage": 0.4892, "dialseg711": 0.6480, "superseg": 0.3806},
    0.75: {"tiage": 0.4724, "dialseg711": 0.6254, "superseg": 0.4275},
}
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


def encode_cached(dataset, split, encoder_name, dialogs):
    safe = encoder_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{dataset}_{split}_{safe}.pkl"
    if cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    print(f"  [encode] {encoder_name} on {dataset}/{split} ({len(dialogs)} dialogs) ...")
    t0 = time.time()
    enc = QueryEncoder(model_name=encoder_name)
    embs = [np.asarray(enc.encode([u for u in utts])) for utts, _ in dialogs]
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump(embs, fh)
    print(f"  [encode] done {time.time()-t0:.0f}s → {cp.name}")
    return embs


def _cos_delta(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return 1.0 - float(np.dot(a, b) / (na * nb))


def causal_delta(emb_seq, m, rho, a, t):
    """δ_adj at transition t-1→t."""
    d_prev = _cos_delta(emb_seq[t - 1], emb_seq[t])
    if d_prev is None:
        return None
    win = emb_seq[max(0, t - m):t][::-1]  # most recent first (i=0 → s_{t-1})
    c = np.zeros_like(win[-1] if isinstance(win, list) else win[0], dtype=np.float64)
    for i, vec in enumerate(win):
        c += (rho ** i) * vec
    d_ctx = _cos_delta(c, emb_seq[t])
    if d_ctx is None:
        return d_prev
    return a * d_prev + (1.0 - a) * d_ctx


def delta_eff_v424(emb_t, emb_d, eta):
    """Mixed δ_eff for v4.2.4 per transition."""
    out = []
    for t in range(1, len(emb_t)):
        d_adj = causal_delta(emb_t, MPNET_M, MPNET_RHO, MPNET_A, t)
        d_model = causal_delta(emb_d, DSE_M, DSE_RHO, DSE_A, t)
        if d_adj is None:
            out.append(None)
            continue
        if d_model is None:
            d_model = 0.0
        d_eff = math.sqrt(eta * d_adj * d_adj + (1.0 - eta) * d_model * d_model)
        out.append(d_eff)
    return out


def f1_best_threshold(deltas_flat, gts_flat, n_grid=200):
    arr = np.array([d for d in deltas_flat if d is not None], dtype=float)
    gt = np.array(gts_flat, dtype=int)
    if len(arr) != len(gt):
        # mismatch (None entries removed). Re-align.
        keep_idx = [i for i, d in enumerate(deltas_flat) if d is not None]
        gt = gt[keep_idx]
    if len(arr) == 0:
        return (0.0, 0.0)
    best = (0.0, float(arr.mean()))
    for th in np.linspace(arr.min(), arr.max(), n_grid):
        pred = (arr >= th).astype(int)
        f = float(f1_score(gt, pred, zero_division=0))
        if f > best[0]:
            best = (f, float(th))
    return best  # (best_f1, best_th)


def calibrate_one(dataset, split, eta):
    dialogs = load_dialogs(dataset, split)
    print(f"\n[calib] {dataset}/{split}, η={eta}")
    print(f"  n_dial={len(dialogs)}, n_turns={sum(len(u) for u,_ in dialogs)}")
    embs_t = encode_cached(dataset, split, ENC_TOPIC, dialogs)
    embs_d = encode_cached(dataset, split, ENC_FLOW, dialogs)

    deltas_flat, gts_flat = [], []
    for (utts, yt), et, ed in zip(dialogs, embs_t, embs_d):
        seq = delta_eff_v424(et, ed, eta)
        for i, d in enumerate(seq):
            deltas_flat.append(d)
            gts_flat.append(yt[i])  # boundary AFTER turn i

    best_f1, best_th = f1_best_threshold(deltas_flat, gts_flat)
    print(f"  → F1-best δ*={best_th:.4f}  train F1={best_f1:.3f}")
    return best_th, best_f1


# ------------------- test eval -------------------

def official_pk_wd(yt, yp):
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


def run_v424_test(dataset, eta, dstar):
    print(f"\n[eval] {dataset}/test, η={eta}, δ*={dstar:.4f}")
    dialogs = load_dialogs(dataset, "test")
    embs_t = encode_cached(dataset, "test", ENC_TOPIC, dialogs)
    embs_d = encode_cached(dataset, "test", ENC_FLOW, dialogs)
    pks, wds, g, p = [], [], [], []
    for (_, yt), et, ed in zip(dialogs, embs_t, embs_d):
        seg = HiOnTopSegmenterV424Exp(
            dim=et.shape[1], alpha=1.0, lmda=10.0,
            delta_star=dstar,
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
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    print(f"  → Score={score:.4f}  F1={f1:.4f}  Pk={pk_m:.4f}  WD={wd_m:.4f}")
    return dict(score=score, f1=f1, pk=pk_m, wd=wd_m)


def main():
    name = "2026-05-21_v424_calib"

    # --- Phase 1: calibrate ---
    calib = {}  # (source_ds, source_split, eta) → (dstar, train_f1)
    for ds, sp in CALIB_SOURCES:
        for eta in ETAS:
            dstar, train_f1 = calibrate_one(ds, sp, eta)
            calib[(ds, sp, eta)] = (dstar, train_f1)

    # --- Phase 2: eval ---
    # For each η, pick δ* per test dataset:
    #   tiage-test       ← tiage-train δ*
    #   dialseg711-test  ← tiage-train δ* (cross-corpus convention)
    #   superseg-test    ← superseg-val δ*  (in-corpus)
    results = {}  # (eta, ds) → metrics
    for eta in ETAS:
        dstar_tiage, _ = calib[("tiage", "train", eta)]
        dstar_super, _ = calib[("superseg", "validation", eta)]
        for ds in TEST_DATASETS:
            dstar = dstar_tiage if ds in ("tiage", "dialseg711") else dstar_super
            results[(eta, ds)] = run_v424_test(ds, eta, dstar)

    # --- REPORT ---
    out = REPO / "outputs" / "experiments" / name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    L = [
        f"# v4.2.4-exp δ* re-calibration (A: tiage-train, B: superseg-validation)",
        "",
        "**Setup**:",
        "- v4.2.4-exp 의 mixed δ_eff = √(η·δ_adj² + (1-η)·δ_model²) 분포가",
        "  mpnet 단독과 다름 → δ*=0.5594 (TIAGE-cfg) 가 부적합할 수 있음.",
        "- 본 실험: η ∈ {0.5, 0.75} 별로 F1-best δ* 를 calibration source 에서",
        "  재산출 후 test 평가.",
        "- Calibration source: **tiage-train** (300 conv, 4692 turn) +",
        "  **superseg-validation** (1322 conv, 17734 turn).",
        "  ⚠ superseg-train 은 6948 conv / 92k turn 으로 CPU 환경에서 인코딩",
        "  불가능 → validation 사용 (v4.1.1.md 의 \"val-cal\" convention).",
        "- δ* 적용 정책 (no test leakage):",
        "  - tiage-test ← tiage-train δ* (in-domain, calibration overlap ⚠)",
        "  - dialseg711-test ← tiage-train δ* (cross-corpus, train split 없음)",
        "  - superseg-test ← superseg-validation δ* (in-corpus)",
        "",
        "## Phase 1 — calibration 결과",
        "",
        "| source | split | η | δ\\* (F1-best) | train F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for (ds, sp), in [(p, ) for p in CALIB_SOURCES]:
        for eta in ETAS:
            dstar, tf1 = calib[(ds, sp, eta)]
            L.append(f"| {ds} | {sp} | {eta} | {dstar:.4f} | {tf1:.3f} |")

    L += [
        "",
        "## Phase 2 — test 평가 (재calibration δ* 사용)",
        "",
        "| η | dataset | δ\\* source | δ\\* | **Score ↑** | F1 ↑ | Pk ↓ | WD ↓ |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for eta in ETAS:
        for ds in TEST_DATASETS:
            if ds in ("tiage", "dialseg711"):
                src = "tiage-train"
                dstar = calib[("tiage", "train", eta)][0]
            else:
                src = "superseg-val"
                dstar = calib[("superseg", "validation", eta)][0]
            m = results[(eta, ds)]
            L.append(
                f"| {eta} | {ds} | {src} | {dstar:.4f} | **{m['score']:.4f}** | "
                f"{m['f1']:.4f} | {m['pk']:.4f} | {m['wd']:.4f} |"
            )

    L += [
        "",
        "## 비교 — 재calibration 전 (δ*=0.5594, TIAGE-cfg) vs 후",
        "",
        "| η | dataset | prior Score | new Score | Δ | v4.1.1 baseline |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for eta in ETAS:
        for ds in TEST_DATASETS:
            prior = PRIOR_V424[eta][ds]
            new = results[(eta, ds)]["score"]
            delta = new - prior
            base = V411_BASELINE[ds]
            sign = "+" if delta >= 0 else ""
            L.append(
                f"| {eta} | {ds} | {prior:.4f} | {new:.4f} | "
                f"{sign}{delta:+.4f} | {base:.4f} |"
            )

    # 3-set mean comparison
    L += [
        "",
        "## 3-set mean Score",
        "",
        "| 변형 | mean Score | vs v4.1.1 (0.5068) |",
        "|---|---:|---:|",
        f"| v4.1.1 (mpnet) | 0.5068 | — |",
    ]
    for eta in ETAS:
        prior_mean = sum(PRIOR_V424[eta].values()) / 3
        new_mean = sum(results[(eta, ds)]["score"] for ds in TEST_DATASETS) / 3
        L.append(
            f"| v4.2.4 @η={eta}, TIAGE-cfg δ* | {prior_mean:.4f} | {prior_mean-0.5068:+.4f} |"
        )
        L.append(
            f"| **v4.2.4 @η={eta}, re-cal δ\\*** | **{new_mean:.4f}** | "
            f"{new_mean-0.5068:+.4f} |"
        )

    L += [
        "",
        "## 해석 / 판정",
        "",
        "(채울 것)",
        "",
        "## 한계",
        "- superseg-train (92k turns) 인코딩 환경 제약으로 validation 사용. ",
        "  train 과 validation 분포 약간 다를 수 있음.",
        "- F1-best δ* 채택. Score-best δ* 와 다를 수 있음 (v4.1.1.md 의 superseg",
        "  val-cal 경험상 F1-best 가 과분절 유발 가능).",
        "- Single trial, seed=0. variance 미측정.",
        "- DSE channel HP (m=2, ρ=0.5, a=0.0) 은 v4.2.2 best 고정. joint",
        "  re-calibration 안 함.",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
