#!/usr/bin/env python3
"""Evaluate Hi-OnTop v3.3.9 (BEST) + prev-cos baseline on the standard
dialogue-segmentation benchmarks Dialseg711 / SuperDialseg (and their
TIAGE copy for cross-check), with the **official SuperDialseg Pk/WD**
metric so numbers are literature-comparable.

Official metric (verbatim from
``benchmarks/superdialseg/src/super_dialseg/metrics/segmentation.py``
L23-33 / L49-59, window_size='auto'): per dialogue,
``n_seg=sum(y_true)+1 ; k=max(2,round(len(y_true)/n_seg/2))`` then
``nltk.metrics.pk/windowdiff`` on the joined 0/1 string; averaged over
dialogues. y_true/y_pred = per-turn boundary labels (1 = last turn of a
segment), last turn forced 0 for BOTH (reproduce/main.py L111-113).
Copied (not imported) to avoid the toolkit's gensim/torch deps — the
formula IS the spec; identical computation.

GT boundary = dataset's per-turn ``segmentation_label`` (authoritative,
last→0). Our segmenter's predicted topic-id sequence → y_pred[i] =
1 if pred[i]!=pred[i+1] else 0, last→0 (same convention).

Methods:
  * ``prevcos``  — 1-line δ_prev<θ baseline. θ = best-F1 on this
    dataset (ORACLE upper bound = the unsupervised-similarity ceiling
    on this benchmark; marked).
  * ``v3.3.9@tiage`` — BEST segmenter, δ* = TIAGE-train 0.5557
    (HONEST zero-shot transfer — no tuning on this data).
  * ``v3.3.9@oracle`` — δ* = best-F1 δ_prev threshold on this dataset
    (UPPER BOUND if δ* were perfectly calibrated here; marked).

Embeddings cached per dataset (encode once, reuse across methods).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.embedding import QueryEncoder  # noqa: E402
from hi_ontop.sem_core_v339 import HiOnTopSegmenterV339  # noqa: E402
from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
TIAGE_DELTA_STAR = 0.5557  # v3.3.9 TIAGE-train δ*
# v4.1.1 TIAGE-train calibration best (v411_delta_star_train.md):
V411_M, V411_RHO, V411_A, V411_DSTAR = 2, 0.7, 0.5, 0.5594


def total_score(f1: float, pk: float, wd: float) -> float:
    """Official SuperDialseg aggregate (segmentation.py compute_total_score
    L91-93, verbatim): 0.5·F1 + 0.25·(1-Pk) + 0.25·(1-WD)."""
    return 0.5 * f1 + 0.25 * (1.0 - pk) + 0.25 * (1.0 - wd)


def load_dialogs(dataset: str, split: str):
    """→ list of (utterances:list[str], y_true:list[int]) per dialogue.
    y_true = segmentation_label, last turn forced 0 (official convention)."""
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


def official_pk_wd(y_true, y_pred):
    """Verbatim SuperDialseg metric (window_size='auto'), per dialogue."""
    n_seg = sum(y_true) + 1
    k = max(2, int(round(len(y_true) / n_seg / 2)))
    ts = "".join(str(v) for v in y_true)
    ps = "".join(str(v) for v in y_pred)
    return float(_nltk_pk(ref=ts, hyp=ps, k=k)), float(_nltk_wd(seg1=ts, seg2=ps, k=k))


def encode_cached(dataset: str, split: str, dialogs):
    cpath = CACHE / f"sds_emb_{dataset}_{split}.pkl"
    if cpath.exists():
        with open(cpath, "rb") as fh:
            return pickle.load(fh)
    enc = QueryEncoder()
    embs = [np.asarray(enc.encode(u)) for u, _ in dialogs]
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cpath, "wb") as fh:
        pickle.dump(embs, fh)
    return embs


def delta_prev_seq(emb):
    out = []
    for i in range(1, len(emb)):
        a, b = emb[i - 1], emb[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out.append(None if na <= 1e-12 or nb <= 1e-12
                   else 1.0 - float(np.dot(a, b) / (na * nb)))
    return out  # length L-1 (transition t→ between turn t and t+1)


def prevcos_pred(emb, theta):
    """y_pred per turn: boundary AFTER turn i if δ_prev(i→i+1) ≥ θ."""
    L = len(emb)
    yp = [0] * L
    dp = delta_prev_seq(emb)
    for i in range(L - 1):
        if dp[i] is not None and dp[i] >= theta:
            yp[i] = 1
    yp[-1] = 0
    return yp


def best_theta(dialogs, embs):
    """ORACLE best-F1 global δ_prev threshold on this dataset."""
    alld, allg = [], []
    for (u, yt), e in zip(dialogs, embs):
        dp = delta_prev_seq(e)
        for i in range(len(e) - 1):
            if dp[i] is not None:
                alld.append(dp[i])
                allg.append(yt[i])  # boundary after turn i
    alld, allg = np.array(alld), np.array(allg)
    best = (0.0, 0.5)
    for th in np.linspace(alld.min(), alld.max(), 120):
        f = f1_score(allg, alld >= th)
        if f > best[0]:
            best = (f, float(th))
    return best[1]


def delta_eff_seq(emb, m, rho, a):
    """v4.1.1 δ_eff = a·δ_prev + (1-a)·δ_ctx, causal (mirrors segmenter)."""
    out = []
    for t in range(1, len(emb)):
        pa, pb = emb[t - 1], emb[t]
        npa, npb = np.linalg.norm(pa), np.linalg.norm(pb)
        d_prev = (None if npa <= 1e-12 or npb <= 1e-12
                  else 1.0 - float(np.dot(pa, pb) / (npa * npb)))
        win = emb[max(0, t - m):t][::-1]
        c = np.zeros_like(emb[t], dtype=np.float64)
        for i, v in enumerate(win):
            c += (rho ** i) * v
        nc, ns = np.linalg.norm(c), np.linalg.norm(emb[t])
        d_ctx = (None if nc <= 1e-12 or ns <= 1e-12
                 else 1.0 - float(np.dot(c, emb[t]) / (nc * ns)))
        if d_prev is None:
            out.append(None)
        elif d_ctx is None:
            out.append(d_prev)
        else:
            out.append(a * d_prev + (1.0 - a) * d_ctx)
    return out


def best_thr_generic(dialogs, embs, sigfn):
    alld, allg = [], []
    for (u, yt), e in zip(dialogs, embs):
        sig = sigfn(e)
        for i in range(len(e) - 1):
            if sig[i] is not None:
                alld.append(sig[i])
                allg.append(yt[i])
    alld, allg = np.array(alld), np.array(allg)
    best = (0.0, 0.5)
    for th in np.linspace(alld.min(), alld.max(), 120):
        f = f1_score(allg, alld >= th)
        if f > best[0]:
            best = (f, float(th))
    return best[1]


def seg_pred(emb, delta_star):
    seg = HiOnTopSegmenterV339(dim=emb.shape[1], alpha=1.0, lmda=10.0,
                            delta_star=delta_star)
    import torch
    torch.manual_seed(0)
    ids = [seg.assign(s.astype(np.float64))[0] for s in emb]
    return [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]


def seg_pred_v411(emb, delta_star):
    seg = HiOnTopSegmenterV411(
        dim=emb.shape[1], alpha=1.0, lmda=10.0, delta_star=delta_star,
        ctx_window=V411_M, ctx_decay=V411_RHO, ctx_blend_a=V411_A)
    import torch
    torch.manual_seed(0)
    ids = [seg.assign(s.astype(np.float64))[0] for s in emb]
    return [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]


def eval_method(name, dialogs, embs, predfn):
    pks, wds = [], []
    gt_all, pr_all = [], []
    t0 = time.perf_counter()
    for (u, yt), e in zip(dialogs, embs):
        yp = predfn(e)
        p, w = official_pk_wd(yt, yp)
        pks.append(p)
        wds.append(w)
        gt_all += yt
        pr_all += yp
    f1 = f1_score(gt_all, pr_all, zero_division=0)
    return {
        "method": name, "pk": float(np.mean(pks)), "wd": float(np.mean(wds)),
        "f1": f1, "n_dial": len(dialogs),
        "pred_rate": float(np.mean(pr_all)), "wall_s": time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["dialseg711", "superseg", "tiage"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default=None)
    ap.add_argument("--calib-dataset", default=None,
                    help="dataset for δ* calibration (no test leakage). "
                         "e.g. superseg")
    ap.add_argument("--calib-split", default="validation",
                    help="split for δ* calibration (default validation)")
    args = ap.parse_args()

    print(f"[1/3] load {args.dataset}/{args.split}")
    dialogs = load_dialogs(args.dataset, args.split)
    nt = sum(len(u) for u, _ in dialogs)
    nb = sum(sum(yt) for _, yt in dialogs)
    print(f"  n_dial={len(dialogs)} n_turn={nt} n_bnd={nb}")

    print("[2/3] encode (cached)")
    embs = encode_cached(args.dataset, args.split, dialogs)
    print(f"  dim={embs[0].shape[1]}")

    print("[3/3] eval")
    theta = best_theta(dialogs, embs)  # δ_prev oracle (prev-cos / v3.3.9)
    theta10 = best_thr_generic(  # δ_eff oracle (v4.1.1 blend)
        dialogs, embs,
        lambda e: delta_eff_seq(e, V411_M, V411_RHO, V411_A))

    # δ* calibrated on a SEPARATE split (no test leakage) — the proper
    # tuned number vs TIAGE-δ* zero-shot and vs test-oracle upper bound.
    cal_rows = []
    if args.calib_dataset:
        cds, csp = args.calib_dataset, args.calib_split
        print(f"  [calib] {cds}/{csp} → δ* (δ_prev & δ_eff)")
        cdia = load_dialogs(cds, csp)
        cemb = encode_cached(cds, csp, cdia)
        cth = best_theta(cdia, cemb)
        cth10 = best_thr_generic(
            cdia, cemb,
            lambda e: delta_eff_seq(e, V411_M, V411_RHO, V411_A))
        print(f"  [calib] δ*_prev={cth:.4f}  δ*_eff={cth10:.4f}")
        cal_rows = [
            eval_method(
                f"v3.3.9 @{cds}-{csp}-δ*={cth:.3f} (calibrated)",
                dialogs, embs, lambda e: seg_pred(e, cth)),
            eval_method(
                f"v4.1.1 @{cds}-{csp}-δ*={cth10:.3f} (calibrated)",
                dialogs, embs, lambda e: seg_pred_v411(e, cth10)),
        ]
    rows = cal_rows + [
        eval_method(f"prevcos @oracleθ={theta:.3f} (similarity ceiling)",
                    dialogs, embs, lambda e: prevcos_pred(e, theta)),
        eval_method(f"v3.3.9 @TIAGE-δ*={TIAGE_DELTA_STAR} (zero-shot)",
                    dialogs, embs, lambda e: seg_pred(e, TIAGE_DELTA_STAR)),
        eval_method(f"v3.3.9 @oracle-δ*={theta:.3f} (upper bound)",
                    dialogs, embs, lambda e: seg_pred(e, theta)),
        eval_method(
            f"v4.1.1 @TIAGE-cfg(m{V411_M},ρ{V411_RHO},a{V411_A},"
            f"δ*{V411_DSTAR}) (zero-shot)",
            dialogs, embs, lambda e: seg_pred_v411(e, V411_DSTAR)),
        eval_method(
            f"v4.1.1 @oracle-δ*={theta10:.3f} (upper bound)",
            dialogs, embs, lambda e: seg_pred_v411(e, theta10)),
    ]
    for r in rows:
        r["score"] = total_score(r["f1"], r["pk"], r["wd"])
        print(f"  {r['method']:54s} Score={r['score']:.3f} F1={r['f1']:.3f}"
              f" Pk={r['pk']:.3f} WD={r['wd']:.3f} "
              f"predr={r['pred_rate']:.3f} ({r['wall_s']:.0f}s)")

    name = args.name or f"2026-05-18_{args.dataset}_{args.split}"
    out = REPO / "outputs" / "experiments" / name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        f"# {args.dataset} {args.split} — official SuperDialseg Pk/WD/F1",
        f"\nn_dial={len(dialogs)} · n_turn={nt} · n_bnd={nb} "
        f"(bnd_rate={nb / nt:.3f})\n",
        "Metric = official SuperDialseg (window='auto', verbatim formula; "
        "literature-comparable). GT = dataset segmentation_label. "
        "**Caveat**: oracle-θ / oracle-δ* are tuned ON this test set = "
        "UPPER BOUNDS, not fair external claims. v3.3.9@TIAGE-δ* = honest "
        "zero-shot transfer (no tuning here).\n",
        "Score = 0.5·F1 + 0.25·(1-Pk) + 0.25·(1-WD) "
        "(official SuperDialseg aggregate).\n",
        "| method | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ | pred_rate | n_dial |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['method']} | {r['score']:.3f} | {r['f1']:.3f} | "
                 f"{r['pk']:.3f} | {r['wd']:.3f} | {r['pred_rate']:.3f} | "
                 f"{r['n_dial']} |")
    L.append(f"\nGT bnd_rate = {nb / nt:.3f} (pred_rate 비교 기준).")
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
