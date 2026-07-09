#!/usr/bin/env python3
"""default Hi-OnTop (= hi_ontop_deneut, de-neut + run-length 적응 β) 의 oracle / deploy
점수를 **스크립트로 직접 재현**한다. AMI(±2 tol, shift0) + DTS 3종(exact, shift-1).

- default 신호/config = `hi_ontop.hi_ontop_deneut.DEFAULTS` (c=1.0, A=2.0, B=1.0, L0=8,
  lam=0.6, g_rho=0.15, rho_min=0.05, R=4, warmup=8).
- **oracle** = gold-reset prototype + raw-EWMA(사용시점만 정규화) de-neut 신호 +
  per-instance(대화/미팅) 최적 임계치 = 신호 천장.
  (raw-EWMA 필수 — hi_ontop_deneut.py L66 note: 매-step 정규화하면 oracle 붕괴.)
- **deploy** = 완전 online `hi_ontop_deneut.segment()` (detected-reset, per-step norm) 그대로 호출.
- 채점 = 공식 채점기: DTS `hi_ontop.dts_scoring`, AMI `hi_ontop.ami_scoring`.
- 정렬: DTS gold=끝-turn → 신호 spike t → 경계 t-1 (shift-1); AMI gold=시작-turn → shift0.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

from sklearn.metrics import f1_score  # noqa: E402

from run_encoder_comparison import load_dialogs, official_pk_wd, CACHE  # noqa: E402
from hi_ontop import ami_scoring as AS  # noqa: E402
from hi_ontop import dts_scoring as DS  # noqa: E402
from hi_ontop import hi_ontop_deneut as HD  # noqa: E402

CFG = HD.DEFAULTS  # default Hi-OnTop config
A, B, L0 = CFG["A"], CFG["B"], CFG["L0"]
LAM, G_RHO, RHO_MIN = CFG["lam"], CFG["g_rho"], CFG["rho_min"]


def _nr(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _nrm_rows(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def deneut_oracle_signal(e):
    """default de-neut 신호 (gold-reset 아님 — 임계만 oracle). 여기선 gold-reset 버전 호출용 래퍼 X.

    raw-EWMA prototype (사용시점만 정규화), 적응 β. gold-reset 은 호출부에서 처리한다.
    """
    raise NotImplementedError


def deneut_signal_goldreset(e, gold):
    """gold-reset prototype + raw-EWMA + 적응 β de-neut 신호 (oracle 신호 천장용).

    hi_ontop_deneut 의 신호식과 동일하되 (1) 경계에서 gold 로 reset, (2) prototype 은
    raw-EWMA 로 누적하고 deneut/cos 계산 시점에만 정규화한다.
    """
    n = len(e); gs = set(gold); s = np.zeros(n)
    m = e[0].copy(); k = 1; g = e[0].copy()
    for t in range(1, n):
        x = e[t]
        beta = min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)
        ghat = _nr(g)
        mc = m - beta * float(m @ ghat) * ghat; mc = _nr(mc)
        xc = x - beta * float(x @ ghat) * ghat; xc = _nr(xc)
        V = (1 - float(xc @ mc)) - LAM * (1 - float(x @ ghat))
        s[t] = V
        if t in gs:
            m = x.copy(); k = 1
        else:
            rho = max(RHO_MIN, 1.0 / (k + 1)); m = (1 - rho) * m + rho * x; k += 1
        g = (1 - G_RHO) * g + G_RHO * x  # raw-EWMA, 사용시점만 정규화
    return s


# ---------------- DTS (exact F1, shift -1) ----------------

def load_dts(ds):
    dl = load_dialogs(ds, "test")
    emb = [_nrm_rows(np.asarray(x, dtype=np.float64))
           for x in pickle.load(open(CACHE / f"enccmp_{ds}_test_minilm-int8.pkl", "rb"))]
    golds = [yt for (_u, yt) in dl]
    return emb, golds


def dts_oracle(emb, golds):
    """per-dialogue 최적 임계 (gold-reset 신호). F1·Score 각각의 천장을 따로 구한다."""
    bestF1, bestSC = [], []
    for e, yt in zip(emb, golds):
        n = len(yt); gold = [i for i, b in enumerate(yt) if b == 1]
        s = deneut_signal_goldreset(e, gold)
        cand = sorted(set(float(s[i]) for i in range(1, n)))
        if len(cand) > 80:
            cand = list(np.quantile(cand, np.linspace(0, 1, 80)))
        bf, bs = 0.0, -1.0
        for thr in cand:
            yp = DS.signal_to_pred(s, thr)  # shift -1, len n, last 0
            ytl = list(yt); ytl[-1] = 0
            f1 = f1_score(ytl, yp, zero_division=0)
            pk, wd = official_pk_wd(ytl, yp)
            sc = 0.5 * f1 + 0.25 * (1 - pk) + 0.25 * (1 - wd)
            if f1 > bf:
                bf = f1
            if sc > bs:
                bs = sc
        bestF1.append(bf); bestSC.append(bs)
    return float(np.mean(bestF1)), float(np.mean(bestSC))


def dts_deploy(emb, golds):
    """완전 online hi_ontop_deneut.segment() (default). 공식 dts_scoring 으로 채점."""
    preds = []
    for e, yt in zip(emb, golds):
        n = len(yt)
        spikes = HD.segment(e)                 # 신호 spike turn (새 segment 첫 turn)
        bset = {p - 1 for p in spikes if 1 <= p <= n - 1}  # shift -1 (끝-turn)
        yp = [1 if i in bset else 0 for i in range(n)]
        preds.append(yp)
    return DS.score_dialogues(golds, preds)


# ---------------- AMI (±2 tol F1, shift 0) ----------------

def load_ami():
    TOPIC = os.path.join(REPO, "data/ami/topic"); EMB = os.path.join(REPO, "outputs/runs/_misc/ami_emb")
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    out = []
    for mid in mids:
        bt = list(json.load(open(f"{TOPIC}/{mid}.json"))["bnd_top"]); bt[-1] = 0
        e = _nrm_rows(np.asarray(pickle.load(open(f"{EMB}/{mid}.pkl", "rb")), dtype=np.float64))
        out.append((e, bt, [i for i, b in enumerate(bt) if b == 1]))
    return out


def ami_oracle(data):
    """per-meeting 최적 임계 (gold-reset, ±2 tol). F1·Score 천장 각각."""
    bestF1, bestSC = [], []
    for e, bt, gold in data:
        n = len(e); s = deneut_signal_goldreset(e, gold)
        cand = sorted(set(float(s[i]) for i in range(1, n - 1)))
        if len(cand) > 100:
            cand = list(np.quantile(cand, np.linspace(0, 1, 100)))
        bf, bs = 0.0, -1.0
        for thr in cand:
            pred = [i for i in range(1, n - 1) if s[i] > thr]
            f1 = AS.tol_f1(gold, pred)
            yp = AS.boundaries_to_pred(n, pred)
            r = AS.score_meetings([AS.boundaries_to_pred(n, gold)], [yp])
            if f1 > bf:
                bf = f1
            if r["score"] > bs:
                bs = r["score"]
        bestF1.append(bf); bestSC.append(bs)
    return float(np.mean(bestF1)), float(np.mean(bestSC))


def ami_deploy(data):
    golds = [AS.boundaries_to_pred(len(bt), gold) for _e, bt, gold in data]
    preds = []
    for e, bt, _g in data:
        spikes = [p for p in HD.segment(e) if 0 < p < len(bt) - 1]  # shift 0 (시작-turn)
        preds.append(AS.boundaries_to_pred(len(bt), spikes))
    return AS.score_meetings(golds, preds)


if __name__ == "__main__":
    print(f"default Hi-OnTop = hi_ontop_deneut, config={CFG}")
    print("=" * 78)

    rows = []
    # DTS 3종 (exact F1, shift -1)
    for ds in ("tiage", "dialseg711", "superseg"):
        emb, golds = load_dts(ds)
        of1, osc = dts_oracle(emb, golds)
        dp = dts_deploy(emb, golds)
        rows.append((ds, "exact", len(golds), of1, osc, dp))
        print(f"[{ds}] n_dialog={len(golds)}  oracle F1={of1:.3f} Score={osc:.3f}  | "
              f"deploy F1={dp['f1']:.3f} Pk={dp['pk']:.3f} WD={dp['wd']:.3f} Score={dp['score']:.3f}",
              flush=True)

    # AMI (±2 tol F1, shift 0)
    ami = load_ami()
    of1, osc = ami_oracle(ami)
    dp = ami_deploy(ami)
    rows.append(("AMI", "±2tol", len(ami), of1, osc, dp))
    print(f"[AMI] n_meeting={len(ami)}  oracle F1={of1:.3f} Score={osc:.3f}  | "
          f"deploy F1={dp['f1']:.3f} Pk={dp['pk']:.3f} WD={dp['wd']:.3f} Score={dp['score']:.3f}",
          flush=True)

    print("=" * 78)
    print(f"{'dataset':<12}{'metric':<8}{'oracleF1':>9}{'oracleScore':>12}{'deployF1':>10}{'deployScore':>12}")
    for ds, met, nn, of1, osc, dp in rows:
        print(f"{ds:<12}{met:<8}{of1:>9.3f}{osc:>12.3f}{dp['f1']:>10.3f}{dp['score']:>12.3f}")
