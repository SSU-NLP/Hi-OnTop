#!/usr/bin/env python3
"""reset 부트스트랩 벽이 인코더-의존인가 — 임의 임베딩으로 재측정.

HANDOFF_01 의 정보-한계 결론(0-lag 판별자 AUC 0.41, deploy ±2F1 ~0.13, oracle 0.687)은
**MiniLM-int8 하나로만** 측정됐다. 더 강한 인코더로 같은 4지표를 재서, 벽이 인코더-독립인지
(약한 int8 아티팩트인지) 가른다.

측정 (모두 AMI 139, 확정 채점기 ami_scoring, shift0, ±2 tol):
  1) oracle 천장 — gold-reset + per-meeting best threshold. de-neut(default 신호) + V_rel(β=0).
  2) deploy — hi_ontop_deneut.segment() c=1.0 / 1.5.
  3) 판별자 AUC(onset>outlier) — 0-lag frozen old prototype 거리 + 1-lag W=2 응집.

사용: python scripts/probe_encoder_wall.py <emb_subdir>   (예: ami_emb / ami_emb_mpnet)
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
sys.path.insert(0, os.path.join(REPO, "src"))
from hi_ontop import ami_scoring as AS  # noqa: E402
from hi_ontop import hi_ontop_deneut as HD  # noqa: E402

LAM, G_RHO, RHO_MIN, WARMUP = 0.6, 0.15, 0.05, 8
A, B, L0 = HD.DEFAULTS["A"], HD.DEFAULTS["B"], HD.DEFAULTS["L0"]


def _nr(v):
    return v / (np.linalg.norm(v) + 1e-12)


def load_ami(sub):
    TOPIC = os.path.join(REPO, "data/ami/topic")
    EMB = os.path.join(REPO, "outputs/runs/_misc", sub)
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    out = []
    for mid in mids:
        bt = list(json.load(open(f"{TOPIC}/{mid}.json"))["bnd_top"]); bt[-1] = 0
        e = np.asarray(pickle.load(open(f"{EMB}/{mid}.pkl", "rb")), dtype=np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        out.append((e, bt, [i for i, b in enumerate(bt) if b == 1]))
    return out


# ---- oracle 신호 (gold-reset, raw-EWMA) ----
def sig_deneut(e, gold):
    n = len(e); gs = set(gold); s = np.zeros(n)
    m = e[0].copy(); k = 1; g = e[0].copy()
    for t in range(1, n):
        x = e[t]; beta = min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)
        gh = _nr(g)
        mc = _nr(m - beta * float(m @ gh) * gh); xc = _nr(x - beta * float(x @ gh) * gh)
        s[t] = (1 - float(xc @ mc)) - LAM * (1 - float(x @ gh))
        if t in gs:
            m = x.copy(); k = 1
        else:
            rho = max(RHO_MIN, 1.0 / (k + 1)); m = (1 - rho) * m + rho * x; k += 1
        g = (1 - G_RHO) * g + G_RHO * x
    return s


def sig_vrel(e, gold):
    n = len(e); gs = set(gold); s = np.zeros(n)
    m = e[0].copy(); k = 1; g = e[0].copy()
    for t in range(1, n):
        x = e[t]
        s[t] = (1 - float(x @ _nr(m))) - LAM * (1 - float(x @ _nr(g)))
        if t in gs:
            m = x.copy(); k = 1
        else:
            rho = max(RHO_MIN, 1.0 / (k + 1)); m = (1 - rho) * m + rho * x; k += 1
        g = (1 - G_RHO) * g + G_RHO * x
    return s


def oracle_f1(data, sigfn):
    F = []
    for e, _bt, gold in data:
        n = len(e); s = sigfn(e, gold)
        cand = sorted(set(float(s[i]) for i in range(1, n - 1)))
        if len(cand) > 100:
            cand = list(np.quantile(cand, np.linspace(0, 1, 100)))
        best = max((AS.tol_f1(gold, [i for i in range(1, n - 1) if s[i] > thr])
                    for thr in cand), default=0.0)
        F.append(best)
    return float(np.mean(F))


# ---- deploy ----
def deploy(data, c):
    golds = [AS.boundaries_to_pred(len(bt), g) for _e, bt, g in data]
    preds = [AS.boundaries_to_pred(len(bt), [p for p in HD.segment(e, c=c) if 0 < p < len(bt) - 1])
             for e, bt, _g in data]
    return AS.score_meetings(golds, preds)


# ---- 판별자 AUC (diag2 복제) ----
def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos = np.asarray(pos); neg = np.asarray(neg)
    al = np.concatenate([pos, neg]); rk = al.argsort().argsort() + 1
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def discriminator_auc(data, c_L=1.5):
    rows = []
    for e, _bt, gold in data:
        gs = gold; n = len(e); a = e[0].copy(); k = 1; g = e[0].copy(); gk = 1
        Wm = WM2 = 0.0; Wn = 0; t = 1
        while t < n:
            x = e[t]; gr = max(G_RHO, 1 / (gk + 1)); g = (1 - gr) * g + gr * x; gk += 1
            V = (1 - float(x @ _nr(a))) - LAM * (1 - float(x @ _nr(g)))
            sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
            L = (Wm + c_L * sd) if Wn >= WARMUP else None
            if L is not None and V > L:
                rec = {"true": any(abs(t - j) <= 2 for j in gs), "rold": 1 - float(x @ _nr(a))}
                for W in (2, 3, 4):
                    seg = e[t:t + W]
                    rec[W] = (float((seg @ seg.T).sum() - len(seg)) / (len(seg) * (len(seg) - 1))) \
                        if len(seg) > 1 else 0.0
                rows.append(rec); a = x.copy(); k = 1; t += 1; continue
            Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
            rho = max(RHO_MIN, 1 / (k + 1)); a = (1 - rho) * a + rho * x; k += 1; t += 1
    tt = [r for r in rows if r["true"]]; ff = [r for r in rows if not r["true"]]
    res = {"n": len(rows), "true_rate": len(tt) / max(len(rows), 1),
           "auc_0lag": _auc([r["rold"] for r in tt], [r["rold"] for r in ff])}
    for W in (2, 3, 4):
        res[f"auc_W{W}"] = _auc([r[W] for r in tt], [r[W] for r in ff])
    return res


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "ami_emb"
    data = load_ami(sub)
    print(f"=== probe_encoder_wall  emb={sub}  AMI {len(data)}미팅, dim={data[0][0].shape[1]} ===")
    print(f"[oracle] de-neut(default) ±2F1 = {oracle_f1(data, sig_deneut):.3f}")
    print(f"[oracle] V_rel(β=0)       ±2F1 = {oracle_f1(data, sig_vrel):.3f}")
    for c in (1.0, 1.5):
        r = deploy(data, c)
        print(f"[deploy] c={c}: ±2F1={r['f1']:.3f} Pk={r['pk']:.3f} WD={r['wd']:.3f} Score={r['score']:.3f}")
    d = discriminator_auc(data)
    print(f"[discrim] provisional={d['n']} true_rate={d['true_rate']:.1%}  "
          f"0-lag AUC={d['auc_0lag']:.3f}  W2={d['auc_W2']:.3f} W3={d['auc_W3']:.3f} W4={d['auc_W4']:.3f}")
    print("  (MiniLM-int8 기준: oracle de-neut 0.372 / V_rel 0.69 / deploy c1.0 ±2F1 0.131 / 0-lag AUC 0.409 / W2 0.665)")
