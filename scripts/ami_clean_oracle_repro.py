"""AMI clean-reset oracle 재현 + prototype 정규화 버그 대조 (2026-06-15).

REPORT(2026-06-10_ami_vrel_localmap) §3·§4 의 clean-oracle 수치를 확정 채점기
(`hi_ontop.ami_scoring`)로 재현하고, 세션 중 발생한 재구현 버그(EWMA prototype 을 매 step
정규화)가 oracle 을 어떻게 깎는지 대조한다.

확인 사항:
  - raw r_active (clean) ≈ 0.488,  V_rel per-meeting oracle ≈ 0.687,
    clean + 단순 μ+cσ(c=2.0) ≈ 0.55~0.63  (전부 REPORT 재현).
  - prototype 매-step 정규화(`--norm-proto`) 시 V_rel oracle 이 ~0.26 으로 붕괴 (버그).
  - deploy(detected-reset)는 raw/norm 무관하게 ±2F1 0.14~0.16 → 병목은 신호가 아니라
    online clean-reset 부트스트랩(HANDOFF_01).

신호: r_active = 1 − cos(x, m̂),  V_rel = r_active − λ·(1 − cos(x, ĝ)),
      m = active prototype(경계서 reset, EWMA rho=max(rho_min,1/(k+1))),
      g = global EWMA(g_rho=0.15).  m̂/ĝ = 사용 시점 정규화.  λ=0.6.  단위정규화 임베딩.
정렬: AMI gold = bnd_top(start-turn), shift 0.  oracle = per-meeting best threshold(±2F1).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hi_ontop import ami_scoring as AS  # noqa: E402

TOPIC = "data/ami/topic"
EMB = "outputs/runs/_misc/ami_emb"
LAM, G_RHO, RHO_MIN = 0.6, 0.15, 0.05


def _nrm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def load_ami() -> list[tuple[np.ndarray, list[int], list[int]]]:
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    out = []
    for m in mids:
        bt = list(json.load(open(f"{TOPIC}/{m}.json"))["bnd_top"]); bt[-1] = 0
        e = _nrm(np.asarray(pickle.load(open(f"{EMB}/{m}.pkl", "rb")), dtype=np.float64))
        out.append((e, bt, [i for i, b in enumerate(bt) if b == 1]))
    return out


def clean_signal(e: np.ndarray, gold: list[int], kind: str, norm_proto: bool) -> np.ndarray:
    """gold-reset prototype 으로 신호. norm_proto=True 면 매 step 정규화(버그 재현용)."""
    n = len(e); gs = set(gold); s = np.zeros(n)
    m = e[0].copy(); k = 1; g = e[0].copy(); gk = 1
    for t in range(1, n):
        x = e[t]
        r_active = 1 - float(x @ (m / (np.linalg.norm(m) + 1e-12)))
        r_global = 1 - float(x @ (g / (np.linalg.norm(g) + 1e-12)))
        s[t] = r_active if kind == "ractive" else r_active - LAM * r_global
        if t in gs:
            m = x.copy(); k = 1
        else:
            rho = max(RHO_MIN, 1.0 / (k + 1)); m = (1 - rho) * m + rho * x
            if norm_proto:
                m = _nrm(m)
            k += 1
        gr = max(G_RHO, 1.0 / (gk + 1)); g = (1 - gr) * g + gr * x
        if norm_proto:
            g = _nrm(g)
        gk += 1
    return s


def deploy_signal(e: np.ndarray, norm_proto: bool, c: float = 1.5,
                  R: int = 4, warmup: int = 8) -> list[int]:
    """detected-reset(online) prototype + μ+cσ 경계 예측."""
    n = len(e); m = e[0].copy(); k = 1; g = e[0].copy(); gk = 1
    Wm = WM2 = 0.0; Wn = 0; pred = []; last = -999
    for t in range(1, n):
        x = e[t]
        v = (1 - float(x @ (m / (np.linalg.norm(m) + 1e-12)))) \
            - LAM * (1 - float(x @ (g / (np.linalg.norm(g) + 1e-12))))
        sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
        thr = (Wm + c * sd) if Wn >= warmup else None
        if thr is not None and v > thr and k >= R and t - last >= R:
            pred.append(t); m = x.copy(); k = 1; last = t
        else:
            Wn += 1; dd = v - Wm; Wm += dd / Wn; WM2 += dd * (v - Wm)
            rho = max(RHO_MIN, 1.0 / (k + 1)); m = (1 - rho) * m + rho * x
            if norm_proto:
                m = _nrm(m)
            k += 1
        gr = max(G_RHO, 1.0 / (gk + 1)); g = (1 - gr) * g + gr * x
        if norm_proto:
            g = _nrm(g)
        gk += 1
    return pred


def oracle_f1(data, kind, norm_proto, tol=2) -> float:
    F = []
    for e, _bt, gold in data:
        s = clean_signal(e, gold, kind, norm_proto); n = len(e)
        cand = sorted(set(float(s[i]) for i in range(1, n - 1)))
        if len(cand) > 100:
            cand = list(np.quantile(cand, np.linspace(0, 1, 100)))
        best = max((AS.tol_f1(gold, [i for i in range(1, n - 1) if s[i] > thr], tol)
                    for thr in cand), default=0.0)
        F.append(best)
    return float(np.mean(F))


def musigma_f1(data, norm_proto, c) -> dict:
    golds, preds = [], []
    for e, bt, gold in data:
        s = clean_signal(e, gold, "vrel", norm_proto); n = len(s)
        Wm = WM2 = 0.0; Wn = 0; pr = []
        for t in range(1, n - 1):
            v = s[t]; sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
            if Wn >= 8 and v > Wm + c * sd:
                pr.append(t)
            Wn += 1; dd = v - Wm; Wm += dd / Wn; WM2 += dd * (v - Wm)
        golds.append(AS.boundaries_to_pred(len(bt), gold))
        preds.append(AS.boundaries_to_pred(len(bt), pr))
    return AS.score_meetings(golds, preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-proto", action="store_true",
                    help="EWMA prototype 매-step 정규화 (재구현 버그 재현)")
    args = ap.parse_args()
    data = load_ami()
    tag = "per-step-norm(버그)" if args.norm_proto else "raw-EWMA(정상)"
    print(f"AMI {len(data)}미팅, bnd_top, 확정 채점기, prototype={tag}")
    print(f"REPORT(2026-06-10): raw r_active 0.488 / V_rel oracle 0.687 / clean+μcσ(c2.0) 0.554")
    print(f"  raw r_active oracle ±2F1 = {oracle_f1(data, 'ractive', args.norm_proto):.3f}")
    print(f"  V_rel       oracle ±2F1 = {oracle_f1(data, 'vrel', args.norm_proto):.3f}")
    for c in (1.5, 2.0):
        r = musigma_f1(data, args.norm_proto, c)
        print(f"  clean + 단순 μ+cσ (c={c}): ±2F1={r['f1']:.3f} Score={r['score']:.3f}")
    print("  --- deploy(detected-reset) ---")
    for c in (1.0, 1.5, 2.0):
        golds = [AS.boundaries_to_pred(len(bt), gold) for _e, bt, gold in data]
        preds = [AS.boundaries_to_pred(len(bt),
                 [p for p in deploy_signal(e, args.norm_proto, c=c) if 0 < p < len(bt) - 1])
                 for e, bt, _g in data]
        r = AS.score_meetings(golds, preds)
        print(f"  deploy c={c}: ±2F1={r['f1']:.3f} Score={r['score']:.3f}")


if __name__ == "__main__":
    main()
