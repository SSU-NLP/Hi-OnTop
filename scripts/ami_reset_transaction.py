"""Reset-as-transaction deploy (codex 설계, 2026-06-15) — AMI 실측.

reset 부트스트랩 공략: detected-reset 의 불가역 hard-reset 을 **가역 split transaction** 으로 교체.
NORMAL 에서 V_t>L 이면 τ 에 provisional 경계를 열고(OPEN_SPLIT), 이후 발화로 old/new dual prototype
(a_old/b)에 대한 응집 증거 S 를 SPRT bound B 로 누적 → confirm(a←b) 또는 cancel(a←a_old replay-merge).
τ 는 고정(위치 정정 없음 → commit-refine 의 lag-이득과 다름). 경계 위치는 0-lag, confirm/cancel 은
prototype 오염만 차단.

신호: V_t=(1−cos(x,a))−λ(1−cos(x,g)), a=active EWMA(reset 가능), g=global EWMA(g_rho). 단위정규화 임베딩.
채점: 확정 채점기 `hi_ontop.ami_scoring`, bnd_top(start-turn), shift 0, ±2 tolerant F1.

clean-reset 천장(같은 신호): per-meeting oracle ±2F1 0.69, 단순 μ+cσ 0.55~0.63.
기존 hard-reset deploy: ±2F1 ~0.15.  ← 이걸 넘는지가 목표.
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


def _nrm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def load_ami():
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    out = []
    for m in mids:
        bt = list(json.load(open(f"{TOPIC}/{m}.json"))["bnd_top"]); bt[-1] = 0
        e = _nrm(np.asarray(pickle.load(open(f"{EMB}/{m}.pkl", "rb")), dtype=np.float64))
        out.append((e, bt, [i for i, b in enumerate(bt) if b == 1]))
    return out


def segment_transaction(e, *, lam=0.6, g_rho=0.15, rho_min=0.05, warmup=8,
                        c_L=2.0, alpha_b=0.30, alpha_a=0.15, n_max=30, m_min=4,
                        return_confirmed=True):
    """가역 split transaction deploy. confirmed 경계 리스트(+ confirm 지연) 반환."""
    n = len(e)
    a = e[0].copy(); k = 1                     # active prototype (NORMAL)
    g = e[0].copy(); gk = 1
    Wm = WM2 = 0.0; Wn = 0                      # online V stats (warmup, μ+cσ)
    state = "NORMAL"
    confirmed = []; provisional = []; delays = []
    # OPEN_SPLIT 상태 변수
    tau = a_old = b = a_shadow = None; buffer = None; S = 0.0
    dm = dM2 = 0.0; dn_ = 0                     # online std of d (scale)

    def au(): return a / (np.linalg.norm(a) + 1e-12)
    def gu(): return g / (np.linalg.norm(g) + 1e-12)

    for t in range(1, n):
        x = e[t]
        gr = max(g_rho, 1.0 / (gk + 1)); g = (1 - gr) * g + gr * x; gk += 1

        if state == "NORMAL":
            V = (1 - float(x @ au())) - lam * (1 - float(x @ gu()))
            sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
            L = (Wm + c_L * sd) if Wn >= warmup else None
            if L is not None and V > L:
                provisional.append(t)
                tau = t; a_old = a.copy(); a_shadow = a.copy(); b = x.copy()
                buffer = [x.copy()]; S = 0.0; dm = dM2 = 0.0; dn_ = 0
                state = "OPEN_SPLIT"
            else:
                Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
                rho = max(rho_min, 1.0 / (k + 1)); a = (1 - rho) * a + rho * x; k += 1

        else:  # OPEN_SPLIT
            buffer.append(x.copy())
            bu = b / (np.linalg.norm(b) + 1e-12)
            su = a_shadow / (np.linalg.norm(a_shadow) + 1e-12)
            d = float(x @ bu) - float(x @ su)          # new vs old 정렬차
            dn_ += 1; ddd = d - dm; dm += ddd / dn_; dM2 += ddd * (d - dm)
            s = math.sqrt(dM2 / (dn_ - 1)) if dn_ > 1 else 1e-6
            s = max(s, 1e-6)
            z = max(-3.0, min(3.0, d / s)); q = 1.0 / (1.0 + math.exp(-z))
            b = _nrm((1 - alpha_b * q) * b + alpha_b * q * x)
            a_shadow = _nrm((1 - alpha_a * (1 - q)) * a_shadow + alpha_a * (1 - q) * x)
            S += z
            m = len(buffer)
            Cb = float(np.mean([bb @ bu for bb in buffer]))
            Ca = float(np.mean([bb @ (a_old / (np.linalg.norm(a_old) + 1e-12)) for bb in buffer]))
            G = Cb - Ca
            B = math.sqrt(2 * m * math.log(math.log2(m + 1) + 1)) if m > 1 else 1e9

            if m >= m_min and S > B and G > 0:         # CONFIRM (최소 증거창 후)
                confirmed.append(tau); delays.append(t - tau)
                a = b.copy(); k = m; state = "NORMAL"
            elif (m >= m_min and S < -B) or (m >= n_max and S <= 0):    # CANCEL → merge
                a = a_old.copy(); kk = k
                for z2 in buffer:
                    rho = max(rho_min, 1.0 / (kk + 1)); a = (1 - rho) * a + rho * z2; kk += 1
                k = kk; state = "NORMAL"
                # 취소된 buffer 구간의 V 도 통계에 반영(근사)
            # else: OPEN_SPLIT 유지

    return (confirmed if return_confirmed else provisional), delays, provisional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["confirmed", "provisional"], default="confirmed")
    ap.add_argument("--c", type=float, default=2.0, help="c_L (provisional 임계)")
    ap.add_argument("--nmax", type=int, default=30)
    ap.add_argument("--m-min", type=int, default=4)
    ap.add_argument("--alpha-b", type=float, default=0.30)
    ap.add_argument("--alpha-a", type=float, default=0.15)
    args = ap.parse_args()
    data = load_ami()
    golds, preds = [], []; all_delays = []; nprov = nconf = 0
    for e, bt, gold in data:
        conf, delays, prov = segment_transaction(
            e, c_L=args.c, n_max=args.nmax, m_min=args.m_min, alpha_b=args.alpha_b, alpha_a=args.alpha_a,
            return_confirmed=(args.mode == "confirmed"))
        all_delays += delays; nprov += len(prov); nconf += len(conf)
        golds.append(AS.boundaries_to_pred(len(bt), gold))
        preds.append(AS.boundaries_to_pred(len(bt), [p for p in conf if 0 < p < len(bt) - 1]))
    r = AS.score_meetings(golds, preds)
    ng = sum(len(g) for _, _, g in data)
    md = float(np.median(all_delays)) if all_delays else 0.0
    mx = max(all_delays) if all_delays else 0
    print(f"transaction deploy [{args.mode}] c_L={args.c} n_max={args.nmax} "
          f"α_b={args.alpha_b} α_a={args.alpha_a}")
    print(f"  gold={ng}  provisional={nprov}  confirmed={nconf}  (cancel률 {1-nconf/max(nprov,1):.2f})")
    print(f"  confirm 지연: median {md:.0f} turn, max {mx} (위치 lag 아님 — τ 고정; 결정 지연)")
    print(f"  ±2F1={r['f1']:.3f}  Pk={r['pk']:.3f}  WD={r['wd']:.3f}  Score={r['score']:.3f}")
    print(f"  (기존 hard-reset 0.15 / clean 천장 0.55~0.69 대비)")


if __name__ == "__main__":
    main()
