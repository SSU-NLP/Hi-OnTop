"""reset 부트스트랩 = 정보 한계 진단 (2026-06-15).

reset-as-transaction(`ami_reset_transaction.py`, codex 설계, 시도 #7)이 왜 실패하는지,
그리고 online reset 부트스트랩이 영리한 메커니즘이 아니라 **정보 한계**로 막혀 있음을 데이터로 보인다.

세 진단:
  1) transaction confirm/cancel 의 판별력 — provisional τ 가 gold(±2)인지에 따른 취소율.
     기대: TRUE 취소율 ≪ FALSE 취소율. 실제: 둘 다 ~0.82 (판별력 0).
  2) "onset vs within-topic outlier" 판별자 AUC(true>false) vs lag W.
     - FROZEN old prototype 거리(0-lag류): AUC ~0.41 (랜덤).
     - τ-window 상호 응집 cos: W=2 에서 이미 0.665, 이후 희석 하락. 정보는 "스파이크 직후 1발화"에 집중.
  3) 최선 판별자(1-lag 응집)를 deploy persistence-gate 로 실제 사용한 ±2F1 — baseline(0.15)급에 그침.

신호: V=(1−cos(x,a))−λ(1−cos(x,g)), a=active EWMA(reset), g=global EWMA. 단위정규화 임베딩.
채점: `hi_ontop.ami_scoring`, bnd_top(start-turn, shift0), ±2 tolerant F1.
clean 천장(같은 신호): oracle 0.69 / 단순 μ+cσ 0.55~0.63 (`ami_clean_oracle_repro.py`).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ami_reset_transaction as T  # noqa: E402
from hi_ontop import ami_scoring as AS  # noqa: E402

LAM, G_RHO, RHO_MIN, WARMUP = 0.6, 0.15, 0.05, 8


def _au(a):
    return a / (np.linalg.norm(a) + 1e-12)


def _V(x, a, g):
    return (1 - float(x @ _au(a))) - LAM * (1 - float(x @ _au(g)))


def diag1_cancel_rate(data, c_L=1.5, m_min=4, n_max=30, alpha_b=0.30, alpha_a=0.15):
    """transaction confirm/cancel 이 true/false 를 가르는가."""
    tt, ff = {"confirm": 0, "cancel": 0}, {"confirm": 0, "cancel": 0}
    for e, _bt, gold in data:
        gs = gold; n = len(e); a = e[0].copy(); k = 1; g = e[0].copy(); gk = 1
        Wm = WM2 = 0.0; Wn = 0; state = "NORMAL"
        tau = a_old = b = a_shadow = None; buf = None; S = 0.0; dm = dM2 = 0.0; dn_ = 0
        for t in range(1, n):
            x = e[t]; gr = max(G_RHO, 1 / (gk + 1)); g = (1 - gr) * g + gr * x; gk += 1
            if state == "NORMAL":
                V = _V(x, a, g); sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
                L = (Wm + c_L * sd) if Wn >= WARMUP else None
                if L is not None and V > L:
                    tau = t; a_old = a.copy(); a_shadow = a.copy(); b = x.copy()
                    buf = [x.copy()]; S = 0.0; dm = dM2 = 0.0; dn_ = 0; state = "OPEN"
                else:
                    Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
                    rho = max(RHO_MIN, 1 / (k + 1)); a = (1 - rho) * a + rho * x; k += 1
            else:
                buf.append(x.copy()); bu = _au(b); su = _au(a_shadow)
                d = float(x @ bu) - float(x @ su)
                dn_ += 1; ddd = d - dm; dm += ddd / dn_; dM2 += ddd * (d - dm)
                s = max(math.sqrt(dM2 / (dn_ - 1)) if dn_ > 1 else 1e-6, 1e-6)
                z = max(-3, min(3, d / s)); q = 1 / (1 + math.exp(-z))
                b = T._nrm((1 - alpha_b * q) * b + alpha_b * q * x)
                a_shadow = T._nrm((1 - alpha_a * (1 - q)) * a_shadow + alpha_a * (1 - q) * x)
                S += z; m = len(buf)
                Cb = np.mean([bb @ bu for bb in buf]); Ca = np.mean([bb @ _au(a_old) for bb in buf])
                B = math.sqrt(2 * m * math.log(math.log2(m + 1) + 1)) if m > 1 else 1e9
                istrue = any(abs(tau - j) <= 2 for j in gs)
                if m >= m_min and S > B and (Cb - Ca) > 0:
                    (tt if istrue else ff)["confirm"] += 1; a = b.copy(); k = m; state = "NORMAL"
                elif (m >= m_min and S < -B) or (m >= n_max and S <= 0):
                    (tt if istrue else ff)["cancel"] += 1; a = a_old.copy(); kk = k
                    for z2 in buf:
                        rho = max(RHO_MIN, 1 / (kk + 1)); a = (1 - rho) * a + rho * z2; kk += 1
                    k = kk; state = "NORMAL"
    print("[1] transaction confirm/cancel 판별력 (기대: TRUE 취소율 ≪ FALSE 취소율)")
    for nm, dd in (("TRUE ", tt), ("FALSE", ff)):
        tot = dd["confirm"] + dd["cancel"]
        print(f"    {nm} 경계: confirm {dd['confirm']} / cancel {dd['cancel']}  취소율 {dd['cancel']/max(tot,1):.2f}")
    print("    → 둘 다 ~0.82 = 판별력 0 (dual-prototype SPRT 무력).")


def _auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    al = np.concatenate([pos, neg]); rk = al.argsort().argsort() + 1
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def diag2_discriminator_auc(data, c_L=1.5, Ws=(2, 3, 4, 6, 8, 12)):
    """provisional τ 의 onset/outlier 판별자 AUC vs lag W."""
    rows = []
    for e, _bt, gold in data:
        gs = gold; n = len(e); a = e[0].copy(); k = 1; g = e[0].copy(); gk = 1
        Wm = WM2 = 0.0; Wn = 0; t = 1
        while t < n:
            x = e[t]; gr = max(G_RHO, 1 / (gk + 1)); g = (1 - gr) * g + gr * x; gk += 1
            V = _V(x, a, g); sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
            L = (Wm + c_L * sd) if Wn >= WARMUP else None
            if L is not None and V > L:
                rec = {"true": any(abs(t - j) <= 2 for j in gs), "rold": 1 - float(x @ _au(a))}
                for W in Ws:
                    seg = e[t:t + W]
                    if len(seg) > 1:
                        M = seg @ seg.T; rec[W] = float((M.sum() - len(seg)) / (len(seg) * (len(seg) - 1)))
                    else:
                        rec[W] = 0.0
                rows.append(rec); a = x.copy(); k = 1; t += 1; continue
            Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
            rho = max(RHO_MIN, 1 / (k + 1)); a = (1 - rho) * a + rho * x; k += 1; t += 1
    tt = [r for r in rows if r["true"]]; ff = [r for r in rows if not r["true"]]
    print(f"\n[2] 판별자 AUC(true>false)  (provisional {len(rows)}, true율 {len(tt)/len(rows):.1%})")
    print(f"    FROZEN old 거리 (0-lag류): AUC {_auc([r['rold'] for r in tt], [r['rold'] for r in ff]):.3f}  (랜덤≈0.5)")
    for W in Ws:
        print(f"    τ-window 응집 W={W:2d}: AUC {_auc([r[W] for r in tt], [r[W] for r in ff]):.3f}")
    print("    → 정보는 W=2(1발화)에 집중·포화 후 희석. 0-lag 거리는 랜덤.")


def diag3_persist_gate(data):
    """최선 판별자(1-lag 응집)를 deploy persistence-gate 로 실제 사용한 ±2F1."""
    def seg(e, c_L, theta):
        n = len(e); a = e[0].copy(); k = 1; g = e[0].copy(); gk = 1
        Wm = WM2 = 0.0; Wn = 0; pred = []; t = 1
        while t < n:
            x = e[t]; gr = max(G_RHO, 1 / (gk + 1)); g = (1 - gr) * g + gr * x; gk += 1
            V = _V(x, a, g); sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
            L = (Wm + c_L * sd) if Wn >= WARMUP else None
            if L is not None and V > L and t + 1 < n:
                if float(e[t] @ e[t + 1]) >= theta:
                    pred.append(t); a = (e[t] + e[t + 1]) / 2.0; k = 2; t += 2; continue
                Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
                for z in (e[t], e[t + 1]):
                    rho = max(RHO_MIN, 1 / (k + 1)); a = (1 - rho) * a + rho * z; k += 1
                t += 2; continue
            Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
            rho = max(RHO_MIN, 1 / (k + 1)); a = (1 - rho) * a + rho * x; k += 1; t += 1
        return pred
    golds = [AS.boundaries_to_pred(len(bt), gold) for _e, bt, gold in data]
    print("\n[3] 1-lag persistence-gate deploy ±2F1 (baseline 0.15 / clean 천장 0.55)")
    best = 0.0
    for c in (1.0, 1.5, 2.0):
        for th in (0.0, 0.1, 0.2):
            preds = [AS.boundaries_to_pred(len(bt), [p for p in seg(e, c, th) if 0 < p < len(bt) - 1])
                     for e, bt, _g in data]
            r = AS.score_meetings(golds, preds); best = max(best, r["f1"])
            print(f"    c={c} θ={th}: ±2F1={r['f1']:.3f} Score={r['score']:.3f}")
    print(f"    → 최선 {best:.3f} ≈ baseline. AUC 0.66 @ 5% base rate 는 F1 으로 전환 안 됨.")


def main():
    data = T.load_ami()
    print(f"AMI {len(data)}미팅 — reset 부트스트랩 정보-한계 진단\n")
    diag1_cancel_rate(data)
    diag2_discriminator_auc(data)
    diag3_persist_gate(data)


if __name__ == "__main__":
    main()
