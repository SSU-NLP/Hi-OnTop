#!/usr/bin/env python3
"""deploy 가 왜 안 되는지 *계량* — 추측 아닌 측정.

세 가지를 잰다:
1. operating point precision/recall 분해 (놓침 vs 헛경보 중 무엇이 주범인가).
2. 신호 분별력 AUC: 진짜 경계 turn 의 V 가 비-경계보다 높은 확률.
   - V_oracle: gold-reset(깨끗) prototype 기준
   - V_deploy: detected-reset(온라인 오염) prototype 기준
   둘의 AUC 차 = '온라인 prototype 오염이 경계 신호를 죽인 정도'(임계 선택과 무관).
3. 진짜 경계에서 V_deploy 가 V_oracle 대비 얼마나 눌렸나(중앙값 비).
"""
from __future__ import annotations
import sys, json, glob, pickle, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")

TOPIC = "data/ami/topic"; AC = "outputs/runs/_misc/ami_emb"


def nr(v): return v / (np.linalg.norm(v) + 1e-9)
def deneut(x, g, b):
    xc = x - b * float(x @ g) * g; return xc / (np.linalg.norm(xc) + 1e-9)
def beta_rl(k, A=2.0, B=1.0, L0=8): return min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)


def _beta(k, bf): return bf if bf is not None else beta_rl(k)


def v_oracle(e, gold, bf=None, lam=0.6, g_rho=0.15, rho_min=0.05):
    """gold-reset 깨끗한 prototype 으로 V[t]. bf=None 적응β, bf=0 면 V_rel."""
    n = len(e); gs = set(gold); V = np.zeros(n); m = e[0].copy(); k = 1; g = e[0].copy()
    for t in range(1, n):
        x = e[t]; b = _beta(k, bf)
        V[t] = (1 - float(deneut(x, g, b) @ deneut(m, g, b))) - lam * (1 - float(x @ g))
        if t in gs: m = x.copy(); k = 1
        else:
            rho = max(rho_min, 1.0 / (k + 1)); m = nr((1 - rho) * m + rho * x); k += 1
        gr = max(g_rho, 1.0 / (t + 1)); g = nr((1 - gr) * g + gr * x)
    return V


def v_deploy(e, bf=None, c=1.5, lam=0.6, g_rho=0.15, rho_min=0.05, R=4, warmup=8):
    """detected-reset(온라인) prototype 으로 V[t] + 예측 경계. 실제 deploy 와 동일 reset."""
    n = len(e); V = np.zeros(n); m = e[0].copy(); k = 1; g = e[0].copy()
    Wm = WM2 = 0.0; Wn = 0; pred = []; last = -999
    for t in range(1, n):
        x = e[t]; b = _beta(k, bf)
        v = (1 - float(deneut(x, g, b) @ deneut(m, g, b))) - lam * (1 - float(x @ g)); V[t] = v
        sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
        thr = (Wm + c * sd) if Wn >= warmup else None
        if thr is not None and v > thr and k >= R and t - last >= R:
            pred.append(t); m = x.copy(); k = 1; last = t
        else:
            Wn += 1; dd = v - Wm; Wm += dd / Wn; WM2 += dd * (v - Wm)
            rho = max(rho_min, 1.0 / (k + 1)); m = nr((1 - rho) * m + rho * x); k += 1
        gr = max(g_rho, 1.0 / (t + 1)); g = nr((1 - gr) * g + gr * x)
    return V, pred


def auc(scores, labels):
    """경계(label=1) V 가 비경계보다 높을 확률 (Mann-Whitney). 0.5=random."""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    alls = np.concatenate([pos, neg]); ranks = alls.argsort().argsort() + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_tol(scores, idx, gold, n, tol):
    lab = np.array([1 if any(abs(i - j) <= tol for j in gold) else 0 for i in idx])
    return auc(scores, lab)


def run_config(mids, bf, tag):
    AUCo0, AUCd0, AUCo2, AUCd2 = [], [], [], []
    tp = fp = fn = 0
    for mid in mids:
        d = json.load(open(f"{TOPIC}/{mid}.json")); bt = list(d["bnd_top"]); n = len(bt); bt[-1] = 0
        gold = [i for i, b in enumerate(bt) if b == 1]
        if n <= 5 or not gold: continue
        e = nr(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl", "rb")), dtype=np.float64))
        Vo = v_oracle(e, gold, bf=bf); Vd, pred = v_deploy(e, bf=bf)
        idx = np.arange(1, n - 1)
        AUCo0.append(auc_tol(Vo[idx], idx, gold, n, 0)); AUCd0.append(auc_tol(Vd[idx], idx, gold, n, 0))
        AUCo2.append(auc_tol(Vo[idx], idx, gold, n, 2)); AUCd2.append(auc_tol(Vd[idx], idx, gold, n, 2))
        ps = set(p for p in pred if 0 < p < n - 1); gs = set(gold)
        tp += sum(1 for j in gs if any(abs(j - p) <= 2 for p in ps))
        fn += sum(1 for j in gs if not any(abs(j - p) <= 2 for p in ps))
        fp += sum(1 for p in ps if not any(abs(j - p) <= 2 for j in gs))
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    print(f"=== {tag} ===")
    print(f"  AUC(exact) : oracle {np.nanmean(AUCo0):.3f}  deploy {np.nanmean(AUCd0):.3f}  "
          f"(오염 비용 {np.nanmean(AUCo0)-np.nanmean(AUCd0):+.3f})")
    print(f"  AUC(±2 tol): oracle {np.nanmean(AUCo2):.3f}  deploy {np.nanmean(AUCd2):.3f}  "
          f"(오염 비용 {np.nanmean(AUCo2)-np.nanmean(AUCd2):+.3f})")
    print(f"  deploy operating point: recall {rec:.3f}  precision {prec:.3f}")


def main():
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    print("질문: (a) 오염이 신호를 죽이나(oracle vs deploy AUC) (b) 신호 자체가 약한가(oracle AUC) "
          "(c) adaptive-β가 AMI에서 V_rel보다 약한가\n")
    run_config(mids, None, "adaptive-β (commit-refine 가 쓰는 신호)")
    run_config(mids, 0.0, "β=0 = V_rel (handoff AMI oracle 0.687 신호)")


if __name__ == "__main__":
    main()
