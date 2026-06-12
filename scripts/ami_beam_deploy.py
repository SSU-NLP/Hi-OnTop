#!/usr/bin/env python3
"""Reversible Suffix Beam (codex 자문) — 오염 회복형 deploy. v2: 제대로 재작성.

online Viterbi-beam: 목적함수 = Σ_seg V(x, suffix-prototype) + c_pen·(#boundaries) 최소화.
- continuation: J += −V (V-단위). reset: J = best_parent.J − c_pen (seed x_t 는 새 seg 가 설명, free-fit 없음).
- top-K + **무-reset(최장 run) 가설 강제 보존** → 초기 잘못된 reset 을 backtrace 가 정정(=오염 회복).
- 경계 = 최종 MAP 가설의 reset path. 신호 = V_rel(β=0). vs commit-refine 0.401 / oracle ±2F1 0.671.
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from ami_adaptive_deneut_deploy import load_ami, ev


def segment_beam(e, lam=0.6, g_rho=0.15, K=12, c_pen=0.3, R=3, keep_noreset=True):
    n = len(e); g = e[0].copy(); gk = 1
    H = [{"sum": e[0].copy(), "J": 0.0, "rl": 1, "starts": (0,)}]
    for t in range(1, n):
        x = e[t]; rg = 1.0 - float(x @ g)
        bp = max(H, key=lambda h: h["J"])
        cand = []
        for h in H:
            p = h["sum"] / (np.linalg.norm(h["sum"]) + 1e-12)
            V = (1.0 - float(x @ p)) - lam * rg
            cand.append({"sum": h["sum"] + x, "J": h["J"] - V, "rl": h["rl"] + 1, "starts": h["starts"]})
        if bp["rl"] >= R:                                   # reset 후보 (best parent 에서, seed free-fit 없음)
            cand.append({"sum": x.copy(), "J": bp["J"] - c_pen, "rl": 1, "starts": bp["starts"] + (t,)})
        cand.sort(key=lambda h: -h["J"])
        H = cand[:K]
        if keep_noreset:                                    # 최장 run(=최근 reset 안 한) 가설 강제 보존
            longest = max(cand, key=lambda h: h["rl"])
            if not any(longest is h for h in H):
                H[-1] = longest
        gr = max(g_rho, 1.0 / (gk + 1)); g = (1 - gr) * g + gr * x; g /= np.linalg.norm(g) + 1e-12; gk += 1
    best = max(H, key=lambda h: h["J"])
    return [b for b in best["starts"] if 0 < b < n - 1]


if __name__ == "__main__":
    AMI = load_ami(); ng = sum(len(d[2]) for d in AMI)
    print(f"Reversible Suffix Beam v2 (V_rel β=0). gold={ng}. 참조: commit-refine Score 0.401/±2F1 0.140, "
          "threshold 0.372, oracle ±2F1 0.671")
    print(f"{'setting':<28}{'pred':>6}{'  ±2F1':>8}{'  Pk':>7}{'  WD':>7}{'  Score':>8}")
    best = (None, -1)
    for K in (8, 16):
        for R in (3, 4):
            for c_pen in (0.18, 0.22, 0.26, 0.30, 0.36, 0.45):
                tot, f2, pk, wd, sc = ev(AMI, lambda e, K=K, c=c_pen, R=R:
                                         segment_beam(e, K=K, c_pen=c, R=R))
                lab = f"beam K={K} R={R} pen={c_pen}"
                if sc > best[1]:
                    best = (lab, sc)
                print(f"{lab:<28}{tot:>6}{f2:>8.3f}{pk:>7.3f}{wd:>7.3f}{sc:>8.3f}", flush=True)
    print(f"\nBEST: {best[0]}  Score={best[1]:.3f}")
