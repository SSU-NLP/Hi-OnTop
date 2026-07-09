#!/usr/bin/env python3
"""LLM-clean-reset deploy (codex §C #2 "label commit ↔ state reset 분리") — 구현·실측 (2026-06-22).

아이디어: oracle(gold-reset) 0.69 vs deploy(detected-reset) 0.13 격차 = **prototype 오염**.
de-neut V 온라인 검출기에서 spike(V>μ+cσ)마다 **LLM 이 confirm 할 때만 active prototype reset**
(reject 면 reset 안 하고 outlier 로 흡수) → prototype 을 gold-reset 처럼 깨끗이 유지 → 신호가 oracle 쪽으로.
경계 출력 = LLM-confirmed reset. 대조군: LLM 없이 spike 마다 reset(=현 오염 deploy).

비용 0: LLM 판정(full-past, judge_binary_neutral_v1, mini)이 ours every-turn 런에서 **전 턴 캐시됨** → 모든 후보 캐시 히트.
AMI n=16, ami_scoring(shift0 ±2tol Score 메인). 참조: oracle ±2F1 0.69 / §H peak+LLM mini 0.32 / 임베딩 deploy 0.13.

사용: python scripts/deploy_llm_clean_reset.py [--n-ami 16] [--model ...mini]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))
import llm_judge_universal as LJ  # noqa: E402
import recompare_binary_judge as RB  # noqa: E402
from hi_ontop import ami_scoring as AS  # noqa: E402

A, B, L0, LAM, G_RHO, RHO_MIN, WARMUP = 2.0, 1.0, 8, 0.6, 0.15, 0.05, 8


def _nr(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _deneut(x, g, beta):
    xc = x - beta * float(x @ g) * g
    return xc / (np.linalg.norm(xc) + 1e-9)


def deploy(e, confirm, c, R=4):
    """de-neut V detector. confirm(t)->bool 이 True 일 때만 reset. 경계 turn 리스트 반환."""
    n = len(e); m = e[0].copy(); k = 1; g = e[0].copy()
    Wm = WM2 = 0.0; Wn = 0; pred = []; last = -999
    for t in range(1, n):
        x = e[t]; beta = min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)
        gh = _nr(g)
        V = (1 - float(_deneut(x, gh, beta) @ _deneut(m, gh, beta))) - LAM * (1 - float(x @ gh))
        sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
        thr = (Wm + c * sd) if Wn >= WARMUP else None
        is_cand = thr is not None and V > thr and k >= R and t - last >= R
        if is_cand and confirm(t):
            pred.append(t); m = x.copy(); k = 1; last = t            # clean reset (확정)
        else:
            Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
            rho = max(RHO_MIN, 1.0 / (k + 1)); m = _nr((1 - rho) * m + rho * x); k += 1  # 흡수
        g = _nr((1 - G_RHO) * g + G_RHO * x)
    return pred


def load(n_ami):
    base = RB.load_ami_turns(n_ami); out = []
    for mid, turns, yt, e in base:
        lines = [f"[{sp}] {tx}" for sp, tx in turns]
        out.append((mid, turns, lines, yt, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ami", type=int, default=16)
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    args = ap.parse_args()
    LJ._load_env(); cli = LJ._client()
    data = load(args.n_ami)
    nturn = sum(len(d[1]) for d in data)

    miss = [0]

    def make_confirm(mid, lines, use_llm):
        def f(t):
            if not use_llm:
                return True
            pr = LJ.build_prompt(lines, t, 0)
            v = LJ.judge(cli, args.model, mid, t, pr)
            if v is None:
                miss[0] += 1
            return v == 1
        return f

    def run(use_llm, c):
        golds, preds = [], []; calls = 0
        for mid, turns, lines, yt, e in data:
            n = len(yt)
            # 후보 수(=LLM 질의수) 세려면 별도 — confirm 호출 횟수 카운트
            cnt = [0]
            base_conf = make_confirm(mid, lines, use_llm)
            def conf(t, cnt=cnt, base_conf=base_conf):
                cnt[0] += 1; return base_conf(t)
            pr = deploy(e, conf, c)
            calls += cnt[0]
            gold = [i for i, b in enumerate(yt) if b == 1]
            golds.append(AS.boundaries_to_pred(n, gold)); preds.append(AS.boundaries_to_pred(n, pr))
        r = AS.score_meetings(golds, preds)
        return r, calls

    print(f"AMI n={len(data)} ({nturn}턴) | de-neut V deploy | {args.model} | Score 메인, 0-lag")
    print(f"참조: oracle ±2F1 0.69 / 임베딩 deploy(오염) ~0.13 / §H peak+LLM(mini) F1 0.32 Score 0.45")
    print(f"{'variant':<24}{'c':>5}{'Score':>8}{'F1':>7}{'Pk':>7}{'WD':>7}{'cand=호출':>10}{'call/turn':>10}")
    for c in (1.0, 0.5, 0.0):
        r, ca = run(False, c)
        print(f"{'대조: spike마다 reset':<24}{c:>5.1f}{r['score']:>8.3f}{r['f1']:>7.3f}{r['pk']:>7.3f}{r['wd']:>7.3f}{ca:>10}{ca/nturn:>10.3f}")
        r, ca = run(True, c)
        tag = " (miss%d)" % miss[0] if miss[0] else ""
        print(f"{'★ LLM-clean-reset':<24}{c:>5.1f}{r['score']:>8.3f}{r['f1']:>7.3f}{r['pk']:>7.3f}{r['wd']:>7.3f}{ca:>10}{ca/nturn:>10.3f}{tag}", flush=True)


if __name__ == "__main__":
    main()
