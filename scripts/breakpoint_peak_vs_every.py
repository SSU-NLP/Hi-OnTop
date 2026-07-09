#!/usr/bin/env python3
"""peak(우리 방법) vs every-turn × prompt(ours/secom) — 캐시에서 재계산 (2026-06-22, 추가 호출 0).

every-turn run 이 모든 interior turn 을 이미 판정·캐시했으므로, **peak 후보만 필터**하면 우리 방법(peak-gate)
결과를 추가 호출 없이 얻는다. prompt(ours=domain-neutral / secom=SeCom-binary) × gate(peak/every) 4셀 비교.
지표: Score(메인)/F1/Pk/WD + prefix-time bin 별 recall·NEW율. AMI, mini, full-past, 0-lag, ami_scoring.

선결: binary_breakpoint_fullpast 를 --prompt secom 및 --prompt ours 로 each 1회 돌려 캐시 채워둘 것.
사용: python scripts/breakpoint_peak_vs_every.py [--n-ami 16] [--model ...mini]
"""
from __future__ import annotations

import argparse
import glob
import json
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

TOPIC = os.path.join(REPO, "data/ami/topic")
BINS = [(0, 4), (4, 10), (10, 20), (20, 30), (30, 999)]


def binof(m):
    for b in BINS:
        if b[0] <= m < b[1]:
            return b
    return BINS[-1]


def metrics(data, judged, gate):
    """judged: {(mid,t):0/1}. gate: 'peak'|'every'. → score dict + 진단."""
    golds, preds = [], []
    bg = {b: 0 for b in BINS}; bc = {b: 0 for b in BINS}
    bt = {b: 0 for b in BINS}; bn = {b: 0 for b in BINS}
    calls = 0
    for mid, turns, times, yt, e in data:
        n = len(yt); gold = [i for i, b in enumerate(yt) if b == 1]
        if gate == "peak":
            cand = set(p for p in LJ.gen_peak(e, k=0.5) if 0 < p < n - 1)
        else:
            cand = set(range(1, n - 1))
        calls += len(cand)
        new = [t for t in range(1, n - 1) if t in cand and judged.get((mid, t)) == 1]
        nset = set(new)
        golds.append(AS.boundaries_to_pred(n, gold)); preds.append(AS.boundaries_to_pred(n, new))
        t0 = times[0] if times else 0.0
        for t in cand:
            b = binof((times[t] - t0) / 60.0); bt[b] += 1
            if t in nset:
                bn[b] += 1
        for g in gold:
            if 0 < g < n - 1:
                b = binof((times[g] - t0) / 60.0); bg[b] += 1
                if any(abs(g - p) <= 2 for p in nset):
                    bc[b] += 1
    r = AS.score_meetings(golds, preds)
    bins = {f"{b[0]}-{b[1]}": (bc[b] / bg[b] if bg[b] else float('nan'),
                               bn[b] / bt[b] if bt[b] else float('nan')) for b in BINS}
    return r, calls, bins


def load(n_ami):
    base = RB.load_ami_turns(n_ami)
    out = []
    for mid, turns, yt, e in base:
        d = json.load(open(f"{TOPIC}/{mid}.json"))
        times = [t.get("start", 0.0) for t in d["turns"]]
        out.append((mid, turns, times, yt, e))
    return out


def collect(data, prompt, cli, model):
    """모든 interior turn 판정(캐시). prompt: ours|secom."""
    judged = {}; miss = 0
    for mid, turns, times, yt, e in data:
        lines = [f"[{sp}] {tx}" for sp, tx in turns]
        n = len(yt)
        for t in range(1, n - 1):
            if prompt == "ours":
                pr = LJ.build_prompt(lines, t, 0); v = LJ.judge(cli, model, mid, t, pr)
            else:
                pr = RB.build(turns, t, 0); v = RB.judge(cli, model, mid, t, pr)
            judged[(mid, t)] = v
            if v is None:
                miss += 1
    return judged, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ami", type=int, default=16)
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    args = ap.parse_args()
    LJ._load_env(); cli = LJ._client()
    data = load(args.n_ami)
    nturn = sum(len(d[1]) for d in data)
    print(f"AMI n={len(data)} ({nturn}턴) | mini | full-past 0-lag | 캐시 재계산 (호출 0 가정)")
    print(f"{'prompt':<8}{'gate':<7}{'Score':>7}{'F1':>7}{'Pk':>7}{'WD':>7}{'calls':>7}{'call/turn':>10}")
    detail = {}
    for prompt in ("ours", "secom"):
        judged, miss = collect(data, prompt, cli, args.model)
        if miss:
            print(f"  ⚠ {prompt}: {miss} 미캐시(=fresh 호출됨)")
        for gate in ("peak", "every"):
            r, calls, bins = metrics(data, judged, gate)
            detail[(prompt, gate)] = bins
            print(f"{prompt:<8}{gate:<7}{r['score']:>7.3f}{r['f1']:>7.3f}{r['pk']:>7.3f}{r['wd']:>7.3f}{calls:>7}{calls/nturn:>10.3f}", flush=True)
    print(f"\n=== prefix-time bin 별 (recall / NEW율) ===")
    print(f"  {'bin(min)':<10}" + "".join(f"{p+'-'+g:>16}" for p in ('ours', 'secom') for g in ('peak', 'every')))
    for b in BINS:
        bn = f"{b[0]}-{b[1]}"
        cells = []
        for p in ('ours', 'secom'):
            for g in ('peak', 'every'):
                rc, nr = detail[(p, g)][bn]
                cells.append(f"{rc:.2f}/{nr:.2f}")
        print(f"  {bn:<10}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
