#!/usr/bin/env python3
"""binary-judge full-past 붕괴지점 기록 (2026-06-22) — 올바른 프레이밍판 §1.5 breakpoint.

사용자 지시: 지금은 V1(full-past)만. bounded(last-64)·v3 제외. **중간 붕괴지점을 기록**(prefix 길이별 recall).
SeCom 최소변경 binary 프롬프트(`baseline_segment_binary_v1.md`)로 **모든 interior turn**을 full-past 맥락 0-lag 판정,
per-checkpoint(prefix turn수·시간·gold여부·pred·근사토큰) 저장 → recall vs prefix-time bin 으로 붕괴 확인.
질문: v1 의 ~20분 붕괴가 'segment-all 태스크 결함'이었나, 아니면 binary 로 바꿔도 긴 full-past 가 붕괴하나?

캐시 공유: `recompare_binary_judge`(RB)의 build/judge 그대로 → 기존 binary_judge_cache 재사용.
사용: python scripts/binary_breakpoint_fullpast.py [--n-ami 16] [--model ...mini]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))
import llm_judge_universal as LJ  # noqa: E402
import recompare_binary_judge as RB  # noqa: E402  (build/judge 재사용 → 캐시 공유)
from hi_ontop import ami_scoring as AS  # noqa: E402

TOPIC = os.path.join(REPO, "data/ami/topic")
OUTDIR = os.path.join(REPO, "outputs/experiments/2026-06-22_binary_breakpoint_fullpast_ami")
os.makedirs(OUTDIR, exist_ok=True)
BINS = [(0, 4), (4, 10), (10, 20), (20, 30), (30, 999)]


def load(n_ami):
    """(mid, turns[(spk,text)], times[s], yt, emb) — RB.load_ami_turns 와 동일 subset 보장."""
    base = RB.load_ami_turns(n_ami)  # (mid, turns, yt, e) sorted+subset
    out = []
    for mid, turns, yt, e in base:
        d = json.load(open(f"{TOPIC}/{mid}.json"))
        times = [t.get("start", 0.0) for t in d["turns"]]
        out.append((mid, turns, times, yt, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ami", type=int, default=16)
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prompt", default="secom", choices=["secom", "ours"],
                    help="secom=baseline_segment_binary_v1 / ours=domain-neutral(LJ.build_prompt)")
    args = ap.parse_args()
    LJ._load_env(); cli = LJ._client()
    data = load(args.n_ami)
    nturn = sum(len(d[1]) for d in data)
    print(f"AMI n={len(data)} ({nturn}턴) | every-turn × full-past | prompt={args.prompt} | {args.model}")

    _judge = RB.judge if args.prompt == "secom" else LJ.judge

    def _mkprompt(turns, t):
        if args.prompt == "secom":
            return RB.build(turns, t, 0)
        lines = [f"[{sp}] {tx}" for sp, tx in turns]  # LJ.load_ami 포맷 = 캐시 일치
        return LJ.build_prompt(lines, t, 0)

    jobs = []
    for mid, turns, times, yt, e in data:
        n = len(yt)
        for t in range(1, n - 1):
            jobs.append((mid, t, _mkprompt(turns, t)))
    res = {}; lock = threading.Lock(); done = [0]
    print(f"  판단 turn = {len(jobs)} (캐시 재사용)…", flush=True)

    def _r(j):
        mid, t, pr = j
        v = _judge(cli, args.model, mid, t, pr)
        with lock:
            res[(mid, t)] = v; done[0] += 1
            if done[0] % 1000 == 0:
                print(f"    {done[0]}/{len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_r, jobs))

    # 채점 + per-checkpoint 기록 + breakpoint bin
    records = []
    golds_pt, preds_pt = [], []
    # bin 집계: gold 경계를 그 시각 bin 에 넣어 recall; pred(NEW) rate by bin
    bin_gold = {b: 0 for b in BINS}; bin_caught = {b: 0 for b in BINS}
    bin_turns = {b: 0 for b in BINS}; bin_new = {b: 0 for b in BINS}

    def binof(tmin):
        for b in BINS:
            if b[0] <= tmin < b[1]:
                return b
        return BINS[-1]

    for mid, turns, times, yt, e in data:
        n = len(yt); gold = [i for i, b in enumerate(yt) if b == 1]
        new = [t for t in range(1, n - 1) if res.get((mid, t)) == 1]
        nset = set(new)
        golds_pt.append(AS.boundaries_to_pred(n, gold)); preds_pt.append(AS.boundaries_to_pred(n, new))
        t0 = times[0] if times else 0.0
        for t in range(1, n - 1):
            tmin = (times[t] - t0) / 60.0
            b = binof(tmin)
            bin_turns[b] += 1
            if t in nset:
                bin_new[b] += 1
            records.append({"mid": mid, "t": t, "tmin": round(tmin, 2),
                            "is_gold": int(any(abs(t - g) <= 2 for g in gold)),
                            "pred": int(t in nset)})
        for g in gold:
            if 0 < g < n - 1:
                tmin = (times[g] - t0) / 60.0; b = binof(tmin)
                bin_gold[b] += 1
                if any(abs(g - p) <= 2 for p in new):
                    bin_caught[b] += 1

    r = AS.score_meetings(golds_pt, preds_pt)
    outd = OUTDIR + ("_ours" if args.prompt == "ours" else "")
    os.makedirs(outd, exist_ok=True)
    json.dump(records, open(os.path.join(outd, "checkpoint_records.json"), "w"))
    json.dump({"score": r, "bins": {f"{b[0]}-{b[1]}": [bin_gold[b], bin_caught[b], bin_turns[b], bin_new[b]] for b in BINS}},
              open(os.path.join(outd, "summary.json"), "w"))

    print(f"\n=== every-turn × full-past (binary) — 전체 ===")
    print(f"  Score={r['score']:.3f} F1={r['f1']:.3f} Pk={r['pk']:.3f} WD={r['wd']:.3f}")
    print(f"\n=== 붕괴지점: prefix 시간 bin 별 ===")
    print(f"  {'bin(min)':<10}{'gold':>6}{'recall':>9}{'NEW율/turn':>12}")
    for b in BINS:
        rec = bin_caught[b] / bin_gold[b] if bin_gold[b] else float('nan')
        nr = bin_new[b] / bin_turns[b] if bin_turns[b] else float('nan')
        print(f"  {str(b[0])+'-'+str(b[1]):<10}{bin_gold[b]:>6}{rec:>9.3f}{nr:>12.3f}")
    print(f"\n  저장: {OUTDIR}/{{checkpoint_records,summary}}.json")
    print(f"  (대조 §1.5 v1 segment-all: recall 0-4분 ~50% → 20분+ 0% 붕괴)")


if __name__ == "__main__":
    main()
