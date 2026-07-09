#!/usr/bin/env python3
"""본격 9셀 비교 (2026-06-22) — AMI, k=1.0, full-past, 0-lag, mini, Score 메인. 전부 캐시(LLM 호출 0).

trigger(LLM 어디서 호출) × prompt + 무-LLM floor:
  {ours, secom} × {every, peak(k=1.0), buffer-10s, buffer-30s}  = 8
  + embedding-only(no LLM; peak k=1.5 신호를 경계로 직접)         = 1   → 9셀
every-turn(ours·secom) 이 모든 interior turn 을 판정·캐시 → peak/buffer 는 그 부분집합 = 캐시 히트.
buffer-Bs = 시간 cadence(B초)로 sample 한 turn 만 judge(고정 cadence vs 적응 peak 대조). 채점 ami_scoring.

선결 캐시: `binary_breakpoint_fullpast.py --prompt {secom,ours}` (n=16 every-turn) 1회씩.
사용: python scripts/compare9_k1.py [--n-ami 16]
"""
from __future__ import annotations

import argparse
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


def load(n_ami):
    base = RB.load_ami_turns(n_ami)
    out = []
    for mid, turns, yt, e in base:
        d = json.load(open(f"{TOPIC}/{mid}.json"))
        times = [t.get("start", 0.0) for t in d["turns"]]
        out.append((mid, turns, times, yt, e))
    return out


def collect(data, prompt, cli, model):
    judged = {}
    for mid, turns, times, yt, e in data:
        lines = [f"[{sp}] {tx}" for sp, tx in turns]; n = len(yt)
        for t in range(1, n - 1):
            if prompt == "ours":
                judged[(mid, t)] = LJ.judge(cli, model, mid, t, LJ.build_prompt(lines, t, 0))
            else:
                judged[(mid, t)] = RB.judge(cli, model, mid, t, RB.build(turns, t, 0))
    return judged


def buffer_triggers(times, n, B):
    """B초 cadence trigger turn 집합 (interior)."""
    out = []; last = -1e9
    for t in range(1, n - 1):
        if times[t] - last >= B:
            out.append(t); last = times[t]
    return set(out)


def score_pred(data, pred_of):
    """pred_of(mid,turns,times,yt,e)->경계 turn 리스트. → (Score,F1,Pk,WD,callrate)."""
    golds, preds = [], []; calls = 0; nturn = 0
    for mid, turns, times, yt, e in data:
        n = len(yt); nturn += n
        pr, c = pred_of(mid, turns, times, yt, e)
        calls += c
        gold = [i for i, b in enumerate(yt) if b == 1]
        golds.append(AS.boundaries_to_pred(n, gold)); preds.append(AS.boundaries_to_pred(n, sorted(pr)))
    r = AS.score_meetings(golds, preds)
    return r["score"], r["f1"], r["pk"], r["wd"], calls / max(nturn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ami", type=int, default=16)
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    args = ap.parse_args()
    LJ._load_env(); cli = LJ._client()
    data = load(args.n_ami)
    nturn = sum(len(d[1]) for d in data)
    J = {p: collect(data, p, cli, args.model) for p in ("ours", "secom")}

    def mk(prompt, trig):
        def f(mid, turns, times, yt, e):
            n = len(yt)
            if trig == "every":
                cand = set(range(1, n - 1))
            elif trig == "peak":
                cand = set(p for p in LJ.gen_peak(e, k=1.0) if 0 < p < n - 1)
            elif trig == "buf10":
                cand = buffer_triggers(times, n, 10.0)
            elif trig == "buf30":
                cand = buffer_triggers(times, n, 30.0)
            pred = [t for t in cand if J[prompt].get((mid, t)) == 1]
            return pred, len(cand)
        return f

    def emb_only(k):
        def f(mid, turns, times, yt, e):
            n = len(yt); pred = [p for p in LJ.gen_peak(e, k=k) if 0 < p < n - 1]
            return pred, 0  # LLM 호출 0
        return f

    print(f"본격 9셀 — AMI n={len(data)} ({nturn}턴), k=1.0(peak), full-past, 0-lag, {args.model}, Score 메인")
    print(f"{'config':<22}{'Score':>7}{'F1':>7}{'Pk':>7}{'WD':>7}{'call/turn':>10}")
    rows = []
    for prompt in ("ours", "secom"):
        for trig, tn in (("peak", "peak k1.0"), ("every", "every-turn"),
                         ("buf10", "buffer-10s"), ("buf30", "buffer-30s")):
            s, f1, pk, wd, cr = score_pred(data, mk(prompt, trig))
            rows.append((f"{prompt} × {tn}", s, f1, pk, wd, cr))
    for k in (1.5,):
        s, f1, pk, wd, cr = score_pred(data, emb_only(k))
        rows.append((f"embedding-only k={k}", s, f1, pk, wd, cr))
    for name, s, f1, pk, wd, cr in rows:
        print(f"{name:<22}{s:>7.3f}{f1:>7.3f}{pk:>7.3f}{wd:>7.3f}{cr:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
