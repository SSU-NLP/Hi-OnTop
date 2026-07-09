#!/usr/bin/env python3
"""spike 후보 recall 상한 측정 — LLM-judge-at-spikes 의 천장 (codex 선결 과제, 2026-06-20).

LLM 은 후보(spike)에 대해서만 binary 판정하므로, **perfect judge 의 천장 = 후보집합의 gold ±2 recall**.
후보 = deploy 검출기(`hi_ontop_deneut`, de-neut V + 적응 μ+cσ, detected-reset)가 flag 하는 turn.
c(σ-배수)를 낮춰 후보를 넓히며 후보율(=LLM 호출 예산) vs recall 상한 트레이드오프를 추적.

지표 (AMI 139, ami_scoring shift0 ±2 tol):
  - cand_rate = 평균 (#후보 / #turn)  ← LLM 호출 예산
  - recall_ceil = per-meeting 평균 (gold 중 ±2 내 후보 있는 비율)  ← perfect judge 최대 recall
  - pj_f1 = per-meeting 평균 tol_f1(gold, {true 후보})  ← perfect judge(precision→1) 실현 ±2F1 천장

사용: python scripts/measure_spike_recall_ceiling.py [emb_subdir ...]   (기본: ami_emb ami_emb_te3large)
"""
from __future__ import annotations

import glob
import json
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


def measure(data, c, R):
    rates, recalls, f1s = [], [], []
    tot_cand = tot_gold = 0
    for e, bt, gold in data:
        n = len(e)
        cand = [p for p in HD.segment(e, c=c, R=R) if 0 < p < n - 1]
        gold_in = [g for g in gold if 0 < g < n - 1]
        rates.append(len(cand) / n)
        if gold_in:
            covered = sum(1 for g in gold_in if any(abs(g - p) <= 2 for p in cand))
            recalls.append(covered / len(gold_in))
            true_cand = [p for p in cand if any(abs(p - g) <= 2 for g in gold_in)]
            f1s.append(AS.tol_f1(gold_in, true_cand))
        tot_cand += len(cand); tot_gold += len(gold_in)
    return (float(np.mean(rates)), float(np.mean(recalls)), float(np.mean(f1s)),
            tot_cand, tot_gold)


if __name__ == "__main__":
    subs = sys.argv[1:] or ["ami_emb", "ami_emb_te3large"]
    for sub in subs:
        data = load_ami(sub)
        ng = sum(len([g for g in gold if 0 < g < len(e) - 1]) for e, _b, gold in data)
        print(f"\n=== {sub}  AMI {len(data)}미팅, gold={ng}, dim={data[0][0].shape[1]} ===")
        print(f"  (참조: deploy c=1.0 ±2F1 0.13~0.14 / oracle 0.71~0.77 / full-context LLM 천장 0.54)")
        print(f"  {'c':>5} {'R':>3} | {'cand_rate':>10} {'recall_ceil':>12} {'perfectF1':>10} | {'#cand':>6}")
        for R in (4, 2):
            for c in (1.5, 1.0, 0.5, 0.0, -0.5, -1.0):
                rate, rec, f1, nc, _ = measure(data, c, R)
                print(f"  {c:>5} {R:>3} | {rate:>10.3f} {rec:>12.3f} {f1:>10.3f} | {nc:>6}", flush=True)
