#!/usr/bin/env python3
"""spike-gated LLM binary judge (codex 1순위, 2026-06-20).

cosine 판별자(랜덤 0.49) 대신, deploy 검출기가 flag 한 후보 spike 에서만 LLM 에 **최근 k턴 raw text**
(0-lag, 미래 금지)를 주고 "현재 발화가 NEW topic 시작인가 SAME 계속인가" binary 판정.
pred = NEW 판정 후보 → `ami_scoring` 으로 ±2F1.  perfect-judge 천장(recall 상한)과 대조.

후보: `hi_ontop_deneut.segment(e, c, R)` (de-neut V + 적응 μ+cσ). 임베딩=minilm-int8(후보생성 인코더 무관 확인됨).
LLM: Crts 프록시(OpenAI 호환). 디스크 캐시(재실행 free) + 스레드 동시성.

사용: python scripts/llm_judge_spikes.py [--c 0.5] [--k 12] [--n 20] [--model openrouter/openai/gpt-4o-mini] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from hi_ontop import ami_scoring as AS  # noqa: E402
from hi_ontop import hi_ontop_deneut as HD  # noqa: E402

TOPIC = os.path.join(REPO, "data/ami/topic")
EMB = os.path.join(REPO, "outputs/runs/_misc/ami_emb")
CACHE_DIR = os.path.join(REPO, "outputs/runs/_misc/llm_judge_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# [SUPERSEDED by judge_binary_neutral_v1] meeting 전용 초기판 — segmentation_prompts/ 단일출처 (byte-exact)
SYS_PROMPT = open(os.path.join(REPO, "scripts/segmentation_prompts/judge_binary_meeting_v0.md")).read()


def _load_env():
    for line in open(os.path.join(REPO, ".env")):
        m = re.match(r"([A-Z_]+)=(.*)", line.strip())
        if m:
            os.environ.setdefault(m.group(1), m.group(2))


def _client():
    from openai import OpenAI
    hdr = {}
    if os.environ.get("CF_ACCESS_CLIENT_ID"):
        hdr["CF-Access-Client-Id"] = os.environ["CF_ACCESS_CLIENT_ID"]
        hdr["CF-Access-Client-Secret"] = os.environ["CF_ACCESS_CLIENT_SECRET"]
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                  base_url=os.environ["OPENAI_BASE_URL"], default_headers=hdr or None)


def load_ami():
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    out = []
    for mid in mids:
        d = json.load(open(f"{TOPIC}/{mid}.json"))
        bt = list(d["bnd_top"]); bt[-1] = 0
        turns = [(t["speaker"], t["text"]) for t in d["turns"]]
        e = np.asarray(pickle.load(open(f"{EMB}/{mid}.pkl", "rb")), dtype=np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        out.append((mid, turns, e, bt, [i for i, b in enumerate(bt) if b == 1]))
    return out


def build_prompt(turns, t, k):
    """최근 k턴(미래 금지) raw text, 현재 발화 강조. k<=0 = 전체 과거(0..t-1, 0-lag full-past)."""
    lo = 0 if k <= 0 else max(0, t - k)
    ctx = "\n".join(f"[{sp}] {tx}" for sp, tx in turns[lo:t])
    cur = f"[{turns[t][0]}] {turns[t][1]}"
    return (f"Recent conversation (most recent last):\n{ctx}\n\n"
            f">>> CURRENT utterance to judge:\n{cur}\n\n"
            f"Does the CURRENT utterance begin a NEW topic segment, or continue the SAME topic? "
            f"Answer one word: NEW or SAME.")


def judge(cli, model, mid, t, prompt):
    key = hashlib.md5(f"{model}|{mid}|{t}|{prompt}".encode()).hexdigest()
    cp = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))["v"]
    for attempt in range(3):
        try:
            r = cli.chat.completions.create(
                model=model, temperature=0.0, max_tokens=4,
                messages=[{"role": "system", "content": SYS_PROMPT},
                          {"role": "user", "content": prompt}])
            txt = (r.choices[0].message.content or "").strip().upper()
            v = 1 if txt.startswith("NEW") else 0
            json.dump({"v": v, "raw": txt}, open(cp, "w"))
            return v
        except Exception as e:
            if attempt == 2:
                print(f"  judge fail {mid}:{t} {str(e)[:60]}", flush=True); return 0
            threading.Event().wait(2 + 2 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--n", type=int, default=20, help="미팅 수 (0=전체)")
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    _load_env(); cli = _client()
    data = load_ami()
    if args.n:
        # 크기 stratified: 길이 정렬 후 균등 샘플
        data = sorted(data, key=lambda d: len(d[1]))
        idx = np.linspace(0, len(data) - 1, args.n).astype(int)
        data = [data[i] for i in sorted(set(idx))]
    print(f"LLM judge: {len(data)}미팅 c={args.c} k={args.k} R={args.R} model={args.model}")

    # 후보 생성
    jobs = []  # (mid, t, prompt)
    meta = {}   # mid -> (turns, bt, gold, n, candidates)
    for mid, turns, e, bt, gold in data:
        n = len(e)
        cand = [p for p in HD.segment(e, c=args.c, R=args.R) if 0 < p < n - 1]
        meta[mid] = dict(turns=turns, bt=bt, gold=[g for g in gold if 0 < g < n - 1], n=n, cand=cand)
        for t in cand:
            jobs.append((mid, t, build_prompt(turns, t, args.k)))
    print(f"  후보 {len(jobs)}개 → LLM 호출(캐시) …", flush=True)

    results = {}
    done = [0]; lock = threading.Lock()

    def run(job):
        mid, t, prompt = job
        v = judge(cli, args.model, mid, t, prompt)
        with lock:
            results[(mid, t)] = v; done[0] += 1
            if done[0] % 200 == 0:
                print(f"    {done[0]}/{len(jobs)}", flush=True)
        return v

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, jobs))

    # 채점 + 후보레벨 confusion
    golds_pt, preds_pt = [], []
    tp = fp = fn_true = n_true_cand = n_new = 0
    perfect_preds = []
    for mid, m in meta.items():
        n, gold, cand = m["n"], m["gold"], m["cand"]
        new = [t for t in cand if results.get((mid, t), 0) == 1]
        n_new += len(new)
        golds_pt.append(AS.boundaries_to_pred(n, gold))
        preds_pt.append(AS.boundaries_to_pred(n, new))
        perfect = [t for t in cand if any(abs(t - g) <= 2 for g in gold)]
        perfect_preds.append(AS.boundaries_to_pred(n, perfect))
        n_true_cand += len(perfect)
        for t in new:
            if any(abs(t - g) <= 2 for g in gold):
                tp += 1
            else:
                fp += 1

    r = AS.score_meetings(golds_pt, preds_pt)
    rceil = AS.score_meetings(golds_pt, perfect_preds)
    prec = tp / max(n_new, 1)
    print("\n=== 결과 ===")
    print(f"  후보 {len(jobs)} / true(±2) {n_true_cand} (false:true ≈ {(len(jobs)-n_true_cand)/max(n_true_cand,1):.1f}:1)")
    print(f"  LLM NEW 판정 {n_new}개  → candidate-level precision(±2) = {prec:.3f}  (tp {tp} / fp {fp})")
    print(f"  meeting ±2F1 = {r['f1']:.3f}  Pk={r['pk']:.3f} WD={r['wd']:.3f} Score={r['score']:.3f}")
    print(f"  --- 대조: perfect-judge 천장 ±2F1 = {rceil['f1']:.3f} / 현 deploy c1.0 ~0.13 / full-ctx LLM ~0.54 ---")


if __name__ == "__main__":
    main()
