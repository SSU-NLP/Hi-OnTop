#!/usr/bin/env python3
"""공정 비교: buffer-cadence(B턴마다 호출) + full-past 입력 vs two-stage(후보마다 호출) + full-past (2026-06-20).

맥락을 둘 다 full-past 로 고정하고 **호출 트리거만** 다르게:
  - two-stage: peak[δ_eff] 후보 turn 마다 1 binary 호출 (~0.25/turn, 0-lag).
  - buffer-cadence: B턴마다 1 호출, full-past 맥락 + "최근 B턴 중 새 화제 시작 turn 들" 을 한 번에 요청 (호출 1/B, lag B).
같은 맥락(full-past)에서 "후보마다 vs B턴마다 호출" 의 Score/호출수/토큰/lag 비교. 채점 ami_scoring(Score 메인).

사용: python scripts/buffer_fullpast.py [--n-ami 20] [--B 4,8,16] [--model ...mini]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
from hi_ontop import ami_scoring as AS  # noqa: E402

CACHE_DIR = os.path.join(REPO, "outputs/runs/_misc/buffer_fullpast_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SYS = (
    "You segment a conversation into topic segments. You see the conversation so far, then a RECENT "
    "WINDOW of the latest utterances (with their indices). List the indices of utterances in the RECENT "
    "WINDOW that BEGIN a new topic (substantially different subject; not a reply/aside/digression). "
    "Return ONLY a JSON array of indices, e.g. [123] or []."
)


def _toks(s):
    return max(1, len(s) // 4)


def build(lines, w0, we):
    """full-past(0..we-1) 맥락 + RECENT WINDOW [w0,we)."""
    hist = "\n".join(f"{i}: {lines[i]}" for i in range(0, we))
    win = "\n".join(f"{i}: {lines[i]}" for i in range(w0, we))
    return (f"Conversation so far:\n{hist}\n\nRECENT WINDOW (judge only these, indices {w0}..{we-1}):\n{win}\n\n"
            f"JSON array of indices in [{w0},{we-1}] that begin a new topic:")


def call(cli, model, cid, w0, we, prompt):
    key = hashlib.md5(f"{model}|{cid}|{w0}|{we}|{prompt}".encode()).hexdigest()
    cp = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))["v"]
    for attempt in range(3):
        try:
            r = cli.chat.completions.create(model=model, temperature=0.0, max_tokens=60,
                                            messages=[{"role": "system", "content": SYS},
                                                      {"role": "user", "content": prompt}])
            txt = r.choices[0].message.content or ""
            arr = []
            for c in reversed(re.findall(r"\[[\d,\s]*\]", txt)):
                try:
                    arr = [int(x) for x in json.loads(c)]; break
                except Exception:
                    continue
            json.dump({"v": arr}, open(cp, "w"))
            return arr
        except Exception:
            if attempt == 2:
                return []
            threading.Event().wait(2 + 2 * attempt)


def run_buffer(data, cli, model, B, workers=12):
    jobs = []
    for cid, lines, e, yt in data:
        n = len(yt)
        for w0 in range(0, n, B):
            we = min(n, w0 + B)
            jobs.append((cid, w0, we, build(lines, w0, we)))
    res = {}; lock = threading.Lock()

    def _r(j):
        cid, w0, we, pr = j
        v = call(cli, model, cid, w0, we, pr)
        with lock:
            res[(cid, w0)] = v
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_r, jobs))

    golds, preds = [], []
    calls = toks = 0; lag_sum = 0.0; lag_n = 0; nturn = 0
    for cid, lines, e, yt in data:
        n = len(yt); nturn += n
        bnd = set()
        for w0 in range(0, n, B):
            we = min(n, w0 + B)
            calls += 1; toks += _toks(SYS) + _toks(build(lines, w0, we))
            for t in res.get((cid, w0), []):
                if 0 < t < n - 1:
                    bnd.add(t); lag_sum += (we - t); lag_n += 1  # 윈도우 끝에서 방출
        gold = [i for i, b in enumerate(yt) if b == 1]
        golds.append(AS.boundaries_to_pred(n, gold)); preds.append(AS.boundaries_to_pred(n, sorted(bnd)))
    r = AS.score_meetings(golds, preds)
    return dict(score=r["score"], f1=r["f1"], calls=calls, callrate=calls / max(nturn, 1),
                ktok=toks / 1000, lag=lag_sum / max(lag_n, 1))


def run_twostage(data, cli, model):
    """우리 방법 — peak 후보마다 full-past binary (캐시 재사용)."""
    golds, preds = [], []
    calls = toks = 0; nturn = 0
    for cid, lines, e, yt in data:
        n = len(yt); nturn += n
        cand = [p for p in LJ.gen_peak(e, k=0.5) if 0 < p < n - 1]
        new = []
        for t in cand:
            pr = LJ.build_prompt(lines, t, 0)
            calls += 1; toks += _toks(LJ.SYS_PROMPT) + _toks(pr)
            if LJ.judge(cli, model, cid, t, pr) == 1:
                new.append(t)
        gold = [i for i, b in enumerate(yt) if b == 1]
        golds.append(AS.boundaries_to_pred(n, gold)); preds.append(AS.boundaries_to_pred(n, new))
    r = AS.score_meetings(golds, preds)
    return dict(score=r["score"], f1=r["f1"], calls=calls, callrate=calls / max(nturn, 1),
                ktok=toks / 1000, lag=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ami", type=int, default=20)
    ap.add_argument("--B", default="4,8,16")
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    LJ._load_env(); cli = LJ._client()
    data = LJ.subset(LJ.load_ami(), args.n_ami)
    print(f"AMI n={args.n_ami} | model={args.model} | 맥락=full-past 고정, 호출 트리거만 비교")
    print(f"{'method':<22}{'Score':>7}{'F1':>7}{'calls':>7}{'call/turn':>10}{'kTok':>9}{'lag(turn)':>10}")
    ts = run_twostage(data, cli, args.model)
    print(f"{'two-stage(후보마다)':<22}{ts['score']:>7.3f}{ts['f1']:>7.3f}{ts['calls']:>7}{ts['callrate']:>10.3f}{ts['ktok']:>9.0f}{ts['lag']:>10.1f}", flush=True)
    for B in [int(x) for x in args.B.split(",")]:
        b = run_buffer(data, cli, args.model, B)
        print(f"{('buffer-cadence B='+str(B)):<22}{b['score']:>7.3f}{b['f1']:>7.3f}{b['calls']:>7}{b['callrate']:>10.3f}{b['ktok']:>9.0f}{b['lag']:>10.1f}", flush=True)


if __name__ == "__main__":
    main()
