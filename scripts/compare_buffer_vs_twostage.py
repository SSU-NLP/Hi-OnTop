#!/usr/bin/env python3
"""버퍼-LLM 분절 vs 우리 two-stage(적응형 peak 후보 + LLM judge) 비용/품질 직접 비교 (2026-06-20).

같은 모델·같은 데이터·공식 채점기(Score 메인). 3축:
  - Score (0.5F1+0.25(1-Pk)+0.25(1-WD)) [[feedback_score_main_metric]]
  - 호출 빈도: #LLM calls, calls/turn
  - latency: emission lag(경계 발생→방출 턴; two-stage=0, buffer=윈도우 대기) + 총 input 토큰(비용/처리 proxy)

버퍼 = tumbling B-turn 윈도우를 LLM 에 통째로 줘 그 안의 NEW-topic 턴을 받음(윈도우 채울 때까지 대기 = lag).
two-stage = peak[δ_eff] 후보(0-lag)마다 full-past binary 판정(이미 캐시됨 → 재계산 무료).

사용: python scripts/compare_buffer_vs_twostage.py [--n-ami 20] [--model ...mini] [--B 8,16,32]
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
from hi_ontop import dts_scoring as DS  # noqa: E402

CACHE_DIR = os.path.join(REPO, "outputs/runs/_misc/buffer_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BUF_PROMPT = (
    "These are consecutive numbered utterances from a conversation (a short window).\n"
    "List the utterance numbers where a NEW topic clearly begins within this window.\n"
    "If none, return []. Return ONLY a JSON array, e.g. [3] or [].\n\nUtterances:\n{block}\n\nJSON array:"
)


def _toks(s: str) -> int:
    return max(1, len(s) // 4)  # chars/4 근사 (양 방법 동일 적용 → 상대비교 공정)


def _score(name, kind, golds, preds):
    if kind == "ami":
        return AS.score_meetings(golds, preds)
    return DS.score_dialogues(golds, preds)


# ---------- two-stage (캐시에서 무료 재계산) ----------
def eval_twostage(data, kind, shift, model):
    _load = LJ.judge
    golds, preds = [], []
    calls = toks = lag_sum = lag_n = 0
    nturn = 0
    for cid, lines, e, yt in data:
        n = len(yt); nturn += n
        cand = [p for p in LJ.gen_peak(e, k=0.5) if 0 < p < n - 1]
        new = []
        for t in cand:
            pr = LJ.build_prompt(lines, t)
            calls += 1; toks += _toks(LJ.SYS_PROMPT) + _toks(pr)
            if _load(LJ._CLI, model, cid, t, pr) == 1:
                new.append(t)
        if kind == "ami":
            gold = [i for i, b in enumerate(yt) if b == 1]
            golds.append(AS.boundaries_to_pred(n, gold))
            preds.append(AS.boundaries_to_pred(n, new))
        else:
            ytl = list(yt); ytl[-1] = 0; bset = {t - 1 for t in new}
            golds.append(ytl); preds.append([1 if (i in bset and i < n - 1) else 0 for i in range(n)])
        # two-stage emission lag = 0 (후보 턴에서 즉시 판정·방출)
    r = _score(None, kind, golds, preds)
    return dict(score=r["score"], f1=r["f1"], calls=calls, callrate=calls / max(nturn, 1),
                ktok=toks / 1000, lag=0.0, nturn=nturn)


# ---------- buffer-LLM ----------
def buffer_call(model, cid, w0, block):
    key = hashlib.md5(f"BUF|{model}|{cid}|{w0}|{block}".encode()).hexdigest()
    cp = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))["v"]
    for attempt in range(3):
        try:
            r = LJ._CLI.chat.completions.create(
                model=model, temperature=0.0, max_tokens=40,
                messages=[{"role": "user", "content": BUF_PROMPT.format(block=block)}])
            txt = r.choices[0].message.content or ""
            arr = []
            for cand in reversed(re.findall(r"\[[\d,\s]*\]", txt)):
                try:
                    arr = [int(x) for x in json.loads(cand)]; break
                except Exception:
                    continue
            json.dump({"v": arr}, open(cp, "w"))
            return arr
        except Exception as e:
            if attempt == 2:
                return []
            threading.Event().wait(2 + 2 * attempt)


def eval_buffer(data, kind, shift, model, B, workers=12):
    # 1) 모든 윈도우 job
    jobs = []
    for cid, lines, e, yt in data:
        n = len(lines)
        for w0 in range(0, n, B):
            block = "\n".join(f"{j+1}. {lines[w0+j]}" for j in range(min(B, n - w0)))
            jobs.append((cid, w0, block))
    res = {}; lock = threading.Lock()

    def run(j):
        cid, w0, block = j
        v = buffer_call(model, cid, w0, block)
        with lock:
            res[(cid, w0)] = v
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run, jobs))

    golds, preds = [], []
    calls = toks = 0; lag_sum = 0.0; lag_n = 0; nturn = 0
    for cid, lines, e, yt in data:
        n = len(yt); nturn += n
        starts = set()
        for w0 in range(0, n, B):
            block = "\n".join(f"{j+1}. {lines[w0+j]}" for j in range(min(B, n - w0)))
            calls += 1; toks += _toks(BUF_PROMPT) + _toks(block)
            we = min(n, w0 + B)  # 윈도우 방출 시점(채워진 뒤)
            for loc in res.get((cid, w0), []):
                t = w0 + (loc - 1)  # 1-indexed → global
                if 0 < t < n - 1:
                    starts.add(t)
                    lag_sum += (we - t); lag_n += 1  # 방출 lag = 윈도우끝 - 경계턴
        if kind == "ami":
            gold = [i for i, b in enumerate(yt) if b == 1]
            golds.append(AS.boundaries_to_pred(n, gold))
            preds.append(AS.boundaries_to_pred(n, sorted(starts)))
        else:
            ytl = list(yt); ytl[-1] = 0; bset = {t - 1 for t in starts}
            golds.append(ytl); preds.append([1 if (i in bset and i < n - 1) else 0 for i in range(n)])
    r = _score(None, kind, golds, preds)
    return dict(score=r["score"], f1=r["f1"], calls=calls, callrate=calls / max(nturn, 1),
                ktok=toks / 1000, lag=lag_sum / max(lag_n, 1), nturn=nturn)


DSETS = [("AMI", 0, "ami"), ("tiage", -1, "dts"), ("dialseg711", -1, "dts"), ("superseg", -1, "dts")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ami", type=int, default=20)
    ap.add_argument("--n-dts", type=int, default=0)
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--B", default="8,16,32")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    LJ._load_env(); LJ._CLI = LJ._client()
    Bs = [int(x) for x in args.B.split(",")]

    data_by = {}
    for name, shift, kind in DSETS:
        d = LJ.load_ami() if kind == "ami" else LJ.load_dts(name)
        data_by[name] = LJ.subset(d, args.n_ami if kind == "ami" else args.n_dts)

    print(f"버퍼 vs two-stage | model={args.model} | AMI n={args.n_ami} DTS n={args.n_dts or 'ALL'}")
    print(f"{'dataset':<11}{'method':<14}{'Score':>7}{'F1':>7}{'calls':>7}{'call/turn':>10}{'kTok':>9}{'lag(turn)':>10}")
    for name, shift, kind in DSETS:
        data = data_by[name]
        ts = eval_twostage(data, kind, shift, args.model)
        print(f"{name:<11}{'two-stage':<14}{ts['score']:>7.3f}{ts['f1']:>7.3f}{ts['calls']:>7}{ts['callrate']:>10.3f}{ts['ktok']:>9.0f}{ts['lag']:>10.1f}", flush=True)
        for B in Bs:
            b = eval_buffer(data, kind, shift, args.model, B)
            print(f"{'':<11}{('buffer B='+str(B)):<14}{b['score']:>7.3f}{b['f1']:>7.3f}{b['calls']:>7}{b['callrate']:>10.3f}{b['ktok']:>9.0f}{b['lag']:>10.1f}", flush=True)


if __name__ == "__main__":
    main()
