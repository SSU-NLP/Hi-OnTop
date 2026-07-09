#!/usr/bin/env python3
"""올바른 프레이밍 재비교 (2026-06-20) — SeCom 프롬프트를 binary 로 최소변경.

사용자 지적: v1/v3 버퍼 베이스라인은 "입력 전체를 분절(segment-all)"이라 잘못. 올바른 설계 =
"맥락 주고 현재 turn 이 경계인지 binary 판단" (= 우리 two-stage judge 프레이밍).
SeCom `baseline_segment_v1.md` 의 Context/criterion 은 유지하고 Question 만 binary 로 바꾼
`baseline_segment_binary_v1.md` 사용 (필요 이상 deviation 없음).

2×2 비교 (AMI, ami_scoring Score 메인, shift0 ±2tol, 전부 0-lag):
  - 어느 turn 판단:  gate(peak[δ_eff] 후보 ~20%)  vs  every(모든 interior turn)
  - 맥락 길이:       full-past  vs  bounded(최근 k턴)
→ segment-all(v1) 붕괴가 태스크 결함이었는지 + 후보게이트 호출절감 효과 확인.

사용: python scripts/recompare_binary_judge.py [--n-ami 20] [--k 64] [--model ...mini]
"""
from __future__ import annotations

import argparse
import hashlib
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
from hi_ontop import ami_scoring as AS  # noqa: E402

PROMPT = open(os.path.join(REPO, "scripts/segmentation_prompts/baseline_segment_binary_v1.md")).read()
CACHE_DIR = os.path.join(REPO, "outputs/runs/_misc/binary_judge_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _toks(s):
    return max(1, len(s) // 4)


def build(turns, t, k):
    """SeCom 포맷 '[Turn i]: (Spk) text', lo..t. k<=0=full-past."""
    lo = 0 if k <= 0 else max(0, t - k)
    block = "\n\n".join(f"[Turn {i}]: ({turns[i][0]}) {turns[i][1]}" for i in range(lo, t + 1))
    return PROMPT.replace("{text_to_be_segmented}", block).replace("{cur}", str(t))


def judge(cli, model, cid, t, prompt):
    key = hashlib.md5(f"BIN|{model}|{cid}|{t}|{prompt}".encode()).hexdigest()
    cp = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))["v"]
    for attempt in range(3):
        try:
            r = cli.chat.completions.create(model=model, temperature=0.0, max_tokens=4,
                                            messages=[{"role": "user", "content": prompt}])
            txt = (r.choices[0].message.content or "").strip().upper()
            v = 1 if "NEW" in txt[:8] else 0
            json.dump({"v": v}, open(cp, "w"))
            return v
        except Exception:
            if attempt == 2:
                return 0
            threading.Event().wait(2 + 2 * attempt)


def run(data, cli, model, gate, k, workers=12):
    # 판단할 turn 집합
    per = []
    for cid, lines, e, yt in data:  # lines unused; turns from raw
        pass
    jobs = []
    meta = []
    for cid, turns, yt, e in data:
        n = len(yt)
        if gate == "peak":
            cand = [p for p in LJ.gen_peak(e, k=0.5) if 0 < p < n - 1]
        else:
            cand = [t for t in range(1, n - 1)]
        meta.append((cid, turns, yt, n, cand))
        for t in cand:
            jobs.append((cid, t, build(turns, t, k)))
    res = {}; lock = threading.Lock(); done = [0]

    def _r(j):
        cid, t, pr = j
        v = judge(cli, model, cid, t, pr)
        with lock:
            res[(cid, t)] = v; done[0] += 1
            if done[0] % 1000 == 0:
                print(f"      {done[0]}/{len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_r, jobs))

    golds, preds = [], []; calls = toks = 0; nturn = 0
    for cid, turns, yt, n, cand in meta:
        nturn += n
        new = [t for t in cand if res.get((cid, t)) == 1]
        for t in cand:
            calls += 1; toks += _toks(build(turns, t, k))
        gold = [i for i, b in enumerate(yt) if b == 1]
        golds.append(AS.boundaries_to_pred(n, gold)); preds.append(AS.boundaries_to_pred(n, new))
    r = AS.score_meetings(golds, preds)
    return dict(score=r["score"], f1=r["f1"], calls=calls, callrate=calls / max(nturn, 1), ktok=toks / 1000)


def load_ami_turns(n_ami):
    """(cid, turns[(spk,text)], yt, emb) — gen_peak 용 emb 포함."""
    import glob, pickle
    TOPIC = os.path.join(REPO, "data/ami/topic"); EMB = os.path.join(REPO, "outputs/runs/_misc/ami_emb")
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
    out = []
    for mid in mids:
        d = json.load(open(f"{TOPIC}/{mid}.json")); bt = list(d["bnd_top"]); bt[-1] = 0
        turns = [(t["speaker"], t["text"]) for t in d["turns"]]
        e = np.asarray(pickle.load(open(f"{EMB}/{mid}.pkl", "rb")), dtype=np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        out.append((mid, turns, bt, e))
    out.sort(key=lambda x: len(x[1]))
    if n_ami and n_ami < len(out):
        idx = sorted(set(np.linspace(0, len(out) - 1, n_ami).astype(int)))
        out = [out[i] for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ami", type=int, default=20)
    ap.add_argument("--k", type=int, default=64, help="bounded 맥락 턴수")
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    LJ._load_env(); cli = LJ._client()
    data = load_ami_turns(args.n_ami)
    nturn = sum(len(d[1]) for d in data)
    print(f"AMI n={len(data)} ({nturn}턴) | SeCom-binary 프롬프트 | model={args.model} | Score 메인, 0-lag")
    print(f"{'judge-turns':<10}{'context':<12}{'Score':>7}{'F1':>7}{'calls':>7}{'call/turn':>10}{'kTok':>9}")
    for gate in ("peak", "every"):
        for k, kn in ((0, "full-past"), (args.k, f"last-{args.k}")):
            print(f"  [{gate} × {kn}] 실행…", flush=True)
            r = run(data, cli, args.model, gate, k, args.workers)
            print(f"{gate:<10}{kn:<12}{r['score']:>7.3f}{r['f1']:>7.3f}{r['calls']:>7}{r['callrate']:>10.3f}{r['ktok']:>9.0f}", flush=True)


if __name__ == "__main__":
    main()
