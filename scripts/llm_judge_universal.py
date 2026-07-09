#!/usr/bin/env python3
"""원툴 spike-gated LLM judge — AMI + DTS(tiage/dialseg711/superseg) 단일 규격 (2026-06-20).

[[feedback_one_tool_no_per_dataset_tuning]]: 데이터셋별 튜닝 금지. 4개 벤치에 **동일 config**:
  - 후보생성: `hi_ontop_deneut.segment(emb, c=C)` (de-neut 신호, 전역 상수 C 1개)
  - 판정: 동일 LLM·동일 도메인중립 프롬프트, full-past(0-lag) raw text → NEW/SAME
  - 경계 = NEW 후보
채점만 각 벤치 gold 규약(= 벤치 정의, method 튜닝 아님):
  - AMI: gold=시작-turn → 경계 turn = spike t (shift0), ±2 tol, `ami_scoring`
  - DTS: gold=끝-turn   → 경계 turn = spike t-1 (shift-1), exact, `dts_scoring`

사용: python scripts/llm_judge_universal.py [--c 0.5] [--n 20] [--model openrouter/openai/gpt-4o]
                                            [--workers 10] [--estimate]
  --n = 데이터셋당 대화(미팅/dialogue) 수 (0=전체). --estimate = LLM 호출 없이 후보수만.
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
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))
from run_encoder_comparison import load_dialogs, delta_eff_seq, CACHE  # noqa: E402
from hi_ontop import ami_scoring as AS  # noqa: E402
from hi_ontop import dts_scoring as DS  # noqa: E402
from hi_ontop import hi_ontop_deneut as HD  # noqa: E402


def gen_peak(e, k=0.5, W=20):
    """적응형 후보생성 (단일 config) — δ_eff peak+prominence. [[measure_candidate_generators]] 우승.
    s_t 가 직전 상승 AND 최근 W 의 robust baseline(med+k·1.4826·MAD) 초과 → 후보. 0-lag, 발화율 stream-적응."""
    s = np.asarray(list(delta_eff_seq(e))); n = len(e); out = []
    for t in range(2, n - 1):
        past = s[max(1, t - W):t]
        if len(past) < 3:
            continue
        med = float(np.median(past)); mad = float(np.median(np.abs(past - med))) + 1e-9
        if s[t] > s[t - 1] and (s[t] - med) > k * 1.4826 * mad:
            out.append(t)
    return out

TOPIC = os.path.join(REPO, "data/ami/topic")
AMI_EMB = os.path.join(REPO, "outputs/runs/_misc/ami_emb")
CACHE_DIR = os.path.join(REPO, "outputs/runs/_misc/llm_judge_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 도메인중립 단일 judge 프롬프트 — segmentation_prompts/ 단일출처 (byte-exact)
SYS_PROMPT = open(os.path.join(REPO, "scripts/segmentation_prompts/judge_binary_neutral_v1.md")).read()


def _nr_rows(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


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
    """→ list of (cid, lines[list[str]], emb, yt[0/1], shift, scorer)."""
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    out = []
    for mid in mids:
        d = json.load(open(f"{TOPIC}/{mid}.json")); bt = list(d["bnd_top"]); bt[-1] = 0
        lines = [f"[{t['speaker']}] {t['text']}" for t in d["turns"]]
        e = _nr_rows(np.asarray(pickle.load(open(f"{AMI_EMB}/{mid}.pkl", "rb")), dtype=np.float64))
        out.append((mid, lines, e, bt))
    return out


def load_dts(ds):
    dl = load_dialogs(ds, "test")
    emb = [_nr_rows(np.asarray(x, dtype=np.float64))
           for x in pickle.load(open(CACHE / f"enccmp_{ds}_test_minilm-int8.pkl", "rb"))]
    out = []
    for i, ((utts, yt), e) in enumerate(zip(dl, emb)):
        lines = [f"- {u}" for u in utts]
        out.append((f"{ds}{i}", lines, e, list(yt)))
    return out


def subset(data, n):
    if not n or n >= len(data):
        return data
    data = sorted(data, key=lambda d: len(d[1]))
    idx = sorted(set(np.linspace(0, len(data) - 1, n).astype(int)))
    return [data[i] for i in idx]


def build_prompt(lines, t, k=0):
    """과거 맥락(0-lag), 현재 발화 강조. k<=0=full-past, k>0=최근 k턴만(토큰 절감)."""
    lo = 0 if k <= 0 else max(0, t - k)
    ctx = "\n".join(lines[lo:t])
    return (f"Conversation so far (most recent last):\n{ctx}\n\n"
            f">>> CURRENT utterance to judge:\n{lines[t]}\n\n"
            f"Does the CURRENT utterance begin a NEW topic, or continue the SAME topic? "
            f"Answer one word: NEW or SAME.")


def judge(cli, model, cid, t, prompt):
    key = hashlib.md5(f"{model}|{cid}|{t}|{prompt}".encode()).hexdigest()
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
                print(f"  judge fail {cid}:{t} {str(e)[:50]}", flush=True); return 0
            threading.Event().wait(2 + 2 * attempt)


DATASETS = [("AMI", 0, "ami"), ("tiage", -1, "dts"), ("dialseg711", -1, "dts"), ("superseg", -1, "dts")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="peak", choices=["peak", "deneut"], help="후보생성기")
    ap.add_argument("--peak-k", type=float, default=0.5)
    ap.add_argument("--c", type=float, default=0.5)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--n", type=int, default=20, help="공통 표본수 (n_ami/n_dts 미지정 시)")
    ap.add_argument("--n-ami", type=int, default=None, help="AMI 미팅 수 (0=전체)")
    ap.add_argument("--n-dts", type=int, default=None, help="DTS dialogue 수/데이터셋 (0=전체)")
    ap.add_argument("--model", default="openrouter/openai/gpt-4o")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--only", default="", help="쉼표구분 데이터셋 필터 (예: AMI)")
    ap.add_argument("--ctx-k", type=int, default=0, help="judge 맥락 턴수 (0=full-past, >0=최근 k턴)")
    args = ap.parse_args()

    print(f"원툴 LLM judge — c={args.c} R={args.R} n={args.n or 'ALL'} model={args.model}")
    if not args.estimate:
        _load_env(); cli = _client()

    # 1) 전체 후보 수집 (모든 데이터셋)
    bundle = {}  # name -> (records list, shift)
    all_jobs = []
    n_ami = args.n_ami if args.n_ami is not None else args.n
    n_dts = args.n_dts if args.n_dts is not None else args.n
    _DSETS = [d for d in DATASETS if not args.only or d[0] in args.only.split(",")]
    for name, shift, kind in _DSETS:
        nn = n_ami if kind == "ami" else n_dts
        data = subset(load_ami() if kind == "ami" else load_dts(name), nn)
        recs = []
        for cid, lines, e, yt in data:
            n = len(yt)
            raw = gen_peak(e, k=args.peak_k) if args.gen == "peak" else HD.segment(e, c=args.c, R=args.R)
            spikes = [p for p in raw if 0 < p < n - 1]
            recs.append((cid, lines, yt, spikes))
            for t in spikes:
                all_jobs.append((name, cid, t, lines))
        bundle[name] = (recs, shift)
        ncand = sum(len(r[3]) for r in recs)
        nturn = sum(len(r[1]) for r in recs)
        ng = sum(sum(1 for b in r[2] if b) for r in recs)
        ktok = sum(len(build_prompt(r[1], t, args.ctx_k)) for r in recs for t in r[3]) / 4 / 1000
        print(f"  [{name}] conv={len(recs)} turns={nturn} gold={ng} cand={ncand} "
              f"(rate {ncand/max(nturn,1):.2f}) ctx_k={args.ctx_k} ~{ktok:.0f}kTok")
    print(f"  총 LLM 호출 후보 = {len(all_jobs)}")
    if args.estimate:
        print("  (--estimate: LLM 미호출)"); return

    # 2) LLM 판정 (동시성 + 캐시)
    results = {}; done = [0]; lock = threading.Lock()

    def run(job):
        name, cid, t, lines = job
        v = judge(cli, args.model, cid, t, build_prompt(lines, t, args.ctx_k))
        with lock:
            results[(cid, t)] = v; done[0] += 1
            if done[0] % 500 == 0:
                print(f"    {done[0]}/{len(all_jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, all_jobs))

    # 3) 채점 (벤치별 규약)
    print(f"\n{'dataset':<12}{'metric':<7}{'cand':>6}{'NEW':>6}{'prec':>7}{'F1':>8}{'Score':>8}{'ceilF1':>8}")
    for name, shift, kind in _DSETS:
        recs, _ = bundle[name]
        if kind == "ami":
            golds, preds, perf = [], [], []
            tp = nnew = 0
            for cid, lines, yt, spikes in recs:
                n = len(yt); gold = [i for i, b in enumerate(yt) if b == 1]
                new = [t for t in spikes if results.get((cid, t)) == 1]
                nnew += len(new)
                tp += sum(1 for t in new if any(abs(t - g) <= 2 for g in gold))
                golds.append(AS.boundaries_to_pred(n, gold))
                preds.append(AS.boundaries_to_pred(n, new))
                perf.append(AS.boundaries_to_pred(n, [t for t in spikes if any(abs(t - g) <= 2 for g in gold)]))
            r = AS.score_meetings(golds, preds); rc = AS.score_meetings(golds, perf)
            ncand = sum(len(x[3]) for x in recs)
            print(f"{name:<12}{'±2tol':<7}{ncand:>6}{nnew:>6}{tp/max(nnew,1):>7.3f}{r['f1']:>8.3f}{r['score']:>8.3f}{rc['f1']:>8.3f}")
        else:
            golds, preds, perf = [], [], []
            tp = nnew = 0
            for cid, lines, yt, spikes in recs:
                n = len(yt); gold = set(i for i, b in enumerate(yt) if b == 1)
                new = [t for t in spikes if results.get((cid, t)) == 1]
                bset = {t - 1 for t in new}
                nnew += len(new)
                tp += sum(1 for t in new if (t - 1) in gold)
                ytl = list(yt); ytl[-1] = 0
                golds.append(ytl)
                preds.append([1 if (i in bset and i < n - 1) else 0 for i in range(n)])
                perf.append([1 if ((i + 1) in {s for s in spikes} and i in gold) else 0 for i in range(n)])
            r = DS.score_dialogues(golds, preds); rc = DS.score_dialogues(golds, perf)
            ncand = sum(len(x[3]) for x in recs)
            print(f"{name:<12}{'exact':<7}{ncand:>6}{nnew:>6}{tp/max(nnew,1):>7.3f}{r['f1']:>8.3f}{r['score']:>8.3f}{rc['f1']:>8.3f}")
    print("  (참조: AMI deploy ±2F1 0.13 / DTS de-neut deploy F1 tiage .10 dlseg .17 super .09 / full-ctx LLM offline AMI Score .54)")


if __name__ == "__main__":
    main()
