#!/usr/bin/env python3
"""LLM 단일호출 latency 전수 측정 (N=1 per model, fresh, RESUME 지원).

각 모델 × 모든 실제 윈도우(=실 워크로드 prompt)에 대해 **개별 호출 latency** 를 측정.
- N=1 per model (모델 내부 순차 → 동시호출 오염 0), 모델끼리는 병렬(다른 endpoint).
- **fresh 호출** (응답 캐시 재사용 금지 — 캐시는 즉답이라 latency 가 0이 됨).
- latency 는 *성공한 create() round-trip* 만 측정(retry backoff 는 제외, 실패는 failure-rate 로 별도).
- **RESUME**: 결과를 즉시 checkpoint JSONL(append+flush)에 기록. 재시작 시 이미 측정한 (model,key)는 skip →
  끊긴 직전에서 바로 이어감. checkpoint 는 /workspace 마운트라 컨테이너 재생성에도 보존.
usage:
  python scripts/llm_latency_full.py            # 측정(이어하기)
  python scripts/llm_latency_full.py --report    # checkpoint 집계(모델별 p50/p95/p99)
handoff HANDOFF_0612 §3z-C(latency). 6모델 확정셋.
"""
from __future__ import annotations
import sys, os, json, time, hashlib, threading, argparse, random
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
TOPIC = REPO / "data" / "ami" / "topic"
CKPT = REPO / "outputs" / "runs" / "_misc" / "latency_full.jsonl"

# ⚠️ DEPRECATED: home-made WIN_PROMPT → 실제 워크로드와 불일치. 정식 latency = secom_latency_full.py.
# 모델셋만 2026-06-12 확정본으로 동기화(stale 방지).
MODELS = ["openrouter/openai/gpt-5-mini", "openrouter/openai/gpt-5-nano",
          "openrouter/qwen/qwen3.5-27b", "openrouter/anthropic/claude-haiku-4.5",
          "openrouter/mistralai/mistral-small-3.1-24b-instruct",
          "openrouter/google/gemma-3-12b-it", "openrouter/google/gemma-3n-e4b-it"]

WIN_PROMPT = """These are consecutive utterances from a meeting (a short window of the conversation).
List the utterance numbers where a NEW topic clearly begins within this window.
If no clear topic change, return [].
Return ONLY a JSON array, e.g. [12] or [].

Utterances:
{block}

JSON array:"""


def build_blocks(subset_path, buffers, offsets):
    """실 워크로드: 모든 (B,offset) 윈도우 block(중복 제거). 반환 [(key, block, n_turns)]."""
    mids = sorted(json.load(open(subset_path)))
    seen = {}
    for mid in mids:
        d = json.load(open(TOPIC / f"{mid}.json")); turns = d["turns"]; n = len(turns)
        if n < 2:
            continue
        t0 = turns[0]["start"]; t_last = turns[-1]["start"]
        for B in buffers:
            for off in offsets:
                w_start = t0; w_end = t0 + (off * B if off > 0 else B)
                while w_start <= t_last:
                    w = [k for k in range(n) if w_start <= turns[k]["start"] < w_end]
                    if len(w) >= 2:
                        block = "\n".join(f"[{k}] ({turns[k]['speaker']}) {turns[k]['text']}" for k in w)
                        h = hashlib.md5(block.encode()).hexdigest()
                        if h not in seen:
                            seen[h] = (block, len(w))
                    w_start = w_end; w_end += B
    return [(h, b, nt) for h, (b, nt) in seen.items()]


def make_client():
    cf = ({"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
           "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]}
          if os.environ.get("CF_ACCESS_CLIENT_ID") and os.environ.get("CF_ACCESS_CLIENT_SECRET") else None)
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                  default_headers=cf)


def load_done():
    """checkpoint 에서 이미 측정한 (model,blockhash) 집합 (성공만 done)."""
    done = set()
    if CKPT.exists():
        for line in open(CKPT):
            try:
                o = json.loads(line)
                if o.get("ok"):
                    done.add((o["model"], o["h"]))
            except Exception:
                pass
    return done


def report():
    by = {}
    for line in open(CKPT):
        try:
            o = json.loads(line)
        except Exception:
            continue
        by.setdefault(o["model"], {"lat": [], "fail": 0, "ret": 0})
        if o.get("ok"):
            by[o["model"]]["lat"].append(o["lat_ms"]); by[o["model"]]["ret"] += o.get("retries", 0)
        else:
            by[o["model"]]["fail"] += 1
    print(f"{'model':<46}{'n':>6}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}{'fail':>6}")
    for m in MODELS:
        if m not in by:
            print(f"{m:<46}{'(미측정)':>6}"); continue
        a = np.array(by[m]["lat"])
        if len(a) == 0:
            print(f"{m:<46}{0:>6}  all-fail {by[m]['fail']}"); continue
        print(f"{m.replace('openrouter/',''):<46}{len(a):>6}{np.percentile(a,50):>8.0f}"
              f"{np.percentile(a,95):>8.0f}{np.percentile(a,99):>8.0f}{a.max():>8.0f}{by[m]['fail']:>6}  (ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(REPO / "outputs/runs/_misc/ami_subset.json"))
    ap.add_argument("--buffers", default="10,30,60,120")
    ap.add_argument("--offsets", default="0,0.5")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(); return

    models = args.models.split(",")
    bufs = [int(x) for x in args.buffers.split(",")]; offs = [float(x) for x in args.offsets.split(",")]
    blocks = build_blocks(args.subset, bufs, offs)
    done = load_done()
    client = make_client()
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    fout = open(CKPT, "a"); lock = threading.Lock()
    prog = {m: [0, 0] for m in models}     # [done_now, fail]
    total = len(blocks)
    print(f"윈도우 {total}개 × {len(models)}모델. 이미 측정 {len(done)}건. RESUME 가능.", flush=True)

    def timed_call(model, block):
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                r = client.chat.completions.create(model=model,
                    messages=[{"role": "user", "content": WIN_PROMPT.format(block=block)}],
                    max_tokens=256, temperature=0.0, extra_body={"reasoning": {"enabled": False}})
                return (time.perf_counter() - t0) * 1000, True, attempt
            except Exception:
                if attempt < 3:
                    time.sleep((2 ** attempt) * 0.5 + random.random())
        return None, False, 3

    def run_model(model):
        t_start = time.perf_counter()
        for h, block, nt in blocks:
            if (model, h) in done:
                continue
            lat, ok, nr = timed_call(model, block)
            rec = {"model": model, "h": h, "ok": ok, "lat_ms": lat, "retries": nr, "n_turns": nt,
                   "n_chars": len(block)}
            with lock:
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                prog[model][0] += 1
                if not ok:
                    prog[model][1] += 1
                if prog[model][0] % 500 == 0:
                    el = time.perf_counter() - t_start
                    print(f"  [{model.split('/')[-1]}] {prog[model][0]} 측정 ({el:.0f}s, "
                          f"{prog[model][0]/max(1e-9,el):.2f}/s, fail {prog[model][1]})", flush=True)

    with __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        for m in models:
            ex.submit(run_model, m)
    print("=== 측정 완료 ===", flush=True); report()


if __name__ == "__main__":
    main()
