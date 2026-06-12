#!/usr/bin/env python3
"""실험2 — LLM 버퍼-폴링 분절. 버퍼 B초로 제한한 윈도우만 LLM에 주고 경계 검출.
버퍼 작을수록 문맥 부족 → 성능↓. Hi-OnTop(0버퍼) 대비 trade-off 곡선용."""
from __future__ import annotations
import json, os, re, sys, argparse
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO/".env")
TOPIC = REPO/"data"/"ami"/"topic"

WIN_PROMPT = """These are consecutive utterances from a meeting (a short window of the conversation).
List the utterance numbers where a NEW topic clearly begins within this window.
If no clear topic change, return [].
Return ONLY a JSON array, e.g. [12] or [].

Utterances:
{block}

JSON array:"""


def parse_bounds(txt, valid):
    for cand in reversed(re.findall(r"\[[\d,\s]*\]", txt or "")):
        try:
            arr = json.loads(cand)
            return sorted({int(x) for x in arr if int(x) in valid})
        except Exception:
            continue
    return []


def tol_f1(gold, pred, tol):
    if not pred or not gold: return 0.0
    pr = sum(1 for i in pred if any(abs(i-j) <= tol for j in gold))/len(pred)
    rc = sum(1 for j in gold if any(abs(i-j) <= tol for i in pred))/len(gold)
    return 2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0


def official_pk_wd(yt, yp):
    """공식 Pk/WD (nltk, k=auto) — run_encoder_comparison.official_pk_wd 와 동일."""
    from nltk.metrics import pk as _pk, windowdiff as _wd
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_pk(ts, ps, k=k)), float(_wd(ts, ps, k=k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/qwen/qwen3.5-27b")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--subset", default=None, help="meeting id 리스트 json (있으면 limit 무시)")
    ap.add_argument("--buffers", default="10,30,60,120")
    ap.add_argument("--offsets", default="0,0.5", help="chunk offset 분수(B*frac), 메트릭 평균 (codex must)")
    ap.add_argument("--workers", type=int, default=10, help="동시 LLM 호출 수 (concurrency)")
    args = ap.parse_args()
    _cf = ({"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
            "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]}
           if os.environ.get("CF_ACCESS_CLIENT_ID") and os.environ.get("CF_ACCESS_CLIENT_SECRET") else None)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                    default_headers=_cf)
    bufs = [int(x) for x in args.buffers.split(",")]
    offs = [float(x) for x in args.offsets.split(",")]
    if args.subset:
        mids = sorted(json.load(open(args.subset)))
    else:
        mids = sorted(m["meeting"] for m in json.load(open(TOPIC/"manifest.json")))[:args.limit]

    import time as _t, hashlib, random, threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import defaultdict
    cache_path = REPO/"outputs"/"runs"/"_misc"/f"llm_cache_{args.model.replace('/','_')}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            try:
                o = json.loads(line); cache[o["k"]] = o["t"]
            except Exception:
                pass
    cf = open(cache_path, "a"); lock = threading.Lock(); errs = [0]

    def _call(block):                                  # 캐시 + retry/backoff
        key = hashlib.md5((args.model + "||" + block).encode()).hexdigest()
        with lock:
            if key in cache:
                return cache[key]
        txt = ""
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=args.model,
                    messages=[{"role": "user", "content": WIN_PROMPT.format(block=block)}],
                    max_tokens=256, temperature=0.0, extra_body={"reasoning": {"enabled": False}})
                msg = r.choices[0].message
                txt = msg.content or getattr(msg, "reasoning_content", None) or ""
                break
            except Exception:
                if attempt == 3:
                    with lock:
                        errs[0] += 1
                else:
                    _t.sleep((2 ** attempt) * 0.5 + random.random())
        with lock:
            cache[key] = txt; cf.write(json.dumps({"k": key, "t": txt}) + "\n"); cf.flush()
        return txt

    # 전 (B,mid,offset) 윈도우 task 수집
    meta = {}
    for mid in mids:
        d = json.load(open(TOPIC/f"{mid}.json"))
        meta[mid] = (d["turns"], d["bnd_top"], len(d["turns"]),
                     [i for i, b in enumerate(d["bnd_top"]) if b == 1])
    plan = []                                          # (B, mid, off, valid_set, block)
    for B in bufs:
        for mid in mids:
            turns, bt, n, gold = meta[mid]; t0 = turns[0]["start"]; t_last = turns[-1]["start"]
            for off in offs:
                w_start = t0; w_end = t0 + (off * B if off > 0 else B)
                while w_start <= t_last:
                    w = [k for k in range(n) if w_start <= turns[k]["start"] < w_end]
                    if len(w) >= 2:
                        block = "\n".join(f"[{k}] ({turns[k]['speaker']}) {turns[k]['text']}" for k in w)
                        plan.append((B, mid, off, set(w[1:]), block))
                    w_start = w_end; w_end += B
    uniq = list({p[4] for p in plan})
    print(f"총 윈도우 {len(plan)} (unique {len(uniq)}), 동시성 {args.workers}, 캐시 {len(cache)}건", flush=True)

    resp = {}; t_start = _t.perf_counter(); done = [0]
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(_call, b): b for b in uniq}
        for fut in as_completed(futs):
            resp[futs[fut]] = fut.result(); done[0] += 1
            if done[0] % 200 == 0:
                rate = done[0] / max(1e-9, _t.perf_counter() - t_start)
                print(f"  {done[0]}/{len(uniq)} ({_t.perf_counter()-t_start:.0f}s, {rate:.1f}/s, err {errs[0]})", flush=True)

    pred_by = defaultdict(set)
    for B, mid, off, valid, block in plan:
        pred_by[(B, mid, off)].update(parse_bounds(resp.get(block, ""), valid))
    results = {}
    for B in bufs:
        ex0, f2, SC = [], [], []; npred = 0; ngold = 0
        for mid in mids:
            turns, bt, n, gold = meta[mid]; yt = [1 if k in set(gold) else 0 for k in range(n)]
            of2, oex, osc, onp = [], [], [], []
            for off in offs:
                pred = sorted(pred_by[(B, mid, off)])
                yp = [1 if k in set(pred) else 0 for k in range(n)]
                _f2 = tol_f1(gold, pred, 2); pk, wd = official_pk_wd(yt, yp)
                of2.append(_f2); oex.append(f1_score(yt, yp, zero_division=0))
                osc.append(0.5 * _f2 + 0.25 * (1 - pk) + 0.25 * (1 - wd)); onp.append(len(pred))
            f2.append(np.mean(of2)); ex0.append(np.mean(oex)); SC.append(np.mean(osc))
            npred += int(np.mean(onp)); ngold += len(gold)
        results[B] = (np.mean(ex0), np.mean(f2), np.mean(SC), npred, ngold, len(uniq))
        print(f"  buffer={B}s: exactF1={np.mean(ex0):.3f} ±2F1={np.mean(f2):.3f} Score={np.mean(SC):.3f} "
              f"pred={npred}/{ngold}", flush=True)
    print(f"[총] unique calls {len(uniq)}, errors {errs[0]}, {_t.perf_counter()-t_start:.0f}s", flush=True)

    print(f"\n=== 실험2 버퍼 곡선 ({args.model}, {len(mids)}미팅) ===")
    print(f"  {'buffer':>8} {'exactF1':>8} {'±2F1':>7} {'Score':>7}")
    for B in bufs:
        e, f, sc, *_ = results[B]
        print(f"  {B:>6}s {e:>8.3f} {f:>7.3f} {sc:>7.3f}")


if __name__ == "__main__":
    main()
