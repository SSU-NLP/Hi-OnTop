#!/usr/bin/env python3
"""AMI 전체 139미팅 LLM full-context 분절 — 표(±2 F1 + Pk/WD)용 LLM 행.
예측을 미팅별 캐시(outputs/runs/_misc/ami_llm_pred/{mid}.json)에 저장 → 죽어도 resume.
병렬 호출(ThreadPool)."""
from __future__ import annotations
import json, os, re, sys, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"scripts"))
load_dotenv(REPO/".env")
TOPIC = REPO/"data"/"ami"/"topic"
PRED = REPO/"outputs"/"runs"/"_misc"/"ami_llm_pred"
PRED.mkdir(parents=True, exist_ok=True)
from run_encoder_comparison import official_pk_wd

PROMPT = """You segment a meeting transcript into topic segments.
Below are numbered utterances from one meeting. A topic boundary is an utterance
that STARTS a new discussion topic (e.g. moving from introductions to project goals,
or from budget to design). Utterance 0 is never a boundary.

Return ONLY a JSON array of the utterance numbers that start a new topic, e.g. [8, 14, 54].
Typical meetings have only 5-12 topic boundaries. Do not over-segment.

Transcript:
{transcript}

JSON array of boundary utterance numbers:"""


def parse_bounds(txt, n):
    for cand in reversed(re.findall(r"\[[\d,\s]*\]", txt or "")):
        try:
            arr = json.loads(cand)
            if arr:
                return sorted({int(x) for x in arr if 0 < int(x) < n})
        except Exception:
            continue
    return []


def tol_f1(gold, pred, t=2):
    if not pred or not gold: return 0.0
    p = sum(1 for i in pred if any(abs(i-j) <= t for j in gold))/len(pred)
    r = sum(1 for j in gold if any(abs(i-j) <= t for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r > 0 else 0.0


def ask_one(client, model, mid):
    cache = PRED/f"{mid}.json"
    if cache.exists():
        return mid, json.load(open(cache))["pred"]
    d = json.load(open(TOPIC/f"{mid}.json")); turns = d["turns"]; n = len(turns)
    transcript = "\n".join(f"[{i}] ({t['speaker']}) {t['text']}" for i, t in enumerate(turns))
    try:
        r = client.chat.completions.create(model=model,
            messages=[{"role": "user", "content": PROMPT.format(transcript=transcript)}],
            max_tokens=400, temperature=0.0, extra_body={"reasoning": {"enabled": False}})
        msg = r.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", None) or ""
        pred = parse_bounds(content, n)
    except Exception as ex:
        print(f"  {mid}: ERROR {ex}", flush=True); return mid, None
    json.dump({"pred": pred}, open(cache, "w"))
    return mid, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/qwen/qwen3.5-27b")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    mids = sorted(p.name[:-5] for p in TOPIC.glob("*.json") if p.name != "manifest.json")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask_one, client, args.model, m): m for m in mids}
        for fut in as_completed(futs):
            mid, pred = fut.result(); done += 1
            print(f"  [{done}/{len(mids)}] {mid}: pred={len(pred) if pred else 0}", flush=True)

    # --- 집계 (±2 F1 + Pk + WD + Score) ---
    F2, PK, WD, SC = [], [], [], []; npred = 0; ngold = 0
    for mid in mids:
        c = PRED/f"{mid}.json"
        if not c.exists(): continue
        pred = json.load(open(c))["pred"]
        d = json.load(open(TOPIC/f"{mid}.json")); bt = list(d["bnd_top"]); n = len(bt); bt[-1] = 0
        gold = [i for i, b in enumerate(bt) if b == 1]
        pred = [p for p in pred if 0 <= p < n and p != n-1]
        npred += len(pred); ngold += len(gold)
        f2 = tol_f1(gold, pred, 2)
        yt = [int(b) for b in bt]; yp = [1 if i in set(pred) else 0 for i in range(n)]
        pk, wd = official_pk_wd(yt, yp)
        F2.append(f2); PK.append(pk); WD.append(wd); SC.append(0.5*f2+0.25*(1-pk)+0.25*(1-wd))
    print(f"\n=== LLM full-context ({args.model}), {len(F2)}미팅 ===")
    print(f"  pred={npred} gold={ngold}")
    print(f"  F1(±2)={np.mean(F2):.3f}  Pk={np.mean(PK):.3f}  WD={np.mean(WD):.3f}  Score={np.mean(SC):.3f}")


if __name__ == "__main__":
    main()
