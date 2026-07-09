#!/usr/bin/env python
"""HANDOFF_02 재채점 — 원칙적 -1 정렬(HANDOFF_04: gold 끝-turn vs LLM start-turn) 반영.
캐시 전용(API 콜 없음 — miss 는 빈응답 처리 + 카운트). baseline/verbatim × {full,120,60} × 7모델, AMI 18-subset.
shift0(구) vs shift-1(원칙) Score 델타도 같이 출력. v2(csec) 는 secom_cache 미저장이라 제외."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import secom_llm_eval as E

SUBSET = str(E.REPO / "outputs/runs/_misc/ami_latency_subset.json")
MODELS = ["openrouter/qwen/qwen3.5-27b", "openrouter/openai/gpt-4o:nitro",
          "openrouter/openai/gpt-4o-mini", "openrouter/anthropic/claude-haiku-4.5",
          "openrouter/mistralai/mistral-small-3.1-24b-instruct",
          "openrouter/google/gemma-3-12b-it", "openrouter/google/gemma-3n-e4b-it"]
BUFFERS = ["full", 120, 60]


def cache_only_caller(model):
    """Caller 캐시만 사용. miss 시 빈응답 반환 + miss 카운트(API 콜 절대 안 함)."""
    c = E.Caller(E.make_client(), model, 1024)
    state = {"hit": 0, "miss": 0}
    import hashlib

    def call(prompt, response_format=None):
        key = hashlib.md5((model + "||" + (response_format and "rf||" or "") + prompt).encode()).hexdigest()
        if key in c.cache:
            state["hit"] += 1; return c.cache[key]
        state["miss"] += 1; return ""
    return call, state


def score(model, prompt, B, meetings):
    E.load_prompts(prompt)
    caller, state = cache_only_caller(model)
    rows = E.run_segment(meetings, caller, ("full" if B == "full" else int(B)), [0.0], 1, "B")
    a = np.array([r[:5] for r in rows])
    return a, state


# 완전 캐시된 config 만(MISS 0) — shift0 vs shift-1 델타 정직 비교용
FULLY_CACHED = [("baseline", "openrouter/qwen/qwen3.5-27b"),
                ("baseline", "openrouter/openai/gpt-4o:nitro"),
                ("verbatim", "openrouter/openai/gpt-4o:nitro")]


if __name__ == "__main__":
    meetings = E.load_meetings(SUBSET)
    print(f"AMI 18-subset, {len(meetings)}미팅 | 채점 = ±2-tol F1 + nltk Pk/WD, 원칙적 -1 정렬(baked in metrics)")
    print("=" * 96)
    for prompt in ["baseline", "verbatim"]:
        print(f"\n### {prompt}  (v1=baseline / verbatim=SeCom원본)")
        print(f"{'model':40s} {'buf':>4s} | {'F1':>5s} {'Pk':>5s} {'WD':>5s} {'Score':>6s} | cache hit/miss")
        for model in MODELS:
            for B in BUFFERS:
                a, st = score(model, prompt, B, meetings)
                tot = st["hit"] + st["miss"]
                flag = "" if st["miss"] == 0 else f"  ⚠ {st['miss']}/{tot} MISS(빈응답 채점)"
                print(f"{model.split('/')[-1]:40s} {str(B):>4s} | {a[:,0].mean():.3f} {a[:,2].mean():.3f} "
                      f"{a[:,3].mean():.3f} {a[:,4].mean():.3f} | {st['hit']}/{tot}{flag}", flush=True)

    print("\n" + "=" * 96)
    print("### shift0(구) vs shift-1(원칙) Score 델타 — 완전캐시 config 만")
    print(f"{'config':48s} {'buf':>4s} | {'shift0':>7s} {'shift-1':>7s} {'Δ':>7s}")
    for prompt, model in FULLY_CACHED:
        for B in BUFFERS:
            E.PRED_SHIFT = 0;  a0, _ = score(model, prompt, B, meetings)
            E.PRED_SHIFT = -1; a1, _ = score(model, prompt, B, meetings)
            s0, s1 = a0[:, 4].mean(), a1[:, 4].mean()
            print(f"{prompt+' '+model.split('/')[-1]:48s} {str(B):>4s} | {s0:7.3f} {s1:7.3f} {s1-s0:+7.3f}")
    E.PRED_SHIFT = -1
