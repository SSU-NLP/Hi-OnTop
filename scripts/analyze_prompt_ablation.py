#!/usr/bin/env python
"""SeCom prompt ablation 진단 (baseline vs verbatim, gpt-4o:nitro, AMI 18-subset).
캐시된 LLM 응답만 재사용(추가 API 0). buffer=full(offline) 기준 실제 예측 경계를 추출해
precision/recall · 경계 변위 · tolerance curve · 프롬프트 간 일치도를 계산한다."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import secom_llm_eval as E

SUBSET = str(E.REPO / "outputs/runs/_misc/ami_latency_subset.json")
MODEL = "openrouter/openai/gpt-4o:nitro"


def preds_full(prompt: str) -> dict:
    """buffer=full(전체 1콜) 예측. 캐시 히트만 사용."""
    E.load_prompts(prompt)
    caller = E.Caller(E.make_client(), MODEL, 1024)
    out = {}
    for mid, turns, n, gold in E.load_meetings(SUBSET):
        p = E.P_SEG.format(text_to_be_segmented=E.fmt_exchanges(turns, list(range(n))))
        txt = caller(p)
        pred = sorted({s for s in E.extract_starts(txt) if 0 < s < n})
        out[mid] = (pred, sorted(gold), n)
    return out


def nearest(a, bs):
    return min((abs(a - b) for b in bs), default=None)


def diagnose(name, data):
    npred = ngold = 0
    dp = []  # 각 pred → 가장 가까운 gold 거리
    dg = []  # 각 gold → 가장 가까운 pred 거리
    far = 0  # gold 와 5 turn 초과로 떨어진 pred (= 노이즈 경계)
    for mid, (pred, gold, n) in data.items():
        npred += len(pred); ngold += len(gold)
        for p in pred:
            d = nearest(p, gold)
            if d is not None:
                dp.append(d)
                if d > 5:
                    far += 1
        for g in gold:
            d = nearest(g, pred)
            if d is not None:
                dg.append(d)
    dp = np.array(dp); dg = np.array(dg)
    # tolerance curve (micro F1 over all meetings)
    print(f"\n===== {name} =====")
    print(f"pred={npred}  gold={ngold}  pred/gold={npred/ngold:.2f}")
    print(f"pred→gold 거리: median={np.median(dp):.1f} mean={dp.mean():.1f}  "
          f"(≤2:{(dp<=2).mean()*100:.0f}%  ≤5:{(dp<=5).mean()*100:.0f}%  >5(noise):{far} = {far/npred*100:.0f}%)")
    print(f"gold→pred 거리: median={np.median(dg):.1f} mean={dg.mean():.1f}  "
          f"(잡힘 ≤2:{(dg<=2).mean()*100:.0f}%  ≤5:{(dg<=5).mean()*100:.0f}%  놓침 >5:{(dg>5).mean()*100:.0f}%)")
    # tolerance F1 curve (micro)
    line = "tolerance F1: "
    for t in [0, 1, 2, 3, 5]:
        # micro P/R over all meetings
        tp_p = sum(1 for mid, (pred, gold, n) in data.items() for p in pred if nearest(p, gold) is not None and nearest(p, gold) <= t)
        tp_r = sum(1 for mid, (pred, gold, n) in data.items() for g in gold if nearest(g, pred) is not None and nearest(g, pred) <= t)
        P = tp_p / npred; R = tp_r / ngold
        f1 = 2 * P * R / (P + R) if P + R else 0
        line += f"±{t}={f1:.3f}({P:.2f}/{R:.2f})  "
    print(line + " [F1(P/R)]")


def compare(base, verb):
    """프롬프트 간 경계 일치도."""
    print("\n===== baseline vs verbatim 경계 일치 =====")
    both = same = within2 = b_only = v_only = 0
    for mid in base:
        bp = set(base[mid][0]); vp = set(verb[mid][0])
        same += len(bp & vp)
        within2 += sum(1 for p in bp if any(abs(p - q) <= 2 for q in vp))
        b_only += len(bp - vp); v_only += len(vp - bp)
        both += len(bp) + len(vp)
    print(f"정확히 동일 위치 경계: {same}  |  ±2 내 일치(baseline 기준): {within2}/{sum(len(base[m][0]) for m in base)}")
    print(f"baseline 총 {sum(len(base[m][0]) for m in base)} · verbatim 총 {sum(len(verb[m][0]) for m in verb)}")


if __name__ == "__main__":
    base = preds_full("baseline")
    verb = preds_full("verbatim")
    diagnose("baseline (full)", base)
    diagnose("verbatim (full)", verb)
    compare(base, verb)
    # 예시 미팅 1개 (최장)
    mid = max(base, key=lambda m: base[m][2])
    print(f"\n===== 예시 미팅 {mid} (n={base[mid][2]} turns) =====")
    print(f"gold     : {base[mid][1]}")
    print(f"baseline : {base[mid][0]}")
    print(f"verbatim : {verb[mid][0]}")
