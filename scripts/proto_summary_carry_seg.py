#!/usr/bin/env python
"""프로토타입 C: 요약-carryover streaming 분절 (SeCom 충실) vs B-mode 120.
매 청크 입력 = (직전까지 committed segment 들의 SeCom summary) + (새 청크 turns).
요약은 별도 콜 없이 SeCom 분절 출력의 summary 필드를 그대로 재사용 → 입력을 짧게(bounded) 유지.
한 모델(gpt-4o:nitro), AMI 18-subset, buffer=120s. quality(후반부/F1) + latency(입력크기·콜시간) 측정."""
from __future__ import annotations
import sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import secom_llm_eval as E

SUBSET = str(E.REPO / "outputs/runs/_misc/ami_latency_subset.json")
MODEL = "openrouter/openai/gpt-4o:nitro"
B = 120

INSTR = """You segment a multi-party meeting into topically coherent segments in a STREAMING fashion. \
Earlier parts of the meeting are already segmented; you are given brief SUMMARIES of those prior segments (oldest first) as context. \
Now segment ONLY the NEW turns below. A new turn may CONTINUE the latest prior topic or START a new topic. \
Each turn is tagged with a global index [Turn N]: (Speaker) text.

Output a single JSON object:
{{"first_continues_prior": true|false,  // does the FIRST new turn continue the latest prior topic?
  "segments": [ {{"start_turn_number": <global index of the first turn of this segment>, "summary": "<=40 words"}}, ... ]}}
Requirements: segments must cover ALL new turns contiguously with no gaps/overlap; the first segment's start_turn_number equals the first new turn's index; create a new segment only at a genuine topic shift; keep summaries concise.

# Prior segment summaries (oldest first)
{prior}

# New turns
{newturns}

# Output
Now output the JSON object only."""


def fmt(turns, idxs):
    return "\n".join(f"[Turn {k}]: ({turns[k]['speaker']}) {turns[k]['text']}" for k in idxs)


def parse(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None, []
    try:
        o = json.loads(m.group(0))
        cont = bool(o.get("first_continues_prior", True))
        segs = o.get("segments", []) or []
        starts = []
        for s in segs:
            v = s.get("start_turn_number")
            if v is not None and str(v) != "":
                starts.append((int(v), str(s.get("summary", ""))[:300]))
        return cont, starts
    except Exception:
        return None, []


def carry_segment(client, turns, n):
    cuts = E.seg_checkpoints(turns, n, B)
    prior = []        # carryover summaries
    pred = set()
    in_chars = []; lat = []
    for kp, ki in cuts:
        prompt = INSTR.format(prior=("\n".join(f"- {s}" for s in prior) or "(none yet)"),
                              newturns=fmt(turns, list(range(kp, ki))))
        in_chars.append(len(prompt))
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=512, temperature=0.0, extra_body={"reasoning": {"enabled": False}})
            out = r.choices[0].message.content or ""
        except Exception as e:
            sys.stderr.write(f"[carry-fail] {type(e).__name__}: {str(e)[:150]}\n"); out = ""
        lat.append(time.time() - t0)
        cont, starts = parse(out)
        if not starts:
            continue
        starts.sort()
        # 첫 new 청크가 새 topic 이면 kp 가 경계
        if cont is False and 0 < kp < n:
            pred.add(kp)
        for i, (s, _) in enumerate(starts):
            if i == 0:
                continue              # 첫 segment 시작=kp (continuation 여부는 위에서 처리)
            if kp < s < ki:
                pred.add(s)
        prior.extend(sm for _, sm in starts if sm)
    return sorted(pred), in_chars, lat


def bmode(prompt):
    E.load_prompts(prompt); c = E.Caller(E.make_client(), MODEL, 1024); out = {}
    for mid, turns, n, gold in E.load_meetings(SUBSET):
        cuts = E.seg_checkpoints(turns, n, B); pred = set(); inch = []
        for kp, ki in cuts:
            p = E.P_SEG.format(text_to_be_segmented=E.fmt_exchanges(turns, list(range(ki))))
            inch.append(len(p))
            for s in E.extract_starts(c(p)):
                if 0 < s < n and kp <= s < ki:
                    pred.add(s)
        out[mid] = (sorted(pred), sorted(gold), n, inch)
    return out


def tail(pred, gold, n):
    tg = [g for g in gold if g > n * 0.5]
    return len([p for p in pred if p > n * 0.5]), sum(1 for g in tg if any(abs(g - p) <= 2 for p in pred)), len(tg)


if __name__ == "__main__":
    meetings = E.load_meetings(SUBSET)
    client = E.make_client()
    carry = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(carry_segment, client, t, n): (mid, sorted(g), n) for mid, t, n, g in meetings}
        for fut in as_completed(futs):
            mid, gold, n = futs[fut]; p, inch, lat = fut.result(); carry[mid] = (p, gold, n, inch, lat)
    bm = bmode("baseline")

    print(f"{'mid':9s} {'n':>5s} {'tG':>3s} | Bmode tail잡음 F1 | carryC tail잡음 F1 | 입력char Bmax/Cmax | C콜시간p50/p95")
    TG = Cbm = Ccv = 0; f1b = []; f1c = []; allc = []
    for mid in sorted(carry, key=lambda m: -carry[m][2]):
        cp, g, n, inch, lat = carry[mid]; bp, _, _, binch = bm[mid]
        _, bc, tg = tail(bp, g, n); _, cc, _ = tail(cp, g, n)
        TG += tg; Cbm += bc; Ccv += cc
        fb = E.tol_f1(g, bp); fc = E.tol_f1(g, cp); f1b.append(fb); f1c.append(fc); allc += lat
        print(f"{mid:9s} {n:5d} {tg:3d} | {bc:5d} {fb:.2f}      | {cc:5d} {fc:.2f}       | {max(binch):6d}/{max(inch):5d}    | {np.percentile(lat,50):.1f}/{np.percentile(lat,95):.1f}s")
    print(f"\nTOTAL 후반부 gold={TG}")
    print(f"  B-mode120 : 후반부잡음 {Cbm} ({Cbm/TG*100:.0f}%)  mean ±2F1 {np.mean(f1b):.3f}")
    print(f"  carry-C   : 후반부잡음 {Ccv} ({Ccv/TG*100:.0f}%)  mean ±2F1 {np.mean(f1c):.3f}")
    bmax = max(max(bm[m][3]) for m in bm); cmax = max(max(carry[m][3]) for m in carry)
    print(f"  입력크기(char) 최대: B-mode {bmax} vs carry-C {cmax}  (C가 {bmax/cmax:.1f}× 작음)")
    print(f"  carry-C 콜시간: p50 {np.percentile(allc,50):.1f}s / p95 {np.percentile(allc,95):.1f}s / max {max(allc):.1f}s")
