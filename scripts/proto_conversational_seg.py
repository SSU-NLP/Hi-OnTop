#!/usr/bin/env python
"""프로토타입: conversational(대화 누적) streaming 분절 vs B-mode 120.
가설 = 매 청크를 작게 주고 대화 history 로 누적하면 긴 미팅 '후반부 붕괴'가 줄어든다.
한 모델(gpt-4o:nitro), AMI 18-subset, buffer=120s. conversational 은 fresh API(미팅내 순차)."""
from __future__ import annotations
import sys, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import secom_llm_eval as E

SUBSET = str(E.REPO / "outputs/runs/_misc/ami_latency_subset.json")
MODEL = "openrouter/openai/gpt-4o:nitro"
B = 120

SYS = (
    "You segment a multi-party meeting transcript into topic segments in a STREAMING fashion. "
    "The meeting arrives in consecutive chunks of turns, each turn tagged with a global index [Turn N]. "
    "For each new chunk, using all earlier chunks as context, identify the turns where a NEW topic begins "
    "(a topic boundary; the first turn of a meeting is not a boundary). "
    "Respond ONLY with a JSON list of integer global turn indices in the CURRENT chunk that start a new topic, "
    "e.g. [142, 167]. If no new topic begins in the chunk, respond []."
)


def fmt(turns, idxs):
    return "\n".join(f"[Turn {k}]: ({turns[k]['speaker']}) {turns[k]['text']}" for k in idxs)


def parse_ints(txt):
    m = re.search(r"\[[^\]]*\]", txt or "")
    if not m:
        return []
    try:
        return [int(x) for x in json.loads(m.group(0))]
    except Exception:
        return [int(x) for x in re.findall(r"\d+", m.group(0))]


def conv_segment(client, turns, n):
    cuts = E.seg_checkpoints(turns, n, B)
    msgs = [{"role": "system", "content": SYS}]
    pred = set()
    for kp, ki in cuts:
        idxs = list(range(kp, ki))
        msgs.append({"role": "user", "content": "New turns (continue segmenting):\n" + fmt(turns, idxs)})
        try:
            r = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=256,
                                               temperature=0.0, extra_body={"reasoning": {"enabled": False}})
            out = r.choices[0].message.content or ""
        except Exception as e:
            sys.stderr.write(f"[conv-fail] {type(e).__name__}: {str(e)[:150]}\n"); out = "[]"
        msgs.append({"role": "assistant", "content": out})
        for s in parse_ints(out):
            if kp <= s < ki and 0 < s < n:
                pred.add(s)
    return sorted(pred)


def bmode_preds(prompt):
    E.load_prompts(prompt); c = E.Caller(E.make_client(), MODEL, 1024); out = {}
    for mid, turns, n, gold in E.load_meetings(SUBSET):
        cuts = E.seg_checkpoints(turns, n, B); pred = set()
        for kp, ki in cuts:
            p = E.P_SEG.format(text_to_be_segmented=E.fmt_exchanges(turns, list(range(ki))))
            for s in E.extract_starts(c(p)):
                if 0 < s < n and kp <= s < ki:
                    pred.add(s)
        out[mid] = (sorted(pred), sorted(gold), n)
    return out


def tail(pred, gold, n):
    tg = [g for g in gold if g > n * 0.5]
    caught = sum(1 for g in tg if any(abs(g - p) <= 2 for p in pred))
    return len([p for p in pred if p > n * 0.5]), caught, len(tg)


if __name__ == "__main__":
    meetings = E.load_meetings(SUBSET)
    client = E.make_client()
    # conversational: 미팅 간 병렬, 미팅 내 순차
    conv = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(conv_segment, client, t, n): (mid, sorted(g), n) for mid, t, n, g in meetings}
        for fut in as_completed(futs):
            mid, gold, n = futs[fut]; conv[mid] = (fut.result(), gold, n)
    bm = bmode_preds("baseline")

    print(f"{'mid':9s} {'n':>5s} {'tailG':>5s} | B-mode120 tailpred/잡음 | conv120 tailpred/잡음 | overallF1 bm/conv")
    TG = Tbm = Tcv = Cbm = Ccv = 0
    f1bm = []; f1cv = []
    for mid in sorted(conv, key=lambda m: -conv[m][2]):
        cp, g, n = conv[mid]; bp = bm[mid][0]
        btp, bc, tg = tail(bp, g, n); ctp, cc, _ = tail(cp, g, n)
        TG += tg; Tbm += btp; Tcv += ctp; Cbm += bc; Ccv += cc
        fb = E.tol_f1(g, bp); fc = E.tol_f1(g, cp); f1bm.append(fb); f1cv.append(fc)
        print(f"{mid:9s} {n:5d} {tg:5d} | {btp:5d} / {bc:5d}          | {ctp:5d} / {cc:5d}         | {fb:.2f}/{fc:.2f}  (pred {len(bp)}/{len(cp)})")
    print(f"\nTOTAL 후반부 gold={TG}")
    print(f"  B-mode120  : tail_pred {Tbm}, 잡음 {Cbm} ({Cbm/TG*100:.0f}%)  | mean ±2F1 {np.mean(f1bm):.3f}")
    print(f"  conv120    : tail_pred {Tcv}, 잡음 {Ccv} ({Ccv/TG*100:.0f}%)  | mean ±2F1 {np.mean(f1cv):.3f}")
