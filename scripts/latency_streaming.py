#!/usr/bin/env python3
"""Hi-OnTop live-streaming latency — 공정 조건(캐시·전처리·batch 금지).

발화가 도착하는 대로 turn 단위 live encode(batch=1, no cache) + segment step 을 재현.
보고: cold-start(모델 로드 + 첫 임베딩) vs warm steady-state(p50/p95/p99/max), segment step,
turn 총 latency, RTF(처리시간/audio길이). subset(ami_subset.json) 사용.
codex 종합감사 must-fix(tail/cold-warm/RTF) 반영. handoff HANDOFF_0612 §3z-B/C.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")

TOPIC = "data/ami/topic"


def main():
    sub = json.load(open("outputs/runs/_misc/ami_subset.json"))
    meetings = []
    for mid in sub:
        d = json.load(open(f"{TOPIC}/{mid}.json"))
        meetings.append((mid, [t["text"] for t in d["turns"]], [t["start"] for t in d["turns"]]))

    # === cold-start: 모델 로드 + 첫 임베딩 ===
    t0 = time.perf_counter()
    from run_encoder_comparison import _encoder
    enc = _encoder("minilm-int8")             # 모델 로드(첫 호출 시)
    load_ms = (time.perf_counter() - t0) * 1000
    first_text = meetings[0][1][0]
    t0 = time.perf_counter(); enc.encode([first_text], normalize_embeddings=True)
    cold_first_ms = (time.perf_counter() - t0) * 1000

    # === warmup ===
    for txt in meetings[0][1][:20]:
        enc.encode([txt], normalize_embeddings=True)

    # === warm steady-state: turn 단위 live encode latency (전 subset) ===
    from hi_ontop.hi_ontop_deneut import _deneut, _beta, _nr   # segment step 비용 측정용
    enc_ms = []           # per-turn encode latency
    seg_ms = []           # per-turn segment-step latency
    rtfs = []
    lam = 0.6; g_rho = 0.15
    for mid, texts, starts in meetings:
        # streaming 재현: 한 발화씩 encode + 1-step segment 갱신
        m = None; g = None; k = 1; Wm = WM2 = 0.0; Wn = 0
        tot_proc = 0.0
        for i, txt in enumerate(texts):
            t0 = time.perf_counter()
            x = np.asarray(enc.encode([txt], normalize_embeddings=True)[0], dtype=np.float64)
            te = (time.perf_counter() - t0) * 1000
            # segment 1-step (threshold reset 의 핵심 연산)
            t1 = time.perf_counter()
            if m is None:
                m = x.copy(); g = x.copy()
            else:
                b = _beta(k, 2.0, 1.0, 8)
                V = (1 - float(_deneut(x, g, b) @ _deneut(m, g, b))) - lam * (1 - float(x @ g))
                sd = (WM2 / (Wn - 1)) ** 0.5 if Wn > 1 else 0.0
                thr = Wm + 1.0 * sd if Wn >= 8 else None
                if thr is not None and V > thr and k >= 4:
                    m = x.copy(); k = 1
                else:
                    Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
                    rho = max(0.05, 1.0 / (k + 1)); m = _nr((1 - rho) * m + rho * x); k += 1
                gr = max(g_rho, 1.0 / (i + 1)); g = _nr((1 - gr) * g + gr * x)
            ts = (time.perf_counter() - t1) * 1000
            enc_ms.append(te); seg_ms.append(ts); tot_proc += (te + ts) / 1000.0
        dur = max(starts) - min(starts)       # audio 길이(초) 근사 = start span
        if dur > 0:
            rtfs.append(tot_proc / dur)

    e = np.array(enc_ms); s = np.array(seg_ms); tot = e + s; r = np.array(rtfs)

    def pc(a):
        return f"p50 {np.percentile(a,50):.3f} | p95 {np.percentile(a,95):.3f} | p99 {np.percentile(a,99):.3f} | max {a.max():.2f}"
    print(f"=== Hi-OnTop live-streaming latency (subset {len(meetings)}미팅, {len(e)} turns, batch=1, no cache) ===")
    print(f"[cold-start]  모델 로드 {load_ms:.0f}ms + 첫 임베딩 {cold_first_ms:.1f}ms = {load_ms+cold_first_ms:.0f}ms")
    print(f"[warm encode/turn ms]   {pc(e)}   mean {e.mean():.3f}")
    print(f"[warm segment/turn ms]  {pc(s)}   mean {s.mean():.4f}")
    print(f"[warm total/turn ms]    {pc(tot)}   mean {tot.mean():.3f}")
    print(f"[RTF per meeting]       median {np.median(r):.5f} | p95 {np.percentile(r,95):.5f} | max {r.max():.5f}")
    print(f"  → RTF<<1 (실시간의 {1/np.median(r):.0f}배 빠름, 안 밀림). audio span 기준.")


if __name__ == "__main__":
    main()
