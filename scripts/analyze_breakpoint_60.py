#!/usr/bin/env python
"""v1(B-mode) 60버퍼 누적 붕괴선 분석 — N=1 직렬, per-checkpoint 저장.
각 체크포인트마다 (prefix turn수/시간, 그 콜의 fair latency, 새 구간 예측경계, 새 구간 gold) 기록 →
prefix-시간별 recall + latency 동시 → 'score·latency 둘 다 양호한 sweet spot' 도출.
HANDOFF_03 원칙: N=1 직렬(공정 latency) + per-meeting 예측 저장(재실행 방지). 캐시 미사용(fresh)."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import secom_llm_eval as E

SUBSET = str(E.REPO / "outputs/runs/_misc/ami_latency_subset.json")
MODEL = "openrouter/qwen/qwen3.5-27b"
B = 60
OUTDIR = E.REPO / "outputs/experiments/2026-06-13_v1-breakpoint-60_ami18"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "checkpoint_records.json"

if __name__ == "__main__":
    E.load_prompts("baseline")  # v1
    client = E.make_client()
    meetings = E.load_meetings(SUBSET)
    recs = []
    for mid, turns, n, gold in meetings:
        cuts = E.seg_checkpoints(turns, n, B)
        for kp, ki in cuts:
            prompt = E.P_SEG.format(text_to_be_segmented=E.fmt_exchanges(turns, list(range(ki))))
            t0 = time.time()
            try:
                r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                   max_tokens=1024, temperature=0.0, timeout=120,
                                                   extra_body={"reasoning": {"enabled": False}})
                out = r.choices[0].message.content or ""
            except Exception as e:
                sys.stderr.write(f"[fail] {type(e).__name__}: {str(e)[:120]}\n"); out = ""
            lat = time.time() - t0
            preds = sorted(s for s in E.extract_starts(out) if kp <= s < ki and 0 < s < n)
            region_gold = sorted(g for g in gold if kp <= g < ki)
            recs.append({"mid": mid, "kp": kp, "ki": ki, "prefix_n": ki,
                         "prefix_time_min": round(turns[ki - 1]["start"] / 60.0, 2),
                         "latency_s": round(lat, 2), "preds": preds, "region_gold": region_gold})
            OUT.write_text(json.dumps(recs, ensure_ascii=False))  # 증분 저장
    # 분석: 미팅별 전체 pred(±2 매칭용) + prefix-시간 bin
    by_mid = {}
    for r in recs:
        by_mid.setdefault(r["mid"], []).extend(r["preds"])
    bins = {}  # 2분 bin -> [gold, caught, lat합, 콜수]
    for r in recs:
        allpred = by_mid[r["mid"]]
        b = int(r["prefix_time_min"] // 2) * 2
        bins.setdefault(b, [0, 0, 0.0, 0])
        bins[b][2] += r["latency_s"]; bins[b][3] += 1
        for g in r["region_gold"]:
            bins[b][0] += 1
            bins[b][1] += int(any(abs(g - p) <= 2 for p in allpred))
    print("prefix시간(분) | gold | 잡힘 | recall | mean_lat(s) | 콜수   [v1 B-mode 60, qwen, N=1]")
    for b in sorted(bins):
        g, ca, lt, nc = bins[b]
        rec = f"{ca/g*100:4.0f}%" if g else "  -"
        print(f"  {b:2d}-{b+2:2d}분 | {g:3d} | {ca:3d} | {rec} | {lt/nc:6.1f} | {nc}")
    json.dump(bins, open(OUTDIR / "breakpoint_bins.json", "w"))
    print(f"\n저장: {OUT} (체크포인트 {len(recs)}개), {OUTDIR/'breakpoint_bins.json'}")
