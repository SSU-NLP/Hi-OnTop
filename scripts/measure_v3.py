#!/usr/bin/env python
"""v3 측정 — raw 슬라이딩 ≤4분 윈도우 (HANDOFF_03 §1.6).
입력 = (현재 청크 직전 누적 ≤4분 raw turns) + (현재 청크). 요약·merge 없음. v1/v2 와 비교용.
18-subset qwen-27b, buffer 120/60, **N=1 직렬 + latency + per-meeting 예측 저장**
([[feedback_handoff03_n1_latency_predictions]] 원칙: 재채점·붕괴선 분석 재실행 방지)."""
from __future__ import annotations
import sys, json, re, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import secom_llm_eval as E

SUBSET = str(E.REPO / "outputs/runs/_misc/ami_latency_subset.json")
MODEL = "openrouter/qwen/qwen3.5-27b"
WINDOW_SEC = 240   # 직전 ≤4분 raw 윈도우 (§1.5 sweet spot)
V3 = (E.REPO / "scripts/segmentation_prompts/baseline_segment_v3.md").read_text()
OUTDIR = E.REPO / "outputs/experiments/2026-06-13_v3_carryover_ami18"


def call(client, prompt, mtok=600):
    t0 = time.time()
    try:
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                           max_tokens=mtok, temperature=0.0, timeout=120,
                                           extra_body={"reasoning": {"enabled": False}})
        out = r.choices[0].message.content or ""
    except Exception as e:
        sys.stderr.write(f"[fail] {type(e).__name__}: {str(e)[:120]}\n"); out = ""
    return out, time.time() - t0, len(prompt)


def fmt_global(turns, idxs):
    return "\n\n".join(f"[Turn {k}]: ({turns[k]['speaker']}) {turns[k]['text']}" for k in idxs)


def parse(txt):
    m = re.search(r"<segmentation>(.*?)</segmentation>", txt or "", re.S)
    body = m.group(1) if m else (txt or ""); segs = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line); st = o.get("start_turn_number")
            if st is None or str(st) == "":
                continue
            segs.append((int(st), bool(o.get("continues_previous", True))))
        except Exception:
            continue
    return segs


def v3_meeting(client, mid, turns, n, B, raw_sink=None):
    """raw 4분 윈도우: 직전 ≤240초 turns(context) + 현재 청크.
    raw_sink(파일핸들) 주어지면 **체크포인트마다 raw 응답·latency·입력 즉시 jsonl append**
    (재채점·붕괴선 재분석 시 API 재실행 0 — N=1 latency 보존)."""
    cuts = E.seg_checkpoints(turns, n, B); pred = set(); lat = []; inch = []
    for kp, ki in cuts:
        t_kp = turns[kp]["start"]
        ctx = [j for j in range(kp) if turns[j]["start"] >= t_kp - WINDOW_SEC]
        recent = fmt_global(turns, ctx) if ctx else "(none — start of meeting)"
        prompt = V3.format(recent_context=recent, text_to_be_segmented=fmt_global(turns, range(kp, ki)))
        out, dt, ln = call(client, prompt); lat.append(dt); inch.append(ln)
        segs = parse(out)
        if raw_sink is not None:        # raw 응답·메타 즉시 저장(증분, 중단보존) — 재현·재채점 원천
            raw_sink.write(json.dumps({"mid": mid, "buffer": B, "kp": kp, "ki": ki,
                                       "n_ctx": len(ctx), "in_len": ln, "latency_s": round(dt, 3),
                                       "out": out, "segs": segs}, ensure_ascii=False) + "\n")
            raw_sink.flush()
        if not segs:
            continue
        segs.sort()
        if not segs[0][1] and 0 < kp < n:        # 첫 new 청크가 새 topic(continues_previous=False) → kp 경계
            pred.add(kp)
        for s, _ in segs[1:]:
            if kp < s < ki:
                pred.add(s)
    return sorted(pred), lat, inch


if __name__ == "__main__":
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    client = E.make_client(); meetings = E.load_meetings(SUBSET)
    if smoke:
        meetings = sorted(meetings, key=lambda x: x[2])[:3]      # 짧은 3 미팅 검증
        print("=== v3 SMOKE (3 짧은 미팅, buffer 120) ===")
        for mid, turns, n, gold in meetings:
            pred, lat, inch = v3_meeting(client, mid, turns, n, 120)   # smoke 는 raw 저장 안 함
            f1 = E.tol_f1(sorted(gold), pred)
            print(f"{mid} n={n} F1={f1:.3f} | pred={pred} | gold={sorted(gold)} | calls={len(lat)} in_max={max(inch)} lat~{np.mean(lat):.1f}s")
        sys.exit(0)
    # run config 메타 1회 저장(재현)
    (OUTDIR / "v3_run_config.json").write_text(json.dumps(
        {"model": MODEL, "window_sec": WINDOW_SEC, "buffers": [120, 60], "subset": SUBSET,
         "n_meet": len(meetings), "prompt": "scripts/segmentation_prompts/baseline_segment_v3.md",
         "scorer": f"E.metrics ±2-tol + nltk Pk/WD, PRED_SHIFT={E.PRED_SHIFT}",
         "workers": "N=1 직렬(latency 공정)"}, ensure_ascii=False, indent=1))
    results = []
    for B in [120, 60]:
        pm = []; ms = []; alllat = []; allinch = []
        raw_path = OUTDIR / f"v3_{B}_raw.jsonl"
        with open(raw_path, "w") as raw_sink:        # 체크포인트별 raw 응답·latency 증분 저장
            for mid, turns, n, gold in meetings:
                pred, lat, inch = v3_meeting(client, mid, turns, n, B, raw_sink)
                m = E.metrics(gold, pred, n)        # (f2, exactf1, pk, wd, score) — ±2-tol (v1/v2 와 동일 채점)
                ms.append(m); alllat += lat; allinch += inch
                pm.append({"mid": mid, "n": n, "preds": pred, "gold": sorted(gold),
                           "F1": float(m[0]), "Pk": float(m[2]), "WD": float(m[3]), "Score": float(m[4]),
                           "lat": [round(x, 2) for x in lat], "inch": inch})
                (OUTDIR / f"v3_{B}_per_meeting.json").write_text(json.dumps(pm, ensure_ascii=False))   # 증분
        a = np.array(ms)
        row = {"method": "v3(raw-4min)", "buffer": str(B), "n_meet": len(a), "n_calls": len(alllat),
               "F1": float(a[:, 0].mean()), "Pk": float(a[:, 2].mean()), "WD": float(a[:, 3].mean()),
               "Score": float(a[:, 4].mean()), "lat_p50": float(np.percentile(alllat, 50)),
               "lat_p95": float(np.percentile(alllat, 95)), "lat_mean": float(np.mean(alllat)),
               "in_max": int(max(allinch)), "lat_all": [round(x, 2) for x in alllat]}
        results.append(row)
        (OUTDIR / "v3_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"v3 {B}: F1 {row['F1']:.3f} Pk {row['Pk']:.3f} WD {row['WD']:.3f} Score {row['Score']:.3f} | "
              f"lat p50 {row['lat_p50']:.1f}s p95 {row['lat_p95']:.1f}s | in_max {row['in_max']}", flush=True)
