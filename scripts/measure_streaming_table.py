#!/usr/bin/env python
"""세 방법론 × {full,120,60} 측정 표: F1/Pk/WD/Score + per-call latency.
- secom(verbatim, B-mode) / baseline(B-mode) / csec(summary-carryover).
- 지표 집계 = 미팅별 metrics 의 mean (secom_llm_eval 와 동일).
- latency = fresh 콜 per-call wall-time (cache 우회), workers 고정. 한 모델(gpt-4o:nitro), AMI 18-subset."""
from __future__ import annotations
import sys, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import secom_llm_eval as E

SUBSET = str(E.REPO / "outputs/runs/_misc/ami_latency_subset.json")
MODEL = "openrouter/qwen/qwen3.5-27b"
WORKERS = 1          # latency 공정 측정 = N=1 순차(동시성 0). 경합 없는 per-call latency.
CSEC = (E.REPO / "scripts/segmentation_prompts/baseline_segment_v2.md").read_text()
LAT = []           # 전역 per-call latency 수집 (현재 측정중 method)


def call(client, prompt, mtok=1024):
    t0 = time.time()
    try:
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                           max_tokens=mtok, temperature=0.0, timeout=120,
                                           extra_body={"reasoning": {"enabled": False}})
        out = r.choices[0].message.content or ""
    except Exception as e:
        sys.stderr.write(f"[fail] {type(e).__name__}: {str(e)[:120]}\n"); out = ""
    dt = time.time() - t0
    return out, dt, len(prompt)


def run_bmode(prompt_name, B):
    """verbatim/baseline B-mode, fresh timed. 반환 rows(metrics)+lat+inchars."""
    E.load_prompts(prompt_name)
    client = E.make_client()
    meetings = E.load_meetings(SUBSET)
    # 모든 (mid, ki) prefix prompt 수집
    uniq = {}
    cuts_by = {}
    for mid, turns, n, gold in meetings:
        if B == "full":
            cuts = [(0, n)]
        else:
            cuts = E.seg_checkpoints(turns, n, B)
        cuts_by[mid] = cuts
        for kp, ki in cuts:
            uniq[(mid, ki)] = E.P_SEG.format(text_to_be_segmented=E.fmt_exchanges(turns, list(range(ki))))
    resp = {}; lat = []; inch = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, client, p): k for k, p in uniq.items()}
        for fut in as_completed(futs):
            out, dt, ln = fut.result(); resp[futs[fut]] = out; lat.append(dt); inch.append(ln)
    rows = []
    for mid, turns, n, gold in meetings:
        pred = set()
        for kp, ki in cuts_by[mid]:
            for s in E.extract_starts(resp.get((mid, ki), "")):
                if 0 < s < n and (kp <= s < ki if B != "full" else True):
                    pred.add(s)
        rows.append(E.metrics(gold, sorted(pred), n))
    return np.array(rows), lat, inch


def parse_csec(txt):
    m = re.search(r"<segmentation>(.*?)</segmentation>", txt or "", re.S)
    body = m.group(1) if m else (txt or "")
    segs = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
            st = o.get("start_turn_number")
            if st is None or str(st) == "":
                continue
            segs.append((int(st), bool(o.get("continues_previous", True)), str(o.get("summary", ""))[:300]))
        except Exception:
            continue
    return segs


def fmt_global(turns, idxs):
    """csec 전용 — 청크가 kp>0 에서 시작하므로 라벨을 **글로벌 turn 인덱스**로(E.fmt_exchanges 는 로컬 0-기반이라 부적합)."""
    return "\n\n".join(f"[Turn {k}]: ({turns[k]['speaker']}) {turns[k]['text']}" for k in idxs)


def csec_meeting(client, turns, n, B):
    cuts = E.seg_checkpoints(turns, n, B)
    prior = []; pred = set(); lat = []; inch = []
    for kp, ki in cuts:
        prompt = CSEC.format(prior_summaries=("\n".join(f"- {s}" for s in prior) or "(none yet)"),
                             text_to_be_segmented=fmt_global(turns, range(kp, ki)))
        out, dt, ln = call(client, prompt, mtok=600); lat.append(dt); inch.append(ln)
        segs = parse_csec(out)
        if not segs:
            continue
        segs.sort()
        first_cont = segs[0][1]
        if not first_cont and 0 < kp < n:
            pred.add(kp)
        for s, _, _ in segs[1:]:
            if kp < s < ki:
                pred.add(s)
        prior.extend(sm for _, _, sm in segs if sm)
    return sorted(pred), lat, inch


def run_csec(B):
    client = E.make_client()
    meetings = E.load_meetings(SUBSET)
    res = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(csec_meeting, client, t, n, B): (mid, sorted(g), n) for mid, t, n, g in meetings}
        for fut in as_completed(futs):
            mid, gold, n = futs[fut]; res[(mid)] = (fut.result(), gold, n)
    rows = []; lat = []; inch = []
    for mid, (r, gold, n) in res.items():
        pred, l, ic = r; lat += l; inch += ic
        rows.append(E.metrics(gold, pred, n))
    return np.array(rows), lat, inch


def fmt_row(name, buf, a, lat, inch):
    return (f"{name:18s} {str(buf):>4s} | F1 {a[:,0].mean():.3f} | Pk {a[:,2].mean():.3f} | "
            f"WD {a[:,3].mean():.3f} | Score {a[:,4].mean():.3f} | "
            f"lat p50 {np.percentile(lat,50):.1f}s p95 {np.percentile(lat,95):.1f}s | "
            f"in~{int(np.mean(inch))}c max{max(inch)}c")


def save(store, name, buf, a, lat, inch):
    """행 완료 즉시 JSON 누적 저장 (6h 작업 중단 대비)."""
    store.append({"method": name, "buffer": str(buf), "n_meet": len(a), "n_calls": len(lat),
                  "F1": float(a[:, 0].mean()), "Pk": float(a[:, 2].mean()), "WD": float(a[:, 3].mean()),
                  "Score": float(a[:, 4].mean()),
                  "lat_p50": float(np.percentile(lat, 50)), "lat_p95": float(np.percentile(lat, 95)),
                  "lat_mean": float(np.mean(lat)), "lat_max": float(max(lat)),
                  "in_mean": float(np.mean(inch)), "in_max": int(max(inch)), "lat_all": [round(x, 2) for x in lat]})
    out = E.REPO / "outputs/experiments/2026-06-12_llm_streaming_modes_ami18/measure_results.json"
    out.write_text(json.dumps(store, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    print("method/buffer        |  ±2F1 |   Pk  |   WD  | Score | per-call latency(N=1순차) | 입력크기")
    print("-" * 110)
    store = []
    # v2(csec) 먼저 — 우선 측정 보장(입력 작아 빠름). 그다음 v1(baseline), verbatim. 버퍼 120/60 먼저(streaming 핵심), full 마지막.
    for buf in [120, 60]:
        a, lat, inch = run_csec(buf)
        print(fmt_row("csec(carryover)", buf, a, lat, inch), flush=True); save(store, "csec(carryover)", buf, a, lat, inch)
    for name, pn in [("baseline(B-mode)", "baseline"), ("SeCom(verbatim)", "verbatim")]:
        for buf in [120, 60, "full"]:
            a, lat, inch = run_bmode(pn, buf)
            print(fmt_row(name, buf, a, lat, inch), flush=True); save(store, name, buf, a, lat, inch)
