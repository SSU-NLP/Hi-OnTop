#!/usr/bin/env python3
"""CSM-online per-turn latency split (Pre. vs Seg., idle CPU, realtime).

dts_result.md 의 CSM-Style 행 latency 컬럼 채우기 위함. Hi-OnTop latency
측정 (`measure_hiontop_latency.py`) 과 동일 정의:
- per-turn 매번 perf_counter
- Pre. = coherence_score (= tokenize + BERT forward) 1 회
- Seg. = push() 안의 나머지 (depth + welford + threshold + emit)
- 첫 발화 제외 (baseline 들과 동일 정책)
- BERT warm-up 10 회 단일-pair forward 먼저 (cold-start 제외)
- cache 없음, batch=1

데이터: Def-DTS 번들 test, 벤치당 500-turn budget subsample (seed=0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.baselines._seg_utils import (  # noqa: E402
    latency_stats, parse_defdts_dialogue)
from hi_ontop.baselines.csm_online import CSMOnlineDelay2  # noqa: E402

DEFDTS = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"


def load_dialogs(ds: str):
    path = DEFDTS / f"{ds}_test.jsonl"
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        utts, bnds = parse_defdts_dialogue(row["dialogue"])
        if len(utts) >= 2:
            out.append((row["id"], utts, bnds))
    return out


def subsample(dialogs, n_turn_budget: int, seed: int = 0) -> list:
    idx = np.random.default_rng(seed).permutation(len(dialogs))
    out, tot = [], 0
    for i in idx:
        d = dialogs[int(i)]
        out.append(d); tot += len(d[1])
        if tot >= n_turn_budget:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-per-bench", type=int, default=500)
    ap.add_argument("--name", default="2026-05-24_csm_latency_split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = REPO / "outputs" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[setup] loading CSM CoherenceNet (bert-base-uncased) ...", flush=True)
    t0 = time.perf_counter()
    seg_prime = CSMOnlineDelay2(device="cpu", alpha=1.0, delay=2, min_gap=2)
    seg_prime._ensure_loaded()
    load_s = time.perf_counter() - t0
    print(f"[setup] loaded in {load_s:.2f}s", flush=True)

    # warm-up: 10 single-pair forwards
    print("[warmup] 10 single-pair forwards ...", flush=True)
    for _ in range(10):
        _ = seg_prime._coherence_score("warm up sentence a", "warm up sentence b")

    model_shared = seg_prime._model
    tok_shared = seg_prime._tokenizer
    dev_shared = seg_prime._device_resolved

    results = {}
    grand_pre, grand_seg = [], []
    for ds in ("tiage", "dialseg711", "superseg"):
        print(f"\n=== {ds} ===", flush=True)
        dialogs = load_dialogs(ds)
        n_dial_full = len(dialogs)
        n_turn_full = sum(len(u) for _, u, _ in dialogs)
        dialogs = subsample(dialogs, args.turns_per_bench, seed=args.seed)
        n_turn = sum(len(u) for _, u, _ in dialogs)
        print(f"  full: {n_dial_full} dial / {n_turn_full} turn → "
              f"sample: {len(dialogs)} dial / {n_turn} turn", flush=True)

        pre_ms, seg_ms, tot_ms = [], [], []
        for did, utts, _ in dialogs:
            seg = CSMOnlineDelay2(device="cpu", alpha=1.0, delay=2, min_gap=2)
            seg._model = model_shared
            seg._tokenizer = tok_shared
            seg._device_resolved = dev_shared

            for i, u in enumerate(utts):
                # mimic push() but with split timing
                seg._utts.append(u)
                seg._t = len(seg._utts)
                t_total0 = time.perf_counter()
                t_pre = 0.0
                if seg._t >= 2:
                    t_pre0 = time.perf_counter()
                    s = seg._coherence_score(seg._utts[-2], seg._utts[-1])
                    t_pre = (time.perf_counter() - t_pre0) * 1000.0
                    seg._scores.append(s)

                t_seg0 = time.perf_counter()
                new_last = len(seg._scores) - seg.delay - 1
                if new_last > seg._evaluated_until:
                    for gi in range(seg._evaluated_until + 1, new_last + 1):
                        depth = seg._depth_at(gi)
                        seg._depths[gi] = depth
                        seg._welf.push(depth)
                        if (seg._welf.n >= seg.warmup_gaps and
                            depth > seg._welf.mean +
                                    seg.alpha * seg._welf.std()):
                            owner_t = gi + 2
                            if owner_t - seg._last_boundary_t >= seg.min_gap:
                                seg._last_boundary_t = owner_t
                    seg._evaluated_until = new_last
                t_seg = (time.perf_counter() - t_seg0) * 1000.0
                t_total = (time.perf_counter() - t_total0) * 1000.0

                if i >= 1:               # exclude turn 0
                    pre_ms.append(t_pre)
                    seg_ms.append(t_seg)
                    tot_ms.append(t_total)

        ps = latency_stats(pre_ms)
        ss = latency_stats(seg_ms)
        ts = latency_stats(tot_ms)
        results[ds] = dict(
            n_dialog_sample=len(dialogs),
            n_turn_sample=n_turn, n_turn_timed=len(tot_ms),
            n_dialog_full=n_dial_full, n_turn_full=n_turn_full,
            pre=ps, seg=ss, total=ts,
        )
        print(f"  Pre.(coherence): mean={ps['mean']:.2f}  p50={ps['p50']:.2f} "
              f"p90={ps['p90']:.2f} ms (n={ps['n']})", flush=True)
        print(f"  Seg.(depth+thr): mean={ss['mean']:.4f} p50={ss['p50']:.4f} "
              f"p90={ss['p90']:.4f} ms", flush=True)
        print(f"  total:           mean={ts['mean']:.2f} p50={ts['p50']:.2f} "
              f"ms", flush=True)
        grand_pre.extend(pre_ms); grand_seg.extend(seg_ms)

    pre_all = latency_stats(grand_pre)
    seg_all = latency_stats(grand_seg)
    tot_all = latency_stats([a + b for a, b in zip(grand_pre, grand_seg)])
    payload = dict(
        method="CSM-online-delay2",
        backbone="bert-base-uncased",
        ckpt="methods/CSM/cpt_277000.pth",
        seed=args.seed, cold_start_load_s=load_s,
        per_bench=results,
        all_bench=dict(pre=pre_all, seg=seg_all, total=tot_all),
        note="per-turn Pre. = _coherence_score (tokenize + BERT forward, "
             "single utt-pair, no cache). Seg. = depth + welford + threshold + "
             "boundary emit. First turn excluded; 10-pair warmup; cold-start "
             "(model load) excluded.")
    (out_dir / "latency.json").write_text(json.dumps(payload, indent=2))

    L = ["# CSM-online-delay2 per-turn latency split (Pre. vs Seg.)",
         "",
         "## 정의",
         "- **Pre.** = `_coherence_score` (tokenize + BERT forward, single pair)",
         "- **Seg.** = `push()` 안의 나머지 (depth peak walk + welford running "
         "threshold + boundary emit)",
         "- per-turn perf_counter 측정, 첫 발화 제외, 10-pair warm-up + "
         f"cold-start ({load_s:.1f}s) 분리.",
         "- 인코더 = bert-base-uncased + lxing532 CoherenceNet head, CPU.",
         "- HP: alpha=1.0, delay=2, min_gap=2, max_seq=128.",
         "",
         "## 결과 (벤치별)",
         "",
         "| 벤치 | n turn | Pre. mean / p50 / p90 (ms) | Seg. mean / p50 / p90 (ms) | Total (ms) |",
         "|---|---:|---:|---:|---:|"]
    for ds in ("tiage", "dialseg711", "superseg"):
        r = results[ds]
        p, s, t = r["pre"], r["seg"], r["total"]
        L.append(f"| {ds} | {r['n_turn_timed']} | "
                 f"{p['mean']:.2f} / {p['p50']:.2f} / {p['p90']:.2f} | "
                 f"{s['mean']:.4f} / {s['p50']:.4f} / {s['p90']:.4f} | "
                 f"{t['mean']:.2f} |")
    L += ["",
          "## cross-benchmark 평균 (dts_result.md 셀)",
          "",
          f"- **Pre.** mean = {pre_all['mean']:.2f} ms (p50 {pre_all['p50']:.2f})",
          f"- **Seg.** mean = {seg_all['mean']:.4f} ms (p50 {seg_all['p50']:.4f})",
          f"- **Total** mean = {tot_all['mean']:.2f} ms",
          f"- cold-start {load_s:.1f}s — per-turn 과 분리.",
          "",
          "## 한계",
          f"- 벤치당 {args.turns_per_bench} turn budget subsample (seed=0). "
          "content-independent latency 가정으로 전수 측정 대비 편차 작음.",
          "- CPU only. GPU 환경에선 Pre. 한 자릿수 ms 가능.",]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
