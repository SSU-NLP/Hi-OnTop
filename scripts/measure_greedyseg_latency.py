#!/usr/bin/env python3
"""GreedySeg-online-delay2 per-turn latency split (Pre. vs Seg., idle CPU).

Pre. = BERT-base forward (single utt, single CLS encoding) (= `_neural_sec`).
Seg. = cosine + argmin greedy + emit (push() 나머지).
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
from hi_ontop.baselines.greedyseg_delay2 import GreedySegOnlineDelay2  # noqa: E402

DEFDTS = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"


def load_dialogs(ds):
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


def subsample(dialogs, budget, seed=0):
    idx = np.random.default_rng(seed).permutation(len(dialogs))
    out, tot = [], 0
    for i in idx:
        d = dialogs[int(i)]; out.append(d); tot += len(d[1])
        if tot >= budget:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-per-bench", type=int, default=500)
    ap.add_argument("--name", default="2026-05-24_greedyseg_latency_split")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[setup] loading GreedySeg BERT (bert-base-uncased) ...", flush=True)
    t0 = time.perf_counter()
    proto = GreedySegOnlineDelay2(device="cpu")
    proto._ensure_model()
    cold_s = time.perf_counter() - t0
    print(f"[setup] loaded in {cold_s:.2f}s", flush=True)

    # warmup
    print("[warmup] 10 forwards ...", flush=True)
    for _ in range(10):
        proto._encode("warm up sentence")

    results = {}
    grand_pre, grand_seg = [], []
    for ds in ("tiage", "dialseg711", "superseg"):
        print(f"\n=== {ds} ===", flush=True)
        dialogs = load_dialogs(ds)
        n_dial_full, n_turn_full = len(dialogs), sum(len(u) for _,u,_ in dialogs)
        dialogs = subsample(dialogs, args.turns_per_bench)
        n_turn = sum(len(u) for _,u,_ in dialogs)
        print(f"  sample {len(dialogs)} dial / {n_turn} turn", flush=True)

        pre_ms, seg_ms, tot_ms = [], [], []
        for did, utts, _ in dialogs:
            seg = GreedySegOnlineDelay2(device="cpu")
            seg._model = proto._model
            seg._tokenizer = proto._tokenizer
            seg._device_resolved = proto._device_resolved
            seg._torch = proto._torch

            for i, u in enumerate(utts):
                nn_before = seg._neural_sec
                t0 = time.perf_counter()
                seg.push(u)
                t_total = (time.perf_counter() - t0) * 1000.0
                t_pre = (seg._neural_sec - nn_before) * 1000.0
                t_seg = t_total - t_pre
                if i >= 1:
                    pre_ms.append(t_pre)
                    seg_ms.append(t_seg)
                    tot_ms.append(t_total)

        ps = latency_stats(pre_ms)
        ss = latency_stats(seg_ms)
        ts = latency_stats(tot_ms)
        results[ds] = dict(n_turn_timed=len(tot_ms), pre=ps, seg=ss, total=ts,
                           n_dial_full=n_dial_full, n_turn_full=n_turn_full)
        print(f"  Pre.: mean={ps['mean']:.2f}  p50={ps['p50']:.2f} ms", flush=True)
        print(f"  Seg.: mean={ss['mean']:.4f}  p50={ss['p50']:.4f} ms", flush=True)
        grand_pre.extend(pre_ms); grand_seg.extend(seg_ms)

    pre_all = latency_stats(grand_pre)
    seg_all = latency_stats(grand_seg)
    tot_all = latency_stats([a+b for a,b in zip(grand_pre, grand_seg)])

    (out_dir / "latency.json").write_text(json.dumps(dict(
        method="GreedySeg-online-delay2", cold_start_s=cold_s, per_bench=results,
        all_bench=dict(pre=pre_all, seg=seg_all, total=tot_all),
        note="Pre. = BERT-base forward (single utt). Seg. = cosine + argmin greedy + emit."),
        indent=2))

    L = ["# GreedySeg-online-delay2 per-turn latency split", "",
         "## 정의",
         "- **Pre.** = BERT-base forward (single utt CLS) (_neural_sec 증분)",
         "- **Seg.** = cosine + argmin greedy bounded-lookahead + emit",
         f"- 10-forward warmup, cold-start (BERT load) {cold_s:.2f}s 분리.",
         "", "## 결과",
         "| 벤치 | n turn | Pre. mean / p50 / p90 (ms) | Seg. mean / p50 / p90 (ms) | Total (ms) |",
         "|---|---:|---:|---:|---:|"]
    for ds in ("tiage", "dialseg711", "superseg"):
        r = results[ds]; p,s,t = r["pre"], r["seg"], r["total"]
        L.append(f"| {ds} | {r['n_turn_timed']} | "
                 f"{p['mean']:.2f} / {p['p50']:.2f} / {p['p90']:.2f} | "
                 f"{s['mean']:.4f} / {s['p50']:.4f} / {s['p90']:.4f} | "
                 f"{t['mean']:.2f} |")
    L += ["", "## cross-benchmark 평균",
          f"- **Pre.** mean = {pre_all['mean']:.2f} ms (p50 {pre_all['p50']:.2f})",
          f"- **Seg.** mean = {seg_all['mean']:.4f} ms (p50 {seg_all['p50']:.4f})",
          f"- **Total** mean = {tot_all['mean']:.2f} ms"]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
