#!/usr/bin/env python3
"""CSM-online-delay2 runner (3-benchmark) — task #3.

Streaming CSM (lxing532 / Dialogue-Topic-Segmenter coherence-model 기반)
의 delay-2 변형. 원본 CSM 의 *score 공식* (coherence-network sigmoid
between adjacent utterances → TextTiling-style depth) 는 보존, 결정
horizon 만 streaming + delay-2 right context 로 변환 (csm_online.py docstring).

데이터 = Def-DTS 번들 `tiage_test/dialseg711_test/superseg_test.jsonl`,
metric = `segeval` Pk/WD + boundary-set F1, `Score = 0.5·F1 + 0.25·(1−Pk)
+ 0.25·(1−WD)`. methods/greedyseg/online/delay2.py 와 같은 harness.

코드 출처 (wrapper, 코드 복사 없음):
- lxing532/Dialogue-Topic-Segmenter (CoherenceNet → `external/Dialogue-Topic-Segmenter/`)
- Coldog2333/SuperDialseg 의 models/csm (참고용, 본 streaming 변형은
  lxing532 CoherenceNet 가중치를 그대로 씀)

ckpt: `outputs/runs/_misc/cpt_277000.pth` (`bert-base-uncased` backbone
으로 학습된 우리 CSM ckpt). 다른 ckpt 면 `--ckpt-path` 로 지정.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hi_ontop.baselines.csm_online import CSMOnlineDelay2
from hi_ontop.baselines._device import resolve_device
from hi_ontop.baselines._seg_utils import (
    DATASETS, boundary_set_f1, latency_stats, load_defdts, pk_wd)

REPO = Path(__file__).resolve().parent.parent.parent.parent
DEFDTS_DIR = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-23_csm_online_delay2")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument("--target-turns", type=int, default=0,
                    help="dataset 별 누적 발화 ≥ N 까지 대화 표본. 0 = 전체.")
    ap.add_argument("--ckpt-path", default="methods/CSM/cpt_277000.pth")
    ap.add_argument("--backbone", default="bert-base-uncased")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="threshold scale: depth > mean + alpha*std fires "
                         "boundary. paper offline CSM uses cut_rate=1.0 "
                         "(modeling_csm.py:108 + utils_texttiling.py:41 → "
                         "threshold = mean + 1.0*std).")
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--min-gap", type=int, default=2)
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "mps", "cpu"))
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[setup] device={device} backbone={args.backbone} "
          f"ckpt={args.ckpt_path}")

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # cold-start: CoherenceNet load 1회 (대화별 공유 모델)
    t_cold0 = time.perf_counter()
    seg_prime = CSMOnlineDelay2(
        ckpt_path=args.ckpt_path, backbone=args.backbone, device=device,
        alpha=args.alpha, delay=args.delay, min_gap=args.min_gap)
    seg_prime._ensure_loaded()
    cold_start_s = time.perf_counter() - t_cold0
    print(f"[setup] cold-start (CSM/BERT load) {cold_start_s:.2f}s")
    model_shared = seg_prime._model
    tok_shared = seg_prime._tokenizer
    dev_shared = seg_prime._device_resolved

    rows: list[dict] = []
    for ds in args.datasets:
        full = load_defdts(ds, DEFDTS_DIR)
        sample, cum = [], 0
        for did, utts, bnds in full:
            sample.append((did, utts, bnds)); cum += len(utts)
            if args.target_turns and cum >= args.target_turns:
                break
        print(f"\n=== {ds}: {len(sample)} dialogues, {cum} turns ===")

        sidecar = exp_dir / f"turns_{ds}.jsonl"
        fh = open(sidecar, "w")
        lat_ms, pk_v, wd_v, f1_v = [], [], [], []
        n_pred_total, n_gold_total, neural_sec = 0, 0, 0.0
        for did, utts, gold in sample:
            seg = CSMOnlineDelay2(
                ckpt_path=args.ckpt_path, backbone=args.backbone,
                device=device, alpha=args.alpha, delay=args.delay,
                min_gap=args.min_gap)
            # share loaded model
            seg._model = model_shared
            seg._tokenizer = tok_shared
            seg._device_resolved = dev_shared

            pred: list[int] = []
            for i, u in enumerate(utts):
                t0 = time.perf_counter()
                new_bs = seg.push(u)
                ms = (time.perf_counter() - t0) * 1000.0
                t_idx = i + 1
                if t_idx >= 2:
                    lat_ms.append(ms)
                fh.write(json.dumps({"ds": ds, "id": did, "t": t_idx,
                                     "ms": ms, "new_bs": new_bs}) + "\n")
                pred.extend(new_bs)
            pred.extend(seg.flush())
            neural_sec += seg._neural_sec

            n = len(utts)
            pk, wd = pk_wd(pred, gold, n)
            f1 = boundary_set_f1(pred, gold)
            pk_v.append(pk); wd_v.append(wd); f1_v.append(f1)
            n_pred_total += len(pred); n_gold_total += len(gold)
        fh.close()

        pk_m = float(np.mean(pk_v)) if pk_v else float("nan")
        wd_m = float(np.mean(wd_v)) if wd_v else float("nan")
        f1_m = float(np.mean(f1_v)) if f1_v else float("nan")
        score = 0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
        st = latency_stats(lat_ms)
        rows.append(dict(ds=ds, n_dial=len(sample), n_turn=cum,
                         pk=pk_m, wd=wd_m, f1=f1_m, score=score,
                         n_pred=n_pred_total, n_gold=n_gold_total,
                         neural_sec=neural_sec, **st))
        r = rows[-1]
        print(f"  Pk={pk_m:.4f} WD={wd_m:.4f} F1={f1_m:.4f} Score={score:.4f}"
              f" | lat/turn mean={st['mean']:.2f}ms p95={st['p95']:.2f}ms "
              f"(n={st['n']}) | pred={n_pred_total} gold={n_gold_total}")

    # REPORT
    L = [
        "# CSM-online-delay2 (3-benchmark)",
        "",
        f"backbone={args.backbone} · ckpt={args.ckpt_path} · "
        f"alpha={args.alpha} · delay={args.delay} · min_gap={args.min_gap} · "
        f"device={device}",
        "데이터 = Def-DTS 번들 test. metric = segeval Pk/WD + boundary-set "
        "F1. Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        f"cold-start (CSM/BERT load) = {cold_start_s:.2f}s",
        "",
        "| dataset | n_dial | n_turn | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/turn(ms) | p95 | pred bs | gold bs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n_dial']} | {r['n_turn']} | "
                 f"{r['pk']:.4f} | {r['wd']:.4f} | {r['f1']:.4f} | "
                 f"{r['score']:.4f} | {r['mean']:.2f} | {r['p95']:.2f} | "
                 f"{r['n_pred']} | {r['n_gold']} |")
    L += ["",
          "## 한계",
          "- ckpt 호환성 — 본 wrapper 는 lxing532 `CoherenceNet` 가중치 "
          "(`bert-base-uncased` backbone) 전제. SuperDialseg 의 CSM ckpt "
          "와는 모듈 구조가 달라 호환 안 됨.",
          "- strict prefix-causal 아님 — `delay=2` right context 의존. "
          "원본 score/HP 보존, *emission* 만 delay-2 (decision-log "
          "2026-05-21 GreedySeg-online-delay2 entry 와 동일 framing)."]
    (exp_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nreport → {exp_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
