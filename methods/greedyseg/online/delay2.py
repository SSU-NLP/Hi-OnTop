#!/usr/bin/env python3
"""GreedySeg-online-delay2 runner (3-benchmark, AUXILIARY → 5행 핵심표 가능).

진짜 streaming 입력 + delay=2 (right context window_size=2 미래 발화)
출력. 원본 SuperDialseg GreedySegmenter 의 score 공식·HP·argmin greedy
선택 그대로 보존 (codex:rescue 2026-05-20/21 검증 통과). 데이터 = Def-DTS
번들 jsonl 3 dataset, metric = segeval Pk/WD + boundary-set F1.

사용::

    uv run python methods/greedyseg/online_delay2.py \\
        --target-turns 0 [--datasets tiage dialseg711 superseg] \\
        [--device auto|cuda|mps|cpu]
"""

from __future__ import annotations

# mps fallback env var 는 torch/transformers import *이전* 설정 필수.
# resolve_device 가 lazy 이지만 모듈 전역 import 가 transformers 를 끌어오므로
# argparse 이전에 가드.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hi_ontop.baselines import GreedySegOnlineDelay2
from hi_ontop.baselines._device import resolve_device
from hi_ontop.baselines._seg_utils import (
    DATASETS,
    boundary_set_f1,
    latency_stats,
    load_defdts,
    pk_wd,
)

REPO = Path(__file__).resolve().parent.parent.parent.parent
DEFDTS_DIR = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-21_greedyseg_online_delay2")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument("--target-turns", type=int, default=0,
                    help="dataset 별 누적 발화 ≥ N 까지 대화 표본. 0 = 전체.")
    ap.add_argument("--backbone", default="bert-base-uncased")
    ap.add_argument("--window-size", type=int, default=2)
    ap.add_argument("--jump-step", type=int, default=2)
    ap.add_argument("--max-seg-round", type=int, default=8)
    ap.add_argument("--sim-threshold", type=float, default=0.6)
    ap.add_argument("--max-seq-length", type=int, default=50)
    ap.add_argument("--device", default="auto",
                    choices=("auto", "cuda", "mps", "cpu"))
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[setup] device={device} backbone={args.backbone}")

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # cold-start: BERT load 1회 (전체 dataset 공유 인스턴스로 가속)
    t_cold0 = time.perf_counter()
    seg_prime = GreedySegOnlineDelay2(
        backbone=args.backbone, window_size=args.window_size,
        jump_step=args.jump_step, max_seg_round=args.max_seg_round,
        sim_threshold=args.sim_threshold, max_seq_length=args.max_seq_length,
        device=device)
    seg_prime._ensure_model()
    cold_start_s = time.perf_counter() - t_cold0
    print(f"[setup] cold-start (BERT load) {cold_start_s:.2f}s")
    # 같은 model/tokenizer 를 후속 dialogue 인스턴스로 공유
    tokenizer = seg_prime._tokenizer
    model = seg_prime._model

    rows: list[dict] = []
    for ds in args.datasets:
        full = load_defdts(ds, DEFDTS_DIR)
        sample: list[tuple[str, list[str], list[int]]] = []
        cum = 0
        for did, utts, bnds in full:
            sample.append((did, utts, bnds))
            cum += len(utts)
            if args.target_turns and cum >= args.target_turns:
                break
        print(f"\n=== {ds}: {len(sample)} dialogues, {cum} turns ===")

        sidecar = exp_dir / f"turns_{ds}.jsonl"
        fh = open(sidecar, "w")
        lat_ms: list[float] = []
        pk_vals: list[float] = []
        wd_vals: list[float] = []
        f1_vals: list[float] = []
        n_pred_total = 0
        n_gold_total = 0
        bert_forwards_total = 0

        for did, utts, gold in sample:
            seg = GreedySegOnlineDelay2(
                backbone=args.backbone, window_size=args.window_size,
                jump_step=args.jump_step, max_seg_round=args.max_seg_round,
                sim_threshold=args.sim_threshold,
                max_seq_length=args.max_seq_length, device=device)
            # share loaded model
            seg._tokenizer = tokenizer
            seg._model = model
            seg._device_resolved = device
            import torch
            seg._torch = torch

            pred: list[int] = []
            for i, u in enumerate(utts):
                t0 = time.perf_counter()
                new_bs = seg.push(u)
                ms = (time.perf_counter() - t0) * 1000.0
                t_idx = i + 1
                if t_idx >= 2:
                    lat_ms.append(ms)
                fh.write(json.dumps({
                    "ds": ds, "id": did, "t": t_idx,
                    "ms": ms, "new_bs": new_bs}) + "\n")
                pred.extend(new_bs)
            pred.extend(seg.flush())
            bert_forwards_total += seg._bert_forwards

            n = len(utts)
            pk_v, wd_v = pk_wd(pred, gold, n)
            f1_v = boundary_set_f1(pred, gold)
            pk_vals.append(pk_v)
            wd_vals.append(wd_v)
            f1_vals.append(f1_v)
            n_pred_total += len(pred)
            n_gold_total += len(gold)
        fh.close()

        pk_m = float(np.mean(pk_vals)) if pk_vals else float("nan")
        wd_m = float(np.mean(wd_vals)) if wd_vals else float("nan")
        f1_m = float(np.mean(f1_vals)) if f1_vals else float("nan")
        score = 0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
        st = latency_stats(lat_ms)
        fpu = bert_forwards_total / cum if cum else float("nan")
        rows.append(dict(
            ds=ds, n_dial=len(sample), n_turn=cum,
            pk=pk_m, wd=wd_m, f1=f1_m, score=score,
            n_pred=n_pred_total, n_gold=n_gold_total,
            bert_forwards_per_utt=fpu, **st))
        r = rows[-1]
        print(f"  Pk={pk_m:.4f} WD={wd_m:.4f} F1={f1_m:.4f} Score={score:.4f}"
              f" | lat/turn mean={st['mean']:.2f}ms p95={st['p95']:.2f}ms "
              f"(n={st['n']}) | bert_fwd/utt={fpu:.2f} | pred={n_pred_total} "
              f"gold={n_gold_total}")

    _write_report(exp_dir, args, device, cold_start_s, rows)


def _write_report(exp_dir: Path, args, device: str, cold_start_s: float,
                   rows: list[dict]) -> None:
    import platform
    import subprocess
    import sys
    import torch
    import transformers

    def g(v, p=4):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.{p}f}"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            text=True).strip()
    except Exception:
        commit = "unknown"
    target = (f"누적 발화 ≥ {args.target_turns} 까지 (small-n indicative)"
              if args.target_turns
              else "전체 test set (전체 대화 사용)")

    L = [
        "# GreedySeg-online-delay2 — bounded-lookahead online baseline",
        "",
        "> **Honest naming (codex 2026-05-21)**: same scoring/selection, "
        "delayed emission (delay=2 — right context window_size=2). "
        "원본 SuperDialseg GreedySegmenter 의 score 공식·HP·argmin greedy "
        "선택 그대로 보존. **5행 핵심표 가능** (본 plan 의 baseline 중 유일).",
        "",
        "## 1. Setup",
        f"- **Method**: `hi_ontop.baselines.GreedySegOnlineDelay2`. backbone="
        f"`{args.backbone}`, window_size={args.window_size}, "
        f"jump_step={args.jump_step}, max_seg_round={args.max_seg_round}, "
        f"sim_threshold={args.sim_threshold}, max_seq_length={args.max_seq_length}.",
        f"- **Device**: `{device}` (auto-resolved; cuda→mps→cpu 우선순위). "
        f"PYTORCH_ENABLE_MPS_FALLBACK=1.",
        f"- **데이터**: Def-DTS 번들 (`benchmarks/Def-DTS/data/DTS_session_datasets/"
        f"*_test.jsonl`) 의 3 dataset. {target}.",
        "- **metric**: `segeval` Pk/WD (per-dialogue → macro mean), "
        "self-implemented boundary-set F1 (per-dialogue → macro mean). "
        "Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        "- **latency**: 매 `push()` 호출 perf_counter (BERT forward 포함). "
        "첫 발화는 표본 제외. cold-start (BERT load) 분리.",
        f"- **cold-start**: BERT/tokenizer load {cold_start_s:.2f}s "
        "(per-turn latency 와 분리).",
        f"- **Environment**: python={sys.version.split()[0]}, torch={torch.__version__}, "
        f"transformers={transformers.__version__}, platform={platform.platform()}, "
        f"commit={commit}.",
        "",
        "## 2. 결과표 (GreedySeg-online-delay2 | bounded-lookahead, BERT)",
        "",
        "| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/turn(ms) ↓ | bert_fwd/utt | pred bs | gold bs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {r['n_dial']}/{r['n_turn']} | {g(r['pk'])} | "
            f"{g(r['wd'])} | {g(r['f1'])} | {g(r['score'])} | "
            f"{g(r['mean'], 2)} | {g(r['bert_forwards_per_utt'], 2)} | "
            f"{r['n_pred']} | {r['n_gold']} |")
    L += [
        "",
        "## 3. latency 분포 (ms/turn, BERT forward 포함)",
        "",
        "| dataset | n | mean | std | p50 | p90 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {g(r['n'],0)} | {g(r['mean'],2)} | "
            f"{g(r['std'],2)} | {g(r['p50'],2)} | {g(r['p90'],2)} | "
            f"{g(r['p95'],2)} | {g(r['p99'],2)} | {g(r['max'],2)} |")
    L += [
        "",
        "## 4. 해석",
        "- 원본 GreedySeg 의 *알고리즘 본질 (BERT cosine score · argmin greedy "
        "selection · HP)* 가 그대로 보존됨. delay=2 는 right context "
        f"(window_size={args.window_size}) 가 buffer 에 도착해야 score 계산 가능"
        "하기 때문 — *boundary 채택* 자체는 비가역 greedy 그대로.",
        "- 실제 emit 시점은 segment 안 `max_seg_round` 후보가 모두 평가된 후 "
        f"(= cut_index + {args.max_seg_round-1} + window_size 발화 후)이므로, "
        "boundary index 가 가리키는 utterance 와 push() 시점 간 차이는 *최대* "
        f"{args.max_seg_round-1 + args.window_size} 발화. interactive 사용 시 "
        "이 lag 명시 필요.",
        "- **TextTiling-streaming (encoder-free, ~0.01ms) 과 같은 latency 표에 "
        "섞지 않음**: encoder cost (BERT forward) 차원이 다름.",
        "- 결정성: device 마다 결정성 보장 다름 (CPU > CUDA > MPS). 본 표 "
        f"수치는 device=`{device}` 1회 측정.",
        "",
        "## 5. 한계 / 검증 미해결",
        "- 원본 GreedySeg paper 점수와 직접 비교 불가: 데이터 (Def-DTS 번들 vs "
        "원 SuperDialseg) + metric (segeval direct vs autoseg) + bounded-"
        "lookahead 인터페이스 차이. *방향성·정상동작* 검증용 보조.",
        f"- HP 미튜닝: 원본 default (`window_size=2, jump_step=2, max_seg_round"
        f"=8, sim_threshold=0.6, max_seq_length=50`) 그대로. dev-set sweep 없음.",
        "- **`segment-concat → [CLS] embedding`** 가정 — 원본 코드의 정확한 pooling "
        "방식 (CLS vs mean) 확인 못함 (superdialseg 로컬 install 없음). 다른 pooling "
        "사용 시 점수 변동 가능.",
        "- BERT 추론 결정성: torch/transformers 버전 동일 + 동일 device 반복 측정"
        "해야 reproducibility 확보. 다른 device 간 직접 비교 금지.",
        "- delay 의 정확한 의미: `delay=2` 는 codex 의 right-context lag 표현. "
        "실제 boundary emit 시점은 segment 길이 + window_size 만큼 lag.",
        "- 표본은 dataset 별 전체 test set (target-turns=0). seed 없음 (BERT "
        "결정성 + 알고리즘 결정성 → seed 무관).",
    ]
    out = exp_dir / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
