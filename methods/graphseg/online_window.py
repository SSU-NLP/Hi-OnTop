#!/usr/bin/env python3
"""GraphSeg-inspired bounded-window runner (3-benchmark, AUXILIARY).

GraphSeg (Glavaš et al. 2016) 의 IC × GloVe + Hungarian + Bron-Kerbosch +
sequential merge 를 window=d 안에서만 적용. *strict online 아님* — codex
2026-05-21 검증: graph 범위·optimization context 가 원본과 다름, 강한
``-online`` 명명 금지 (정직 명명 = ``GraphSeg-inspired bounded-window``,
short ``GraphSeg-window-d``). AUXILIARY only.

사용::

    uv run python methods/graphseg/online_window.py --target-turns 0 \\
        [--datasets ...] [--window-d 10] [--tau 0.25]
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hi_ontop.baselines import GraphSegWindowD
from hi_ontop.baselines._seg_utils import (
    DATASETS,
    boundary_set_f1,
    latency_stats,
    load_defdts,
    pk_wd,
)

REPO = Path(__file__).resolve().parent.parent.parent
DEFDTS_DIR = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-21_graphseg_window_d")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument("--target-turns", type=int, default=0,
                    help="dataset 별 누적 발화 ≥ N 까지 대화 표본. 0 = 전체.")
    ap.add_argument("--window-d", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.25,
                    help="edge sim threshold")
    ap.add_argument("--min-seg-size", type=int, default=3)
    ap.add_argument("--glove-path",
                    default=str(REPO / "benchmarks" / "glove" / "glove.6B.300d.txt"))
    ap.add_argument("--no-pos-filter", action="store_true",
                    help="POS content-word filter 비활성화")
    ap.add_argument("--freq-source", default="brown",
                    choices=("brown", "none"))
    args = ap.parse_args()

    if not Path(args.glove_path).exists():
        raise SystemExit(
            f"GloVe 없음: {args.glove_path}\n"
            "다운로드: wget https://nlp.stanford.edu/data/glove.6B.zip -P benchmarks/glove/ && "
            "unzip benchmarks/glove/glove.6B.zip -d benchmarks/glove/")

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # cold-start: GloVe + IC table 1회 load (전체 dataset 공유)
    t_cold0 = time.perf_counter()
    seg_prime = GraphSegWindowD(
        window_d=args.window_d, sim_threshold=args.tau,
        min_seg_size=args.min_seg_size, glove_path=args.glove_path,
        freq_source=args.freq_source,
        use_pos_filter=not args.no_pos_filter)
    seg_prime._ensure_resources()
    cold_start_s = time.perf_counter() - t_cold0
    print(f"[setup] cold-start (GloVe + IC table) {cold_start_s:.2f}s")
    print(f"[setup] glove vocab={len(seg_prime._glove)}, "
          f"window_d={args.window_d}, tau={args.tau}, "
          f"min_seg={args.min_seg_size}, pos_filter={not args.no_pos_filter}")

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

        for did, utts, gold in sample:
            seg = GraphSegWindowD(
                window_d=args.window_d, sim_threshold=args.tau,
                min_seg_size=args.min_seg_size, glove_path=args.glove_path,
                freq_source=args.freq_source,
                use_pos_filter=not args.no_pos_filter)
            # share loaded resources
            seg._glove = seg_prime._glove
            seg._glove_dim = seg_prime._glove_dim
            seg._ic_table = seg_prime._ic_table

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
        rows.append(dict(
            ds=ds, n_dial=len(sample), n_turn=cum,
            pk=pk_m, wd=wd_m, f1=f1_m, score=score,
            n_pred=n_pred_total, n_gold=n_gold_total, **st))
        r = rows[-1]
        print(f"  Pk={pk_m:.4f} WD={wd_m:.4f} F1={f1_m:.4f} Score={score:.4f}"
              f" | lat/turn mean={st['mean']:.2f}ms p95={st['p95']:.2f}ms "
              f"(n={st['n']}) | pred={n_pred_total} gold={n_gold_total}")

    _write_report(exp_dir, args, cold_start_s, len(seg_prime._glove), rows)


def _write_report(exp_dir: Path, args, cold_start_s: float, glove_vocab: int,
                   rows: list[dict]) -> None:
    import platform
    import subprocess
    import sys

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
        "# GraphSeg-inspired bounded-window (short: `GraphSeg-window-d`) — AUXILIARY",
        "",
        "> **Honest naming (codex 2026-05-21)**: *original global graph mechanism "
        "is not preserved*. GraphSeg (Glavaš et al. 2016) 의 sentence similarity "
        "(IC × GloVe + Hungarian) + Bron-Kerbosch maximal clique + sequential "
        "merge 3-phase 를 **window 안에서만** 적용. 강한 `-online` 명명 금지.",
        "> **AUXILIARY only** — 원본 GraphSeg paper 결과와 같은 표 등재 금지.",
        "",
        "## 1. Setup",
        f"- **Method**: `hi_ontop.baselines.GraphSegWindowD`. window_d={args.window_d}, "
        f"sim_threshold (τ)={args.tau}, min_seg_size={args.min_seg_size}, "
        f"freq_source=`{args.freq_source}`, "
        f"use_pos_filter={not args.no_pos_filter}.",
        f"- **Embedding**: GloVe 6B.300d (vocab={glove_vocab}). "
        f"IC weighting from NLTK brown corpus.",
        f"- **데이터**: Def-DTS 번들 (`benchmarks/Def-DTS/data/DTS_session_datasets/"
        f"*_test.jsonl`). {target}.",
        "- **metric**: `segeval` Pk/WD (per-dialogue → macro mean) + boundary-set "
        "F1. Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        f"- **cold-start**: GloVe + IC table load {cold_start_s:.2f}s (per-turn "
        "latency 와 분리).",
        f"- **Environment**: python={sys.version.split()[0]}, "
        f"platform={platform.platform()}, commit={commit}. "
        "GraphSeg-window-d 는 PyTorch 무관 (numpy/scipy/networkx).",
        "",
        "## 2. 결과표 (GraphSeg-inspired bounded-window | window-local clique)",
        "",
        "| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/turn(ms) ↓ | pred bs | gold bs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {r['n_dial']}/{r['n_turn']} | {g(r['pk'])} | "
            f"{g(r['wd'])} | {g(r['f1'])} | {g(r['score'])} | "
            f"{g(r['mean'], 2)} | {r['n_pred']} | {r['n_gold']} |")
    L += [
        "",
        "## 3. latency 분포 (ms/turn, Bron-Kerbosch + Hungarian 포함)",
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
        "- **원본 GraphSeg 의 어떤 본질이 보존되고 어느 본질이 양보됐는가** "
        "(codex 2026-05-21):",
        "  - 보존: sentence similarity 공식 (IC × GloVe + Hungarian), "
        "Bron-Kerbosch maximal clique, sequential merge 3-phase, "
        "content-word POS filter.",
        "  - 양보: full-dialogue graph (→ window=d), global clique structure "
        "(→ window-local), single-pass global merge (→ sliding window 마다 "
        "재계산), backtracking (= 불가, boundary 비가역).",
        "- 따라서 본 결과는 *GraphSeg 원본 점수와 직접 비교 불가*. AUXILIARY "
        "online auxiliary baseline.",
        "- **TextTiling-streaming (encoder-free, ~0.01ms) / GreedySeg-online-"
        "delay2 (BERT, ~10ms) 와 같은 latency 표에 직접 비교 금지** — "
        "encoder/연산 카테고리가 다름. 비교 시 *어느 본질이 보존되고 어느 본질이 "
        "양보됐는지* 열 분리 (codex 권고).",
        "",
        "## 5. 한계 / 검증 미해결",
        f"- 원본 GraphSeg paper 점수와 직접 비교 불가: 데이터 (Def-DTS vs 원 "
        "SemEval 2016 dialogue datasets), metric (segeval direct vs 원 paper), "
        "그리고 window-local maximal clique 자체가 algorithmic 차이.",
        f"- HP 미튜닝: τ={args.tau}, window_d={args.window_d}, "
        f"min_seg_size={args.min_seg_size} 모두 codex 권고 default. dev-set "
        "sweep 없음.",
        "- **word frequency 출처** = NLTK brown corpus (Wikipedia 보다 작은 "
        "domain). 원본은 더 큰 corpus. IC 가중치 분포 차이가 결과에 영향 가능.",
        "- **POS filter** = NLTK averaged_perceptron_tagger. 원본 paper 의 "
        "POS-filter 와 정확히 같지 않을 수 있음.",
        "- **boundary lag-emission**: 한 boundary 가 여러 window 평가에 걸쳐 "
        "재출현 가능. 본 구현은 *최초 발견 즉시 비가역 채택* (codex 'graph 범위 "
        "변경' 의 자연스러운 부산물).",
        "- 결정성: numpy/scipy/networkx 모두 결정적 → seed 무관, byte-identical 결과.",
    ]
    out = exp_dir / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
