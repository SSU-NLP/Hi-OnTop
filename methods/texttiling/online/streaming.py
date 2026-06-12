#!/usr/bin/env python3
"""TextTiling-online-streaming runner (3-benchmark, AUXILIARY baseline).

진짜 streaming TextTiling (``hi_ontop.baselines.StreamingTextTiling``) 를
Def-DTS 번들 데이터 (tiage / dialseg711 / superseg) 의 test set 에 적용.
metric 은 ``segeval`` 직접 호출 (Def-DTS 의 autoseg 거치지 않음). 데이터
로드 외에는 Def-DTS 의존 없음.

원본 NLTK TextTiling 의 점수를 재현하지 *않는다* (causal running threshold
≠ offline global threshold). 따라서 이름·표를 분리해서 보고. decision-log
2026-05-20 codex:rescue 위임 결과 참조.

사용::

    uv run python methods/texttiling/online_streaming.py \\
        --target-turns 100 [-w 5] [-k 3] [--c 0.5] [--min-gap 3] \\
        [--datasets tiage dialseg711 superseg]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hi_ontop.baselines import StreamingTextTiling
from hi_ontop.baselines._seg_utils import (
    DATASETS,
    boundary_set_f1 as _f1,
    bnds_to_masses as _bnds_to_masses,
    latency_stats as _stats,
    load_defdts,
    pk_wd,
)

REPO = Path(__file__).resolve().parent.parent.parent.parent
DEFDTS_DIR = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"


def _load_defdts(dataset: str) -> list[tuple[str, list[str], list[int]]]:
    return load_defdts(dataset, DEFDTS_DIR)


# -------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_texttiling_streaming")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument("--target-turns", type=int, default=100,
                    help="dataset 별 누적 발화 ≥ N 까지 대화 표본 (small-n "
                    "indicative). 0 = 전체 test set.")
    # tiage 의 평균 대화 길이 ~16발화 × ~5단어 ≈ 80토큰 짧아 w=10,k=6 (=
    # SuperDialseg 기본) 면 2k=12 pseudo-sentence 못 채우고 boundary 0 개.
    # 짧은 대화 데이터에 맞춰 w=5, k=3 으로 축소.
    ap.add_argument("-w", type=int, default=5)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--c", type=float, default=0.5,
                    help="cutoff = mean(depth) + c·std(depth)")
    ap.add_argument("--min-gap", type=int, default=3)
    ap.add_argument("--warmup-gaps", type=int, default=3)
    args = ap.parse_args()

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ds in args.datasets:
        full = _load_defdts(ds)
        sample: list[tuple[str, list[str], list[int]]] = []
        cum = 0
        for did, utts, bnds in full:
            sample.append((did, utts, bnds))
            cum += len(utts)
            if args.target_turns and cum >= args.target_turns:
                break
        print(f"=== {ds}: {len(sample)} dialogues, {cum} turns ===")

        sidecar = exp_dir / f"turns_{ds}.jsonl"
        fh = open(sidecar, "w")
        lat_ms: list[float] = []
        pk_vals: list[float] = []
        wd_vals: list[float] = []
        f1_vals: list[float] = []
        n_pred_total = 0
        n_gold_total = 0

        for did, utts, gold in sample:
            seg = StreamingTextTiling(
                w=args.w, k=args.k, c=args.c,
                min_gap=args.min_gap, warmup_gaps=args.warmup_gaps)
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
            flush_bs = seg.flush()
            pred.extend(flush_bs)
            n = len(utts)
            pk_v, wd_v = pk_wd(pred, gold, n)
            f1_v = _f1(pred, gold)
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
        st = _stats(lat_ms)
        rows.append(dict(
            ds=ds, n_dial=len(sample), n_turn=cum,
            pk=pk_m, wd=wd_m, f1=f1_m, score=score,
            n_pred=n_pred_total, n_gold=n_gold_total, **st))
        print(f"  Pk={pk_m:.4f} WD={wd_m:.4f} F1={f1_m:.4f} "
              f"Score={score:.4f} | lat/turn mean={st['mean']:.3f}ms "
              f"p95={st['p95']:.3f}ms (n={st['n']}) | "
              f"pred={n_pred_total} gold={n_gold_total}")

    _write_report(exp_dir, args, rows)


def _write_report(exp_dir: Path, args, rows: list[dict]) -> None:
    def g(v, p=4):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.{p}f}"

    target = (f"누적 발화 ≥ {args.target_turns} 까지 앞에서 자름 (small-n indicative)"
              if args.target_turns
              else "전체 test set (전체 대화 사용)")
    L = [
        "# TextTiling-online-streaming — AUXILIARY baseline "
        "(3-benchmark, per-turn latency)",
        "",
        "> **Paper-ready honest naming (codex 2026-05-21)**: `Streaming-TT-inspired` "
        "(or `CausalTextTiling-Streaming`). 원본 NLTK/SuperDialseg TextTiling 의 "
        "결정 규칙 (global mean−std/2 threshold, bilateral depth, paragraph "
        "min_gap) 모두 변경 → running mean+c·std, one-sided depth, utterance "
        "min_gap. **TextTiling 의 online 변형이 아니라 *TextTiling-inspired 별도 "
        "method***. 원본 paper 결과와 같은 표 등재 금지.",
        "> AUXILIARY (codex:rescue 2026-05-20/21 위임 권고). 핵심 비교값 = "
        "**per-turn latency (ms)**.",
        "",
        "## 1. Setup",
        f"- **Method**: `hi_ontop.baselines.StreamingTextTiling` (block-cosine "
        f"incremental, Welford running threshold).",
        f"  w={args.w}, k={args.k}, c={args.c} (cutoff=mean+c·std of depth), "
        f"min_gap={args.min_gap}, warmup_gaps={args.warmup_gaps}. "
        f"※ short-dialogue (tiage ~16 발화) 대응 runner default; class default 는 "
        f"NLTK 호환 w=10/k=6.",
        f"- **데이터**: Def-DTS 번들 (`benchmarks/Def-DTS/data/DTS_session_datasets/"
        f"*_test.jsonl`) 의 3 dataset. 데이터 로드 외 Def-DTS 의존 없음.",
        f"- **표본 정의**: dataset 별 {target}.",
        "- **metric**: `segeval` Pk/WD (per-dialogue → macro mean), "
        "self-implemented boundary-set F1 (per-dialogue → macro mean). "
        "Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        "- **latency**: 매 `push()` 호출 perf_counter (CPU only, "
        "calls/turn=0, tokens/turn=0). 첫 발화는 표본 제외.",
        "",
        "## 2. 보조표 (TextTiling-online-streaming | past-only, non-LLM)",
        "",
        "| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/turn(ms) ↓ | pred bs | gold bs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {r['n_dial']}/{r['n_turn']} | {g(r['pk'])} | "
            f"{g(r['wd'])} | {g(r['f1'])} | {g(r['score'])} | "
            f"{g(r['mean'], 3)} | {r['n_pred']} | {r['n_gold']} |")
    L += [
        "",
        "## 3. latency 분포 (ms/turn)",
        "",
        "| dataset | n | mean | std | p50 | p90 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {g(r['n'],0)} | {g(r['mean'],3)} | "
            f"{g(r['std'],3)} | {g(r['p50'],3)} | {g(r['p90'],3)} | "
            f"{g(r['p95'],3)} | {g(r['p99'],3)} | {g(r['max'],3)} |")
    L += [
        "",
        "## 4. 해석",
        "- streaming 의 per-turn O(w) 비용은 nltk prefix-recompute 의 O(t) "
        "비용보다 *원리적으로* 작음 (긴 대화일수록 격차 확대). 세 dataset 모두 "
        "ms 단위 이하의 latency 가 나오면 baseline 의 핵심 주장 (낮은 "
        "per-turn latency) 이 검증됨.",
        "- Pk/WD/F1 = INDICATIVE. running threshold 가 미래를 모르므로 NLTK "
        "원본의 global threshold 와 다른 boundary set 을 만든다. 이 격차 자체가 "
        "*의도된 algorithmic 차이*. 같은 이름 대신 `TextTiling-online-streaming` "
        "을 쓰는 이유.",
        "- dataset 별 분포 차이: tiage (~16 발화) 짧음 → 2k pseudo-sentence "
        "채우기 빠듯 (under-seg 편향), dialseg711·superseg (수십~수백 발화) "
        "에선 running threshold 가 안정화됨.",
        "- causal lag: gap score 는 right block (k pseudo-sentence) 이 닫혀야 "
        "계산되므로, boundary 채택 시점은 *대상 발화 인덱스보다 늦은 push() 호출 "
        "안* 에서 발생. metric 계산에는 영향 없음 (대화 종료시 boundary set 동일).",
        "",
        "## 5. 한계 / 검증 미해결",
        f"- 표본: {target}. seed/반복 없음 (알고리즘은 결정적이라 seed 무관).",
        "- one-sided depth (right_peak 미사용) → 원본 양방향 depth 와 boundary "
        "criterion 자체 다름.",
        f"- c={args.c}, min_gap={args.min_gap} 모두 dev-set sweep 없이 codex 권고값 "
        f"(class default c=0.5, min_gap=4; runner default 는 짧은 tiage 대화에 "
        f"맞춰 조정). tuning 여지 있음.",
        "- utterance ↔ pseudo-sentence 매핑: pseudo-sentence 가 발화 경계를 넘을 "
        "수 있어 detected boundary 의 utterance 귀속이 ±몇 발화 단위로 미세하게 "
        "shift 됨. Pk/WD 의 window-tolerant 성격으로 흡수되지만 F1 (정확 일치) 은 "
        "그만큼 낮을 수 있음.",
        "- NLTK 원본 / `run_texttiling_prefix.py` (prefix-recompute) 와의 "
        "boundary diff 비교는 별도 작업.",
    ]
    out = exp_dir / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"report → {out}")


if __name__ == "__main__":
    main()
