#!/usr/bin/env python3
"""Hi-OnTop per-turn latency (realtime, no encoder cache, single-utt forward).

dts_result.md 의 Ours 행 Pre./Seg. 가 'host-shared accounting (encoder cost
amortized away)' 로 보고돼 있어 *실제 online 배포 조건* 의 per-turn cost
가 표에 반영돼 있지 않다. 본 runner 는 각 turn 의 (encode_one + assign)
을 매번 새로 계산하여 perf_counter 로 측정한다 — 캐시·배치 없이.

methodology (다른 online baseline REPORT 들과 동일):
- 매 turn 의 encode 와 assign 을 각각 perf_counter 로 시간 측정 후 ms.
- 첫 발화 (turn 0) 는 표본 제외 — TextTiling/GreedySeg/GraphSeg 와 동일 정책.
- encoder warm-up : 본격 측정 전 10 utt forward (모델 lazy init / 첫
  call jit 비용 제외). 이 시간은 cold-start, per-turn 과 분리.
- 인코더 = MPNet (`multi-qa-mpnet-base-dot-v1`, sentence-transformers,
  CPU). δ* 는 본 측정에서 무관 (encode + assign 시간은 boundary 결정
  여부와 독립).

latency 가 δ* 무관이므로 한 번 측정한 결과를 Ours p70/p75/p80/oracle
4 행 모두에 적용한다.
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
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

DEFDTS = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"
ENCODERS = {
    "mpnet": {"model": "sentence-transformers/multi-qa-mpnet-base-dot-v1"},
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "minilm-int8": {"model": "sentence-transformers/all-MiniLM-L6-v2",
                    "backend": "onnx",
                    "file_name": "onnx/model_quint8_avx2.onnx"},
}
M, RHO, A = 2, 0.7, 0.5


def load_dialogs(ds: str) -> list[tuple[str, list[str], list[int]]]:
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


def subsample(dialogs: list, n_turn_budget: int, seed: int = 0) -> list:
    """seed shuffle → 누적 turn 수가 budget 도달할 때까지 자른다."""
    idx = np.random.default_rng(seed).permutation(len(dialogs))
    out = []
    tot = 0
    for i in idx:
        d = dialogs[int(i)]
        out.append(d)
        tot += len(d[1])
        if tot >= n_turn_budget:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns-per-bench", type=int, default=500,
                    help="per-benchmark turn budget for timing (subsample)")
    ap.add_argument("--name", default="2026-05-23_hiontop_latency_realtime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", choices=list(ENCODERS), default="mpnet")
    args = ap.parse_args()
    ENCODER_MODEL = ENCODERS[args.encoder]["model"]
    enc_cfg = ENCODERS[args.encoder]

    out_dir = REPO / "outputs" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[encoder] loading {ENCODER_MODEL} (CPU, {args.encoder}) ...", flush=True)
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    if enc_cfg.get("backend") == "onnx":
        enc = SentenceTransformer(
            ENCODER_MODEL, backend="onnx",
            model_kwargs={"provider": "CPUExecutionProvider",
                          "file_name": enc_cfg["file_name"]})
    else:
        enc = SentenceTransformer(ENCODER_MODEL, device="cpu")
    load_s = time.perf_counter() - t0
    print(f"[encoder] loaded in {load_s:.2f}s", flush=True)

    # warm-up : 10 forwards (jit / lazy init 비용 제외, baseline 들과 동일)
    print("[warmup] 10 single-utt forwards ...", flush=True)
    for _ in range(10):
        enc.encode(["warm up sentence"], normalize_embeddings=True,
                   show_progress_bar=False)

    results = {}
    grand_enc, grand_seg = [], []
    for ds in ("tiage", "dialseg711", "superseg"):
        print(f"\n=== {ds} ===", flush=True)
        dialogs = load_dialogs(ds)
        n_dial_full = len(dialogs)
        n_turn_full = sum(len(u) for _, u, _ in dialogs)
        dialogs = subsample(dialogs, args.turns_per_bench, seed=args.seed)
        n_turn = sum(len(u) for _, u, _ in dialogs)
        print(f"  full: {n_dial_full} dial / {n_turn_full} turn → "
              f"sample: {len(dialogs)} dial / {n_turn} turn (seed={args.seed})",
              flush=True)

        enc_ms, seg_ms, tot_ms = [], [], []
        for did, utts, _ in dialogs:
            seg = HiOnTop(dim=768, delta_star=0.5594,
                          ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
            for i, u in enumerate(utts):
                t_enc0 = time.perf_counter()
                s = enc.encode([u], normalize_embeddings=True,
                               show_progress_bar=False)[0]
                t_enc = (time.perf_counter() - t_enc0) * 1000.0
                t_seg0 = time.perf_counter()
                seg.assign(np.asarray(s, dtype=np.float64))
                t_seg = (time.perf_counter() - t_seg0) * 1000.0
                if i >= 1:                       # exclude turn 0
                    enc_ms.append(t_enc)
                    seg_ms.append(t_seg)
                    tot_ms.append(t_enc + t_seg)

        es = latency_stats(enc_ms)
        ss = latency_stats(seg_ms)
        ts = latency_stats(tot_ms)
        results[ds] = dict(
            n_dialog_sample=len(dialogs),
            n_turn_sample=n_turn,
            n_turn_timed=len(tot_ms),
            n_dialog_full=n_dial_full,
            n_turn_full=n_turn_full,
            enc=es, seg=ss, total=ts,
        )
        print(f"  encode(Pre.):  mean={es['mean']:.2f} p50={es['p50']:.2f} "
              f"p90={es['p90']:.2f} ms (n={es['n']})", flush=True)
        print(f"  segment(Seg.): mean={ss['mean']:.4f} p50={ss['p50']:.4f} "
              f"p90={ss['p90']:.4f} ms", flush=True)
        print(f"  total:         mean={ts['mean']:.2f} p50={ts['p50']:.2f} "
              f"p90={ts['p90']:.2f} ms", flush=True)
        grand_enc.extend(enc_ms); grand_seg.extend(seg_ms)

    # cross-benchmark aggregate
    enc_all = latency_stats(grand_enc)
    seg_all = latency_stats(grand_seg)
    tot_all = latency_stats([a + b for a, b in zip(grand_enc, grand_seg)])

    # JSON dump
    payload = dict(
        encoder=ENCODER_MODEL,
        hp=dict(m=M, rho=RHO, a=A),
        seed=args.seed,
        cold_start_load_s=load_s,
        per_bench=results,
        all_bench=dict(enc=enc_all, seg=seg_all, total=tot_all),
        note=("per-turn = encode(single utt, no cache) + assign(Hi-OnTop). "
              "First turn of each dialogue excluded (warmup convention). "
              "Encoder warmup 10 fwd before timing; cold-start excluded."),
    )
    (out_dir / "latency.json").write_text(json.dumps(payload, indent=2))

    # REPORT.md
    L = [
        "# Hi-OnTop per-turn latency (realtime, no encoder cache)",
        "",
        "## 1. 측정 정의",
        "",
        "- per-turn = **encode(단일 발화, 캐시 없음) + HiOnTop.assign**.",
        "- 매 turn perf_counter 로 encode 와 assign 시간을 따로 측정 (ms).",
        "- 첫 발화 (turn 0) 는 baseline 들과 동일하게 표본 제외.",
        "- encoder warmup 10 forwards (cold-start, model load 비용 분리).",
        f"- encoder = `{ENCODER_MODEL}` (CPU).",
        f"- HP: m={M}, ρ={RHO}, a={A}. seed={args.seed}.",
        "- δ\\* 무관 — encode+assign 시간은 boundary 결정 여부와 독립이므로",
        "  Ours 의 모든 percentile/oracle 행에 동일 latency 적용.",
        "",
        "## 2. 결과 (벤치별)",
        "",
        "| 벤치 | n turn (timed) | Pre. encode (ms) mean / p50 / p90 | "
        "Seg. assign (ms) mean / p50 / p90 | Total (ms) mean / p50 / p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in ("tiage", "dialseg711", "superseg"):
        r = results[ds]
        e, s, t = r["enc"], r["seg"], r["total"]
        L.append(
            f"| {ds} | {r['n_turn_timed']} | "
            f"{e['mean']:.2f} / {e['p50']:.2f} / {e['p90']:.2f} | "
            f"{s['mean']:.4f} / {s['p50']:.4f} / {s['p90']:.4f} | "
            f"{t['mean']:.2f} / {t['p50']:.2f} / {t['p90']:.2f} |")
    L += [
        "",
        "## 3. cross-benchmark 평균 (Ours 행에 단일 latency 보고용)",
        "",
        f"- **Pre. (encode)**: mean = {enc_all['mean']:.2f} ms · p50 "
        f"{enc_all['p50']:.2f} · p90 {enc_all['p90']:.2f} (n={enc_all['n']})",
        f"- **Seg. (assign)**: mean = {seg_all['mean']:.4f} ms · p50 "
        f"{seg_all['p50']:.4f} · p90 {seg_all['p90']:.4f}",
        f"- **Total (Pre.+Seg.)**: mean = {tot_all['mean']:.2f} ms · p50 "
        f"{tot_all['p50']:.2f} · p90 {tot_all['p90']:.2f}",
        f"- cold-start (model load): {load_s:.2f} s — per-turn 과 분리.",
        "",
        "## 4. 표 셀 매핑 (dts_result.md 의 Online Ours 4 행)",
        "",
        f"- **Pre.** 셀 = {enc_all['mean']:.0f} ms (cross-bench mean, encoder "
        "single-utt forward).",
        f"- **Seg.** 셀 = {seg_all['mean']:.3f} ms (cross-bench mean, HiOnTop.assign).",
        "- p70/p75/p80/oracle 4 행 모두 동일 (δ\\* 무관).",
        "",
        "## 5. 한계 / 검증 미해결",
        "",
        f"- 표본 = 벤치당 {args.turns_per_bench} turn budget subsample "
        "(seed=0). content-independent 한 latency 특성상 전수 측정 대비 "
        "편차 작을 것이나, 완전 일치 보장은 분포 가정에 의존.",
        "- CPU only (machine: 본 측정 머신). GPU 환경에서는 encode "
        "비용 한 자릿수 ms 수준으로 떨어질 가능성 — 그 경우 별도 측정 필요.",
        "- 인코더 lazy init / jit / GC 영향 — warmup 10 fwd 로 1차 제거, "
        "잔여 노이즈는 p50 사용시 영향 작음.",
        "- baseline (GreedySeg 13 ms 등) 와 동일 single-utt forward 정의로 "
        "비교 가능. metric 가족 차이는 별도 — 본 REPORT 는 latency 만.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out_dir/'REPORT.md'}", flush=True)
    print(f"DONE → {out_dir/'latency.json'}", flush=True)


if __name__ == "__main__":
    main()
