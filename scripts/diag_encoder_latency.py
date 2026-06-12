#!/usr/bin/env python3
"""Phase 1+2 — 인코더 후보 per-utterance latency + ONNX 임베딩 동등성.

온라인 배포는 발화 1개씩 인코딩 → per-utterance 단건 latency 가 정직한
지표 (batch 아님). 4 후보를 같은 발화 표본·같은 CPU 에서 비교:

  mpnet-PyTorch / mpnet-ONNX / MiniLM-PyTorch / MiniLM-ONNX

또 ONNX 가 PyTorch 와 임베딩이 동일한지 (cos≈1) 검증 — 동일하면 ONNX
변형의 segmentation Score 는 PyTorch 결과를 그대로 상속할 수 있다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.baselines._seg_utils import parse_defdts_dialogue  # noqa: E402

DEFDTS = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"
MPNET = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def sample_utterances(per_ds: int = 200) -> list[str]:
    out = []
    for ds in ("tiage", "dialseg711", "superseg"):
        utts = []
        for ln in (DEFDTS / f"{ds}_test.jsonl").read_text().splitlines():
            if not ln.strip():
                continue
            u, _ = parse_defdts_dialogue(json.loads(ln)["dialogue"])
            utts.extend(u)
            if len(utts) >= per_ds:
                break
        out.extend(utts[:per_ds])
    return out


def load_encoder(model_name: str, backend: str):
    from sentence_transformers import SentenceTransformer
    if backend == "onnx":
        return SentenceTransformer(model_name, backend="onnx",
                                   model_kwargs={"provider": "CPUExecutionProvider"})
    return SentenceTransformer(model_name, device="cpu")


def per_utt_latency(model, utts: list[str], warmup: int = 15) -> dict:
    for u in utts[:warmup]:                       # warm-up (graph/cache)
        model.encode(u, normalize_embeddings=True, show_progress_bar=False)
    lat = []
    for u in utts:
        t0 = time.perf_counter()
        model.encode(u, normalize_embeddings=True, show_progress_bar=False)
        lat.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(lat)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)))


def embed_all(model, utts):
    return np.asarray(model.encode(utts, normalize_embeddings=True,
                                   show_progress_bar=False))


def main():
    utts = sample_utterances(200)
    print(f"표본 발화 {len(utts)}개 (tiage/dialseg711/superseg 각 200)\n", flush=True)

    variants = [
        ("mpnet  PyTorch", MPNET, "pt"),
        ("mpnet  ONNX", MPNET, "onnx"),
        ("MiniLM PyTorch", MINILM, "pt"),
        ("MiniLM ONNX", MINILM, "onnx"),
    ]
    rows, embs = {}, {}
    for label, name, backend in variants:
        print(f"[load] {label} ...", flush=True)
        try:
            model = load_encoder(name, backend)
            lat = per_utt_latency(model, utts)
            embs[label] = embed_all(model, utts)
            rows[label] = lat
            print(f"  {label}: mean={lat['mean']:.2f}ms p50={lat['p50']:.2f} "
                  f"p90={lat['p90']:.2f} p99={lat['p99']:.2f}", flush=True)
        except Exception as e:
            rows[label] = None
            print(f"  {label}: FAILED — {type(e).__name__}: {e}", flush=True)

    # ONNX ↔ PyTorch 임베딩 동등성
    def cos_eq(a, b):
        if a is None or b is None:
            return None
        return float(np.mean([np.dot(x, y) for x, y in zip(a, b)]))

    eq_mpnet = cos_eq(embs.get("mpnet  PyTorch"), embs.get("mpnet  ONNX"))
    eq_minilm = cos_eq(embs.get("MiniLM PyTorch"), embs.get("MiniLM ONNX"))

    out_dir = REPO / "outputs" / "experiments" / "2026-05-23_encoder_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [
        "# 인코더 후보 per-utterance latency (Phase 1+2)",
        "",
        f"표본 = Def-DTS 번들 test 발화 {len(utts)}개. 단건(turn당 1발화) "
        "인코딩 — 온라인 배포 조건. CPU. warm-up 15회 제외.",
        "",
        "| 인코더 | mean (ms) | p50 | p90 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, _, _ in variants:
        r = rows[label]
        if r:
            L.append(f"| {label} | **{r['mean']:.2f}** | {r['p50']:.2f} | "
                     f"{r['p90']:.2f} | {r['p99']:.2f} |")
        else:
            L.append(f"| {label} | FAILED | — | — | — |")
    L += [
        "",
        "## ONNX ↔ PyTorch 임베딩 동등성 (표본 평균 cos)",
        "",
        f"- mpnet : cos = {eq_mpnet:.6f}" if eq_mpnet is not None
        else "- mpnet : 측정 불가",
        f"- MiniLM: cos = {eq_minilm:.6f}" if eq_minilm is not None
        else "- MiniLM: 측정 불가",
        "",
        "cos ≈ 1.0 이면 ONNX 변형의 segmentation Score 는 PyTorch 결과를 "
        "그대로 상속 가능 (재평가 불필요).",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out_dir/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
