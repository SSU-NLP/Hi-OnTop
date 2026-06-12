"""Per-turn CPU encode latency for the Hi-OnTop mpnet encoder.

Hi-OnTop' ``assign()`` is genuinely per-turn, but the SeCom adapter encodes a
whole session in one batched call. The baseline segmenters (CSM / GreedySeg /
GraphSeg) are measured turn-by-turn via streaming ``push()``. For an
apples-to-apples *online* End-to-end latency, the encoder must also be timed
in batch-size-1, one-turn-at-a-time mode on the **same local CPU**.

This script encodes MTB+ exchanges one at a time with the local
sentence-transformers mpnet model (CPU) and reports per-turn ms statistics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"),
    )
    ap.add_argument(
        "--model", default="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    ap.add_argument("--n_turns", type=int, default=200, help="sample size")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--backend", choices=["torch", "onnx"], default="torch",
        help="torch = sentence-transformers default; onnx = ONNX runtime (e.g. int8).",
    )
    ap.add_argument(
        "--onnx_file", default="onnx/model_quint8_avx2.onnx",
        help="ONNX file inside model repo (only for --backend onnx).",
    )
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "outputs/experiments/2026-05-21_v413_secom_swap/encode_latency_cpu.json"
        ),
    )
    args = ap.parse_args()

    # flatten exchanges
    turns: list[str] = []
    with open(args.data) as f:
        for line in f:
            for sess in json.loads(line)["sessions"]:
                turns.extend(sess)
    sample = turns[: args.warmup + args.n_turns]
    print(f"loaded {len(turns)} turns, sampling {len(sample)} "
          f"({args.warmup} warmup + {args.n_turns} timed)", flush=True)

    from sentence_transformers import SentenceTransformer

    if args.backend == "onnx":
        model = SentenceTransformer(
            args.model, backend="onnx",
            model_kwargs={"provider": "CPUExecutionProvider",
                          "file_name": args.onnx_file},
        )
        print(f"model on CPU (ONNX): {args.model} :: {args.onnx_file}",
              flush=True)
    else:
        model = SentenceTransformer(args.model, device="cpu")
        print(f"model on CPU (torch): {args.model}", flush=True)

    per_turn_ms: list[float] = []
    for i, text in enumerate(sample):
        t0 = time.perf_counter()
        model.encode([text], batch_size=1, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= args.warmup:
            per_turn_ms.append(dt)

    arr = np.array(per_turn_ms)
    out = {
        "model": args.model,
        "backend": args.backend,
        "onnx_file": args.onnx_file if args.backend == "onnx" else None,
        "device": "cpu",
        "mode": "batch_size=1 per-turn (online streaming)",
        "n_timed": int(arr.size),
        "n_warmup": args.warmup,
        "encode_ms_per_turn": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
