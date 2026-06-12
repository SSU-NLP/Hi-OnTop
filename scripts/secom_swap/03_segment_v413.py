"""v4.1.3 segmentation runner — mirrors SeCom's ``experiment/segment.py``.

Reads the same input JSONL (``data/mtbp/mtbp.jsonl``) and writes the same
output schema (each sample gains ``sample["segments"]: List[List[str]]``),
so SeCom's downstream pipeline (``compress.py`` → ``retrieve.py`` →
``chat.py``) consumes our segments transparently.

Differences vs SeCom's segment.py:
- No LLM call. Uses :class:`hi_ontop.secom_adapter.HiOnTopSecomSegmenter`.
- Records per-conversation latency (encode + assign).
- Records boundary-strength histogram (very_weak / weak / normal / strong).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from hi_ontop.secom_adapter import HiOnTopSecomSegmenter

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--load_path",
        default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"),
    )
    ap.add_argument(
        "--save_path",
        default=str(
            REPO_ROOT
            / "benchmarks/SeCom/experiment/result/mtbp/v413seg_mtbp.jsonl"
        ),
    )
    ap.add_argument(
        "--latency_path",
        default=str(
            REPO_ROOT
            / "outputs/experiments/2026-05-21_v413_secom_swap/latency_v413.json"
        ),
    )
    ap.add_argument(
        "--encoder",
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    )
    ap.add_argument(
        "--encoder_backend",
        choices=["api", "local", "onnx"],
        default="api",
        help=(
            "api = Crts /v1/embeddings (default); local = sentence-transformers; "
            "onnx = sentence-transformers ONNX backend (e.g. MiniLM-int8)."
        ),
    )
    ap.add_argument(
        "--onnx_file", default="onnx/model_quint8_avx2.onnx",
        help="ONNX file inside model repo (only for --encoder_backend onnx).",
    )
    ap.add_argument("--delta_star", type=float, required=True)
    ap.add_argument("--dim", type=int, default=768)
    # Hi-OnTop dual-signal blend (ablation): a=0.0 → ctx-only, a=1.0 → prev-only,
    # default 0.5 = paper baseline.
    ap.add_argument("--ctx_blend_a", type=float, default=0.5)
    ap.add_argument("--ctx_window", type=int, default=2)
    ap.add_argument("--ctx_decay", type=float, default=0.7)
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latency_path).parent.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(Path(args.load_path))
    print(f"n_conv: {len(data)}")

    if args.encoder_backend == "api":
        from dotenv import load_dotenv

        load_dotenv()
        from hi_ontop.embedding import make_encoder

        encoder = make_encoder(backend="api", model=args.encoder)
        print(f"encoder: Crts API ({args.encoder})")
    else:
        from sentence_transformers import SentenceTransformer

        if args.encoder_backend == "onnx":
            encoder = SentenceTransformer(
                args.encoder, backend="onnx",
                model_kwargs={"provider": "CPUExecutionProvider",
                              "file_name": args.onnx_file},
            )
            print(f"encoder: ONNX ({args.encoder} :: {args.onnx_file})")
        else:
            encoder = SentenceTransformer(args.encoder)
            print(f"encoder: local sentence-transformers ({args.encoder})")
    seg = HiOnTopSecomSegmenter(
        encoder=encoder,
        dim=args.dim,
        delta_star=args.delta_star,
        hiontop_kwargs={
            "ctx_window": args.ctx_window,
            "ctx_decay": args.ctx_decay,
            "ctx_blend_a": args.ctx_blend_a,
        },
    )

    per_conv_latency: list[dict] = []
    results: list[dict] = []

    for idx, sample in enumerate(tqdm(data, desc="segmenting")):
        segments = seg.segment(sample["sessions"])
        sample["segments"] = segments
        results.append(sample)
        lat = seg.last_latency.asdict()
        lat["conversation_id"] = sample["conversation_id"]
        lat["n_segments"] = len(segments)
        per_conv_latency.append(lat)
        print(
            f"  conv {idx} ({sample['conversation_id']}): "
            f"{lat['n_exchanges']} ex → {len(segments)} segs, "
            f"{lat['total_sec']*1000:.1f}ms total, "
            f"{lat['sec_per_exchange']*1000:.2f}ms/ex"
        )

    with open(args.save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ex_total = sum(lat["n_exchanges"] for lat in per_conv_latency)
    n_seg_total = sum(lat["n_segments"] for lat in per_conv_latency)
    total_sec = sum(lat["total_sec"] for lat in per_conv_latency)
    encode_sec = sum(lat["encode_sec"] for lat in per_conv_latency)
    segment_sec = sum(lat["segment_sec"] for lat in per_conv_latency)

    summary = {
        "encoder": args.encoder,
        "delta_star": args.delta_star,
        "n_conv": len(per_conv_latency),
        "n_exchanges": n_ex_total,
        "n_segments": n_seg_total,
        "avg_exchanges_per_segment": n_ex_total / max(1, n_seg_total),
        "boundary_strength_total": seg.boundary_strength_total,
        "total_sec": total_sec,
        "encode_sec": encode_sec,
        "segment_sec": segment_sec,
        "ms_per_exchange_total": total_sec * 1000 / max(1, n_ex_total),
        "ms_per_exchange_segment_only": segment_sec * 1000 / max(1, n_ex_total),
        "per_conv": per_conv_latency,
    }
    with open(args.latency_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nlatency summary -> {args.latency_path}")
    print(
        f"v4.1.3: {n_seg_total} segments / {n_ex_total} exchanges, "
        f"{summary['ms_per_exchange_total']:.2f} ms/exchange total, "
        f"{summary['ms_per_exchange_segment_only']:.3f} ms/exchange (assign-only)"
    )


if __name__ == "__main__":
    main()
