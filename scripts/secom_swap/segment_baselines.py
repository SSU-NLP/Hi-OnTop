"""Unified runner for online baseline segmenters → SeCom segments JSONL.

Supports four methods, each consuming the MTB+ session-list input the same
way SeCom's segment.py does (session-by-session, fresh segmenter per
session) and producing SeCom-compatible output (``segments: List[List[str]]``).

Methods
-------
- ``texttiling`` — StreamingTextTiling (NLTK-style block cosine + Welford
  threshold + min_gap; no model).
- ``greedyseg``  — GreedySegOnlineDelay2 (BERT bounded-lookahead).
- ``graphseg``   — GraphSegWindow (IC × GloVe, Bron-Kerbosch clique).
- ``csm``        — Hi-OnTop CSM-online (DSE-BERT + our trained ckpt + depth
  threshold, delay-2 window). Implementation in this file (lightweight).

Outputs
-------
- ``--save_path`` (JSONL): copy of input with ``segments`` field added.
- ``--latency_path`` (JSON): timing roll-up across all sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.secom_swap._streaming_adapter import (
    StreamingSegmentLatency,
    run_streaming_segmenter,
)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _build_factory(method: str, args: argparse.Namespace):
    """Returns a factory ``() -> StreamingSegmenter`` that re-uses any
    heavy loaded resources (GloVe vocab, BERT weights, CSM ckpt) across
    sessions — only the per-session segmenter STATE is fresh."""

    if method == "texttiling":
        from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling
        return lambda: StreamingTextTiling(
            w=args.tt_w, k=args.tt_k, c=args.tt_c,
            min_gap=args.tt_min_gap, warmup_gaps=args.tt_warmup,
        )

    if method == "greedyseg":
        from hi_ontop.baselines.greedyseg_delay2 import GreedySegOnlineDelay2
        # Share BERT weights across sessions by warming one instance then
        # cloning its _model/_tokenizer references.
        prototype = GreedySegOnlineDelay2(
            backbone=args.gs_backbone,
            device=args.device,
            window_size=args.gs_window,
            sim_threshold=args.gs_threshold,
            max_seq_length=args.gs_max_len,
        )
        prototype._ensure_model()

        def factory():
            inst = GreedySegOnlineDelay2(
                backbone=args.gs_backbone,
                device=args.device,
                window_size=args.gs_window,
                sim_threshold=args.gs_threshold,
                max_seq_length=args.gs_max_len,
            )
            inst._tokenizer = prototype._tokenizer
            inst._model = prototype._model
            inst._device_resolved = prototype._device_resolved
            # Also share lazily-imported torch handle to avoid re-import per session.
            if hasattr(prototype, "_torch"):
                inst._torch = prototype._torch
            return inst
        return factory

    if method == "graphseg":
        from hi_ontop.baselines.graphseg_window import GraphSegWindowD
        print(f"[graphseg] loading GloVe ({args.glove_path}) — first instance...", flush=True)
        prototype = GraphSegWindowD(
            glove_path=args.glove_path,
            window_d=args.graphseg_window,
            sim_threshold=args.graphseg_threshold,
            min_seg_size=args.graphseg_min_seg,
        )
        # Force GloVe + IC table load once via _ensure_resources
        prototype._ensure_resources()
        print(f"[graphseg] loaded: glove={len(prototype._glove)} words, "
              f"ic_table={len(prototype._ic_table) if prototype._ic_table else 0} words",
              flush=True)

        def factory():
            inst = GraphSegWindowD(
                glove_path=args.glove_path,
                window_d=args.graphseg_window,
                sim_threshold=args.graphseg_threshold,
                min_seg_size=args.graphseg_min_seg,
            )
            # share loaded GloVe + IC tables (no re-load)
            inst._glove = prototype._glove
            inst._glove_dim = prototype._glove_dim
            inst._ic_table = prototype._ic_table
            return inst
        return factory

    if method == "csm":
        from hi_ontop.baselines.csm_online import CSMOnlineDelay2
        prototype = CSMOnlineDelay2(
            ckpt_path=args.csm_ckpt,
            device=args.device,
            alpha=args.csm_alpha,
            delay=args.csm_delay,
            warmup_gaps=args.csm_warmup,
            min_gap=args.csm_min_gap,
        )
        prototype._ensure_loaded()

        def factory():
            inst = CSMOnlineDelay2(
                ckpt_path=args.csm_ckpt,
                device=args.device,
                alpha=args.csm_alpha,
                delay=args.csm_delay,
                warmup_gaps=args.csm_warmup,
                min_gap=args.csm_min_gap,
            )
            inst._model = prototype._model
            inst._tokenizer = prototype._tokenizer
            inst._device_resolved = prototype._device_resolved
            return inst
        return factory

    raise ValueError(f"unknown method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["texttiling", "greedyseg", "graphseg", "csm"])
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--latency_path", required=True)
    ap.add_argument("--device", default="cpu")
    # TextTiling
    ap.add_argument("--tt_w", type=int, default=10)
    ap.add_argument("--tt_k", type=int, default=6)
    ap.add_argument("--tt_c", type=float, default=0.5)
    ap.add_argument("--tt_min_gap", type=int, default=4)
    ap.add_argument("--tt_warmup", type=int, default=3)
    # GreedySeg
    ap.add_argument("--gs_backbone", default="bert-base-uncased")
    ap.add_argument("--gs_window", type=int, default=2)
    ap.add_argument("--gs_threshold", type=float, default=0.5)
    ap.add_argument("--gs_max_len", type=int, default=50)
    # GraphSeg
    ap.add_argument("--glove_path", default="benchmarks/glove/glove.6B.300d.txt")
    ap.add_argument("--graphseg_window", type=int, default=10)
    ap.add_argument("--graphseg_threshold", type=float, default=0.25)
    ap.add_argument("--graphseg_min_seg", type=int, default=3)
    # CSM
    ap.add_argument("--csm_ckpt",
                    default="outputs/runs/_misc/cpt_277000.pth")
    ap.add_argument("--csm_alpha", type=float, default=0.0,
                    help="threshold = mean_depth + alpha * std_depth")
    ap.add_argument("--csm_delay", type=int, default=2)
    ap.add_argument("--csm_warmup", type=int, default=3)
    ap.add_argument("--csm_min_gap", type=int, default=2)
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latency_path).parent.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(Path(args.load_path))
    print(f"method: {args.method}, n_conv: {len(data)}", flush=True)

    # _build_factory now warms heavy models inside (prototype pattern); the
    # returned factory creates lightweight per-session segmenters that share
    # the cached weights.
    factory = _build_factory(args.method, args)

    per_conv_latency = []
    results = []
    agg = StreamingSegmentLatency()
    for idx, sample in enumerate(tqdm(data, desc=args.method)):
        segments, lat = run_streaming_segmenter(sample["sessions"], factory)
        sample["segments"] = segments
        results.append(sample)
        per_conv_latency.append({
            "conversation_id": sample["conversation_id"],
            "n_sessions": lat.n_sessions,
            "n_exchanges": lat.n_exchanges,
            "n_segments": len(segments),
            "total_sec": lat.total_sec,
            "ms_per_exchange": lat.ms_per_exchange,
        })
        agg.n_sessions += lat.n_sessions
        agg.n_exchanges += lat.n_exchanges
        agg.push_sec += lat.push_sec
        agg.flush_sec += lat.flush_sec
        agg.total_sec += lat.total_sec
        agg.neural_sec += lat.neural_sec
        agg.preprocess_sec += lat.preprocess_sec
        agg.per_turn.extend(lat.per_turn)
        print(
            f"  conv {idx}: {lat.n_exchanges} ex → {len(segments)} segs, "
            f"{lat.total_sec*1000:.0f}ms total, {lat.ms_per_exchange:.2f}ms/ex",
            flush=True,
        )

    with open(args.save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_seg_total = sum(c["n_segments"] for c in per_conv_latency)
    summary = {
        "method": args.method,
        "n_conv": len(per_conv_latency),
        "n_exchanges": agg.n_exchanges,
        "n_segments": n_seg_total,
        "avg_exchanges_per_segment": agg.n_exchanges / max(1, n_seg_total),
        **agg.asdict(),
        "per_conv": per_conv_latency,
    }
    with open(args.latency_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{args.method}: {n_seg_total} segments / {agg.n_exchanges} ex, "
          f"{agg.ms_per_exchange:.2f} ms/ex (push) — saved {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
