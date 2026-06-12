"""δ* calibration for Hi-OnTop on Long-MT-Bench+ with the mpnet encoder.

Hi-OnTop' default ``delta_star`` was tuned on TIAGE *train* with Hi-OnTop's bge
encoder. SeCom uses ``multi-qa-mpnet-base-dot-v1`` (different embedding
geometry), so the cosine-distance distribution differs → ``delta_star``
must be re-estimated. Long-MT-Bench+ has no ground-truth boundaries, so we
use an unsupervised percentile heuristic (p80 ≈ ~20% of turns become
boundaries ≈ 2-3 segments per 13-turn session).

Two modes
---------
- ``delta_prev``: percentiles of raw ``δ_prev = 1 - cos(s_{t-1}, s_t)``.
  This is **m-independent** (no causal window).
- ``delta_eff``: percentiles of ``δ_eff = a·δ_prev + (1-a)·δ_ctx`` — the
  quantity Hi-OnTop actually thresholds. ``δ_ctx`` depends on the causal
  window ``m`` (``ctx_window``), so this mode **must be re-run whenever
  ``ctx_window`` changes** (e.g. the m=3 → m=2 default change). This is the
  principled target for δ* since δ* is compared against δ_eff, not δ_prev.

Encoder backend
---------------
``--encoder_backend api`` (default) routes embeddings through the Crts
``/v1/embeddings`` endpoint; ``local`` uses sentence-transformers. The Crts
mpnet output is bit-identical to the local model (cos=1.0), so the choice
does not change the calibrated value — ``api`` just offloads CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_mtbp(path: Path) -> list[dict]:
    data = []
    with path.open() as f:
        for line in f:
            data.append(json.loads(line))
    return data


def percentile_block(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p60": float(np.percentile(arr, 60)),
        "p65": float(np.percentile(arr, 65)),
        "p70": float(np.percentile(arr, 70)),
        "p75": float(np.percentile(arr, 75)),
        "p80": float(np.percentile(arr, 80)),
        "p85": float(np.percentile(arr, 85)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"),
    )
    ap.add_argument(
        "--encoder",
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    )
    ap.add_argument(
        "--encoder_backend", choices=["api", "local", "onnx"], default="api",
        help="api = Crts /v1/embeddings (default); local = sentence-transformers; "
             "onnx = sentence-transformers ONNX backend (e.g. MiniLM-int8).",
    )
    ap.add_argument(
        "--onnx_file", default="onnx/model_quint8_avx2.onnx",
        help="ONNX file inside the model repo (only for --encoder_backend onnx).",
    )
    ap.add_argument("--n_conv", type=int, default=11)
    ap.add_argument(
        "--mode", choices=["delta_prev", "delta_eff"], default="delta_eff",
        help="delta_eff = the quantity Hi-OnTop thresholds (m-dependent).",
    )
    ap.add_argument("--ctx_window", type=int, default=2)
    ap.add_argument("--ctx_decay", type=float, default=0.7)
    ap.add_argument("--ctx_blend_a", type=float, default=0.5)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "outputs/experiments/2026-05-21_v413_secom_swap"
            / "delta_star_calibration.json"
        ),
    )
    args = ap.parse_args()

    data = load_mtbp(Path(args.data))[: args.n_conv]
    print(f"calibrating on {len(data)} conversations · mode={args.mode}", flush=True)

    # ── encoder ──────────────────────────────────────────────────────────
    if args.encoder_backend == "api":
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        from hi_ontop.embedding import make_encoder

        enc = make_encoder(backend="api", model=args.encoder)

        def encode(texts: list[str]) -> np.ndarray:
            return np.asarray(enc.encode(texts), dtype=np.float64)

        print(f"encoder: Crts API ({args.encoder})", flush=True)
    else:
        from sentence_transformers import SentenceTransformer

        if args.encoder_backend == "onnx":
            st = SentenceTransformer(
                args.encoder, backend="onnx",
                model_kwargs={"provider": "CPUExecutionProvider",
                              "file_name": args.onnx_file},
            )
            print(f"encoder: ONNX ({args.encoder} :: {args.onnx_file})",
                  flush=True)
        else:
            st = SentenceTransformer(args.encoder)
            print(f"encoder: local sentence-transformers ({args.encoder})",
                  flush=True)

        def encode(texts: list[str]) -> np.ndarray:
            return st.encode(
                texts, batch_size=64, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float64)

    # ── flatten + encode once ────────────────────────────────────────────
    flat_sents: list[str] = []
    session_spans: list[tuple[int, int]] = []  # (start_idx, length)
    for conv in data:
        for sess in conv["sessions"]:
            if len(sess) < 2:
                continue
            session_spans.append((len(flat_sents), len(sess)))
            flat_sents.extend(sess)
    print(
        f"total sentences to encode: {len(flat_sents)} "
        f"across {len(session_spans)} sessions", flush=True,
    )
    vecs = encode(flat_sents)
    print(f"encoded shape={vecs.shape}", flush=True)

    # ── collect deltas ───────────────────────────────────────────────────
    all_deltas: list[float] = []
    per_session_n_turns: list[int] = []
    if args.mode == "delta_prev":
        for start, n in session_spans:
            sv = vecs[start : start + n]
            cos = (sv[:-1] * sv[1:]).sum(axis=1)
            all_deltas.extend((1.0 - cos).tolist())
            per_session_n_turns.append(n)
    else:  # delta_eff — replay each session through Hi-OnTop
        from hi_ontop.hi_ontop import HiOnTop

        for start, n in session_spans:
            sv = vecs[start : start + n]
            seg = HiOnTop(
                dim=args.dim,
                delta_star=1.0,  # irrelevant — δ_eff is independent of δ*
                ctx_window=args.ctx_window,
                ctx_decay=args.ctx_decay,
                ctx_blend_a=args.ctx_blend_a,
            )
            for v in sv:
                seg.assign(v)
            # turn 0 has δ_eff=0.0 (no s_{t-1}); keep turns 1..n-1
            for rec in seg.history()[1:]:
                all_deltas.append(float(rec["delta_eff"]))
            per_session_n_turns.append(n)

    arr = np.array(all_deltas)
    pct = percentile_block(arr)
    summary = {
        "encoder": args.encoder,
        "encoder_backend": args.encoder_backend,
        "mode": args.mode,
        "ctx_window": args.ctx_window if args.mode == "delta_eff" else None,
        "ctx_decay": args.ctx_decay if args.mode == "delta_eff" else None,
        "ctx_blend_a": args.ctx_blend_a if args.mode == "delta_eff" else None,
        "n_conv": len(data),
        "n_sessions": len(per_session_n_turns),
        "n_delta_samples": int(arr.size),
        args.mode: pct,
        "session_turns": {
            "mean": float(np.mean(per_session_n_turns)),
            "min": int(np.min(per_session_n_turns)),
            "max": int(np.max(per_session_n_turns)),
        },
        "candidate_delta_star": {
            "p60": pct["p60"], "p65": pct["p65"],
            "p70": pct["p70"], "p75": pct["p75"],
            "p80": pct["p80"], "p85": pct["p85"],
            "p90": pct["p90"],
        },
        "recommended_initial": pct["p80"],
        "note": (
            f"mode={args.mode}: p80 ≈ boundary at ~20% of turns ≈ 2-3 segments "
            "per 13-turn session. "
            + (
                f"δ_eff at ctx_window={args.ctx_window} — the quantity Hi-OnTop "
                "actually thresholds; re-run if ctx_window changes."
                if args.mode == "delta_eff"
                else "raw δ_prev — m-independent proxy; δ* is applied to δ_eff "
                "so prefer --mode delta_eff."
            )
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
