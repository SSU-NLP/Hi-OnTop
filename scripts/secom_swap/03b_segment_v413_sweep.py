"""Run v4.1.3 segmentation with multiple δ* candidates and pick the one whose
total n_segments is closest to the baseline.

Why: comparing v4.1.3 vs baseline downstream QA is only fair if both produce
roughly the same number of memory units. Picking δ* post-hoc to match
n_segments removes "memory granularity" as a confound and isolates the
effect of *which* boundaries are placed (the actual segmentation quality).

Outputs:
- segments JSONL for the chosen δ* (compatible with subsequent compress/
  retrieve/chat/eval stages)
- ``latency_ours.json`` (timing + boundary-strength rollup for chosen δ*)
- ``delta_star_sweep.json`` (per-δ*-candidate n_segments / latency, plus the
  selection rationale)

Reuses the encoder cache across all candidates (mpnet runs once per
session, then assign() is called per-candidate). Cheap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"))
    ap.add_argument("--baseline_path", required=True,
                    help="segments JSONL from baseline (gpt-4o-mini) — for n_segments matching")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--latency_path", required=True)
    ap.add_argument("--sweep_path", required=True)
    ap.add_argument("--deltas", default="0.10,0.15,0.20,0.25,0.30,0.40",
                    help="comma-separated δ* candidates")
    ap.add_argument("--encoder", default="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    args = ap.parse_args()

    candidates = [float(x) for x in args.deltas.split(",")]
    print(f"δ* candidates: {candidates}", flush=True)

    data = load_jsonl(Path(args.load_path))
    baseline = {s["conversation_id"]: s for s in load_jsonl(Path(args.baseline_path))}
    baseline_n_seg = {cid: len(b["segments"]) for cid, b in baseline.items()}
    baseline_total = sum(baseline_n_seg.values())
    print(f"baseline total n_segments = {baseline_total} (per-conv: {baseline_n_seg})", flush=True)

    # Encode all sentences once, per conversation × session
    print(f"loading encoder {args.encoder}", flush=True)
    enc = SentenceTransformer(args.encoder)

    # cache: per (conv_idx, sess_idx) → encoded vectors
    encoded: dict[tuple[int, int], np.ndarray] = {}
    encode_start = time.perf_counter()
    flat_sents = []
    spans = []  # (conv_idx, sess_idx, start_in_flat, length)
    for cidx, conv in enumerate(data):
        for sidx, sess in enumerate(conv["sessions"]):
            spans.append((cidx, sidx, len(flat_sents), len(sess)))
            flat_sents.extend(sess)
    print(f"encoding {len(flat_sents)} sentences in one batch", flush=True)
    vecs = enc.encode(flat_sents, batch_size=64, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float64)
    for cidx, sidx, st, n in spans:
        encoded[(cidx, sidx)] = vecs[st : st + n]
    encode_sec = time.perf_counter() - encode_start
    print(f"encoded all in {encode_sec:.1f}s", flush=True)

    # Import segmenter
    from hi_ontop.hi_ontop import HiOnTop

    sweep_results = []
    per_candidate_segments: dict[float, list[dict]] = {}
    for delta_star in candidates:
        t0 = time.perf_counter()
        cand_segs_per_conv: list[dict] = []
        per_turn_times: list[float] = []
        band_total = {"very_weak": 0, "weak": 0, "normal": 0, "strong": 0}
        for cidx, conv in enumerate(data):
            sample_out = {"conversation_id": conv["conversation_id"],
                          "sessions": conv["sessions"],
                          "questions": conv["questions"],
                          "answers": conv["answers"],
                          "segments": []}
            for sidx, sess in enumerate(conv["sessions"]):
                seg = HiOnTop(dim=768, delta_star=delta_star)
                vecs_sess = encoded[(cidx, sidx)]
                current: list[str] = []
                for v, ex in zip(vecs_sess, sess):
                    tt = time.perf_counter()
                    _, is_bnd = seg.assign(v)
                    per_turn_times.append(time.perf_counter() - tt)
                    if is_bnd and current:
                        sample_out["segments"].append(current)
                        current = []
                    current.append(ex)
                if current:
                    sample_out["segments"].append(current)
                band = seg.boundary_strength()
                for k, v in band.items():
                    band_total[k] += v
            cand_segs_per_conv.append(sample_out)
        dt = time.perf_counter() - t0
        n_seg_total = sum(len(s["segments"]) for s in cand_segs_per_conv)
        per_conv_seg = {s["conversation_id"]: len(s["segments"]) for s in cand_segs_per_conv}
        per_conv_delta = sum(abs(per_conv_seg[c] - baseline_n_seg[c]) for c in per_conv_seg)
        sweep_results.append({
            "delta_star": delta_star,
            "n_segments_total": n_seg_total,
            "abs_delta_baseline_total": abs(n_seg_total - baseline_total),
            "abs_delta_baseline_per_conv_sum": per_conv_delta,
            "assign_only_sec": dt,
            "ms_per_exchange_assign_only": dt * 1000 / max(1, len(per_turn_times)),
            "p50_ms_per_turn": float(np.median(per_turn_times)) * 1000,
            "p95_ms_per_turn": float(np.percentile(per_turn_times, 95)) * 1000,
            "boundary_strength": band_total,
            "per_conv_n_segments": per_conv_seg,
        })
        per_candidate_segments[delta_star] = cand_segs_per_conv
        print(f"  δ*={delta_star}: n_seg={n_seg_total} (baseline={baseline_total}, "
              f"Δ={abs(n_seg_total-baseline_total):+d}), {dt*1000/max(1,len(per_turn_times)):.3f} ms/ex",
              flush=True)

    # Pick best: minimize per_conv_delta (more local), tiebreak by total delta
    best = min(sweep_results,
               key=lambda r: (r["abs_delta_baseline_per_conv_sum"], r["abs_delta_baseline_total"]))
    chosen_delta = best["delta_star"]
    print(f"\nchosen δ* = {chosen_delta} (per-conv delta sum = "
          f"{best['abs_delta_baseline_per_conv_sum']}, total delta = "
          f"{best['abs_delta_baseline_total']})", flush=True)

    # Write chosen segments
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for s in per_candidate_segments[chosen_delta]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"saved chosen segments -> {args.save_path}")

    # Write latency JSON for the chosen δ* (compatible with summarize.py format)
    n_ex_total = sum(len(sess) for conv in data for sess in conv["sessions"])
    chosen_metrics = next(r for r in sweep_results if r["delta_star"] == chosen_delta)
    latency = {
        "encoder": args.encoder,
        "delta_star": chosen_delta,
        "n_conv": len(data),
        "n_exchanges": n_ex_total,
        "n_segments": chosen_metrics["n_segments_total"],
        "avg_exchanges_per_segment": n_ex_total / max(1, chosen_metrics["n_segments_total"]),
        "boundary_strength_total": chosen_metrics["boundary_strength"],
        "encode_sec": encode_sec,
        "segment_sec": chosen_metrics["assign_only_sec"],
        "total_sec": encode_sec + chosen_metrics["assign_only_sec"],
        "ms_per_exchange_total": (encode_sec + chosen_metrics["assign_only_sec"]) * 1000 / max(1, n_ex_total),
        "ms_per_exchange_segment_only": chosen_metrics["ms_per_exchange_assign_only"],
        "ms_per_turn_assign_p50": chosen_metrics["p50_ms_per_turn"],
        "ms_per_turn_assign_p95": chosen_metrics["p95_ms_per_turn"],
        # alias for summarize.py compatibility
        "ms_per_exchange": (encode_sec + chosen_metrics["assign_only_sec"]) * 1000 / max(1, n_ex_total),
    }
    Path(args.latency_path).write_text(json.dumps(latency, indent=2, ensure_ascii=False))
    print(f"latency -> {args.latency_path}")

    # Write full sweep results
    Path(args.sweep_path).write_text(json.dumps({
        "candidates": candidates,
        "chosen_delta_star": chosen_delta,
        "baseline_total_n_segments": baseline_total,
        "baseline_per_conv": baseline_n_seg,
        "encode_sec": encode_sec,
        "results": sweep_results,
    }, indent=2, ensure_ascii=False))
    print(f"sweep -> {args.sweep_path}")


if __name__ == "__main__":
    main()
