"""Compare two segmentation methods' boundary placement.

Reads two segments JSONL (e.g. baseline gpt-4o-mini vs ours v4.1.3) and
reports:
- per-conv segment counts
- boundary-placement agreement (Pk-like + WindowDiff approximation if both
  produce same total exchange count)
- token-level overlap (proportion of consecutive (i, i+1) exchanges that
  fall in the same segment in both methods)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def segments_to_boundary_vector(segments: list[list[str]]) -> list[int]:
    """Return a 0/1 vector where position i = 1 iff a boundary follows exchange i.

    len(vec) = total_exchanges - 1.
    """
    flat_lens = [len(seg) for seg in segments]
    bnd: list[int] = []
    for i, n in enumerate(flat_lens):
        # exchanges 0..n-1 inside this segment: no internal boundaries
        bnd.extend([0] * (n - 1))
        # boundary AFTER the last exchange of this segment, unless last segment
        if i < len(flat_lens) - 1:
            bnd.append(1)
    return bnd


def per_conv_compare(a_segs: list[list[str]], b_segs: list[list[str]]) -> dict:
    # Filter out empty segments (LLM segmenter occasionally outputs
    # ``num_exchanges: 0`` lines; downstream stages skip them but they
    # pollute boundary-vector arithmetic).
    a_segs = [s for s in a_segs if len(s) > 0]
    b_segs = [s for s in b_segs if len(s) > 0]
    a_total = sum(len(s) for s in a_segs)
    b_total = sum(len(s) for s in b_segs)
    if a_total != b_total:
        return {
            "n_seg_a": len(a_segs),
            "n_seg_b": len(b_segs),
            "n_ex_a": a_total,
            "n_ex_b": b_total,
            "error": "exchange count mismatch — cannot compute boundary agreement",
        }
    a_bnd = segments_to_boundary_vector(a_segs)
    b_bnd = segments_to_boundary_vector(b_segs)
    assert len(a_bnd) == len(b_bnd)
    n = len(a_bnd)
    n_match = sum(1 for x, y in zip(a_bnd, b_bnd) if x == y)
    n_both_bnd = sum(1 for x, y in zip(a_bnd, b_bnd) if x == 1 and y == 1)
    n_a_bnd = sum(a_bnd)
    n_b_bnd = sum(b_bnd)
    # boundary F1
    prec = n_both_bnd / n_a_bnd if n_a_bnd else 0.0
    rec = n_both_bnd / n_b_bnd if n_b_bnd else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n_seg_a": len(a_segs),
        "n_seg_b": len(b_segs),
        "n_ex": a_total,
        "n_boundary_a": n_a_bnd,
        "n_boundary_b": n_b_bnd,
        "position_agreement": n_match / n if n else 0.0,
        "boundary_precision_a_vs_b": prec,
        "boundary_recall_a_vs_b": rec,
        "boundary_f1": f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="segments JSONL #1 (reference)")
    ap.add_argument("--b", required=True, help="segments JSONL #2 (compared)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.a))}
    b_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.b))}
    shared = sorted(set(a_data) & set(b_data))
    print(f"shared conversations: {len(shared)}")

    per_conv = []
    sums = {"n_match": 0, "n_total": 0, "n_both_bnd": 0,
            "n_a_bnd": 0, "n_b_bnd": 0, "n_seg_a": 0, "n_seg_b": 0}
    for cid in shared:
        a_segs = a_data[cid]["segments"]
        b_segs = b_data[cid]["segments"]
        row = per_conv_compare(a_segs, b_segs)
        row["conversation_id"] = cid
        per_conv.append(row)
        if "error" in row:
            print(f"  {cid}: {row['error']}")
            continue
        sums["n_match"] += int(row["position_agreement"] * (row["n_ex"] - 1))
        sums["n_total"] += row["n_ex"] - 1
        sums["n_both_bnd"] += int(row["boundary_precision_a_vs_b"] * row["n_boundary_a"])
        sums["n_a_bnd"] += row["n_boundary_a"]
        sums["n_b_bnd"] += row["n_boundary_b"]
        sums["n_seg_a"] += row["n_seg_a"]
        sums["n_seg_b"] += row["n_seg_b"]

    overall = {}
    if sums["n_total"] > 0:
        overall["position_agreement"] = sums["n_match"] / sums["n_total"]
    if sums["n_a_bnd"] > 0:
        overall["boundary_precision_a_vs_b"] = sums["n_both_bnd"] / sums["n_a_bnd"]
    if sums["n_b_bnd"] > 0:
        overall["boundary_recall_a_vs_b"] = sums["n_both_bnd"] / sums["n_b_bnd"]
    if "boundary_precision_a_vs_b" in overall and "boundary_recall_a_vs_b" in overall:
        p, r = overall["boundary_precision_a_vs_b"], overall["boundary_recall_a_vs_b"]
        overall["boundary_f1"] = 2 * p * r / (p + r) if (p + r) else 0.0
    overall["n_seg_a_total"] = sums["n_seg_a"]
    overall["n_seg_b_total"] = sums["n_seg_b"]
    overall["a_file"] = args.a
    overall["b_file"] = args.b

    summary = {"overall": overall, "per_conv": per_conv}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
