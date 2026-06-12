#!/usr/bin/env python3
"""Pick top-20 (α, λ, cos) configs from v1 ∪ v3.1.1 results, ranked by acc.

Strategy (b): for v1 (no `cos`) rows, expand to (α, λ, best_cos) where
``best_cos`` is the cos that gave the highest v3.1.1 acc for the same
(α, λ) pair. Then merge with v3.1.1 entries and pick top-20 unique
(α, λ, cos) tuples by acc.

Reads:
    outputs/sweeps/2026-05-05_locomo_alpha_lambda_cos/summary_table.csv
        (produced by aggregate_locomo_alphalambda_results.py)

Writes:
    outputs/sweeps/2026-05-05_locomo_alpha_lambda_cos/top20_for_v321.txt
        one ``alpha lambda cos`` per line.

Also prints lines to stdout so the v3.2.1 sweep shell can pipe them.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "outputs" / "sweeps/2026-05-05_locomo_alpha_lambda_cos" / "summary_table.csv"
OUT_PATH = REPO / "outputs" / "sweeps/2026-05-05_locomo_alpha_lambda_cos" / "top20_for_v321.txt"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found — run aggregator first", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(CSV_PATH.open()))

    # v3.1.1 rows: (α, λ, cos) → acc
    v31 = {}
    # v1 rows: (α, λ) → acc
    v1 = {}
    for r in rows:
        try:
            acc = float(r["acc"])
        except (KeyError, ValueError):
            continue
        method = r.get("method", "")
        try:
            alpha = int(r["α"])
            lmda = int(r["λ"])
        except (KeyError, ValueError):
            continue
        # v3.1.1 entries (Bounded Cosine MAP, the source of cos info)
        if method in ("v3.1", "v3.1.1"):
            try:
                cos = float(r["cos"])
            except (KeyError, ValueError):
                continue
            v31[(alpha, lmda, cos)] = acc
        # v1 entries (Gaussian, no cos)
        elif method in ("full", "v1"):
            v1[(alpha, lmda)] = acc

    # Strategy (b): expand v1 entries using best v3.1.1 cos for same (α, λ).
    expanded_v1 = {}
    for (a, l), acc in v1.items():
        cos_options = [(c, ac) for (a_, l_, c), ac in v31.items() if a_ == a and l_ == l]
        if not cos_options:
            # No matching v3.1.1 row — fallback cos=0.5
            best_cos = 0.5
        else:
            best_cos = max(cos_options, key=lambda x: x[1])[0]
        # Use the v1 acc (since this row represents a v1-strong (α, λ) zone)
        expanded_v1[(a, l, best_cos)] = max(expanded_v1.get((a, l, best_cos), 0.0), acc)

    # Merge: prefer max acc per unique tuple
    combined = dict(v31)
    for k, v in expanded_v1.items():
        combined[k] = max(combined.get(k, 0.0), v)

    # Sort by acc desc, take top-20
    ranked = sorted(combined.items(), key=lambda kv: -kv[1])[:20]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for (a, l, c), acc in ranked:
            line = f"{a} {l} {c}  # acc={acc:.4f}"
            f.write(line + "\n")
            print(line)
    print(f"\n# wrote {OUT_PATH} with {len(ranked)} configs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
