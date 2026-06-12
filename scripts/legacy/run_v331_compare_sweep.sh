#!/usr/bin/env bash
# v3.3.1 vs prior versions + RAG baselines comparison sweep on LoCoMo.
# Methods: hi-ontop-full-v1, hi-ontop-full-v3.1.1, hi-ontop-full-v3.2.1,
#          hi-ontop-full-v3.3.1, rag, rag-summary, rag-observation, sliding.
#
# Per-run STM top-K stats (T1/T2/T3 turn-counts) for hi-ontop-full-* methods are
# captured into <run_dir>/stm_topk.json + .rounds.jsonl via the
# HIONTOP_STM_TOPK_STATS_PATH env var. RAG / sliding do not emit these.
#
# TIAGE benchmark is not covered here — run_experiment.py only supports
# longmemeval/locomo. Use scripts/run_tiage_v3_compare.py for TIAGE.
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR="outputs/sweeps/2026-05-07_v331_compare"
mkdir -p "$OUT_DIR"

run_one() {
  local benchmark="$1"
  local method="$2"
  local data="$3"
  local method_tag="${method//./_}"
  method_tag="${method_tag//-/_}"
  local run_dir="$OUT_DIR/${benchmark}/${method_tag}"
  local exp_id="20260507_v331_compare_${benchmark}_${method_tag}"
  local log="$run_dir/run.log"
  local topk="$run_dir/stm_topk.json"

  mkdir -p "$run_dir"
  # Wipe any pre-existing topk so file-existence gate doesn't suppress recording.
  rm -f "$topk" "${topk%.json}.rounds.jsonl" 2>/dev/null || true
  {
    echo "=== START $(date -Is) ==="
    echo "benchmark=${benchmark}"
    echo "method=${method}"
    echo "data=${data}"
    echo "exp_id=${exp_id}"
    echo "run_dir=${run_dir}"
    echo
  } > "$log"

  WANDB_MODE=disabled \
  UV_CACHE_DIR=/tmp/uv-cache \
  HIONTOP_STM_TOPK_STATS_PATH="$topk" \
  uv run python scripts/run_experiment.py \
    --method "$method" \
    --benchmark "$benchmark" \
    --data "$data" \
    --limit 50 --stratify \
    --questions-per-round 50 \
    --exp-id "$exp_id" \
    --results-root "$run_dir/results" \
    --no-token-count \
    --no-thinking \
    --workers 100 \
    >> "$log" 2>&1
  local rc=$?
  echo "$rc" > "$run_dir/exit_code.txt"
  echo "[exit] ${exp_id} rc=${rc}" >> "$log"
  echo "=== END $(date -Is) ===" >> "$log"
}

# Hi-OnTop lineage (RNN-free → RNN, in chronological order).
run_one locomo hi-ontop-full-v1       benchmarks/locomo/data/locomo10.json
run_one locomo hi-ontop-full-v3.1.1   benchmarks/locomo/data/locomo10.json
run_one locomo hi-ontop-full-v3.2.1   benchmarks/locomo/data/locomo10.json
run_one locomo hi-ontop-full-v3.3.1   benchmarks/locomo/data/locomo10.json

# RAG baselines.
run_one locomo rag                 benchmarks/locomo/data/locomo10.json
run_one locomo rag-summary         benchmarks/locomo/data/locomo10.json
run_one locomo rag-observation     benchmarks/locomo/data/locomo10.json
run_one locomo sliding             benchmarks/locomo/data/locomo10.json

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

out_dir = Path("outputs/sweeps/2026-05-07_v331_compare")
rows = [
    ("hi-ontop-full-v1", "locomo"),
    ("hi-ontop-full-v3.1.1", "locomo"),
    ("hi-ontop-full-v3.2.1", "locomo"),
    ("hi-ontop-full-v3.3.1", "locomo"),
    ("rag", "locomo"),
    ("rag-summary", "locomo"),
    ("rag-observation", "locomo"),
    ("sliding", "locomo"),
]


def load_topk(run_dir: Path) -> dict:
    p = run_dir / "stm_topk.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


print("| method | bench | n | acc | err | T1μ | T2μ | T3μ | T1max |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for method, benchmark in rows:
    method_tag = method.replace(".", "_").replace("-", "_")
    run_dir = out_dir / benchmark / method_tag
    summary_path = next(
        (run_dir / "results" / "experiments").glob("*/summary.json"),
        None,
    ) if (run_dir / "results" / "experiments").exists() else None
    if summary_path is None:
        print(f"| {method} | {benchmark} | - | - | - | - | - | - | - |")
        continue
    summary = json.loads(summary_path.read_text())
    n = summary.get("n_questions", "-")
    acc = summary.get("accuracy_overall", summary.get("primary_metric"))
    err = summary.get("error_rate")
    topk = load_topk(run_dir)
    t1 = topk.get("top1_per_round", {}).get("mean")
    t2 = topk.get("top2_per_round", {}).get("mean")
    t3 = topk.get("top3_per_round", {}).get("mean")
    t1mx = topk.get("top1_per_round", {}).get("max")

    def f(v, dp=3):
        return "-" if v is None else f"{float(v):.{dp}f}"

    print(
        f"| {method} | {benchmark} | {n} | {f(acc)} | {f(err)} | "
        f"{f(t1, 1)} | {f(t2, 1)} | {f(t3, 1)} | {t1mx if t1mx is not None else '-'} |"
    )
PY
