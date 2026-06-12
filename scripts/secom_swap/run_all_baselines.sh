#!/bin/bash
# Chain all remaining baselines + zero/full evaluations sequentially.
# Run AFTER TextTiling-compress finishes (otherwise OOM with LLMLingua-2 +
# bertscore + GraphSeg/CSM models all in memory).
set -e

REPO=/home/namchailin/Hi-OnTop
cd "$REPO"
EXP=outputs/experiments/2026-05-21_v413_secom_swap
RESULT=benchmarks/SeCom/experiment/result/mtbp

run_chain() {
  local method=$1
  local seg=$RESULT/$method/segments.jsonl
  local comp=$RESULT/$method/segments_comp.jsonl
  local ret=$RESULT/$method/retrieved.jsonl
  local chat=$RESULT/$method/chat.jsonl

  # 0) segment if not yet
  [ -f "$seg" ] || {
    echo "=== $method: segment ==="
    mkdir -p "$(dirname "$seg")"
    EXTRA_ARGS=""
    if [ "$method" = "graphseg" ]; then
      EXTRA_ARGS="--glove_path benchmarks/glove/glove.6B.300d.txt"
    fi
    if [ "$method" = "csm" ]; then
      export CSM_REPO_DIR=/home/namchailin/Hi-OnTop/external/Dialogue-Topic-Segmenter
      EXTRA_ARGS="--csm_alpha -0.5"
    fi
    uv run python scripts/secom_swap/segment_baselines.py \
      --method "$method" \
      --load_path benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl \
      --save_path "$seg" \
      --latency_path "$EXP/latency_$method.json" $EXTRA_ARGS
  }

  [ -f "$comp" ] || {
    echo "=== $method: compress ==="
    uv run python scripts/secom_swap/05_compress.py \
      --load_path "$seg" --save_path "$comp"
  }
  [ -f "$ret" ] || {
    echo "=== $method: retrieve ==="
    uv run python scripts/secom_swap/06_retrieve.py \
      --load_path "$comp" --save_path "$ret" --topk 1 --device cpu
  }
  [ -f "$chat" ] || {
    echo "=== $method: chat ==="
    uv run python scripts/secom_swap/07_chat.py \
      --load_path "$ret" --save_path "$chat" \
      --model openai/gpt-4o-mini --workers 8
  }
  [ -f "$EXP/metrics_$method.json" ] || {
    echo "=== $method: eval ==="
    uv run python scripts/secom_swap/08_eval.py \
      --load_path "$chat" --save_path "$EXP/metrics_$method.json" \
      --judge_model openai/gpt-4o --judge_workers 8
  }
}

run_zero_full_eval() {
  local mode=$1
  local chat=$RESULT/$mode/chat.jsonl
  [ -f "$EXP/metrics_$mode.json" ] || {
    echo "=== $mode: eval ==="
    uv run python scripts/secom_swap/08_eval.py \
      --load_path "$chat" --save_path "$EXP/metrics_$mode.json" \
      --judge_model openai/gpt-4o --judge_workers 8
  }
}

# Order: zero/full evals (cheap, already chat'd) → CSM (our trained model,
# highest paper interest) → TextTiling → GraphSeg → GreedySeg.
run_zero_full_eval zero
run_zero_full_eval full
run_chain csm
run_chain texttiling
run_chain graphseg
run_chain greedyseg

echo "=== ALL DONE ==="
ls -la $EXP/metrics_*.json
