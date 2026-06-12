#!/bin/bash
# Re-run chat + eval for all downstream_task.md baselines with chat=gpt-4o.
# Segmentation/compress/retrieve from the original chat=gpt-4o-mini run is
# reused (LLM-agnostic). Only 07_chat.py / 07b_chat_no_retrieve.py + 08_eval.py
# are re-executed.
#
# Output:
#   - benchmarks/SeCom/experiment/result/mtbp/<dir>/chat_gpt4o.jsonl
#   - outputs/experiments/2026-05-21_v413_secom_swap/metrics_<method>_gpt4o.json
#
# Model spec: per memory feedback_api_endpoint, all model slugs sent to Crts
# are prefixed with openrouter/.
set -uo pipefail   # intentionally NOT -e: one method's failure should not abort the rest

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap
DATA=$REPO/benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl

CHAT_MODEL="openrouter/openai/gpt-4o"
JUDGE_MODEL="openrouter/openai/gpt-4o"
CHAT_WORKERS=24
JUDGE_WORKERS=24
NO_THINK="off"        # gpt-4o is not a thinking model; do not send reasoning_effort

cd "$REPO"

# (paper_method_name, result_dir_name, mode)   mode: zero | full | retrieve
METHODS=(
  "zero|zero|zero"
  "full|full|full"
  "texttiling|texttiling|retrieve"
  "graphseg|graphseg|retrieve"
  "greedyseg|greedyseg|retrieve"
  "csm|csm|retrieve"
  "roberta|roberta|retrieve"
  "ours_p60|ours_p60|retrieve"
  "ours_p70|ours_p70|retrieve"
  "ours_minilm_p60|ours_minilm_p60|retrieve"
  "ours_minilm_p70|ours_minilm_p70|retrieve"
  "ours_minilm_p80|ours_minilm_p80|retrieve"
  "ours_int8_p60|ours_int8_p60|retrieve"
  "ours_int8_p70|ours_int8_p70|retrieve"
  "ours_int8_p80|ours_int8_p80|retrieve"
  "ours_mpnet_bestp|ours_mpnet_bestp|retrieve"
  "ours_minilm_bestp|ours_minilm_bestp|retrieve"
  "ours_minilm_int8_bestp|ours_minilm_int8_bestp|retrieve"
  "ours_mpnet_a0|ours_mpnet_a0|retrieve"
  "ours_mpnet_a1|ours_mpnet_a1|retrieve"
  "gpt5seg|gpt5seg|retrieve"
  "qwen35_122bseg|qwen35_122bseg|retrieve"
  "qwen27bseg|qwen27bseg|retrieve"
  "gpt4omini|gpt4omini|retrieve"
  "qwen35_4bseg|qwen35_4bseg|retrieve"
  "llama32_3bseg|llama32_3bseg|retrieve"
  "ministral3_3bseg|ministral3_3bseg|retrieve"
  "qwen35_2bseg|qwen35_2bseg|retrieve"
)

ok=0; fail=0; skip=0
START=$(date +%s)

for entry in "${METHODS[@]}"; do
  IFS='|' read -r method dir mode <<<"$entry"
  metrics_path="$EXP/metrics_${method}_gpt4o.json"
  chat_path="$RESULT/$dir/chat_gpt4o.jsonl"

  if [ -f "$metrics_path" ]; then
    score=$(python -c "import json;print(json.load(open('$metrics_path')).get('gpt4_score_x10','-'))" 2>/dev/null)
    echo "[skip] $method (exists, gpt4=$score)"
    skip=$((skip+1)); continue
  fi

  echo "================================================================"
  echo "[$(date +%H:%M:%S)] $method  (mode=$mode, dir=$dir)"
  echo "================================================================"

  if [ "$mode" = "zero" ] || [ "$mode" = "full" ]; then
    uv run python scripts/secom_swap/07b_chat_no_retrieve.py \
      --load_path "$DATA" \
      --save_path "$chat_path" \
      --mode "$mode" \
      --model "$CHAT_MODEL" \
      --workers "$CHAT_WORKERS"
  else
    uv run python scripts/secom_swap/07_chat.py \
      --load_path "$RESULT/$dir/retrieved.jsonl" \
      --save_path "$chat_path" \
      --model "$CHAT_MODEL" \
      --workers "$CHAT_WORKERS" \
      --no_think "$NO_THINK"
  fi
  chat_rc=$?

  if [ $chat_rc -ne 0 ] || [ ! -s "$chat_path" ]; then
    echo "[FAIL chat] $method  (rc=$chat_rc)"
    fail=$((fail+1)); continue
  fi

  uv run python scripts/secom_swap/08_eval.py \
    --load_path "$chat_path" \
    --save_path "$metrics_path" \
    --judge_model "$JUDGE_MODEL" \
    --judge_workers "$JUDGE_WORKERS"
  eval_rc=$?

  if [ $eval_rc -ne 0 ] || [ ! -f "$metrics_path" ]; then
    echo "[FAIL eval] $method  (rc=$eval_rc)"
    fail=$((fail+1)); continue
  fi

  score=$(python -c "import json;print(json.load(open('$metrics_path')).get('gpt4_score_x10','-'))" 2>/dev/null)
  echo "[ok] $method  gpt4=$score"
  ok=$((ok+1))
done

END=$(date +%s); ELAPSED=$((END-START))
echo
echo "================================================================"
echo "DONE. ok=$ok fail=$fail skip=$skip  elapsed=${ELAPSED}s"
echo "================================================================"
