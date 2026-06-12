#!/usr/bin/env bash
# SeCom/baseline 풀 스윕 드라이버 (resumable).
# 6모델 × {segment: full,10,30,60,120 / incremental} × {baseline, verbatim}.
# 각 combo 의 summary line 을 RESULTS 에 태그와 함께 append. 이미 있으면 skip.
# 모델별 응답캐시(secom_cache_*.jsonl)가 있어 중간에 죽어도 직전부터 재개.
set -u
cd "$(dirname "$0")/.."
# 사용법: bash secom_sweep.sh [SUBSET_JSON]  (default=37 dev; 139=outputs/runs/_misc/ami_full139.json)
SUBSET="${1:-outputs/runs/_misc/ami_subset.json}"
STAG="$(basename "$SUBSET" .json)"          # 결과 태그에 subset 포함 → 37/139 충돌 방지
RESULTS=outputs/runs/_misc/secom_sweep_results.tsv
LOGDIR=outputs/runs/_misc/secom_sweep_logs
mkdir -p "$LOGDIR"
touch "$RESULTS"

# 모델셋 확정 2026-06-12 (사용자, 전 실험 공통). gpt-5-mini/nano · gemma-3n 추가, gpt-4o 계열 제외.
MODELS=(
  "openrouter/openai/gpt-5-mini"
  "openrouter/openai/gpt-5-nano"
  "openrouter/qwen/qwen3.5-27b"
  "openrouter/anthropic/claude-haiku-4.5"
  "openrouter/mistralai/mistral-small-3.1-24b-instruct"
  "openrouter/google/gemma-3-12b-it"
  "openrouter/google/gemma-3n-e4b-it"
)
PROMPTS=(baseline verbatim)
# (mode buffer workers) 묶음 — 싼 segment 먼저, incremental 마지막
SEGBUFS=(full 120 60 30 10)

run_one() {  # tag mode buffer prompt model workers
  local tag="$1" mode="$2" buf="$3" prompt="$4" model="$5" workers="$6"
  if grep -qF "$tag" "$RESULTS"; then echo "SKIP $tag"; return; fi
  local log="$LOGDIR/$(echo "$tag" | tr '/:| ' '____').log"
  echo "RUN  $tag"
  if [ "$mode" = incremental ]; then
    python scripts/secom_llm_eval.py --mode incremental --prompt "$prompt" \
      --model "$model" --subset "$SUBSET" --workers "$workers" > "$log" 2>&1
  else
    python scripts/secom_llm_eval.py --mode segment --buffer "$buf" --prompt "$prompt" \
      --model "$model" --subset "$SUBSET" --workers "$workers" --context B > "$log" 2>&1
  fi
  local line; line="$(grep -m1 '±2F1' "$log")"
  if [ -n "$line" ]; then printf '%s\t%s\n' "$tag" "$line" >> "$RESULTS"; echo "DONE $tag | $line";
  else echo "FAIL $tag (no summary, see $log)"; fi
}

# 1) segment 버퍼들 (빠름, window 완전병렬)
for prompt in "${PROMPTS[@]}"; do
  for buf in "${SEGBUFS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_one "$STAG|seg|$buf|$prompt|$model" segment "$buf" "$prompt" "$model" 16
    done
  done
done

# 2) incremental (느림, 미팅순차/미팅병렬)
for prompt in "${PROMPTS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_one "$STAG|inc|-|$prompt|$model" incremental - "$prompt" "$model" 37
  done
done

echo "=== SWEEP COMPLETE ==="
