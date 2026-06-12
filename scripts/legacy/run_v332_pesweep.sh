#!/usr/bin/env bash
# v3.3.2 (shared-GRU PE + surprise hard boundary) sweep on LoCoMo.
#
# Built on v3.3.1 HP sweep (2026-05-08 01:37) result:
#   best v3.3.1 = (cos=0.9, alpha=100, beta=0.25) → acc=0.263.
# Fix that as base; sweep (pe_threshold, lmda, promotion_threshold,
# tau, rnn_train_steps) on top.
#
# Each run captures STM top-K stats (T1/T2/T3) so we can verify whether
# the surprise rule + alternate axes actually break mega-topic.
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR="outputs/sweeps/2026-05-08_v332_pesweep"
mkdir -p "$OUT_DIR"

# (cos, alpha, beta, pe, lmda, promotion, tau, train_steps, label)
configs=(
  # 1) pe sweep at best v3.3.1 HP, others at default
  "0.9  100  0.25  1.0  10  0.5  50   1   pe-disabled"
  "0.9  100  0.25  0.3  10  0.5  50   1   pe0.3"
  "0.9  100  0.25  0.5  10  0.5  50   1   pe0.5"
  "0.9  100  0.25  0.7  10  0.5  50   1   pe0.7"
  # 2) pe=0.5 fixed, vary lmda
  "0.9  100  0.25  0.5  0   0.5  50   1   pe0.5_lmda0"
  "0.9  100  0.25  0.5  1   0.5  50   1   pe0.5_lmda1"
  # 3) pe=0.5 fixed, lower promotion threshold
  "0.9  100  0.25  0.5  10  0.2  50   1   pe0.5_prom0.2"
  # 4) pe=0.5 fixed, lower tau (prior matters more)
  "0.9  100  0.25  0.5  10  0.5  10   1   pe0.5_tau10"
  # 5) pe=0.5 fixed, more RNN training
  "0.9  100  0.25  0.5  10  0.5  50   3   pe0.5_train3"
  # 6) combined aggressive: low lmda + low promotion + pe=0.5
  "0.9  100  0.25  0.5  1   0.2  50   1   pe0.5_lmda1_prom0.2"
  # 7) combined ultra: pe=0.3 + lmda=1 + prom=0.2
  "0.9  100  0.25  0.3  1   0.2  50   1   pe0.3_lmda1_prom0.2"
  # 8) very high pe (relax surprise) at base
  "0.9  100  0.25  0.9  10  0.5  50   1   pe0.9"
)

run_one() {
  local cos="$1"
  local alpha="$2"
  local beta="$3"
  local pe="$4"
  local lmda="$5"
  local prom="$6"
  local tau="$7"
  local train="$8"
  local lbl="$9"
  local tag="$lbl"
  tag="${tag//./p}"
  local run_dir="$OUT_DIR/$tag"
  local exp_id="20260508_v332_${tag}"
  local log="$run_dir/run.log"
  local topk="$run_dir/stm_topk.json"

  mkdir -p "$run_dir"
  rm -f "$topk" "${topk%.json}.rounds.jsonl" 2>/dev/null || true
  {
    echo "=== START $(date -Is) ==="
    echo "method=hi-ontop-full-v3.3.2"
    echo "cos=${cos} alpha=${alpha} beta=${beta} pe=${pe} lmda=${lmda} prom=${prom} tau=${tau} train_steps=${train}"
    echo "exp_id=${exp_id}"
    echo
  } > "$log"

  WANDB_MODE=disabled \
  UV_CACHE_DIR=/tmp/uv-cache \
  HIONTOP_STM_TOPK_STATS_PATH="$topk" \
  uv run python scripts/run_experiment.py \
    --method hi-ontop-full-v3.3.2 \
    --benchmark locomo \
    --data benchmarks/locomo/data/locomo10.json \
    --limit 50 --stratify \
    --questions-per-round 50 \
    --exp-id "$exp_id" \
    --results-root "$run_dir/results" \
    --no-token-count --no-thinking --workers 100 \
    --cos-threshold "$cos" --alpha "$alpha" --beta "$beta" \
    --pe-threshold "$pe" --lmda "$lmda" \
    --promotion-threshold "$prom" --tau "$tau" \
    --rnn-train-steps "$train" \
    >> "$log" 2>&1
  local rc=$?
  echo "$rc" > "$run_dir/exit_code.txt"
  echo "[exit] ${exp_id} rc=${rc}" >> "$log"
  echo "=== END $(date -Is) ===" >> "$log"
}

for cfg in "${configs[@]}"; do
  read -r cos alpha beta pe lmda prom tau train lbl <<< "$cfg"
  run_one "$cos" "$alpha" "$beta" "$pe" "$lmda" "$prom" "$tau" "$train" "$lbl"
done

# Final report after all runs.
uv run python scripts/aggregate_v332_sweep.py
