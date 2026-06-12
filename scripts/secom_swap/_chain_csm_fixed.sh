#!/usr/bin/env bash
# Autonomous chain: compress → retrieve → chat → eval for CSM-fixed MTB+.
# Stops on first error. Writes per-step success markers + log.

set -e
cd /home/namchailin/Hi-OnTop

EXP=outputs/experiments/2026-05-24_secom_csm_fixed
RDIR=benchmarks/SeCom/experiment/result/mtbp/csm
LOG=$EXP/chain.log
mkdir -p "$EXP"

echo "[chain] $(date) start" | tee -a "$LOG"

# 1) compress
if [ ! -f "$RDIR/segments_comp.jsonl" ]; then
  echo "[chain] compress …" | tee -a "$LOG"
  uv run python scripts/secom_swap/05_compress.py \
    --load_path "$RDIR/segments.jsonl" --save_path "$RDIR/segments_comp.jsonl" \
    >> "$LOG" 2>&1
  echo "[chain] compress done $(date)" | tee -a "$LOG"
else
  echo "[chain] compress already done" | tee -a "$LOG"
fi

# 2) retrieve
if [ ! -f "$RDIR/retrieved.jsonl" ]; then
  echo "[chain] retrieve …" | tee -a "$LOG"
  uv run python scripts/secom_swap/06_retrieve.py \
    --load_path "$RDIR/segments_comp.jsonl" --save_path "$RDIR/retrieved.jsonl" \
    --topk 1 --device cpu >> "$LOG" 2>&1
  echo "[chain] retrieve done $(date)" | tee -a "$LOG"
else
  echo "[chain] retrieve already done" | tee -a "$LOG"
fi

# 3) chat (gpt-4o-mini)
if [ ! -f "$RDIR/chat.jsonl" ]; then
  echo "[chain] chat …" | tee -a "$LOG"
  uv run python scripts/secom_swap/07_chat.py \
    --load_path "$RDIR/retrieved.jsonl" --save_path "$RDIR/chat.jsonl" \
    --model openai/gpt-4o-mini --workers 8 >> "$LOG" 2>&1
  echo "[chain] chat done $(date)" | tee -a "$LOG"
else
  echo "[chain] chat already done" | tee -a "$LOG"
fi

# 4) eval (gpt-4o judge)
if [ ! -f "$EXP/metrics_csm.json" ]; then
  echo "[chain] eval …" | tee -a "$LOG"
  uv run python scripts/secom_swap/08_eval.py \
    --load_path "$RDIR/chat.jsonl" --save_path "$EXP/metrics_csm.json" \
    --judge_model openai/gpt-4o --judge_workers 8 >> "$LOG" 2>&1
  echo "[chain] eval done $(date)" | tee -a "$LOG"
else
  echo "[chain] eval already done" | tee -a "$LOG"
fi

echo "[chain] $(date) ALL DONE" | tee -a "$LOG"
touch "$EXP/_chain_done"
