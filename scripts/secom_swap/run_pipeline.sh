#!/bin/bash
# Full SeCom-swap pipeline. Run from repo root.
# Methods: "ours" (v4.1.3 segment) and "gpt4omini" (gpt-4o-mini segment).
# Both share: mpnet retriever (CPU), LLMLingua-2 compression, gpt-4o-mini chat.
#
# Pre-req: .env has OPENAI_API_KEY + OPENAI_BASE_URL (Crts).
# Data: benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl (run 01_prepare_data.py first).
#
# Outputs go under benchmarks/SeCom/experiment/result/mtbp/<method>/ and
# outputs/experiments/2026-05-21_v413_secom_swap/.
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap
mkdir -p "$RESULT/ours" "$RESULT/gpt4omini" "$EXP"

DELTA_STAR="${DELTA_STAR:-0.30}"   # override from calibration JSON

cd "$REPO"

echo "=== 1) prepare data ==="
uv run python scripts/secom_swap/01_prepare_data.py

# ─── Method 1: Ours (v4.1.3 seg) ─────────────────────────────────────────────
echo "=== 2a) segment (ours, v4.1.3) ==="
uv run python scripts/secom_swap/03_segment_v413.py \
    --delta_star "$DELTA_STAR" \
    --save_path "$RESULT/ours/segments.jsonl" \
    --latency_path "$EXP/latency_ours.json"

# ─── Method 2: GPT-4o-mini seg ───────────────────────────────────────────────
echo "=== 2b) segment (gpt-4o-mini) ==="
uv run python scripts/secom_swap/04_segment_baseline.py \
    --segment_model openai/gpt-4o-mini \
    --save_path "$RESULT/gpt4omini/segments.jsonl" \
    --latency_path "$EXP/latency_gpt4omini.json"

# ─── Compress + Retrieve + Chat + Eval for each method ───────────────────────
for METHOD in ours gpt4omini; do
    echo "=== $METHOD: compress ==="
    uv run python scripts/secom_swap/05_compress.py \
        --load_path "$RESULT/$METHOD/segments.jsonl" \
        --save_path "$RESULT/$METHOD/segments_comp.jsonl"

    echo "=== $METHOD: retrieve (topk=1) ==="
    uv run python scripts/secom_swap/06_retrieve.py \
        --load_path "$RESULT/$METHOD/segments_comp.jsonl" \
        --save_path "$RESULT/$METHOD/retrieved.jsonl" \
        --topk 1

    echo "=== $METHOD: chat (gpt-4o-mini) ==="
    uv run python scripts/secom_swap/07_chat.py \
        --load_path "$RESULT/$METHOD/retrieved.jsonl" \
        --save_path "$RESULT/$METHOD/chat.jsonl" \
        --model openai/gpt-4o-mini --workers 8

    echo "=== $METHOD: eval ==="
    uv run python scripts/secom_swap/08_eval.py \
        --load_path "$RESULT/$METHOD/chat.jsonl" \
        --save_path "$EXP/metrics_$METHOD.json"
done

echo
echo "=== DONE. Summary: ==="
ls -la "$EXP/"
