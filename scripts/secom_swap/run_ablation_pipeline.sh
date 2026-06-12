#!/bin/bash
# NON-LATENCY BLOCK — compress / retrieve / chat / eval for ablation runs.
# Run AFTER all latency measurements completed (run_ablation_segments.sh +
# any other pending latency work).
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap

cd "$REPO"

for METHOD in ours_mpnet_a0 ours_mpnet_a1; do
    DIR="$RESULT/$METHOD"
    if [ ! -f "$DIR/segments.jsonl" ]; then
        echo "ERROR: $DIR/segments.jsonl missing — run run_ablation_segments.sh first" >&2
        exit 1
    fi
    echo
    echo "##################################################################"
    echo "###  $METHOD pipeline  (NON-LATENCY)"
    echo "##################################################################"

    echo "=== [$METHOD] 2) compress (LLMLingua-2) ==="
    if [ -f "$DIR/segments_comp.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/05_compress.py \
            --load_path "$DIR/segments.jsonl" \
            --save_path "$DIR/segments_comp.jsonl"
    fi

    echo "=== [$METHOD] 3) retrieve (mpnet+FAISS, topk=1) ==="
    if [ -f "$DIR/retrieved.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/06_retrieve.py \
            --load_path "$DIR/segments_comp.jsonl" \
            --save_path "$DIR/retrieved.jsonl" \
            --topk 1
    fi

    echo "=== [$METHOD] 4) chat (gpt-4o-mini, workers=8) ==="
    if [ -f "$DIR/chat.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/07_chat.py \
            --load_path "$DIR/retrieved.jsonl" \
            --save_path "$DIR/chat.jsonl" \
            --model openai/gpt-4o-mini --workers 8
    fi

    echo "=== [$METHOD] 5) eval ==="
    uv run python scripts/secom_swap/08_eval.py \
        --load_path "$DIR/chat.jsonl" \
        --save_path "$EXP/metrics_${METHOD}.json"
done

echo
echo "=== DONE — ablation metrics:"
for m in ours_mpnet_a0 ours_mpnet_a1; do
    F="$EXP/metrics_${m}.json"
    [ -f "$F" ] && { echo "  $(basename $F):"; cat "$F"; echo; }
done
