#!/bin/bash
# Long-MT-Bench+ downstream eval — MiniLM (fp32) × p60/p70/p80 percentile sweep.
#
# δ* values from outputs/.../delta_star_calibration_minilm_p_sweep (deploy = MTB+):
#   MiniLM (fp32):  p60 ≈ 0.5242 · p70 ≈ 0.7033 · p80 ≈ 0.8726
#                   (from 2026-05-25_llm_distillation_calib/results.json sweep)
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap
ENC="sentence-transformers/all-MiniLM-L6-v2"
DIM=384

cd "$REPO"

declare -A DSTARS=(
    [p60]="0.5242"
    [p70]="0.7033"
    [p80]="0.8726"
)

for P in p60 p70 p80; do
    METHOD="ours_minilm_${P}"
    DELTA="${DSTARS[$P]}"
    DIR="$RESULT/$METHOD"
    mkdir -p "$DIR"
    echo
    echo "##################################################################"
    echo "###  $METHOD  (δ* = $DELTA, encoder=MiniLM fp32, dim=$DIM)"
    echo "##################################################################"

    echo "=== [$METHOD] 1) segment ==="
    if [ -f "$DIR/segments.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/03_segment_v413.py \
            --encoder "$ENC" --encoder_backend local \
            --dim "$DIM" --delta_star "$DELTA" \
            --save_path "$DIR/segments.jsonl" \
            --latency_path "$EXP/latency_${METHOD}.json"
    fi

    echo "=== [$METHOD] 2) compress (LLMLingua-2) ==="
    if [ -f "$DIR/segments_comp.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/05_compress.py \
            --load_path "$DIR/segments.jsonl" \
            --save_path "$DIR/segments_comp.jsonl"
    fi

    echo "=== [$METHOD] 3) retrieve ==="
    if [ -f "$DIR/retrieved.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/06_retrieve.py \
            --load_path "$DIR/segments_comp.jsonl" \
            --save_path "$DIR/retrieved.jsonl" \
            --topk 1
    fi

    echo "=== [$METHOD] 4) chat (gpt-4o-mini) ==="
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
echo "=== DONE — metrics:"
for P in p60 p70 p80; do
    F="$EXP/metrics_ours_minilm_${P}.json"
    echo "  $(basename $F):"; cat "$F"; echo
done
