#!/bin/bash
# Long-MT-Bench+ downstream eval — p60/p70/p80 percentile sweep for Hi-OnTop δ*.
#
# δ* values from outputs/.../delta_star_calibration_p_sweep.json (delta_eff
# mode, m=2, ρ=0.7, a=0.5):
#   p60 = 0.3903 · p70 = 0.4799 · p80 = 0.5983
#
# Each percentile runs full pipeline: segment → compress → retrieve → chat → eval.
# Outputs separated into result/mtbp/ours_p{60,70,80}/ and metrics_ours_p{xx}.json.
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap

cd "$REPO"

declare -A DSTARS=(
    [p60]="0.3903"
    [p70]="0.4799"
    [p80]="0.5983"
)

for P in p60 p70 p80; do
    METHOD="ours_${P}"
    DELTA="${DSTARS[$P]}"
    DIR="$RESULT/$METHOD"
    mkdir -p "$DIR"
    echo
    echo "##################################################################"
    echo "###  $METHOD  (δ* = $DELTA)"
    echo "##################################################################"

    echo "=== [$METHOD] 1) segment (v4.1.3, δ*=$DELTA) ==="
    if [ -f "$DIR/segments.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/03_segment_v413.py \
            --delta_star "$DELTA" \
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
echo "=== DONE — metrics:"
for P in p60 p70 p80; do
    echo "  metrics_ours_${P}.json:"
    cat "$EXP/metrics_ours_${P}.json"
    echo
done
