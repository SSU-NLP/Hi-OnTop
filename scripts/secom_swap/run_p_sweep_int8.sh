#!/bin/bash
# Long-MT-Bench+ downstream eval — MiniLM-int8 ONNX × p60/p70/p80 percentile sweep.
#
# δ* values from delta_star_calibration_minilm_int8_p_sweep.json (mode=delta_eff,
# m=2, ρ=0.7, a=0.5, encoder=all-MiniLM-L6-v2 ONNX quint8_avx2):
#   p60 = 0.5227 · p70 = 0.7049 · p80 = 0.8704
#
# Outputs separated into result/mtbp/ours_int8_p{60,70,80}/ and
# metrics_ours_int8_p{xx}.json. MPNet pipeline 결과는 ours_p60 에 그대로.
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap
ENC="sentence-transformers/all-MiniLM-L6-v2"
ONNX="onnx/model_quint8_avx2.onnx"
DIM=384

cd "$REPO"

declare -A DSTARS=(
    [p60]="0.5227"
    [p70]="0.7049"
    [p80]="0.8704"
)

for P in p60 p70 p80; do
    METHOD="ours_int8_${P}"
    DELTA="${DSTARS[$P]}"
    DIR="$RESULT/$METHOD"
    mkdir -p "$DIR"
    echo
    echo "##################################################################"
    echo "###  $METHOD  (δ* = $DELTA, encoder=MiniLM-int8 ONNX, dim=$DIM)"
    echo "##################################################################"

    echo "=== [$METHOD] 1) segment (v4.1.3, MiniLM-int8, δ*=$DELTA) ==="
    if [ -f "$DIR/segments.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/03_segment_v413.py \
            --encoder "$ENC" --encoder_backend onnx --onnx_file "$ONNX" \
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
    echo "  metrics_ours_int8_${P}.json:"
    cat "$EXP/metrics_ours_int8_${P}.json"
    echo
done
