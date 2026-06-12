#!/bin/bash
# Long-MT-Bench+ downstream eval — Hi-OnTop with LLM-distill macro best_p.
#
# best_p values from outputs/experiments/2026-05-25_llm_distillation_calib/
# results.json (macro across 3 LLM refs: GPT-5 / Qwen3.5-27B / Qwen3.5-122B-A10B):
#
#   MPNet        best_p=71  δ*=0.4854
#   MiniLM       best_p=72  δ*=0.7638
#   MiniLM-int8  best_p=72  δ*=0.7678
#
# Each row = full SeCom pipeline (segment → compress → retrieve → chat → eval)
# with the macro best_p δ*. Outputs into result/mtbp/ours_<enc>_bestp/ +
# metrics_ours_<enc>_bestp.json.
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap

cd "$REPO"

# label  encoder  backend  onnx_file  dim  delta*  p
RUNS=(
    "mpnet|sentence-transformers/multi-qa-mpnet-base-dot-v1|local|-|768|0.4854|71"
    "minilm|sentence-transformers/all-MiniLM-L6-v2|local|-|384|0.7638|72"
    "minilm_int8|sentence-transformers/all-MiniLM-L6-v2|onnx|onnx/model_quint8_avx2.onnx|384|0.7678|72"
)

for spec in "${RUNS[@]}"; do
    IFS='|' read -r LABEL ENC BACKEND ONNX DIM DELTA P <<< "$spec"
    METHOD="ours_${LABEL}_bestp"
    DIR="$RESULT/$METHOD"
    mkdir -p "$DIR"

    echo
    echo "##################################################################"
    echo "###  $METHOD  (encoder=$ENC, backend=$BACKEND, δ*=$DELTA, p=$P)"
    echo "##################################################################"

    SEG_ARGS=(--encoder "$ENC" --encoder_backend "$BACKEND"
              --dim "$DIM" --delta_star "$DELTA"
              --save_path "$DIR/segments.jsonl"
              --latency_path "$EXP/latency_${METHOD}.json")
    if [[ "$BACKEND" == "onnx" ]]; then
        SEG_ARGS+=(--onnx_file "$ONNX")
    fi

    echo "=== [$METHOD] 1) segment (Hi-OnTop, δ*=$DELTA) ==="
    if [ -f "$DIR/segments.jsonl" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/03_segment_v413.py "${SEG_ARGS[@]}"
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
for spec in "${RUNS[@]}"; do
    IFS='|' read -r LABEL _ _ _ _ _ _ <<< "$spec"
    METHOD="ours_${LABEL}_bestp"
    echo "  metrics_${METHOD}.json:"
    cat "$EXP/metrics_${METHOD}.json"
    echo
done
