#!/bin/bash
# LATENCY BLOCK only — segment + latency measurement for ablation a=0.0, a=1.0.
# Run on idle CPU after all SeCom main sweeps done.
#
# δ* values from outputs/experiments/2026-05-26_ablation_blend/calib.json
# (MTB+ pool p70 percentile, computed numpy-only):
#   MPNet  a=0.0 (ctx-only)  δ*=0.4622
#   MPNet  a=1.0 (prev-only) δ*=0.4957
#
# Outputs:
#   benchmarks/SeCom/experiment/result/mtbp/ours_mpnet_a{0,1}/segments.jsonl
#   outputs/experiments/2026-05-21_v413_secom_swap/latency_ours_mpnet_a{0,1}.json
#
# After all segments + latency measurements done → run_ablation_pipeline.sh
# (which does compress/retrieve/chat/eval; NOT latency-critical).
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
RESULT=$REPO/benchmarks/SeCom/experiment/result/mtbp
EXP=$REPO/outputs/experiments/2026-05-21_v413_secom_swap
ENC="sentence-transformers/multi-qa-mpnet-base-dot-v1"
DIM=768

cd "$REPO"

declare -A DSTARS=(
    [a0]="0.4622"   # ctx only
    [a1]="0.4957"   # prev only
)
declare -A BLENDS=(
    [a0]="0.0"
    [a1]="1.0"
)

for A in a0 a1; do
    METHOD="ours_mpnet_${A}"
    DELTA="${DSTARS[$A]}"
    BLEND="${BLENDS[$A]}"
    DIR="$RESULT/$METHOD"
    mkdir -p "$DIR"
    echo
    echo "##################################################################"
    echo "###  $METHOD  (a=$BLEND, δ*=$DELTA, MPNet)  [LATENCY BLOCK]"
    echo "##################################################################"

    if [ -f "$DIR/segments.jsonl" ] && [ -f "$EXP/latency_${METHOD}.json" ]; then
        echo "  → cached, skip"
    else
        uv run python scripts/secom_swap/03_segment_v413.py \
            --encoder "$ENC" --encoder_backend local \
            --dim "$DIM" --delta_star "$DELTA" \
            --ctx_blend_a "$BLEND" \
            --ctx_window 2 --ctx_decay 0.7 \
            --save_path "$DIR/segments.jsonl" \
            --latency_path "$EXP/latency_${METHOD}.json"
    fi
done

echo
echo "=== LATENCY BLOCK done. Next: run_ablation_pipeline.sh"
