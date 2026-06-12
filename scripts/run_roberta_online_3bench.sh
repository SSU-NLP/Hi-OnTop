#!/bin/bash
# Run online RoBERTa 3-bench full eval on CPU.
# Single-thread RoBERTa forward, ~2 hours.
# Safe to run AFTER all latency-critical SeCom work done (no latency overlap).
set -euo pipefail

REPO=/home/namchailin/Hi-OnTop
cd "$REPO"

echo "=== RoBERTa online 3-bench full eval (CPU, ~2h) ==="
echo "Started: $(date +%H:%M:%S)"
nice -n 10 uv run python methods/RoBERTa/online/segment.py \
    --name 2026-05-26_roberta_online_full
echo "Finished: $(date +%H:%M:%S)"
echo
echo "Report: outputs/experiments/2026-05-26_roberta_online_full/REPORT.md"
