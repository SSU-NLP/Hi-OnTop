#!/usr/bin/env bash
# v3.3.1 HP sweep on LoCoMo — search (cos_threshold, α, β) that breaks
# mega-topic collapse. Default (0.7, 1, 0.5) gives T1μ=386 (mega). Try 5
# configs ranging from "slight tighten" to "very aggressive".
#
# Each run captures STM top-K stats (T1/T2/T3) so we can verify whether
# mega is actually breaking.
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR="outputs/sweeps/2026-05-08_v331_hpsweep"
mkdir -p "$OUT_DIR"

# (cos_threshold, alpha, beta)
configs=(
  "0.85  10   0.5"
  "0.90  10   0.5"
  "0.85  100  0.25"
  "0.90  100  0.25"
  "0.95  100  0.1"
)

run_one() {
  local cos="$1"
  local alpha="$2"
  local beta="$3"
  local tag="cos${cos}_a${alpha}_b${beta}"
  tag="${tag//./p}"
  local run_dir="$OUT_DIR/$tag"
  local exp_id="20260508_v331_hp_${tag}"
  local log="$run_dir/run.log"
  local topk="$run_dir/stm_topk.json"

  mkdir -p "$run_dir"
  rm -f "$topk" "${topk%.json}.rounds.jsonl" 2>/dev/null || true
  {
    echo "=== START $(date -Is) ==="
    echo "method=hi-ontop-full-v3.3.1"
    echo "cos_threshold=${cos}  alpha=${alpha}  beta=${beta}"
    echo "exp_id=${exp_id}"
    echo
  } > "$log"

  WANDB_MODE=disabled \
  UV_CACHE_DIR=/tmp/uv-cache \
  HIONTOP_STM_TOPK_STATS_PATH="$topk" \
  uv run python scripts/run_experiment.py \
    --method hi-ontop-full-v3.3.1 \
    --benchmark locomo \
    --data benchmarks/locomo/data/locomo10.json \
    --limit 50 --stratify \
    --questions-per-round 50 \
    --exp-id "$exp_id" \
    --results-root "$run_dir/results" \
    --no-token-count --no-thinking --workers 100 \
    --cos-threshold "$cos" --alpha "$alpha" --beta "$beta" \
    >> "$log" 2>&1
  local rc=$?
  echo "$rc" > "$run_dir/exit_code.txt"
  echo "[exit] ${exp_id} rc=${rc}" >> "$log"
  echo "=== END $(date -Is) ===" >> "$log"
}

for cfg in "${configs[@]}"; do
  read -r cos alpha beta <<< "$cfg"
  run_one "$cos" "$alpha" "$beta"
done

python - <<'PY'
import json
from pathlib import Path

out = Path("outputs/sweeps/2026-05-08_v331_hpsweep")
print("\n| cos | α | β | acc | multi-hop | single-hop | T1μ | T2μ | T3μ | T1max | n_topics_avg |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for d in sorted(out.iterdir()):
    if not d.is_dir() or not (d / "exit_code.txt").exists():
        continue
    sp = list((d / "results" / "experiments").glob("*/summary.json"))
    if not sp: continue
    s = json.loads(sp[0].read_text())
    tk = d / "stm_topk.json"
    t = json.loads(tk.read_text()) if tk.exists() else {}
    t1 = t.get("top1_per_round",{})
    t2 = t.get("top2_per_round",{})
    t3 = t.get("top3_per_round",{})
    # parse cos/alpha/beta from tag
    tag = d.name
    cos = tag.split("_")[0].replace("cos","").replace("p",".")
    al = tag.split("_a")[1].split("_")[0]
    be = tag.split("_b")[1].replace("p",".")
    acc = s.get("accuracy_overall", 0)
    mh = s.get("accuracy_by_qtype/multi-hop", 0)
    sh = s.get("accuracy_by_qtype/single-hop", 0)
    # average #topics from rounds.jsonl
    rj = d / "stm_topk.rounds.jsonl"
    if rj.exists():
        nts = [json.loads(l).get("n_stm_topics", 0) for l in rj.open() if l.strip()]
        nt_avg = sum(nts)/len(nts) if nts else 0
    else:
        nt_avg = 0
    def f(v, dp=3):
        return "-" if v is None else f"{float(v):.{dp}f}"
    print(f"| {cos} | {al} | {be} | {f(acc)} | {f(mh)} | {f(sh)} | "
          f"{f(t1.get('mean'),1)} | {f(t2.get('mean'),1)} | {f(t3.get('mean'),1)} | "
          f"{t1.get('max','-')} | {f(nt_avg,2)} |")
PY
