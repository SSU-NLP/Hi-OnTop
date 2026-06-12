# Hi-OnTop per-turn latency (realtime, no encoder cache)

## 1. 측정 정의

- per-turn = **encode(단일 발화, 캐시 없음) + HiOnTop.assign**.
- 매 turn perf_counter 로 encode 와 assign 시간을 따로 측정 (ms).
- 첫 발화 (turn 0) 는 baseline 들과 동일하게 표본 제외.
- encoder warmup 10 forwards (cold-start, model load 비용 분리).
- encoder = `sentence-transformers/multi-qa-mpnet-base-dot-v1` (CPU).
- HP: m=2, ρ=0.7, a=0.5. seed=0.
- δ\* 무관 — encode+assign 시간은 boundary 결정 여부와 독립이므로
  Ours 의 모든 percentile/oracle 행에 동일 latency 적용.

## 2. 결과 (벤치별)

| 벤치 | n turn (timed) | Pre. encode (ms) mean / p50 / p90 | Seg. assign (ms) mean / p50 / p90 | Total (ms) mean / p50 / p90 |
|---|---:|---:|---:|---:|
| tiage | 468 | 153.14 / 105.76 / 219.80 | 0.3578 / 0.2310 / 0.3579 | 153.50 / 106.04 / 220.46 |
| dialseg711 | 498 | 106.88 / 97.15 / 156.20 | 0.2728 / 0.2292 / 0.3440 | 107.15 / 97.42 / 156.43 |
| superseg | 468 | 130.36 / 102.07 / 189.60 | 0.2753 / 0.2216 / 0.3380 | 130.63 / 102.40 / 189.90 |

## 3. cross-benchmark 평균 (Ours 행에 단일 latency 보고용)

- **Pre. (encode)**: mean = 129.64 ms · p50 100.56 · p90 184.46 (n=1434)
- **Seg. (assign)**: mean = 0.3014 ms · p50 0.2270 · p90 0.3454
- **Total (Pre.+Seg.)**: mean = 129.94 ms · p50 100.81 · p90 184.70
- cold-start (model load): 87.41 s — per-turn 과 분리.

## 4. 표 셀 매핑 (dts_result.md 의 Online Ours 4 행)

- **Pre.** 셀 = 130 ms (cross-bench mean, encoder single-utt forward).
- **Seg.** 셀 = 0.301 ms (cross-bench mean, HiOnTop.assign).
- p70/p75/p80/oracle 4 행 모두 동일 (δ\* 무관).

## 5. 한계 / 검증 미해결

- 표본 = 벤치당 500 turn budget subsample (seed=0). content-independent 한 latency 특성상 전수 측정 대비 편차 작을 것이나, 완전 일치 보장은 분포 가정에 의존.
- CPU only (machine: 본 측정 머신). GPU 환경에서는 encode 비용 한 자릿수 ms 수준으로 떨어질 가능성 — 그 경우 별도 측정 필요.
- 인코더 lazy init / jit / GC 영향 — warmup 10 fwd 로 1차 제거, 잔여 노이즈는 p50 사용시 영향 작음.
- baseline (GreedySeg 13 ms 등) 와 동일 single-utt forward 정의로 비교 가능. metric 가족 차이는 별도 — 본 REPORT 는 latency 만.
