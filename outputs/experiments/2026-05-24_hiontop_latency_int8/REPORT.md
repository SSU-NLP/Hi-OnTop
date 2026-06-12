# Hi-OnTop per-turn latency (realtime, no encoder cache)

## 1. 측정 정의

- per-turn = **encode(단일 발화, 캐시 없음) + HiOnTop.assign**.
- 매 turn perf_counter 로 encode 와 assign 시간을 따로 측정 (ms).
- 첫 발화 (turn 0) 는 baseline 들과 동일하게 표본 제외.
- encoder warmup 10 forwards (cold-start, model load 비용 분리).
- encoder = `sentence-transformers/all-MiniLM-L6-v2` (CPU).
- HP: m=2, ρ=0.7, a=0.5. seed=0.
- δ\* 무관 — encode+assign 시간은 boundary 결정 여부와 독립이므로
  Ours 의 모든 percentile/oracle 행에 동일 latency 적용.

## 2. 결과 (벤치별)

| 벤치 | n turn (timed) | Pre. encode (ms) mean / p50 / p90 | Seg. assign (ms) mean / p50 / p90 | Total (ms) mean / p50 / p90 |
|---|---:|---:|---:|---:|
| tiage | 468 | 11.62 / 9.86 / 17.76 | 0.2655 / 0.2035 / 0.2959 | 11.89 / 10.05 / 17.99 |
| dialseg711 | 498 | 11.18 / 9.97 / 16.40 | 0.2307 / 0.2013 / 0.2818 | 11.41 / 10.26 / 16.65 |
| superseg | 468 | 12.34 / 10.28 / 18.70 | 0.2423 / 0.2010 / 0.2731 | 12.58 / 10.49 / 19.00 |

## 3. cross-benchmark 평균 (Ours 행에 단일 latency 보고용)

- **Pre. (encode)**: mean = 11.70 ms · p50 10.01 · p90 17.61 (n=1434)
- **Seg. (assign)**: mean = 0.2459 ms · p50 0.2016 · p90 0.2852
- **Total (Pre.+Seg.)**: mean = 11.95 ms · p50 10.26 · p90 17.91
- cold-start (model load): 34.04 s — per-turn 과 분리.

## 4. 표 셀 매핑 (dts_result.md 의 Online Ours 4 행)

- **Pre.** 셀 = 12 ms (cross-bench mean, encoder single-utt forward).
- **Seg.** 셀 = 0.246 ms (cross-bench mean, HiOnTop.assign).
- p70/p75/p80/oracle 4 행 모두 동일 (δ\* 무관).

## 5. 한계 / 검증 미해결

- 표본 = 벤치당 500 turn budget subsample (seed=0). content-independent 한 latency 특성상 전수 측정 대비 편차 작을 것이나, 완전 일치 보장은 분포 가정에 의존.
- CPU only (machine: 본 측정 머신). GPU 환경에서는 encode 비용 한 자릿수 ms 수준으로 떨어질 가능성 — 그 경우 별도 측정 필요.
- 인코더 lazy init / jit / GC 영향 — warmup 10 fwd 로 1차 제거, 잔여 노이즈는 p50 사용시 영향 작음.
- baseline (GreedySeg 13 ms 등) 와 동일 single-utt forward 정의로 비교 가능. metric 가족 차이는 별도 — 본 REPORT 는 latency 만.
