# GreedySeg-online-delay2 per-turn latency split

## 정의
- **Pre.** = BERT-base forward (single utt CLS) (_neural_sec 증분)
- **Seg.** = cosine + argmin greedy bounded-lookahead + emit
- 10-forward warmup, cold-start (BERT load) 24.74s 분리.

## 결과
| 벤치 | n turn | Pre. mean / p50 / p90 (ms) | Seg. mean / p50 / p90 (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| tiage | 468 | 199.88 / 0.00 / 498.67 | 2.0918 / 0.0134 / 5.2435 | 201.97 |
| dialseg711 | 498 | 311.33 / 0.00 / 1146.27 | 3.5336 / 0.0146 / 12.8400 | 314.86 |
| superseg | 468 | 180.41 / 0.00 / 469.72 | 2.1481 / 0.0119 / 5.6765 | 182.55 |

## cross-benchmark 평균
- **Pre.** mean = 232.23 ms (p50 0.00)
- **Seg.** mean = 2.6109 ms (p50 0.0130)
- **Total** mean = 234.84 ms
