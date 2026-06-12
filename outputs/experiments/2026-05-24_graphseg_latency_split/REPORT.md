# GraphSeg-window per-turn latency split

## 정의
- **Pre.** = tokenize + POS + GloVe lookup/stack (_preprocess_sec + _neural_sec)
- **Seg.** = IC-weighted cosine + Hungarian + graph + clique + emit
- cold-start (GloVe load) 102.94s 분리.

## 결과
| 벤치 | n turn | Pre. mean / p50 / p90 (ms) | Seg. mean / p50 / p90 (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| tiage | 468 | 5.3699 / 1.4591 / 12.4260 | 3.8797 / -0.0359 / 0.8462 | 9.2496 |
| dialseg711 | 498 | 5.9175 / 5.6856 / 11.5846 | 0.2279 / 0.2961 / 1.1403 | 6.1454 |
| superseg | 468 | 3.0954 / 1.1459 / 8.5714 | -0.1126 / -0.0071 / 0.4803 | 2.9828 |

## cross-benchmark 평균
- **Pre.** mean = 4.8178 ms (p50 2.1340)
- **Seg.** mean = 1.3086 ms (p50 -0.0059)
- **Total** mean = 6.1264 ms
