# TextTiling-streaming per-turn latency split

## 정의
- **Pre.** = tokenize + stopword + bag-of-words (`_preprocess_sec` 증분)
- **Seg.** = block-cosine depth + welford threshold + boundary emit
- per-turn perf_counter, 첫 발화 제외, no neural model (warmup 불필요).

## 결과
| 벤치 | n turn | Pre. mean / p50 / p90 (ms) | Seg. mean / p50 / p90 (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| tiage | 468 | 0.0182 / 0.0156 / 0.0255 | 0.0039 / 0.0013 / 0.0023 | 0.0221 |
| dialseg711 | 498 | 0.0179 / 0.0157 / 0.0263 | 0.0248 / 0.0014 / 0.0624 | 0.0427 |
| superseg | 468 | 0.0248 / 0.0167 / 0.0382 | 0.0080 / 0.0013 / 0.0046 | 0.0328 |

## cross-benchmark 평균
- **Pre.** mean = 0.0203 ms (p50 0.0159)
- **Seg.** mean = 0.0125 ms (p50 0.0013)
- **Total** mean = 0.0327 ms
