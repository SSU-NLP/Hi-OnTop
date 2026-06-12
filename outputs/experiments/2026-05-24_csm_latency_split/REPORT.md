# CSM-online-delay2 per-turn latency split (Pre. vs Seg.)

## 정의
- **Pre.** = `_coherence_score` (tokenize + BERT forward, single pair)
- **Seg.** = `push()` 안의 나머지 (depth peak walk + welford running threshold + boundary emit)
- per-turn perf_counter 측정, 첫 발화 제외, 10-pair warm-up + cold-start (29.7s) 분리.
- 인코더 = bert-base-uncased + lxing532 CoherenceNet head, CPU.
- HP: alpha=1.0, delay=2, min_gap=2, max_seq=128.

## 결과 (벤치별)

| 벤치 | n turn | Pre. mean / p50 / p90 (ms) | Seg. mean / p50 / p90 (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| tiage | 468 | 292.16 / 254.06 / 350.50 | 0.0446 / 0.0318 / 0.0412 | 292.21 |
| dialseg711 | 498 | 282.71 / 253.34 / 341.53 | 0.0374 / 0.0324 / 0.0404 | 282.75 |
| superseg | 468 | 274.87 / 249.64 / 323.15 | 0.0288 / 0.0311 / 0.0374 | 274.90 |

## cross-benchmark 평균 (dts_result.md 셀)

- **Pre.** mean = 283.23 ms (p50 252.23)
- **Seg.** mean = 0.0369 ms (p50 0.0319)
- **Total** mean = 283.27 ms
- cold-start 29.7s — per-turn 과 분리.

## 한계
- 벤치당 500 turn budget subsample (seed=0). content-independent latency 가정으로 전수 측정 대비 편차 작음.
- CPU only. GPU 환경에선 Pre. 한 자릿수 ms 가능.
