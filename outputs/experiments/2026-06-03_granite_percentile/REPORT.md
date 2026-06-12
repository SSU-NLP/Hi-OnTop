# Hi-OnTop Granite 인코더 비교 (2026-06-03_granite_percentile)

## 실험 setup
- 인코더: `ibm-granite/granite-embedding-97m-multilingual-r2` (dim=384, multilingual, ONNX quint8_avx2)
- 비교 기준: 기존 MiniLM-int8 (dim=384, 11.7 ms/turn Pre.)
- 데이터: superdialseg_data (TIAGE / Dialseg711 / SuperDialseg)
- 메트릭: Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)
- HP: m=2, ρ=0.7, a=0.5 (Hi-OnTop canonical)
- δ* calibration: calib split δ_eff 의 percentile (p60/p70/p80)
- seed: 0 (rng), dialseg711 = test 70:30 split

## 결과

| percentile | TIAGE Score | DS711 Score | SDS Score | **mean** |
|---|---:|---:|---:|---:|
| p60 | 0.4391 | 0.4875 | 0.4359 | **0.4542** |
| p70 | 0.4324 | 0.5476 | 0.4280 | **0.4693** |
| p80 | 0.4269 | 0.5844 | 0.3973 | **0.4695** |

### 상세 (Pk / WD / F1 / Score, 각 dataset)

| p | dataset | δ* | Pk | WD | F1 | Score |
|---|---|---:|---:|---:|---:|---:|
| p60 | tiage | 0.2131 | 0.4527 | 0.5838 | 0.3965 | 0.4391 |
| p60 | dialseg711 | 0.2199 | 0.4142 | 0.5716 | 0.4678 | 0.4875 |
| p60 | superseg | 0.2096 | 0.4937 | 0.5813 | 0.4093 | 0.4359 |
| p70 | tiage | 0.2241 | 0.4595 | 0.5491 | 0.3691 | 0.4324 |
| p70 | dialseg711 | 0.2352 | 0.3612 | 0.4583 | 0.5050 | 0.5476 |
| p70 | superseg | 0.2249 | 0.4995 | 0.5463 | 0.3788 | 0.4280 |
| p80 | tiage | 0.2368 | 0.4518 | 0.4975 | 0.3285 | 0.4269 |
| p80 | dialseg711 | 0.2511 | 0.3211 | 0.3720 | 0.5154 | 0.5844 |
| p80 | superseg | 0.2414 | 0.5112 | 0.5319 | 0.3161 | 0.3973 |

## MiniLM-int8 비교 (dts_result.md 기준)

| percentile | TIAGE | DS711 | SDS | mean |
|---|---:|---:|---:|---:|
| p60 (int8) | 0.470 | 0.503 | 0.423 | 0.465 |
| p70 (int8) | 0.489 | 0.566 | 0.401 | 0.485 |
| p80 (int8) | 0.493 | 0.607 | 0.365 | 0.488 |

## 한계
- encoder latency 미측정 (별도 measure_hiontop_latency 실행 필요)
- trust_remote_code=True 필요 (보안 주의)
- 버전 경고: sentence-transformers 3.4.1 vs 모델 생성 5.1.1
