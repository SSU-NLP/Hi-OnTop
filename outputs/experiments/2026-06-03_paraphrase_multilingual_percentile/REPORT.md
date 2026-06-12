# Hi-OnTop Paraphrase-Multilingual-MiniLM-L12-v2 인코더 비교 (2026-06-03_paraphrase_multilingual_percentile)

## 실험 setup
- 인코더: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dim=384, multilingual, ONNX quint8_avx2)
- 비교 기준: 기존 MiniLM-int8 (dim=384, 11.7 ms/turn Pre.)
- 데이터: superdialseg_data (TIAGE / Dialseg711 / SuperDialseg)
- 메트릭: Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)
- HP: m=2, ρ=0.7, a=0.5 (Hi-OnTop canonical)
- δ* calibration: calib split δ_eff 의 percentile (p60/p70/p80)
- seed: 0 (rng), dialseg711 = test 70:30 split

## 결과

| percentile | TIAGE Score | DS711 Score | SDS Score | **mean** |
|---|---:|---:|---:|---:|
| p60 | 0.4416 | 0.5022 | 0.4223 | **0.4554** |
| p70 | 0.4521 | 0.5586 | 0.4113 | **0.4740** |
| p80 | 0.4297 | 0.5869 | 0.3885 | **0.4684** |

### 상세 (Pk / WD / F1 / Score, 각 dataset)

| p | dataset | δ* | Pk | WD | F1 | Score |
|---|---|---:|---:|---:|---:|---:|
| p60 | tiage | 0.7203 | 0.4546 | 0.5842 | 0.4027 | 0.4416 |
| p60 | dialseg711 | 0.6798 | 0.3945 | 0.5564 | 0.4797 | 0.5022 |
| p60 | superseg | 0.7151 | 0.5026 | 0.6065 | 0.3991 | 0.4223 |
| p70 | tiage | 0.7635 | 0.4368 | 0.5209 | 0.3830 | 0.4521 |
| p70 | dialseg711 | 0.7355 | 0.3447 | 0.4440 | 0.5116 | 0.5586 |
| p70 | superseg | 0.7751 | 0.5125 | 0.5756 | 0.3667 | 0.4113 |
| p80 | tiage | 0.8088 | 0.4362 | 0.4859 | 0.3205 | 0.4297 |
| p80 | dialseg711 | 0.7918 | 0.3164 | 0.3678 | 0.5160 | 0.5869 |
| p80 | superseg | 0.8342 | 0.5181 | 0.5517 | 0.3119 | 0.3885 |

## MiniLM-int8 비교 (dts_result.md 기준)

| percentile | TIAGE | DS711 | SDS | mean |
|---|---:|---:|---:|---:|
| p60 (int8) | 0.470 | 0.503 | 0.423 | 0.465 |
| p70 (int8) | 0.489 | 0.566 | 0.401 | 0.485 |
| p80 (int8) | 0.493 | 0.607 | 0.365 | 0.488 |

## 한계
- encoder latency 미측정 (별도 measure_hiontop_latency 실행 필요)
- 버전 경고: sentence-transformers 3.4.1 vs 모델 생성 5.1.1
