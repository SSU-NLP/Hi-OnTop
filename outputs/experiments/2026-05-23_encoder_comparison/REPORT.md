# 인코더 비교 — mpnet / MiniLM / MiniLM-int8 (superdialseg_data)

데이터 = superdialseg_data, 공식 SuperDialseg metric (Score=0.5F1+0.25(1−Pk)+0.25(1−WD)).
δ* = 인코더별 train split 보정 (서브샘플 ≤400). dialseg711 은 train split 부재 → test 70:30 (70%를 보정).
Hi-OnTop HP: m=2, ρ=0.7, a=0.5.

## Ours (p80)  — δ* = train δ_eff 80-percentile

| 인코더 | tiage | dialseg711 | superseg | **mean-3** |
|---|---:|---:|---:|---:|
| mpnet | 0.4331 | 0.6162 | 0.4304 | **0.4932** |
| minilm | 0.4694 | 0.6040 | 0.3720 | **0.4818** |
| minilm-int8 | 0.4925 | 0.6074 | 0.3650 | **0.4883** |

## Ours (best-Score)  — δ* = train split F1 최대화

| 인코더 | tiage | dialseg711 | superseg | **mean-3** |
|---|---:|---:|---:|---:|
| mpnet | 0.4528 | 0.6299 | 0.4626 | **0.5151** |
| minilm | 0.4745 | 0.6086 | 0.4347 | **0.5059** |
| minilm-int8 | 0.4844 | 0.6150 | 0.4336 | **0.5110** |

## δ* 값 (인코더 × 벤치)

| 인코더 | 벤치 | δ* p80 | δ* best-Score | calib |
|---|---|---:|---:|---|
| mpnet | tiage | 0.6007 | 0.5737 | train split (calib 300 / test 100) |
| mpnet | dialseg711 | 0.5937 | 0.6144 | test 70:30 split (calib 498 / test 213) |
| mpnet | superseg | 0.6125 | 0.5534 | train split (calib 400 / test 1322) |
| minilm | tiage | 0.8192 | 0.8178 | train split (calib 300 / test 100) |
| minilm | dialseg711 | 0.8011 | 0.8178 | test 70:30 split (calib 498 / test 213) |
| minilm | superseg | 0.8352 | 0.6856 | train split (calib 400 / test 1322) |
| minilm-int8 | tiage | 0.8223 | 0.7873 | train split (calib 300 / test 100) |
| minilm-int8 | dialseg711 | 0.8029 | 0.8280 | test 70:30 split (calib 498 / test 213) |
| minilm-int8 | superseg | 0.8409 | 0.6042 | train split (calib 400 / test 1322) |

## 한계
- δ* 는 인코더별로 다름 (mpnet ~0.56, MiniLM ~0.77) — 인코더마다 재보정 필수.
- dialseg711 은 superdialseg_data 에 train split 부재 → test 70:30 seeded 분할 (no leakage).
- 베이스라인(TextTiling/GreedySeg/GraphSeg) 비교는 별도 — 그들도 superdialseg_data 에서 재측정 필요.
