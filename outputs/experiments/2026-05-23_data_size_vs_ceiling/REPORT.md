# 데이터 크기 vs 천장 — 3 벤치 × 3 인코더 격자

**질문**: 천장 (test-side oracle Score) 이 데이터 크기 (대화 수 / 턴 수) 와 상관 있나? 아니면 인코더 + 벤치 난이도 가 지배적인가?

**방법**: 각 (인코더, 벤치) 셀에서 test 자체에서 δ* sweep [0.35, 0.95] (data-snooping 허용) → calib 와 무관한 알고리즘+데이터 천장.

**HP**: Hi-OnTop m=2, ρ=0.7, a=0.5. metric = 공식 SuperDialseg (0.5F1+0.25(1−Pk)+0.25(1−WD)).

## 데이터 크기

| 벤치 | calib dialog | calib turn | test dialog | test turn | note |
|---|---:|---:|---:|---:|---|
| tiage | 300 | 4692 | 100 | 1564 | train split |
| dialseg711 | 498 | 13467 | 213 | 5883 | test 70:30 |
| superseg | 6948 | 92151 | 1322 | 17328 | train split |

## test-side oracle Score (천장) — 인코더 × 벤치

| 벤치 | MPNet | MiniLM | MiniLM-int8 |
|---|---:|---:|---:|
| tiage | 0.4729 | 0.4849 | 0.4889 |
| dialseg711 | 0.6299 | 0.6090 | 0.6012 |
| superseg | 0.4643 | 0.4382 | 0.4360 |

## test-side oracle δ* — 인코더 × 벤치

| 벤치 | MPNet | MiniLM | MiniLM-int8 |
|---|---:|---:|---:|
| tiage | 0.543 | 0.767 | 0.777 |
| dialseg711 | 0.614 | 0.818 | 0.828 |
| superseg | 0.543 | 0.645 | 0.625 |

## 해석 가이드

- 같은 벤치 안에서 인코더별 천장 차 작으면 → **벤치 난이도** 지배적.
- 같은 인코더 안에서 벤치별 천장 차 크면 → **데이터 특성** 지배적.
- 데이터 크기 (turn 수) 큰 벤치에서 천장이 *낮으면* → 크기 ≠ 점수. 어려운 데이터가 큰 경우.
- δ* 가 인코더 마다 크게 다르면 → 인코더 별 cosine 분포 자체가 달라 calib 불가피.
