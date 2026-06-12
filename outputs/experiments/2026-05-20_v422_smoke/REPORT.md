# v4.2.2 smoke — encoder swap (mpnet → DSE-BERT) on ['tiage', 'dialseg711', 'superseg']

**Setup**:
- Algorithm = v4.1.1 (causal-window δ identity segmenter), encoder only swapped.
- v4.1.1 baseline: encoder=`sentence-transformers/multi-qa-mpnet-base-dot-v1`, m=2, ρ=0.7, a=0.5, δ*=0.5594 (TIAGE-train calibration).
- v4.2.2 variant: encoder=`aws-ai/dse-bert-base`, m=2, ρ=0.5, a=0.0, δ*=0.4569 (TIAGE-train, DSE-BERT calibration).
- Dialseg711 has no train split → uses TIAGE-train δ\* per encoder (cross-corpus).
- Metric: official SuperDialseg (Pk/WD k=auto, F1 binary), Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).

## 결과

| arm | dataset | m | ρ | a | δ\* | **Score ↑** | F1 ↑ | Pk ↓ | WD ↓ | wall(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4.1.1 (mpnet) | tiage | 2 | 0.7 | 0.5 | 0.5594 | **0.4675** | 0.4102 | 0.4421 | 0.5082 | 5 |
| v4.2.2 (DSE-BERT) | tiage | 2 | 0.5 | 0 | 0.4569 | **0.4747** | 0.4420 | 0.4366 | 0.5488 | 2 |
| v4.1.1 (mpnet) | dialseg711 | 2 | 0.7 | 0.5 | 0.5594 | **0.5897** | 0.5493 | 0.3248 | 0.4151 | 23 |
| v4.2.2 (DSE-BERT) | dialseg711 | 2 | 0.5 | 0 | 0.4569 | **0.4547** | 0.4587 | 0.4472 | 0.6514 | 45 |
| v4.1.1 (mpnet) | superseg | 2 | 0.7 | 0.5 | 0.5594 | **0.4631** | 0.4323 | 0.4711 | 0.5410 | 23 |
| v4.2.2 (DSE-BERT) | superseg | 2 | 0.5 | 0 | 0.4569 | **0.3886** | 0.3488 | 0.5332 | 0.6098 | 27 |

## 해석

(빈 칸 — 숫자 보고 채워야 함. 사용자 / Claude 가 작성)

## 판정

- v4.2.2 Score 가 v4.1.1 대비 모든 데이터셋에서 노이즈 밖 우위면 → 승격,   methodology v4.2.2.md '결과' 섹션 + decision-log entry append.
- 일부 우위 / 일부 회귀면 → 부분 채택, v4.1.1 default 유지 + v4.2.2 옵션 보존.
- 우위 없으면 → negative result, v4.1.1.md '고려 중인 변형' 한 줄 기록.

## 한계 / 검증 미해결

- seed=0 (v4.1.1 default) 단일 run — variance 미측정.
- δ\* 는 TIAGE-train 만 calibration. superseg-train 도 별도 δ\* 산출하면   더 공정 (현재는 cross-corpus TIAGE δ\* 그대로 사용).
- DSE-BERT 의 max_seq_length=512 default — 긴 발화 truncation 검증 필요.
- 단일 (m, ρ, a) 만 — encoder 별 ctx_window 최적값 다를 수 있음 (sweep 미수행).
