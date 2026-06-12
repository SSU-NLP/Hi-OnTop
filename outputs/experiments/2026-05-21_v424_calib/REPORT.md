# v4.2.4-exp δ* re-calibration (A: tiage-train, B: superseg-validation)

**Setup**:
- v4.2.4-exp 의 mixed δ_eff = √(η·δ_adj² + (1-η)·δ_model²) 분포가
  mpnet 단독과 다름 → δ*=0.5594 (TIAGE-cfg) 가 부적합할 수 있음.
- 본 실험: η ∈ {0.5, 0.75} 별로 F1-best δ* 를 calibration source 에서
  재산출 후 test 평가.
- Calibration source: **tiage-train** (300 conv, 4692 turn) +
  **superseg-validation** (1322 conv, 17734 turn).
  ⚠ superseg-train 은 6948 conv / 92k turn 으로 CPU 환경에서 인코딩
  불가능 → validation 사용 (v4.1.1.md 의 "val-cal" convention).
- δ* 적용 정책 (no test leakage):
  - tiage-test ← tiage-train δ* (in-domain, calibration overlap ⚠)
  - dialseg711-test ← tiage-train δ* (cross-corpus, train split 없음)
  - superseg-test ← superseg-validation δ* (in-corpus)

## Phase 1 — calibration 결과

| source | split | η | δ\* (F1-best) | train F1 |
|---|---|---:|---:|---:|
| tiage | train | 0.5 | 0.5260 | 0.462 |
| tiage | train | 0.75 | 0.5209 | 0.458 |
| superseg | validation | 0.5 | 0.3801 | 0.463 |
| superseg | validation | 0.75 | 0.4529 | 0.465 |

## Phase 2 — test 평가 (재calibration δ* 사용)

| η | dataset | δ\* source | δ\* | **Score ↑** | F1 ↑ | Pk ↓ | WD ↓ |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.5 | tiage | tiage-train | 0.5260 | **0.4991** | 0.4420 | 0.4138 | 0.4739 |
| 0.5 | dialseg711 | tiage-train | 0.5260 | **0.5690** | 0.5552 | 0.3505 | 0.4839 |
| 0.5 | superseg | superseg-val | 0.3801 | **0.4230** | 0.4496 | 0.4757 | 0.7316 |
| 0.75 | tiage | tiage-train | 0.5209 | **0.4826** | 0.4493 | 0.4346 | 0.5335 |
| 0.75 | dialseg711 | tiage-train | 0.5209 | **0.5455** | 0.5330 | 0.3697 | 0.5143 |
| 0.75 | superseg | superseg-val | 0.4529 | **0.4419** | 0.4501 | 0.4753 | 0.6573 |

## 비교 — 재calibration 전 (δ*=0.5594, TIAGE-cfg) vs 후

| η | dataset | prior Score | new Score | Δ | v4.1.1 baseline |
|---:|---|---:|---:|---:|---:|
| 0.5 | tiage | 0.4892 | 0.4991 | ++0.0099 | 0.4675 |
| 0.5 | dialseg711 | 0.6480 | 0.5690 | -0.0790 | 0.5897 |
| 0.5 | superseg | 0.3806 | 0.4230 | ++0.0424 | 0.4631 |
| 0.75 | tiage | 0.4724 | 0.4826 | ++0.0102 | 0.4675 |
| 0.75 | dialseg711 | 0.6254 | 0.5455 | -0.0799 | 0.5897 |
| 0.75 | superseg | 0.4275 | 0.4419 | ++0.0144 | 0.4631 |

## 3-set mean Score

| 변형 | mean Score | vs v4.1.1 (0.5068) |
|---|---:|---:|
| v4.1.1 (mpnet) | 0.5068 | — |
| v4.2.4 @η=0.5, TIAGE-cfg δ* | 0.5059 | -0.0009 |
| **v4.2.4 @η=0.5, re-cal δ\*** | **0.4970** | -0.0098 |
| v4.2.4 @η=0.75, TIAGE-cfg δ* | 0.5084 | +0.0016 |
| **v4.2.4 @η=0.75, re-cal δ\*** | **0.4900** | -0.0168 |

## 해석 / 판정

(채울 것)

## 한계
- superseg-train (92k turns) 인코딩 환경 제약으로 validation 사용. 
  train 과 validation 분포 약간 다를 수 있음.
- F1-best δ* 채택. Score-best δ* 와 다를 수 있음 (v4.1.1.md 의 superseg
  val-cal 경험상 F1-best 가 과분절 유발 가능).
- Single trial, seed=0. variance 미측정.
- DSE channel HP (m=2, ρ=0.5, a=0.0) 은 v4.2.2 best 고정. joint
  re-calibration 안 함.
