# v4.2.4-exp η sweep — DSE 가 δ_model 자리, η ∈ [1.0, 0.75, 0.5, 0.25, 0.0]

**Setup**:
- mpnet channel (parent): encoder=`sentence-transformers/multi-qa-mpnet-base-dot-v1`, m=2, ρ=0.7, a=0.5, δ*=0.5594 (v4.1.1 TIAGE-train best).
- DSE channel (δ_model slot): encoder=`aws-ai/dse-bert-base`, m_dse=2, ρ_dse=0.5, a_dse=0.0 (v4.2.2 TIAGE-train best for DSE).
- δ_eff² = η·δ_adj² + (1−η)·δ_model² (v4.1.1 식 그대로, RNN PE 자리에 DSE causal-window PE 주입).
- δ* = 0.5594 (mpnet 그대로 — DSE δ_model 의 raw scale 이 약간 다름, η sweep 으로 effective blend 탐색).
- Sanity: η=1.0 == v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상).

## Score matrix (행=η, 열=dataset, **bold**=row best)

| η | tiage | dialseg711 | superseg | mean |
|---:|---:|---:|---:|---:|
| 1.00 | 0.4675 | 0.5897 | 0.4631 | 0.5068 |
| 0.75 | 0.4724 | 0.6254 | 0.4275 | 0.5084 |
| 0.50 | 0.4892 | 0.6480 | 0.3806 | 0.5060 |
| 0.25 | 0.4860 | 0.6455 | 0.3368 | 0.4894 |
| 0.00 | 0.4488 | 0.6352 | 0.3036 | 0.4625 |

## Detailed metrics

| η | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |
|---:|---|---:|---:|---:|---:|
| 1.00 | tiage | 0.4675 | 0.4102 | 0.4421 | 0.5082 |
| 1.00 | dialseg711 | 0.5897 | 0.5493 | 0.3248 | 0.4151 |
| 1.00 | superseg | 0.4631 | 0.4323 | 0.4711 | 0.5410 |
| 0.75 | tiage | 0.4724 | 0.3982 | 0.4307 | 0.4761 |
| 0.75 | dialseg711 | 0.6254 | 0.5919 | 0.2981 | 0.3841 |
| 0.75 | superseg | 0.4275 | 0.3824 | 0.5023 | 0.5524 |
| 0.50 | tiage | 0.4892 | 0.4028 | 0.4089 | 0.4397 |
| 0.50 | dialseg711 | 0.6480 | 0.6176 | 0.2796 | 0.3635 |
| 0.50 | superseg | 0.3806 | 0.3151 | 0.5365 | 0.5711 |
| 0.25 | tiage | 0.4860 | 0.3819 | 0.3994 | 0.4203 |
| 0.25 | dialseg711 | 0.6455 | 0.6124 | 0.2786 | 0.3641 |
| 0.25 | superseg | 0.3368 | 0.2449 | 0.5602 | 0.5826 |
| 0.00 | tiage | 0.4488 | 0.3206 | 0.4140 | 0.4322 |
| 0.00 | dialseg711 | 0.6352 | 0.5991 | 0.2847 | 0.3728 |
| 0.00 | superseg | 0.3036 | 0.1918 | 0.5755 | 0.5935 |

## Per-dataset best

| dataset | best η | Score | vs v4.1.1 (η=1.0) |
|---|---:|---:|---:|
| tiage | 0.50 | 0.4892 | ++0.0217 |
| dialseg711 | 0.50 | 0.6480 | ++0.0583 |
| superseg | 1.00 | 0.4631 | ++0.0000 |

## 해석 / 판정

(채울 것)

## 한계
- Raw scale mismatch: mpnet δ ≈ 0.4~0.6, DSE δ ≈ 0.3~0.5. η blend 가 raw 로 일어남.
- δ\* 은 mpnet 기준 (0.5594). η<1 일 때 effective δ_eff scale 이 변하므로 δ\* 재calibration 가능.
- Single trial per η, no seed variance.
- DSE channel HP 는 v4.2.2 best 고정. 재calibration 시 best η 다를 가능성.
