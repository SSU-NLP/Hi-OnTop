# v4.2.5-exp η sweep — CSM (finetuned BERT-base) 이 δ_model 자리

**Setup**: mpnet (parent), m=2, ρ=0.7, a=0.5, δ*=0.5594. CSM tag=`csm_bert_base` (precomputed δ = 1 − p_coh). Blend mode = raw.

## Score matrix (행=η, 열=dataset, mean = row mean)

| η | tiage | mean |
|---:|---:|---:|
| 1.00 | 0.4675 | 0.4675 |
| 0.75 | 0.4840 | 0.4840 |
| 0.50 | 0.4856 | 0.4856 |
| 0.25 | 0.4875 | 0.4875 |
| 0.00 | 0.4875 | 0.4875 |

## Detailed metrics

| η | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |
|---:|---|---:|---:|---:|---:|
| 1.00 | tiage | 0.4675 | 0.4102 | 0.4421 | 0.5082 |
| 0.75 | tiage | 0.4840 | 0.4951 | 0.4531 | 0.6011 |
| 0.50 | tiage | 0.4856 | 0.4946 | 0.4512 | 0.5958 |
| 0.25 | tiage | 0.4875 | 0.4956 | 0.4483 | 0.5929 |
| 0.00 | tiage | 0.4875 | 0.4956 | 0.4483 | 0.5929 |

## 정직 비교: single η (mean-best, no test leak)

| | best single η | mean Score | vs v4.1.1 |
|---|---:|---:|---:|
| v4.1.1 (η=1) | — | 0.4675 | — |
| **v4.2.5 (raw)** | 0.25 | 0.4875 | +0.0200 |

## Per-dataset best (⚠ test leak, ablation 참고용만)

| dataset | best η | Score | vs v4.1.1 |
|---|---:|---:|---:|
| tiage | 0.25 | 0.4875 | +0.0200 |

## 해석 / 판정

(채울 것 — sweep 결과 보고 작성)

## 한계 / 검증 미해결
- **Domain shift**: CSM 학습 corpus (DailyDialog NSP triplet) → TIAGE / Dialseg711 / SuperDialseg.
- **단일 ckpt** (cpt_277000.pth), seed variance 미측정.
- **δ\* re-calibration**: TIAGE-train 직접 calib 미수행 (현재는 test mean sanity 또는 raw).
- **v4.3.2 (continuous regression) 와 직접 비교**: 같은 η 와 동일 setup 에서 두 head 가 어디가 다른지 분리 필요.
