# v4.3.2-exp η sweep — frozen ST5 + learned NextEmbedHead 가 δ_model 자리

**Setup**:
- mpnet channel (parent): `sentence-transformers/multi-qa-mpnet-base-dot-v1`, m=2, ρ=0.7, a=0.5, δ*=0.5594 (v4.1.1 TIAGE-train).
- ST5 channel (δ_model slot): frozen `sentence-transformers/sentence-t5-base`, NextEmbedHead tag=`st5_m5_tx_h8_l1_ff1024` (trained on DailyDialog), δ_model = 1−cos(\hat{s}_t, s_st5[t]).
- δ_eff² = η·δ_adj² + (1−η)·δ_model² (v4.1.1 식 그대로).
- δ* = 0.5594 (mpnet 그대로; ST5 cosine-distance scale ≈ 0.3~0.7).
- Sanity: η=1.0 == v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상).

## Score matrix (행=η, 열=dataset, mean = row mean)

| η | tiage | mean |
|---:|---:|---:|
| 1.00 | 0.4675 | 0.4675 |
| 0.75 | 0.4094 | 0.4094 |
| 0.50 | 0.2989 | 0.2989 |
| 0.25 | 0.2805 | 0.2805 |
| 0.00 | 0.2805 | 0.2805 |

## Detailed metrics

| η | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |
|---:|---|---:|---:|---:|---:|
| 1.00 | tiage | 0.4675 | 0.4102 | 0.4421 | 0.5082 |
| 0.75 | tiage | 0.4094 | 0.2564 | 0.4273 | 0.4478 |
| 0.50 | tiage | 0.2989 | 0.0309 | 0.4326 | 0.4334 |
| 0.25 | tiage | 0.2805 | 0.0000 | 0.4389 | 0.4389 |
| 0.00 | tiage | 0.2805 | 0.0000 | 0.4389 | 0.4389 |

## Per-dataset best

| dataset | best η | Score | vs v4.1.1 (η=1.0) |
|---|---:|---:|---:|
| tiage | 1.00 | 0.4675 | +0.0000 |

## 해석 / 판정

(채울 것 — η sweep 결과 본 후 작성)

## 한계 / 검증 미해결
- **Domain shift**: DailyDialog (open-domain conversation) → TIAGE (자연 대화) / Dialseg711 (정형) / SuperDialseg (gov FAQ).
  NextEmbedHead 가 in-domain 으로 학습된 dynamics 가 일반화하는지 미검증.
- **δ\* re-calibration 미수행** — ST5 cosine 분포가 mpnet 과 다르면 후속 필요.
- **Head architecture**: small MLP only. Transformer / GRU 비교 미수행.
- **m=5 causal window** — m sweep 미수행.
- **DailyDialog 학습 hyperparameter** (lr/epoch/batch) sweep 미수행.
- **Single seed, single ckpt** — head variance 미측정.
- **CSM 과 직접 baseline 비교 미수행** — train_csm_hf.py 산물과 별도 표 필요.
- **TIAGE-test 는 in-domain 가능성** (mpnet δ\* 가 TIAGE-train 기반).
