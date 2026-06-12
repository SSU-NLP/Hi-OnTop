# v4.3.2-exp η sweep — frozen ST5 + learned NextEmbedHead 가 δ_model 자리

**Setup**:
- mpnet channel (parent): `sentence-transformers/multi-qa-mpnet-base-dot-v1`, m=2, ρ=0.7, a=0.5, δ*=0.5594 (v4.1.1 TIAGE-train).
- ST5 channel (δ_model slot): frozen `sentence-transformers/sentence-t5-base`, NextEmbedHead tag=`st5_m5_tx_h8_l1_ff1024` (trained on DailyDialog), δ_model = 1−cos(\hat{s}_t, s_st5[t]).
- δ_eff² = η·δ_adj² + (1−η)·δ_model² (v4.1.1 식 그대로).
- δ* = 0.5594 (mpnet 그대로; ST5 cosine-distance scale ≈ 0.3~0.7).
- Sanity: η=1.0 == v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상).

## Score matrix (행=η, 열=dataset, mean = row mean)

| η | tiage | dialseg711 | superseg | mean |
|---:|---:|---:|---:|---:|
| 1.00 | 0.4675 | 0.5897 | 0.4631 | 0.5068 |
| 0.75 | 0.4573 | 0.5997 | 0.4498 | 0.5023 |
| 0.50 | 0.4414 | 0.5679 | 0.4375 | 0.4823 |
| 0.25 | 0.4247 | 0.5116 | 0.4138 | 0.4500 |
| 0.00 | 0.3931 | 0.4492 | 0.3957 | 0.4127 |

## Detailed metrics

| η | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |
|---:|---|---:|---:|---:|---:|
| 1.00 | tiage | 0.4675 | 0.4102 | 0.4421 | 0.5082 |
| 1.00 | dialseg711 | 0.5897 | 0.5493 | 0.3248 | 0.4151 |
| 1.00 | superseg | 0.4631 | 0.4323 | 0.4711 | 0.5410 |
| 0.75 | tiage | 0.4573 | 0.4035 | 0.4445 | 0.5333 |
| 0.75 | dialseg711 | 0.5997 | 0.5694 | 0.3158 | 0.4240 |
| 0.75 | superseg | 0.4498 | 0.4211 | 0.4829 | 0.5601 |
| 0.50 | tiage | 0.4414 | 0.3942 | 0.4554 | 0.5673 |
| 0.50 | dialseg711 | 0.5679 | 0.5444 | 0.3386 | 0.4786 |
| 0.50 | superseg | 0.4375 | 0.4070 | 0.4887 | 0.5753 |
| 0.25 | tiage | 0.4247 | 0.3862 | 0.4704 | 0.6034 |
| 0.25 | dialseg711 | 0.5116 | 0.4930 | 0.3781 | 0.5615 |
| 0.25 | superseg | 0.4138 | 0.3755 | 0.5015 | 0.5943 |
| 0.00 | tiage | 0.3931 | 0.3619 | 0.4975 | 0.6539 |
| 0.00 | dialseg711 | 0.4492 | 0.4356 | 0.4244 | 0.6501 |
| 0.00 | superseg | 0.3957 | 0.3523 | 0.5071 | 0.6146 |

## Per-dataset best

| dataset | best η | Score | vs v4.1.1 (η=1.0) |
|---|---:|---:|---:|
| tiage | 1.00 | 0.4675 | +0.0000 |
| dialseg711 | 0.75 | 0.5997 | +0.0100 |
| superseg | 1.00 | 0.4631 | +0.0000 |

## 해석 / 판정

**Clear negative (single η 기준), no promotion.**

### 정직 비교 (single η, no test leak)

η 를 dataset 마다 다르게 고르면 test 누설. 정직 비교는 mean Score 가
best 인 single η 를 모든 dataset 동일 적용.

| | best single η | mean Score | vs v4.1.1 |
|---|---:|---:|---:|
| v4.1.1 (η=1) | — | 0.5068 | — |
| v4.2.4 (raw, frozen DSE) | 0.75 | 0.5084 | +0.16pp (noise 의심) |
| **v4.3.2 (이 보고, calibrated)** | **0.75** | **0.5023** | **−0.45pp** |

→ v4.3.2 는 어떤 single η 로도 v4.1.1 못 이김. **clear negative**.

### Per-dataset best (⚠ test leak, ablation 진단용만 — promotion 근거 X)

| dataset | best η | Score | vs v4.1.1 | 비고 |
|---|---:|---:|---:|---|
| TIAGE | 1.00 | 0.4675 | 0.00 | head 무용 |
| Dialseg711 | 0.75 | 0.5997 | +1.00pp | 약신호, v4.2.4 그림자 |
| SuperDialseg | 1.00 | 0.4631 | 0.00 | head 무용 |

Per-dataset best 는 *어디서 신호 약하게라도 있었는지* 진단 목적만. test
셋 sweep 으로 best 고른 것이므로 deploy / paper 결론 근거 사용 금지.

### Boundary signal 진단 (TIAGE test)

- boundary-after δ_model = 0.1799 ± 0.0293 (n=315)
- non-boundary δ_model = 0.1692 ± 0.0312 (n=1149)
- diff +0.0107, Cohen's d = 0.348, p = 5.5e-08
- → 통계 유의하나 분류로는 너무 약함 (분포 ~86% 겹침). regression-to-mean.

### 판정

v4.3.2-exp suffix 유지, v4.1.1 default 무변경. **single η 기준 clear
negative.** Dialseg711 의 약신호 (per-dataset best 기준 +1pp) 는 v4.2.4
패턴의 *부분적* 재현이지만 promotion 근거 아님.

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
