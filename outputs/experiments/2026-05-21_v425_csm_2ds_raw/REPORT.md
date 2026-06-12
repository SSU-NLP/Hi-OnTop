# v4.2.5-exp η sweep — CSM (finetuned BERT-base) 이 δ_model 자리

**Setup**: mpnet (parent), m=2, ρ=0.7, a=0.5, δ*=0.5594. CSM tag=`csm_bert_base` (precomputed δ = 1 − p_coh). Blend mode = raw.

## Score matrix (행=η, 열=dataset, mean = row mean)

| η | tiage | dialseg711 | mean |
|---:|---:|---:|---:|
| 1.00 | 0.4675 | 0.5897 | 0.5286 |
| 0.75 | 0.4840 | 0.4072 | 0.4456 |
| 0.50 | 0.4856 | 0.4041 | 0.4448 |
| 0.25 | 0.4875 | 0.4040 | 0.4458 |
| 0.00 | 0.4875 | 0.4040 | 0.4458 |

## Detailed metrics

| η | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |
|---:|---|---:|---:|---:|---:|
| 1.00 | tiage | 0.4675 | 0.4102 | 0.4421 | 0.5082 |
| 1.00 | dialseg711 | 0.5897 | 0.5493 | 0.3248 | 0.4151 |
| 0.75 | tiage | 0.4840 | 0.4951 | 0.4531 | 0.6011 |
| 0.75 | dialseg711 | 0.4072 | 0.4207 | 0.4919 | 0.7208 |
| 0.50 | tiage | 0.4856 | 0.4946 | 0.4512 | 0.5958 |
| 0.50 | dialseg711 | 0.4041 | 0.4167 | 0.4932 | 0.7239 |
| 0.25 | tiage | 0.4875 | 0.4956 | 0.4483 | 0.5929 |
| 0.25 | dialseg711 | 0.4040 | 0.4166 | 0.4932 | 0.7238 |
| 0.00 | tiage | 0.4875 | 0.4956 | 0.4483 | 0.5929 |
| 0.00 | dialseg711 | 0.4040 | 0.4166 | 0.4932 | 0.7238 |

## 정직 비교: single η (mean-best, no test leak)

| | best single η | mean Score | vs v4.1.1 |
|---|---:|---:|---:|
| v4.1.1 (η=1) | — | 0.5286 | — |
| **v4.2.5 (raw)** | 1.00 | 0.5286 | -0.0000 |

## Per-dataset best (⚠ test leak, ablation 참고용만)

| dataset | best η | Score | vs v4.1.1 |
|---|---:|---:|---:|
| tiage | 0.25 | 0.4875 | +0.0200 |
| dialseg711 | 1.00 | 0.5897 | -0.0000 |

## 해석 / 판정

**판정: NEGATIVE (single-η mean-best 기준)** — raw blend 에서 v4.2.5 가
v4.1.1 와 동률 (η=1.0 즉 CSM off 가 mean-best). CSM 을 섞는 순간 (η<1)
mean Score 가 -8.3pp 폭락 (0.5286 → 0.4456).

데이터셋별 패턴 (⚠ test-leak 정보, 분석 참고용):
- **tiage** — CSM 단방향 개선. η↓ → Score↑, η=0 (CSM only) 가 +2.0pp.
  F1 +8.5pp (0.41 → 0.50) 가 주된 ↑ 동력 (Pk/WD 는 거의 무변).
  Boundary 감지 자체는 개선되나 위치 정확도 (Pk/WD) 는 그대로 — CSM
  이 "다른 topic 으로의 전환" 을 잘 catch 하지만 시점이 noisy.
- **dialseg711** — CSM 섞는 순간 **-18.6pp 붕괴** (0.5897 → 0.4040).
  Pk 0.32 → 0.49 (random 수준), WD 0.42 → 0.72 로 boundary 위치 자체가
  와해. 즉 v4.1.1 의 mpnet 채널이 dialseg711 에서 매우 잘 작동하는데
  CSM 의 noisy boundary 신호가 그것을 overrule.

원인 분석:
1. **Raw scale mismatch** — δ_csm distribution 이 거의 binary (mean ≈
   0.5, std ≈ 0.5, min 0 max 1 — sigmoid 의 saturated 형태). mpnet δ_adj
   는 continuous (≈ 0.4-0.7 분포). raw blend 에서 CSM 의 0/1 spike 가
   δ_eff 를 dominate → mpnet 신호 묻힘.
2. **Domain shift** — dialseg711 (Wikipedia-style sectioned text) 은
   DailyDialog NSP triplet 와 매우 멀음. CSM 의 incoherence prob 가
   dialseg711 에선 사실상 random.
3. **F1 vs Pk/WD 분리** (tiage) — CSM 이 segment count 는 늘리지만
   *어디서* 자를지는 약한 신호.

**다음 단계 결정 (사용자 중단 요청)**:
- 사용자 결정: superseg precompute 중단, 현재까지 결과 저장
- calibrated z-blend (scale-mismatch fix) 미수행 — 향후 회수 가능 (캐시
  보존됨)
- superseg cache 미생성 — 3 데이터셋 풀 비교 미완

## 한계 / 검증 미해결
- **superseg 미수행** — 사용자 중단 (CPU cost). dialseg711 + tiage 2
  데이터셋만 비교.
- **calibrated z-blend 미수행** — raw scale-mismatch 가 dialseg711 붕괴의
  주원인일 가능성, calibrated 로 회복 여부 미검증.
- **Domain shift**: CSM 학습 corpus (DailyDialog NSP triplet) → TIAGE /
  Dialseg711. dialseg711 domain 이 가장 멀음.
- **단일 ckpt** (cpt_277000.pth), seed variance 미측정.
- **δ\* re-calibration**: TIAGE-train 직접 calib 미수행 (현재는 test mean
  sanity 또는 raw).
- **v4.3.2 (continuous regression) 와 직접 비교**: 같은 η 와 동일 setup
  에서 두 head 가 어디가 다른지 분리 필요. 단 mean-best η 기준 둘 다
  네거티브.
