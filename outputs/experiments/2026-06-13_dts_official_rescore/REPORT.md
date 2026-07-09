# DTS 공식 재채점 — δ_eff(DTS primary) vs Hi-OnTop-DeNeut(de-neut+threshold)

## 실험 setup
- 목적: HANDOFF_01 off-by-one 버그 없이, 확정 공식 채점기로 두 분절 모델 DTS 비교.
- 채점기: `hi_ontop.dts_scoring` (= 레포 `SegmentationEvaluation`, per-dialogue binary
  F1 + nltk Pk/WD(window=avg seg/2) + Score, 끝-turn 정렬). 검증: HANDOFF_04 / `scripts/validate_official_scorer.py` (paper TextTiling 3-decimal 재현).
- 데이터: superdialseg_data test (tiage 100 / dialseg711 711 / superseg 1322), MiniLM-int8 cached.
- δ_eff = `HiOnTop`(m=2,ρ=0.7,a=0.5), label-free percentile δ* (p60/p70/p80, dts_result §3.2). **best percentile 은 데이터셋마다 다름**(superseg=p60, tiage=p70, ds711=p80).
- Hi-OnTop-DeNeut = `hi_ontop_deneut.segment()` (de-neut+적응β 신호 + threshold deploy, 단일 calibration-free 임계).
- 정렬: 두 신호 모두 스파이크(새 segment 첫 turn) → 경계 t-1 매핑 (끝-turn 규약).

## 결과 (MiniLM-int8, per-dialogue 공식 Score)

| 데이터셋 | δ_eff p60 | δ_eff p70 | δ_eff p80 | δ_eff best | Hi-OnTop-DeNeut |
|---|--:|--:|--:|--:|--:|
| tiage | 0.462 | **0.476** | 0.465 | **0.476** (p70) | 0.302 |
| dialseg711 | 0.515 | 0.573 | **0.602** | **0.602** (p80) | 0.367 |
| superseg | **0.405** | 0.378 | 0.340 | **0.405** (p60) | 0.293 |

## 해석
- **δ_eff 는 낮지 않다 — published 범위 그대로** (best percentile, official per-dialogue): tiage 0.476(p70) /
  dialseg711 0.602(p80) / superseg 0.405(p60). dts_result(pooled) 0.489/0.607/0.423 대비 official
  per-dialogue 가 0.01~0.02 낮을 뿐(규약 차이). best percentile 은 데이터셋마다 다름(p80 일괄은 superseg·tiage 과소평가).
- **DTS 3개 전부 δ_eff > Hi-OnTop-DeNeut**. Hi-OnTop-DeNeut 이 낮은 건 (1) de-neut 신호가 AMI drift 용이라
  sharp-seam DTS 에 부적합 + (2) per-dataset percentile 튜닝 없이 단일 calibration-free 임계 사용. (회귀 아님 — 도메인 불일치.)
- HANDOFF_01 의 'de-neut 우위/superseg 벽 돌파' 는 off-by-one 버그 산물(무효).

## 판정
- **DTS primary 디폴트 = δ_eff (`HiOnTop`)** (정상 성능). Hi-OnTop-DeNeut 은 DTS 미적합 → AMI/drift 한정.

## 한계 / 검증 미해결
- MiniLM-int8 단일 인코더. baseline(GreedySeg/CSM 등) 공식 재채점은 GPU(torch sm_120)·ckpt 부재로 보류(HANDOFF_04 §3).
- AMI(±2 metric)는 off-by-one 영향 미미 — 별도 재확인(HANDOFF_04 §5d).
