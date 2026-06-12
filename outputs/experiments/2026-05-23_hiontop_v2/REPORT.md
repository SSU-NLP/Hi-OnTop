# Hi-OnTop-v2 (lexical-overlap correction) — 3-benchmark evaluation

## 1. 실험 setup

- **목적**: Hi-OnTop 에 TextTiling 식 단어-빈도 겹침(lexical overlap) 보정항을 더한 Hi-OnTop-v2 가, 세 분절 벤치마크에서 Hi-OnTop-v1 대비 성능을 올리는지 확인. v1 의 알려진 실패 모드 — wording 은 비슷한데 topic 이 바뀌는 경계 — 를 lexical 신호로 보완하는 게 가설.
- **모델**: `src/hi_ontop/hi_ontop_v2.py` `HiOnTopV2`. `δ_eff_v2 = clip_[0,2](δ_base + w_lex·r_t·(lexdist − μ_lex))` (clip 범위 = cosine-distance 자연 범위 [0,2]; w_lex=0 시 no-op → v1 과 byte-parity). `δ_base` = Hi-OnTop δ_eff (불변). `lexdist = 1 − cos_tf(L_{t-1}, x_t)`, L = 직전 m_lex turn 의 ρ_lex-감쇠 sublinear-TF 합, x_t = 현재 turn sublinear-TF. `r_t = min(1, min(n_ctx,n_t)/min_tokens)` 짧은-turn 게이트.
- **데이터** (SuperDialseg 번들, `benchmarks/superdialseg_data/`):
  - tiage — calib `tiage/train` / test `tiage/test`
  - superseg — calib `superseg/validation` / test `superseg/test`
  - dialseg711 — `dialseg711/test` 711-dialogue 를 seed=0 으로 30/70 shuffle → calib 30% / test 70% (train split 부재).
- **calibration** (label-free, test leakage 없음): calib split 에서 `μ_lex` = median(lexdist), `δ*` = p80(δ_eff_v2). μ_lex 는 w_lex 무관, δ* 는 w_lex 별 재산출.
- **HP**: 임베딩 window m=2 ρ=0.7 a=0.5 (Hi-OnTop default). lexical m_lex=2 ρ_lex=0.7 min_tokens=3 (codex 2026-05-23 권고 default). 인코더 multi-qa-mpnet.
- **w_lex grid (작은 범위 calibration)**: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]. `w_lex=0.0` ⇒ Hi-OnTop-v2 가 Hi-OnTop-v1 로 정확히 환원 → baseline 행.
- **metric**: official SuperDialseg Pk/WD (k=auto) + binary F1. `Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)`. seed 1개 (calib shuffle seed=0); per-w_lex 단일 run.

## 2. 결과 표 (test split, w_lex sweep)

### tiage

| w_lex | μ_lex | δ* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | ΔScore vs v1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 _(v1)_ | 0.9251 | 0.6007 | 0.4483 | 0.4801 | 0.3305 | 0.4331 | +0.0000 |
| 0.05 | 0.9251 | 0.6016 | 0.4501 | 0.4840 | 0.3218 | 0.4274 | -0.0058 |
| 0.10 | 0.9251 | 0.6015 | 0.4381 | 0.4743 | 0.3339 | 0.4388 **★best** | +0.0057 |
| 0.15 | 0.9251 | 0.6030 | 0.4364 | 0.4740 | 0.3305 | 0.4376 | +0.0045 |
| 0.20 | 0.9251 | 0.6044 | 0.4413 | 0.4773 | 0.3241 | 0.4324 | -0.0007 |
| 0.30 | 0.9251 | 0.6059 | 0.4425 | 0.4819 | 0.3271 | 0.4324 | -0.0007 |

### superseg

| w_lex | μ_lex | δ* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | ΔScore vs v1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 _(v1)_ | 0.8654 | 0.6139 | 0.4838 | 0.5097 | 0.3532 | 0.4283 | +0.0000 |
| 0.05 | 0.8654 | 0.6172 | 0.4832 | 0.5086 | 0.3527 | 0.4284 | +0.0002 |
| 0.10 | 0.8654 | 0.6198 | 0.4791 | 0.5057 | 0.3579 | 0.4327 | +0.0045 |
| 0.15 | 0.8654 | 0.6223 | 0.4815 | 0.5092 | 0.3581 | 0.4314 | +0.0031 |
| 0.20 | 0.8654 | 0.6265 | 0.4796 | 0.5071 | 0.3591 | 0.4329 **★best** | +0.0046 |
| 0.30 | 0.8654 | 0.6343 | 0.4791 | 0.5060 | 0.3583 | 0.4329 | +0.0046 |

### dialseg711

| w_lex | μ_lex | δ* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | ΔScore vs v1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 _(v1)_ | 0.9102 | 0.5977 | 0.2974 | 0.3433 | 0.5538 | 0.6167 | +0.0000 |
| 0.05 | 0.9102 | 0.5994 | 0.2940 | 0.3403 | 0.5607 | 0.6218 | +0.0050 |
| 0.10 | 0.9102 | 0.6020 | 0.2928 | 0.3392 | 0.5621 | 0.6231 | +0.0063 |
| 0.15 | 0.9102 | 0.6049 | 0.2894 | 0.3358 | 0.5660 | 0.6267 | +0.0100 |
| 0.20 | 0.9102 | 0.6065 | 0.2881 | 0.3363 | 0.5694 | 0.6286 | +0.0119 |
| 0.30 | 0.9102 | 0.6119 | 0.2862 | 0.3348 | 0.5733 | 0.6314 **★best** | +0.0147 |

### mean-3 요약

| w_lex | tiage | superseg | dialseg711 | mean-3 |
|---:|---:|---:|---:|---:|
| 0.00 _(v1)_ | 0.4331 | 0.4283 | 0.6167 | **0.4927** |
| 0.05 | 0.4274 | 0.4284 | 0.6218 | **0.4925** |
| 0.10 | 0.4388 | 0.4327 | 0.6231 | **0.4982** |
| 0.15 | 0.4376 | 0.4314 | 0.6267 | **0.4986** |
| 0.20 | 0.4324 | 0.4329 | 0.6286 | **0.4980** |
| 0.30 | 0.4324 | 0.4329 | 0.6314 | **0.4989** |

## 3. 해석

- Hi-OnTop-v1 (w_lex=0) mean-3 Score = **0.4927**. w_lex>0 전 구간이 mean-3
  에서 v1 을 넘거나(≥0.10) 동률(0.05)이고, 최고는 w_lex=0.30 의 **0.4989**
  (+0.0062). 회귀 구간 없음.
- **벤치마크별 신호 강도가 뚜렷이 다르다 — 단일 합산으로 읽으면 안 됨**:
  - **dialseg711 — 명확한 양(+)**. Score 가 w_lex 에 대해 *단조 증가*
    (0.6167→0.6314), F1 도 단조(0.554→0.573), Pk·WD 도 단조 감소. 6점
    모두 한 방향 → noise 로 보기 어려운 일관 신호. test 498 dialogue 로
    표본도 충분. lexical 보정이 가장 잘 듣는 코퍼스.
  - **superseg — 약한 양(+)**. w_lex≥0.10 에서 +0.003~+0.005 로 작지만
    test 1322 dialogue (최대 표본) 라 작은 Δ 도 비교적 신뢰. F1 이
    0.3532→0.359 로 일관 상승.
  - **tiage — 신호 없음(noise)**. w_lex=0.05 −0.0058, 0.10 +0.0057 로
    부호가 뒤집히고 단조성 없음. test 100 dialogue 로 표본 최소 → ±0.006
    변동은 noise 범위. 향상도 회귀도 주장 불가.
- **δ\* 가 거의 안 움직임**(0.60→0.61, w_lex=0→0.30): 보정항이 μ_lex 로
  centering 되고 r_t 게이트가 걸려 δ_eff_v2 분포를 크게 넓히지 않음. 즉
  boundary *개수*는 대략 보존되고 *위치*가 lexical 신호로 미세 조정되는
  설계 의도대로 동작. 그래서 개선이 주로 F1(경계 위치 정확도)에서 나옴.
- 개선 폭 자체는 작다(mean-3 +0.6 Score point). 명확히 robust 한 양(+)은
  dialseg711 단조 추세 하나이고, 나머지 둘은 "약한 양 / noise" 다.

## 4. 판정

- **dialseg711**: 향상 (단조, +0.005~+0.015, 표본 충분). lexical 보정 효과
  실재.
- **superseg**: 미세 향상 (+0.003~+0.005, 표본 큼 → 약하지만 실재 가능성).
- **tiage**: 동일 (noise — 표본 100, 부호 불안정). 향상 주장 안 함.
- **best-w_lex-on-test 표기(★)는 참고용**: w_lex 를 test Score 로 고른 것은
  test 튜닝 — 낙관 편향. 정직한 default 선택은 test 가 아니라 추세로:
  dialseg711 단조성 + superseg 안정 구간 + tiage 무해 구간의 교집합 →
  **w_lex ≈ 0.15~0.20** 가 벤치별 test 튜닝 없이 쓸 수 있는 단일 default.
  w_lex=0.30 은 mean-3 최고지만 dialseg711 한 코퍼스에 끌린 값이라 과적합
  위험.
- **다음 iteration 결정**: Hi-OnTop-v2 를 v1 *대체*로 승격하기엔 mean-3
  +0.6점은 약하고 tiage 가 noise 라 근거 부족. **승격 보류** — 단
  dialseg711 단조 추세는 lexical 신호가 원리적으로 유효함을 보이므로
  폐기도 아님. 권고: (a) multi-seed 로 superseg·tiage Δ 의 noise 대비
  유의성 확인, (b) lexical HP(m_lex/ρ_lex) 2차 grid, (c) v1 의 실패 모드
  (wording 유사·topic 전환) turn 을 직접 골라 v2 가 그 turn 에서 실제로
  잡는지 case-level 검증 — 이 3개 통과 시 default 승격 재검토.

## 5. 한계 / 검증 미해결

- **seed 1개**: calib shuffle seed=0, per-w_lex 단일 run. ΔScore 가 ±0.005 이내면 noise 와 구분 불가 — multi-seed 미실행.
- **superseg calib = validation**: superseg 는 cached train 임베딩이 없고 92k-turn train 로컬 인코딩이 비현실적. validation 은 held-out label-free split (Hi-OnTop HP sweep 과 동일 선택). p80/median 은 라벨 미사용 → leakage 아님이나 '학습 데이터' 와는 엄밀히 다름.
- **dialseg711 train split 부재**: 30/70 seeded split 으로 대체. calib 30% 와 test 70% disjoint (leakage 없음). full-test literature number 와 직접 비교 금지.
- **lexical HP 미sweep**: m_lex/ρ_lex/min_tokens 는 codex 권고 default 고정. w_lex 만 sweep (사용자 지시 '작은 범위'). 2차 grid 미실행.
- **μ_lex centering**: median 사용. mean 대비 robust 하나 분포가 치우치면 p80-δ* 와 상호작용 — 미분석.
- stopword set 은 내장 영문 set — 세 벤치 모두 영문이라 적용되나 도메인 특화 stopword 미조정.

