# REPORT — TIAGE test: 구현된 전 hi-ontop-full 계열 segmentation 비교 (12 method × 3 seed)

experiment: `2026-05-17_tiage_all_hiontop_full`
generated: 2026-05-17 (DTS 전환 후 첫 확정 측정 / branch `dts`)
path: `outputs/experiments/2026-05-17_tiage_all_hiontop_full/`

---

## 1. 실험 setup

- **목적**: QA→DTS task 전환(decision-log 2026-05-17) 후, 이제까지 구현한
  모든 hi-ontop-full segmenter 가 **default HP 에서 TIAGE topic-shift 를
  얼마나 잘 감지하는지** 단일 기준으로 확정. codex DTS 진단(과분절·과병합
  경계, boundary-F1 단독 부적합)을 실데이터로 검증.
- **데이터**: `benchmarks/tiage/data/personachat/anno/test/anno_test.json`.
  n_convs=**100**, n_turns=**1564** (≈15.6 turn/conv),
  n_shifts=**315** (GT topic-shift 라벨 `1` 개수).
  - boundary 위치 수 = 1564−100 = **1464** (conv 내부 인접 turn 쌍).
  - GT boundary rate ≈ 315/1464 ≈ **0.215**. GT segment/conv ≈
    315/100 + 1 ≈ **4.15** (이 값이 "적정 n_topics" 기준선).
- **방법 (12)**: `v1, v3.1.1, v3.3.1, v3.3.2, v3.3.3, v3.3.4,
  v3.3.3-2, v3.3.4-2, v3.3.5, v3.3.6, v3.3.7, v3.3.8` —
  `scripts/run_tiage_full_compare.py` 의 `METHODS` 전체.
- **HP (실제 사용)**: 각 method 의 LoCoMo 권장 default
  (decision-log 2026-05-08~). v3.3.5~8 = **α=1, λ=10 + 클래스 default**
  (cos_threshold=0.9, pe_var_sigma0_sq=0.04, **pe_prior=1.0**). v1
  α=1/λ=10/σ₀²=0.01. α=100 regime 은 미사용(degenerate 기지).
- **seed**: RNG 사용 method(v3.3.x) = seed {0,1,2} 3-run.
  비-RNG(v1, v3.1.1) = 1-run (deterministic).
- **metric**: turn-level **boundary F1 / P / R** —
  pred shift = `topic_id[i] != topic_id[i-1]`, GT shift = anno 라벨.
  보조: **n_topics/conv** (과분절↔과병합 직접 지표), ms/turn.
- **encoder**: `QueryEncoder` (env backend, dim **768**). 전 method
  동일 임베딩 1회 인코딩 후 공유 → method 간 비교 공정.
- **baseline**: 비교 자체가 cross-method. 외부 SOTA seg 모델 미포함.

## 2. 결과 (TIAGE test, 12 method × 3 seed)

**ARI ↑ = primary** (GT segment 분할 vs 예측 topic-id 분할 대화별
`adjusted_rand_score` macro 평균; 과분절·과병합·collapse 동시 페널티,
collapse-immune). Pk/WD ↓ = 대화별 macro, k = 평균 ref seg 길이의 절반.
collapse = 1-topic 으로 병합된 대화 비율; **†** = degenerate(collapse
≥50%) → F1/Pk/WD best 에서 제외. 행은 ARI 내림차순.
(2026-05-17 재측정 — ARI/collapse 컬럼 추가; F1/P/R/Pk/WD/n_topics 는
직전 측정과 동일 = deterministic 재현 확인. ms/turn 만 재측정값.)

| method | n_seeds | ARI ↑ (m ± s) | F1 ↑ | P | R | Pk ↓ | WD ↓ | n_topics | collapse | ms/turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v3.3.6** | 3 | **0.359** ± 0.000 | 0.234 | 0.213 | 0.260 | 0.517 | 0.610 | 4.8 | 0% | 39.90 |
| v3.3.5 | 3 | 0.343 ± 0.000 | 0.279 | 0.230 | 0.352 | 0.497 | **0.598** | **5.8** | 0% | 11.72 |
| v3.3.7 | 3 | 0.334 ± 0.000 | 0.209 | 0.182 | 0.244 | 0.521 | 0.619 | 5.2 | 0% | 41.90 |
| v1 | 1 | 0.269 ± 0.000 | 0.363 | 0.252 | 0.654 | **0.467** | 0.721 | 7.5 | 0% | 0.49 |
| v3.1.1 | 1 | 0.111 ± 0.000 | **0.377** | 0.235 | 0.956 | 0.478 | 0.913 | 13.3 | 0% | 0.31 |
| v3.3.8† | 3 | 0.031 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.510 | 0.510 | **1.0** | **99%** | 138.15 |
| v3.3.1 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.59 |
| v3.3.2 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.02 |
| v3.3.3 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.14 |
| v3.3.3-2 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.36 |
| v3.3.4 | 3 | 0.000 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.26 |
| v3.3.4-2 | 3 | 0.000 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.48 |

GT 기준선: n_topics/conv ≈ **4.15**, boundary rate ≈ **0.215**.
std 전부 0.000 (deterministic). **best ARI = v3.3.6 0.359** (primary);
best F1=v3.1.1(0.377, degenerate 제외 후에도 1등이나 ARI 0.111 = 남발),
best Pk=v1(0.467), best WD=v3.3.5(0.598). v3.3.8† collapse 99% =
guard 가 정상 격리 (F1/Pk/WD best 에서 자동 제외) — §3 참조.

## 3. 해석

**세 구간으로 갈린다 — 전부 GT 를 비껴감.**

1. **극단 과분절 (v3.3.1~v3.3.4-2, 6종)**: n_topics/conv=15.6 =
   turn/conv 와 동일 → **매 turn 이 새 topic**. R=1.000(모든 GT
   boundary 적중하나 모든 turn 을 boundary 로 찍어서), P=0.215 =
   GT base rate 그대로. = "분절 안 함"과 정보량 동일. v3.3.8 동기 문서의
   `L0=−0.125`(fresh-baseline 이 cos 0.9 예측 가정) 가 TIAGE 짧은 turn
   에서 fresh 를 항상 이기게 만드는 구조적 이탈이 그대로 재현.
2. **중간 과분절 (v1 7.5, v3.1.1 13.3)**: GT(4.15)보다 2~3× 많이 쪼갬.
   v3.1.1 이 "best F1 0.377" 인 건 **R=0.956 로 boundary 를 남발**해
   recall 로 F1 을 끌어올린 것 (P=0.235). 품질이 좋은 게 아님.
3. **과병합 (v3.3.5→3.3.8)**: SEM2 충실 복원이 topic 수를 단조 감소
   시킴 — 5.8→4.8→5.2→**1.0**. v3.3.5(5.8)가 GT(4.15)에 **가장 근접**
   하나 boundary-F1 은 0.279 로 v3.1.1 보다 낮음. v3.3.8 `pe_prior=1.0`
   = **전 대화 1 topic mega-collapse (F1=0)** — codex 가 지적한 구조적
   문제가 idx374 단일 케이스뿐 아니라 **TIAGE 100 conv 에서도 재현**
   (N=1 과적합 아님이 확정).

**핵심: boundary-F1 단독은 over-segmentation 에 의해 게임됨.** v3.1.1
이 1등인 유일한 이유는 boundary 남발(R 0.956)이고, GT topic 수에 가장
가까운 v3.3.5 는 오히려 F1 낮음. F1 의 P/R 비대칭이 "많이 찍을수록
유리"하게 작동 → codex Q1 권고(ARI 로 과분절·과병합 동시 페널티,
boundary-F1 은 보조)가 데이터로 입증됨. n_topics/conv 를 같이 보지
않으면 v3.1.1 을 "최고"로 오판한다.

**Pk/WD 가 F1 의 over-segmentation 게임을 부분적으로 교정한다.**
- **WD ↓ 가 F1 과 정반대 순위를 준다**: F1 1등 v3.1.1 의 WD 는 0.913
  (거의 최악) — boundary 남발이 WD 의 윈도우 불일치를 최대화. 반대로
  과병합군 v3.3.5~7 은 WD 0.598~0.619 로 **전 method 중 가장 낮음**.
  즉 "near-miss 를 부분 인정"하는 WD 기준에선 v3.3.5 계열이 v3.1.1 보다
  **명백히 우수** → §4 "v3.3.5 가 구조적으로 유망" 판정을 WD 가 독립
  지지(기존엔 n_topics 근접성만 근거였음, §5 약점 보강).
- **Pk 는 변별력이 약하다**: 전 method 0.467~0.521 로 ±0.05 폭. GT
  boundary rate 0.215 에서 Pk 의 base(무경계 가정) ≈ 0.5 근처라, 과분절
  /과병합 모두 0.5 부근으로 수렴 → Pk 단독으로는 method 구분 불가.
  v1 이 best Pk(0.467)인 것도 중간 과분절의 우연 (F1·WD 와 불일치).
- **v3.3.8 의 best WD(0.510) 는 함정**: 전 대화 1 topic =
  pred boundary 0 개 → WD 가 "ref 의 듬성한 boundary 만큼"만 불일치라
  낮게 나옴. F1=0 / n_topics=1.0 와 같이 보면 **degenerate 임이 자명**.
  best-WD 라벨을 단독으로 신뢰하면 안 되는 직접 사례 → "단일 metric
  최댓값 추종 금지" (§4 ARI+collapse guard 도입 근거 강화).
- **결론**: WD 추가로 "v3.1.1 1등은 metric artifact, v3.3.5 가 실질
  유망"이 **두 번째 독립 metric 으로 확정**. 단 Pk 무변별 + WD 의
  degenerate 취약성 → codex Q1 의 ARI(과분절·과병합 동시 페널티 +
  collapse 면역) 필요성은 그대로 유효.

**ARI(primary)가 F1 환상을 완전히 무너뜨리고 순위를 재정의한다.**
- **과분절군 ARI ≈ 0.000~0.001**: v3.3.1~3.3.4-2 는 F1 0.354 로 "중간"
  처럼 보였으나 ARI 가 GT 분할과의 합치도를 직접 재니 **거의 0** —
  매-turn 분절이 GT segment 구조와 무상관임이 정량 확정. F1 의 "boundary
  남발 보상" 이 ARI 에선 작동 안 함 (과분절 = 동일 GT 세그먼트를 무수히
  쪼갬 → Rand index 페널티). v3.1.1 도 ARI **0.111** 로 추락 (F1 1등
  0.377 ↔ ARI 5위) → "F1 1등 = artifact" 가 primary metric 으로 종결.
- **상위권이 학습형 v3.3.5~7 로 역전**: ARI 1~3위 = v3.3.6(0.359) ≳
  v3.3.5(0.343) > v3.3.7(0.334), 전부 collapse 0%. 즉 GT 분할 재현
  관점에선 **과병합 계열이 과분절·중간군을 압도**. F1/WD 와 또 다른
  순위 — 특히 **ARI 는 v3.3.6 ≳ v3.3.5** 로, WD(v3.3.5 우위)·n_topics
  (v3.3.5 가 GT 4.15 근접)와 *불일치*. v3.3.6 의 per-topic+persistence+
  replay 보강이 boundary 위치(WD)·topic 수(n_topics)로는 손해지만 GT
  cluster 구조 재현(ARI)에선 근소 우위 → "v3.3.6 보강이 TIAGE 에서
  순수 역효과"라던 직전(Pk/WD only) 판정은 **부분 정정 필요**: 세
  지표가 v3.3.5↔v3.3.6 을 서로 다르게 줄세움 = default HP 에선
  우열 미확정, HP sweep 없이 baseline 확정 불가(§4·§5).
- **collapse guard 정상 작동**: v3.3.8 ARI **0.031**, collapse **99%**
  → `†` 자동 부착, F1/Pk/WD best 에서 제외. 직전 "best WD=v3.3.8(0.510)
  함정"이 guard 로 코드 레벨에서 차단됨이 실증 (best WD 가 v3.3.5 로
  교정). 단일 metric 최댓값 추종 위험이 구조적으로 해소.
- **결론(갱신)**: ARI 도입으로 (i) F1 1등 artifact 종결, (ii) 상위권이
  학습형 v3.3.5~7 로 확정, (iii) 단 **v3.3.5 vs v3.3.6 우열은 ARI·WD·
  n_topics 가 상충** → default HP 단독 판정 불가. codex 권고대로 v3.3.5
  잠정 baseline + v3.3.5/6/7 HP sweep(ARI/WD/n_topics 동시)으로 확정.

## 4. 판정

- **iteration 내 분류**: default HP 기준 **전 method DTS 실패**.
  - v3.3.1~v3.3.4-2: 사용 불가 (every-turn = 분절 정보 0).
  - v1/v3.1.1: 과분절. F1 1등은 metric artifact.
  - v3.3.5~7: 과병합이나 **v3.3.5 가 topic 수로는 GT 최근접** → 구조적
    으로 가장 유망. v3.3.8: mega-collapse, default `pe_prior=1.0`
    production 부적합 확정.
- **다음 iteration 결정 (DTS task)**:
  1. **ARI + collapse guard 도입** (codex Q1): `run_tiage_full_compare.py`
     metric 에 ARI / Boundary-F1 분리 + max_share/new_rate 추가. 현행
     boundary-F1 단독 비교는 의사결정 근거로 폐기.
  2. **TIAGE-specific HP sweep** (codex Q3·Q4): default HP 에선 method
     차이가 "붕괴 vs 붕괴"라 신호 없음. v3.3.5 중심 + `pe_prior`,
     `lmda`, `alpha`, `pe_var_sigma0_sq` joint sweep 으로 v3.3.5~8 이
     GT(4.15 topic)에 수렴하는 영역 탐색.
  3. **pe_prior calibration**: idx374 N=1 의 ~0.4 가 TIAGE 에서도
     유효한지 ARI 기준 재측정. default 1.0 보존(원칙값), production 후보
     승격 금지.
- **구조 변경 아직 보류**: codex Q3 — calibration/eval 먼저. 구조 변경
  필요 판명 시 SEM2 "fresh prior predictive 의 데이터 기반 calibration"
  복원 후보.

## 5. 한계 / 검증 미해결

- **metric (해소, 2026-05-17)**: Pk / WindowDiff **+ ARI(primary) +
  collapse guard 산출 완료** (§2·§3). 기존 #1 미해결("ARI 미산출")
  종결 — ARI 가 과분절(ARI≈0)·F1 1등 artifact(v3.1.1 ARI 0.111)·
  collapse(v3.3.8 † 자동 제외)를 모두 정량 분리. "best=v3.1.1(F1)"
  은 metric artifact 로 확정 폐기(primary=ARI). **남은 metric gap
  없음**; 단 ARI 자체는 boundary 위치 정밀도(WD)·topic 수(n_topics)
  와 상충 가능 — 그래서 단일 metric 아닌 ARI+WD+n_topics 3-축 동시
  판정을 §4 가 채택. (Pk 는 무변별 0.467~0.521 로 사실상 비활성, 보고만.)
- **v3.3.5 vs v3.3.6 우열 미확정 (신규 #1 미해결)**: ARI 는 v3.3.6
  ≳ v3.3.5, WD·n_topics 는 v3.3.5 우위 → 3-축 상충. default HP
  (LoCoMo값) 단독으로는 baseline 확정 불가. v3.3.5/6/7 HP sweep
  (ARI/WD/n_topics 동시) 전까지 v3.3.5 는 **잠정** baseline.
- **default HP 가 TIAGE 미보정**: 전부 LoCoMo 권장값. TIAGE 에서
  method 변별이 안 됨(붕괴 vs 붕괴) → HP sweep 전엔 "어느 v3.3.x 가
  낫다" 결론 금지. v3.3.5 우위는 topic 수 근접성만 근거 (약함).
- **단일 split (test), 단일 데이터셋 (TIAGE)**: TopiOCQA segmentation
  GT 미포함. codex Q4 의 "짧은 GT seg vs 긴 distractor-heavy" 2-tier
  중 짧은 tier 만 측정. 일반화엔 TopiOCQA + long-conv stress 필요.
- **encoder backend 미기록**: `QueryEncoder` env default (dim 768).
  임베딩 공간 의존(`pe_prior` 작동값이 embedding-dependent, v3.3.8 doc)
  이라 backend 명시 필요 — 다음 run 부터 REPORT 에 backend 문자열 기록.
- **n_topics/conv 는 ARI 대체 아님**: GT 근접해도 boundary 위치가
  틀릴 수 있음(같은 개수, 다른 자리). v3.3.5 "유망"은 ARI 로 재확인 전
  까지 잠정.
- **v1/v3.1.1 n_seeds=1**: deterministic 이라 무방하나 v3.3.x 와 동일
  3-run 표기 아님(표 n_seeds 열 참조).
