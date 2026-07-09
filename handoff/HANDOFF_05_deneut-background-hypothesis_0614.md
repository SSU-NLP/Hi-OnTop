# HANDOFF_05 — "안정 배경 → de-neut 적절" 가설의 인과 검증

**지정일**: 2026-06-14 · **상태**: 문제 정의 (가설 미검증) · **부모**: HANDOFF_01(deploy-oracle-gap) / HANDOFF_04(공식 채점기)

## 0. 문제 정의

### 배경
de-neut 신호(`hi_ontop_deneut`)가 도메인별로 정반대 성능을 보인다 (공식 per-dialogue oracle, 규약정렬, MiniLM-int8):
| | de-neut oracle | δ_eff oracle (baseline) |
|---|--:|--:|
| AMI ±2F1 (shift0) | **0.342** | 0.214 |
| dialseg711 Score | 0.504 | **0.720** |
| tiage Score | 0.487 | **0.598** |
| superseg Score | 0.544 | 0.567 |

(※ AMI 정렬 = **shift 0** — AMI gold=시작-turn 이라 신호와 같은 규약. 이전 표의 0.370/0.225 는 shift -1(filler-lag overfit);
±2 tolerance 가 ±1 잔차 흡수해 결론 동일: shift 0/−1 둘 다 de-neut>δ_eff. DTS 는 gold=끝-turn 이라 shift -1 이 규약정렬. [[feedback_ami_gold_start_turn_convention]])

이 패턴에 **"AMI는 안정적 global 배경(한 회의 공통주제)이 있어 de-neut(=배경성분 제거)이 이득, DTS(concat)는 무관 대화
이어붙임이라 뺄 배경이 없어 de-neut이 화제정보까지 깎음"** 이라는 설명을 붙였다.

### ★ 문제: 이 설명은 *가설*이며 인과 검증이 안 됐다
1. **인과 미검증**: AMI vs DTS 차이를 만드는 게 정말 "배경 안정성"인지, 아니면 **공변 요인**(segment 길이·경계 sparsity·drift 점진성·대화 길이·코퍼스 도메인)인지 분리 안 함. 관찰된 상관일 뿐.
2. **원인을 측정조차 못 함**: "배경 안정성"을 online 측정하려는 4개 지표가 전부 실패 →
   - run-length 게이트: deploy 과분절 악순환.
   - burstiness 게이트(δ_prev tail/rough): LOO overfit gap 0.13 (`scripts/deploy_loo_validation.py`).
   - background-stability raw(centroid_shift g_fast/g_slow): 단일AMI 0.055 < DTS 0.098 로 aggregate 구분은 되나 **seam 국소화 실패 + 길이 confound**(`scripts/diag_bg_instability.py`).
   - codex 길이-robust S_bg(normalized excess drift): **부호 역전**(AMI 0.038 < DTS 0.200) + corr(S_bg,len)=**−0.97**(`scripts/diag_sbg_lengthrobust.py`). 원인 = d_obs가 "배경변화"가 아니라 "회의 내부 화제drift"까지 잡음 — **"배경"과 "화제"가 같은 임베딩 공간이라 분리 불가**.
   ⇒ 원인(배경)을 신뢰성 있게 잴 수 없으니, "배경이 de-neut 이득을 결정한다"를 직접 못 보임.

### 목표
**"안정 global 배경의 존재 여부가 de-neut 이득의 인과 원인"임을 확실히 검증(또는 반증)** — 공변 요인 통제 + 인과 조작 실험으로.

### 성공기준
- (a) **인과 조작**: 데이터에 배경을 인위로 *추가/제거* 했을 때 de-neut 이득이 그에 따라 *생기거나 사라지면* → 가설 확정. 안 따라가면 → 반증(다른 요인).
- (b) **공변 통제**: segment 길이·경계 밀도·drift 점진성·대화 길이를 맞춘 상태에서도 배경 유무만으로 de-neut 이득 차이가 재현.
- 둘 다 충족해야 "배경 때문" 주장을 논문에 쓸 수 있다.

## 1. 검증 설계 (제안 — 미실행)

### 1a. 인과 조작 (핵심, 가장 깨끗함)
같은 데이터에서 **배경 성분만** 조작하여 de-neut 이득 변화를 본다 (나머지 요인 자동 통제):
- **배경 주입 (DTS→배경 있음)**: 무관 concat 대화의 모든 발화에 공통 벡터 `b` 주입 `x' = normalize(x + γ·b)`. 안정 배경이 생김.
  → de-neut이 이제 **이득으로 돌아서면**(δ_eff 대비 회복/초과) 가설 지지.
- **배경 제거 (AMI→배경 없음)**: 회의 공통방향 `b`(회의 평균)를 전 발화에서 사영제거 `x' = normalize(x − (x·b)b)`.
  → de-neut 이득이 **사라지면** 가설 지지.
- γ sweep으로 "배경 강도 → de-neut 이득" **단조 관계**가 나오면 인과 강함. (oracle 기준으로 먼저, 채점 = HANDOFF_04 공식 scorer.)

### 1b. 공변 통제 (확증)
- segment 길이/경계밀도/대화길이를 매칭한 subset 비교 (AMI 짧은 구간 vs DTS, 또는 길이 stratify).
- drift 점진성: AMI 내부도 sharp 전환 구간 vs 점진 구간 나눠 de-neut 이득 비교.
- 코퍼스 도메인 confound: AMI(한 도메인) vs DTS(혼합) 외에, **단일 도메인 긴 대화 다수**(예: superseg 한 대화 = 한 배경)에서도 검증.

### 1c. 직접 측정 (oracle 측 배경 척도)
online 불가했으니, **라벨/oracle 허용한 배경 척도**로 먼저 가설 성립을 확인:
- 각 stream의 gold-topic centroid들의 *공통성분 크기* (배경 강도) 측정 → de-neut 이득과의 상관.
- 공통성분 크다(AMI) → de-neut 이득 ↑, 작다(DTS) → 이득 ↓ 인지.

## 2. 현재까지의 증거 (재현, 수록)
- de-neut vs δ_eff oracle 표(위 §0). 스크립트: `scripts/diag_deneut_dts_signal.py`(신호 변형 분해 — dialseg에서 δ_ctx 0.778 > δ_eff 0.749 > de-neut 변형 0.50~0.63), `scripts/ami_alignment_recheck.py`(AMI 규약정렬 shift0: de-neut 0.342>δ_eff 0.214; shift -1=0.370/0.225 는 filler overfit).
- 신호 분해(공식 oracle, dialseg): λ·r_global 제거 +0.11, de-neut(global제거)+평균prototype vs δ_ctx −0.14 → **DTS에선 de-neut 연산이 신호를 깎음**.
- reset 타이밍은 주범 아님(`scripts/diag_deneut_reset_timing.py`: end→start 수정해도 +0.006~0.035).
- 배경 측정 4종 실패(§0-2, 스크립트 위).
- 공식 채점기 = HANDOFF_04 (`src/hi_ontop/dts_scoring.py`).

## 3. 시도 ledger (논문 수록 후보 — 인과 검증)
*(아직 없음 — §1 설계가 첫 시도. 위 §2는 "가설을 만든 관찰"이지 "가설 검증"이 아님.)*

## 4. 한계 / 주의
- 배경 주입/제거 시 임베딩이 인코더 manifold를 벗어날 수 있음(정규화·γ 범위 점검). 합성 신호 artifact 경계.
- oracle 측 검증이 성립해도 deploy(online·calibration-free)는 별개 미해결(HANDOFF_01) — 본 handoff는 *가설의 인과성*만 다룸.

## 5. 적응-c 검증 (2026-06-15) — dead end + c-tension 확정

**동기**: 현 디폴트 c=1.0 이 AMI 과분절(pred 4105 vs gold 938)·DTS 미분절. c-sweep 결과 best-c 가 도메인마다 **정반대**:
**AMI=1.5(sparse), DTS=0~0.75(dense)** (`scripts/diag_ami_c_sweep.py`, `diag_dts_threshold_isolation.py`). 단일 고정 c 불가능 → 적응-c 시도.

**구현·검증 (확정 채점기, `scripts/diag_adaptive_c_{otsu,knee}.py`)**:
| 적응-c | dialseg | tiage | superseg | AMI |
|---|--:|--:|--:|--:|
| Otsu | +0.009 | +0.029 | +0.051 | **−0.046** (AMI 과분절) |
| knee(elbow) | 0.334 | 0.334 | 0.336 | **0.159** (pred 14959=16×gold) |
| 고정 c=1.0 | 0.367 | 0.302 | 0.293 | 0.282 |

**결론 (★ 2026-06-15 정정)**: 적응-c 는 **deploy(detected-reset) 신호 위에서** Otsu·knee 둘 다 AMI 과분절로 실패.
c-tension(AMI 고-c vs DTS 저-c)도 임계 튜닝만으론 해결 불가.

⚠️ **단, 이전에 적은 근본 원인 "de-neut V 분포에 분리구조 없음 / 신호 한계"는 틀렸음 — 재구현 버그(prototype 매-step
정규화) 산물이었다.** **clean(gold-reset) + raw-EWMA prototype** 으로 재측정하면 V 분포는 경계를 **깨끗이 가른다**:
per-meeting oracle ±2F1 **0.687**(V_rel raw-EWMA 0.706), **단순 μ+cσ(c=2.0)만으로도 ±2F1 0.55~0.63**(LLM급) —
REPORT(2026-06-10) §3·§4 + 확정 채점기(`ami_scoring`) 재현 확인 (`scripts/ami_clean_oracle_repro.py [--norm-proto]`
로 재현·버그 대조; raw r_active 0.486≈REPORT 0.488. 옛 `/tmp/repro_687.py`·`/tmp/clean_musigma.py` 는 승격됨).
즉 신호엔 분리구조가 **있다**.
적응-c·deploy 가 실패하는 진짜 원인은 **detected-reset 오염**(prototype 이 두 topic 으로 더럽혀짐) → 동일 신호의 deploy ±2F1 이
0.14~0.16 으로 붕괴([[HANDOFF_01]] reset 부트스트랩, 헤드룸 ~0.5). **병목 = 신호가 아니라 online clean-reset 부트스트랩.**
(주의: `ami_deploy_failure_anatomy.py` 의 "oracle AUC 0.50" 도 같은 매-step 정규화 버그 — clean 신호 약함의 근거로 쓰지 말 것.)
