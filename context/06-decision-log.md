# 설계 결정 로그

모든 주요 설계 결정의 **근거와 날짜**를 기록한다.
판단이 바뀔 때마다 기존 내용을 덮어쓰지 말고 **append**.

## 형식
```
YYYY-MM-DD: <결정 사항 한 줄>
근거: <관찰/실험/논문>
영향 범위: <어떤 문서/코드가 바뀌나>
대안: <고려했다가 기각한 옵션들과 기각 이유>
```
---

## 기록

> **범위 노트.** 이 로그는 **2026-05-17 화제분절(DTS) 올인 전환** 이후의 Hi-OnTop
> 분절기 설계 결정만 담는다. 그 이전(2026-04 ~ 2026-05-16)의 결정 이력은 화제분절기가
> *대화 메모리 관리 시스템의 한 구성요소*였던 시기의 기록(LTM/STM/retrieval/QA 평가 등
> 메모리 파이프라인 결정 포함)이라 본 분절 레포 scope 밖이며 제거했다. 분절기 알고리즘의
> *기반 설계*(SEM/SEM2 계승, 사건모델 옵션 A, sCRP + Gaussian, regime split)는
> `context/01-hi-ontop-design.md` · `context/02-math-model.md` · `context/methodology/`
> 에 자급적으로 문서화되어 있다.

## 2026-05-17 — 주 task 를 Long-Conversation QA → DTS(Dialogue Topic Segmentation) 로 전환, 별도 브랜치 `dts` 분리 (사용자 결정)

**결정**: 프로젝트의 1차 평가 task 를 downstream QA(LoCoMo/LongMemEval accuracy)에서 **DTS(Dialogue Topic Segmentation) 자체 품질**로 전환한다. 작업은 신규 브랜치 **`dts`** 에서 진행하고 `main` 은 QA-baseline 시점으로 보존. 현재 uncommitted 변경(v3.3.5~8, tiage/longmemeval inspection 스크립트 등 ~20파일)은 `dts` 브랜치로 이월(checkout -b, working tree 그대로).

**이유 (근거 데이터)**:
- LongMemEval oracle 500Q 에서 천장(full-context)조차 RAG 대비 **+0.6pt** (full 0.734 vs rag 0.728, ~2.7× 토큰). oracle 은 distractor 거의 없어 RAG cosine top-k 도 증거 대부분 회수 → "관련 세션 통째 prefill" 전략의 구조적 헤드룸이 작다 (2026-05-16 entry cross-ref). QA 우위 입증 경로가 데이터상 막힘.
- 따라서 QA accuracy 우위는 목표에서 제외하고, SEM/SEM2 계승 정체성의 핵심인 **분절 품질**(과분절·과병합 동시 회피)을 1차 목표로 재설정. 평가 체계는 직전 codex 위임 권고(segmentation primary = TIAGE/TopiOCQA/pseudo 대비 ARI·Boundary-F1, retrieval primary = topic-level evidence_recall@K, collapse guard = max_share/new_rate/raw_topics) 를 그대로 채택.
- 브랜치 분리: QA-지향 코드/문서 상태를 `main` 에 보존해 회귀 비교 가능하게 유지하고, DTS 재설계의 파괴적 변경을 격리.

**영향**:
- 브랜치 `dts` 생성(현 작업 브랜치). docs 반영 범위 = decision-log(본 entry) + handoff 만 (사용자 지정). methodology DTS metric 정의는 2026-05-17 codex 위임 entry 들로 이미 확정 — 중복 작성 안 함.
- 완료: TIAGE test 12-method×3-seed 비교(`outputs/experiments/2026-05-17_tiage_all_hiontop_full/REPORT.md`) — DTS 첫 확정 측정. 결과: default HP 에서 전 method 실패(v3.3.1~3.3.4-2 every-turn, v1/v3.1.1 과분절, v3.3.5~7 과병합, v3.3.8 1-topic mega-collapse). codex 진단(boundary-F1 게임됨, pe_prior=1.0 붕괴)이 TIAGE 100conv 에서 재현 — idx374 N=1 과적합 아님 확정. 다음: ARI+collapse guard metric 도입 → TIAGE HP sweep.
- 후속 DTS 작업의 primary metric/데이터 셋업은 위 codex 권고 따름.

---

## 2026-05-17 — RNN(learned scene dynamics) 제거/교체 보류 + ARI·collapse guard 도입 (codex 위임 2건 결정)

**질문 (사용자)**: ① v3.3.6(per-topic+persistence+history replay)이 v3.3.5보다 TIAGE에서 나쁘니 learned dynamics RNN을 아예 빼버려야 하나? ② 빼는 게 아니라 f 아키텍처를 RNN보다 현대적 모델(Transformer/SSM 등)로 교체하면?

**결정 (codex `codex:rescue` 2회 위임, 한국어 답변)**:
- ① **(C) 결론 보류 — ARI+HP sweep 선행.** v3.3.6 부진은 "RNN 무용 증거" 아님 — RNN을 제거한 게 아니라 per-topic 분리+persistence prior+history replay 보강 조합이 TIAGE 짧은 대화에서 역효과. + default HP(LoCoMo값, TIAGE 미보정)에서 v3.3.8 mega-collapse까지 나오는 "붕괴 vs 붕괴" 상태라 method 변별 신호 자체가 불안정 → 이 상태로 구조 결정은 위험. RNN 제거(A)는 SEM 핵심 component 제거라 3-step 정당화 부담, 데이터 지지 약함.
- ② **(3) 아키텍처 교체는 부차적 — 학습 신호/데이터 문제 먼저.** SEM2 기본 f = `GRUEvent`(코드에 `LinearEvent`/`StationaryEvent`/`SimpleRNN`/LSTM 계열 존재) → f 교체는 learned dynamics 유지하는 한 SEM 정합 범주(제거보다 정당화 부담 훨씬 낮음). 그러나 TIAGE 15턴/topic·저표본·online·CPU-only 제약상 Transformer/Mamba가 RNN보다 낫다는 근거 없음(과적합·과함). v3.3.5 학습부실 원인 = "RNN이라서"가 아니라 공유 EventRNN+단발 3-step 학습+cross-topic 간섭+unseeded+trained 전환 즉시 랜덤예측 사용 → **저표본 online 학습 + likelihood/fresh-baseline calibration 문제**. 아키텍처 교체로 안 풀림. 교체한다면 더 큰 모델이 아니라 더 단순한 것(`persistence`/`learned linear`/`small GRU`/`EMA-mixer`)이 후보 — 그것도 ARI+sweep 으로 "dynamics 실패 vs calibration 실패" 가린 뒤.

**확정 액션 (분기 없음)**:
1. **v3.3.5 를 현 best baseline 으로 고정.** v3.3.6/7/8 보강은 채택 안 함(폐기 아님, 비교군 유지).
2. **ARI + collapse guard 를 `scripts/run_tiage_full_compare.py` 에 도입** (본 entry 와 함께 구현 완료). ARI = GT segment partition vs 예측 topic-id partition, 대화별 `adjusted_rand_score` macro 평균 — 과분절·과병합·collapse 동시 페널티, collapse-immune → **primary metric**(REPORT 정렬·best 기준 = ARI). collapse_rate = 1-topic 병합 대화 비율; ≥50% 면 `†` 표시 + F1/Pk/WD best 선정에서 제외(v3.3.8 "best WD=0.510" 같은 무경계 artifact 차단). Pk 무변별(0.467~0.521)·WD degenerate 취약성 보강 = ARI 도입 직접 동기.
3. **v3.3.5/6/7 HP sweep** (alpha·lmda·pe_prior) — ARI/WD/n_topics 동시. TIAGE 는 항상 **full(100 conv × 3 seed), subset/sanity 금지**(사용자 결정). GPU 복구 불가(드라이버 11040, CUDA 비활성) 확정 → CPU-only. sweep 가능화 = **임베딩 1회 인코딩 후 캐시 재사용**(HP 무관 불변) + HP 조합 코어 병렬. Crts 프록시는 LLM/임베딩 API 라 세그멘터(EventRNN/sCRP) 연산 오프로드 불가 — 임베딩 단계만 대체 가능하나 캐시가 더 우월.
4. sweep 후 분기: HP 보정해도 v3.3.5 가 GT(4.15 topic) 수렴 실패 → dynamics 실패 → `persistence`/`learned linear`/`small GRU`/`EMA-mixer` ablation(Transformer/SSM 아님). 보정으로 수렴 → calibration 문제, 아키텍처 불변. RNN 제거(A)는 단순 dynamics baseline 에도 패배 확인 시에만 SEM 3-step 정당화와 함께 재상정.

**영향**: `run_tiage_full_compare.py` metric 확장(ARI/collapse_rate 컬럼, ARI 정렬, degenerate † guard). 기존 TIAGE REPORT 3종(`2026-05-17_tiage_v33x_compare`/`_a1`/`_all_hiontop_full`)은 Pk/WD 까지만 반영됨 — ARI 는 다음 full run 에서 추가. methodology/infrastructure DTS metric § 갱신 대상. handoff 갱신 대상.

---

## 2026-05-18 — SEM-vs-prev-cos 근본진단 → v3.3.9 (prev-cos 를 SEM PE 로 정합 복원) (codex 위임 2건)

**진단 (codex `codex:rescue` 위임 #1)**: 같은 인코더·같은 gt_shifts+F1 에서 1줄 baseline "직전 발화 cosine < θ → shift" 가 F1 **0.433~0.440** (3 인코더 multi-qa/all-mpnet/MiniLM 모두, ≈ SOTA 0.427). 반면 우리 SEM 전 변형 F1 0.25~0.36, v1 0.363. **인코더 무관·eval 무관 → SEM 파이프라인 자체가 인접 의미거리 신호를 1줄 threshold 보다 못 살리고 죽임.** 근본원인 = fresh/untrained baseline `L0` 오보정 + scalar cosine-PE σ² 포화 (v3.3.5 `L0=−0.125` 과분절 / v3.3.8 `L0=−12.5` collapse; sCRP·RNN 단독 주범 아님). prev-cos 는 SEM2 untrained `predict_next` = identity dynamics `f(x)=x_prev` 의 cosine PE 그 자체 → SEM 밖 heuristic 아님.

**결정**: codex 권고 **(A) prev-cos 신호를 SEM likelihood/PE 에 SEM-정합 복원** 채택. **CLAUDE.md cosine 금지는 retrieval 한정** — segmentation 의 prediction-error 는 본질적으로 인접 의미거리이고 SEM PE 가 prev-cos 의 일반화이므로, segmentation 단계 `cos(s_{t-1},s_t)` 사용은 retrieval-cosine 금지 위반이 **아니라 SEM2 cold-start 복원**임을 명시(예외 아님, 규칙 범위 밖). SEM 계승 3-step: (1) SEM2 에 있음(identity dynamics PE), (2) sCRP/Bayes/local MAP 와 충돌 없음(prior 상쇄만, 제거 아님; η<1 시 learned f 혼합 유지), (3) 본 entry.

**구현 → v3.3.9 (`sem_core_v339.py`, v3.3.8 기반)**:
- repeat(k=prev_k, trained·untrained 공통) `Lδ(δ_eff)`, `δ_eff²=η·δ_prev²+(1−η)·δ_model²`, η 기본 1.0.
- **회귀 (codex 위임 #2)**: v3.3.9-pre 가 ARI 0.36→0.126, n_topics 2.4 과병합. 원인 = 고정 `r0=λ/α` prior mismatch(`P_prev/α` 가 turn 진행에 커져 threshold 밀림) + σδ² 를 전체 δ_prev population variance(경계 점프 혼합)로 online calibration → discrimination 씻김.
- **codex calibration-fix (#2)**: ① σδ² = **고정** softness temperature `c·δ*²` (c 기본 1/16; online calibration 폐기) ② fresh baseline 을 **time-dependent prior-corrected** `B0_t = Lδ(δ*) + log(P_prev/P_new)` → `_scores` 의 prior 항과 상쇄되어 prev-vs-fresh MAP 결정이 **매 turn `δ_eff < δ*` hard cut 과 정확 동치** ③ untrained prev 도 δ_prev gate(η=1 ⇒ δ_eff=δ_prev; 기존 `_l0()` 무조건 병합 bias 제거) ④ `δ*` 를 **TIAGE train split** prev-cos 에서 재추정(0.5557 @ cos_th 0.4443, F1 0.437, n_conv=300) — test leakage 제거(이전 0.536 은 test 유래라 폐기).

**결과 (TIAGE dev full, α=1 λ=10, δ* train, seed0)**: ARI 0.352 / WD 0.628 / **F1 0.440** / Pk 0.412 / n_topics 7.1 / collapse 0%. **F1 0.257→0.440 = prev-cos baseline·SOTA F1 동급 회복** (SEM 이 신호 더 이상 죽이지 않음 = 진단 근본목표 달성, bimodal 붕괴 해소). 단 WD 0.628 여전히 나쁨(약한 과분절 nt 7.1 ≫ GT 4.15) → 구조 붕괴는 해소, calibration 거리 문제로 전환.

**영향**:
- 신규 `src/hi_ontop/sem_core_v339.py` + `context/methodology/v3.3.9.md` + `run_tiage_full_compare.py` 배선(import/factory/METHODS/SWEEP_CLASSES) + handoff 갱신.
- **full TIAGE test 3-seed 확정 (2026-05-18, `outputs/experiments/2026-05-18_tiage_v339_full/`)**: v3.3.9 = ARI 0.408 / F1 0.437 / Pk 0.415 / WD 0.605 / nt 6.9 / collapse 0% — 13-method 중 **primary(ARI)·F1·Pk 동시 best** (이전 best ARI v3.3.6 0.359 → +0.049; F1 = prev-cos test 0.433 동급/근소상회 = SEM 이 1줄 baseline 따라잡음, 진단 근본목표 완수). dev 0.440↔test 0.437 일관. **v3.3.9 = 새 baseline** (v3.3.5 잠정 baseline 대체 확정). 잔여 약점 = WD 0.605 ≫ SOTA 0.420 (약과분절 nt 6.9 ≫ GT 4.15; best WD 여전히 v3.3.5 0.598).
- **codex 위임 #3 (2026-05-18, RNN→BERT 질문)**: 권고 **(C) BERT-f 보류**. 병목은 f 아키텍처 아니라 calibration — v3.3.9 가 η=1(RNN 미사용)로 F1 회복. BERT-f 는 RNN 보다 파라미터 커서 TIAGE 저표본·CPU·online 에서 v3.3.5 학습부실 악화 위험; supervised BERT classifier 는 SEM unsupervised 철학 정면충돌(+그 "0.55~0.65"가 supervised 면 비교 unfair). 순서 = (i) v3.3.9 test 확정[완료] → (ii) strong similarity 천장(TextTiling-SBERT depth/windowed/adaptive) 우리 eval 측정 → (iii) BERT-f 는 η<1 에서 δ_model 이 δ_prev 를 dev 에서 유의 개선할 때만 승격. + v3.3.9 HP sweep(δ*·σδ_c·η·α·λ, n_topics→4.15 로 WD/ARI 동반개선).
- **SOTA caveat (codex #1·#2)**: SOTA(F1 0.427/Pk 0.40/WD 0.42) 출처·split·boundary 정의·threshold 선택 방식 사용자 미확정 → 외부 SOTA 초과 주장 금지. "같은 인코더·eval 에서 SEM<prev-cos→v3.3.9 가 prev-cos 동급 회복" 결론은 caveat 무관 유효. HP(δ*/σδ_c/r0-strategy)는 train/dev-tune 에서만 선택, test 1회 평가 절차 준수.
- Stage A sweep 은 103/162 BrokenPipe 중단(partial 보존); codex 진단상 구조문제라 재개 critical-path 아님 — v3.3.9 sweep 으로 대체.

---

## 2026-05-18 — 판정 지표를 ARI-primary → **WD/F1/Pk target** 으로 변경 (사용자 결정, 2026-05-17 ARI-primary entry 대체)

**결정 (사용자)**: TIAGE 평가의 판정·정렬·best 기준을 **WD↓ / F1↑ / Pk↓** 로 한다. 이유: 외부 비교 — TIAGE/segmentation 문헌·SOTA 는 Pk/WD/F1 로 보고하고 ARI 를 안 쓴다. 내부 robustness 용으로 ARI 를 primary 로 뒀던 2026-05-17 entry(codex 백킹)를 본 entry 가 **대체**한다.

**유지하는 가드 (제거 시 과거 오판 재발 — 반드시 동반)**:
- **ARI 를 가드 컬럼으로 존치**: F1 은 과분절로 게임됨 (v3.3.1~3.3.4-2 가 매-turn 경계로 F1 0.354 받는데 ARI≈0 = GT 분할 무상관). F1 단독 정렬 시 이 degenerate 가 상위. ARI 컬럼이 over-seg 탐지기.
- **collapse-guard † 존치**: collapse≥50% method 는 *모든* best 에서 제외. 없으면 v3.3.8 (1-topic, F1=0) 이 "best WD 0.510" 으로 다시 뽑힘 (무경계 artifact).
- **n_topics 컬럼 존치** (GT≈4.15) — 과분절·과병합 직접 가시화.
- best F1/WD/Pk 는 전부 non-degenerate(collapse<50%) 한정.

**구현**: `run_tiage_full_compare.py` 메인 REPORT + sweep REPORT 둘 다 — 정렬 key `ari→f1` desc, 선두 컬럼 `F1/WD/Pk`, `ARI(guard)/n_topics/collapse` 후미 가드, best F1/WD/Pk(target) + best ARI(guard, non-target) 라벨. docstring/헤더 문구 갱신. f1_s(std) 집계 추가.

**영향**: 향후 모든 TIAGE REPORT 가 WD/F1/Pk 정렬·판정. 기존 산출(`2026-05-18_tiage_v339_full` 등)의 "best ARI" 해석은 가드로 강등 — v3.3.9 가 F1 0.437/Pk 0.415 로 target 기준에서도 best(non-degenerate)임은 불변(WD 0.605 는 약점 그대로). methodology v3.3.9.md / handoff 갱신 대상. SOTA 비교 caveat(출처·split·supervised 미확정, 외부 초과주장 금지) 그대로 유효.

---

## 2026-05-18 — 프로젝트 default segmenter 를 v3.3.9 (현 BEST) 로 (사용자 지시)

**결정 (사용자: "지금까지의 BEST를 디폴트로")**: `orchestrator.HiOnTop` 기본 `version` 을 `"v2"` → **`"v3.3.9"`** 로 변경. v3.3.9 분기 추가(없었음). 근거: full TIAGE test (target WD/F1/Pk + ARI guard) 13-method 중 v3.3.9 가 F1 0.437 / WD 0.605 / Pk 0.415 / ARI 0.408 로 4개 전부 best(non-degenerate). v3.3.9 `__init__` 기본값(eta_prev=1.0·delta_star=0.5557·sigma_delta_c=0.0625·α=1·λ=10)이 곧 그 BEST config 이므로 orchestrator 분기는 공유 HP 만 passthrough, v3.3.9-specific calibration default 는 미override (BEST 위임). run_tiage `_factory_v339` 도 이미 클래스 default(=BEST) 사용.

**caveat**: orchestrator default 변경은 `version` 명시 안 한 호출자(구 QA 파이프라인)에 영향. 실험 harness(`experiment.py`/`run_experiment.py`/`run_tiage_full_compare.py`)는 version 을 명시 전달하므로 무영향 — 본 변경은 *canonical default* 설정 성격. RNN 처리(A3 identity 고정)·v3.3.10(causal-window) 설계 결정은 미반영(별도, decision-log 2026-05-18 RNN/A3 entry 참조). δ*=0.5557 은 TIAGE train 의존(타 벤치 재calibration 필요).

---

## 2026-05-18 — 표준 segmentation 벤치 도입(SuperDialseg/Dialseg711) + v3.3.10 실벤치 검증 (사용자 지시)

**결정/근거**: TIAGE 단일(잡담형, 노이즈 큼) 한계 → 표준 dialogue-segmentation 벤치 **SuperDialseg + Dialseg711** 도입. 소스 = GitHub `coldog2333/SuperDialseg`(clone, `benchmarks/superdialseg/`+`benchmarks/superdialseg_data/`, gitignored). 신규 `scripts/run_superdialseg_eval.py` — 그들 **공식 Pk/WD/F1 + Score**(=0.5·F1+0.25·(1−Pk)+0.25·(1−WD), `metrics/segmentation.py` verbatim, window='auto') 사용해 문헌-비교 가능. GT = dataset `segmentation_label`(last→0 공식 규약). 임베딩 캐시·우리 segmenter→seg-id→boundary 변환.

**SuperDialseg test 결과 (1322 dial/17328 turn/bnd 0.232, `outputs/experiments/2026-05-18_superseg_test_v3310/`)**:
- prev-cos@oracleθ(천장): Score 0.447 / F1 0.458 / Pk 0.467 / WD 0.660
- v3.3.9 @TIAGE-δ* zero-shot: Score 0.460 / F1 **0.449** / Pk 0.468 / WD 0.589 / predr 0.418
- **v3.3.10 @TIAGE-cfg(m2,ρ0.7,a0.5,δ*0.5594) zero-shot: Score 0.463 / F1 0.432 / Pk 0.471 / WD 0.541 / predr 0.316**
- 전 oracle-δ* 행 수렴 → F1 0.458 / Score ~0.447

**판정**:
1. **v3.3.10 > v3.3.9 (Score·WD·과분절억제)** — Score 0.460→0.463, WD 0.589→0.541, predr 0.418→0.316(GT 0.232 근접). codex causal-window FP-감소 예측이 *구조화된 실벤치*에서 실증. **TIAGE-train "+0.006=노이즈" 판정은 TIAGE 잡담 특유의 가림**이었음 — v3.3.10 무가치 아님 (이전 ledger/판정 정정).
2. 단 F1 은 v3.3.9 우위(0.449>0.432): v3.3.10 경계 덜 찍어 recall↓ ↔ WD/Score 이득. "이긴다"는 지표 의존(Score=공식 종합 기준 v3.3.10).
3. **인접-cosine unsupervised 천장 ≈ F1 0.46 / Score 0.45** (prev-cos·v3.3.9·v3.3.10 oracle 전부 수렴). 문헌 0.55~0.65 는 unsupervised 도달 불가 → **supervised regime** 강하게 시사 (codex SOTA-caveat 실증). 인코더·split·supervised 미확정이라 외부 우열 주장 금지.

**영향**: 신규 벤치/스크립트/데이터 + ledger 를 전 seg벤치로 확장(`outputs/reports/tiage_hp_sweep_ledger.md`) + handoff 갱신. 다음: Dialseg711 eval(어댑터 준비됨), superseg-val δ* 재calibration(현재 TIAGE-δ* zero-shot 전이라 향상 여지), 인코더 정합 ablation, η=1 RNN-skip fast-path(eval ~4배 가속 — RNN dead-weight forward 제거). v3.3.10 은 SuperDialseg Score-우위라 default 승격 후보(단 F1 trade·δ* 미보정이라 보류, superseg-val calibration 후 재판정).

---

## 2026-05-18 — Dialseg711 결과 + v3.3.10 → v4.1.1 rename (새 메이저 라인) (사용자 지시)

**Dialseg711 test (711d/19350t/bnd 0.142, 공식 SuperDialseg metric, `outputs/experiments/2026-05-18_dialseg711_test_v3310/`)**:
- prev-cos@oracleθ(천장): Score 0.590 / F1 0.536 / Pk 0.322 / WD 0.389
- v3.3.9 @TIAGE-δ* zero-shot: Score 0.514 / F1 0.492 / Pk 0.394 / WD 0.534
- **v4.1.1(구3.3.10) @TIAGE-cfg zero-shot: Score 0.590 / F1 0.549 / Pk 0.325 / WD 0.415**
- v4.1.1 @oracle-δ*: Score 0.629 / F1 0.560 / Pk 0.285 / WD 0.319

**판정**: v4.1.1 zero-shot 이 **튜닝 0 으로 유사도 천장(Score 0.590) 매칭**, v3.3.9 대비 **+0.076 Score**(F1 0.492→0.549). superseg 에 이어 두 번째 표준벤치에서 causal-window 우위 일관 — TIAGE-train "+0.006=노이즈" 는 잡담형 특유 가림이었음 확정. dialseg711(합성·경계 선명)은 유사도 방법 적합 벤치, oracle Pk 0.285/WD 0.319 = unsupervised 문헌 경쟁권(단 oracle=test-tuned 상한, 외부주장 불가; zero-shot 0.590 이 정직값=천장).

**rename 결정 (사용자)**: v3.3.10 → **v4.1.1**. 이유: v3.3.x = "SEM2 RNN scene-dynamics 복원" 계열인데, η-ablation(RNN 무용~유해) + v3.3.9(prev-cos=identity PE 정합복원)로 사실상 learned-dynamics 를 버리고 identity+calibrated-MAP 로 전환됨. v4.1.1(identity+causal-window)은 v3.3.x 정체성과 질적으로 다른 라인 → **메이저 분리**. RNN 코드는 `use_rnn` 플래그로 보존(제거 아님, η=1 자동 skip).

**구현**: `git mv sem_core_v3310.py→sem_core_v411.py`, `calibrate_v3310_delta_star.py→calibrate_v411_delta_star.py` (이력 보존). 코드 일괄 치환 HiOnTopSegmenterV3310→V411 / v3310→v411 / V3310_→V411_ / v3.3.10→v4.1.1 (run_tiage_full_compare·run_superdialseg_eval·factory·METHODS·SWEEP). 신규 `context/methodology/v4.1.1.md`. **pre-rename 산출물**(`outputs/experiments/2026-05-18_{superseg,dialseg711}_test_v3310/`, ledger 표의 exp명)은 *역사적 명칭(v3310) 유지* — 커밋된 참조·git history 보존. 이후 신규 산출물부터 v411.

**영향**: orchestrator default 는 여전히 v3.3.9 (v4.1.1 승격은 superseg-val δ* 재calibration 후 재판정 — 보류). 다음: superseg val-calibrated 재평가(`--calib-dataset superseg --calib-split validation`, v411 코드). ledger/handoff 갱신. git: dts push (정책 2026-05-18: 명시요청 시 Claude 직접 실행).

---

## 2026-05-19 — superseg val-calibrated δ* 재평가: "miscalibration artifact" 가설 반증

**실행**: `2026-05-18_superseg_test_v411_calib` — superseg validation 에서 best-F1 δ* 산출(δ*_prev=0.520, δ*_eff=0.502; TIAGE 0.556/0.559 보다 낮음) → superseg test 재평가(v3.3.9·v4.1.1, v411 코드).

**결과 (공식 metric, GT bnd 0.232)**:
- v3.3.9: TIAGE-δ* zero-shot Score 0.460 → superseg-val-δ* **0.451** (하락). F1 0.449→0.455(↑) but WD 0.589→0.637·pred 0.418→0.500(과분절↑).
- v4.1.1: TIAGE-cfg zero-shot Score 0.463 → val-cal **0.453** (하락). F1 0.432→0.451(↑) but WD 0.541→0.615·pred 0.316→0.461.

**판정**:
1. **"superseg 저조는 TIAGE-δ* miscalibration artifact" 가설 = 반증.** val-calibration 이 Score 를 오히려 낮춤. 앞선 superseg 수치는 calibration 탓이 아니라 **진짜 task 난이도** — 전 method F1~0.45/Score~0.45 가 인접-cosine 유사도 천장(F1 0.46)에 붙어있음.
2. **원인 = calibration objective 불일치**: val-cal 이 *F1* 을 최대화 → F1-최적 δ* 는 과분절(pred_rate 0.46~0.50 ≫ GT 0.232) → WD/Pk 악화 → Score 순손실. (F1-게임-by-과분절; ARI 가드 존재 이유와 동일 패턴.) **교훈: target=Score/WD 면 δ* 를 F1 로 calibrate 하면 안 됨 — Score/WD-aware calibration 필요.**
3. **v4.1.1 ≥ v3.3.9 일관** (Score 두 regime, zero-shot WD 크게 우위). causal-window 가치 유지.
4. **v4.1.1 운용점 = TIAGE-cfg zero-shot (Score 0.463) 확정.** val-calibrated 승격 불가(Score 더 낮음). orchestrator default 는 v3.3.9 유지(v4.1.1 default 승격은 TIAGE/Dialseg711 재확인 + Score-aware calib 후 재판정 — 보류).
5. SOTA/문헌 0.55~0.65 가 unsupervised 로 superseg 에서 도달 불가(천장 F1 0.46) 재확인 → supervised regime. 외부 초과주장 금지(인코더·split·supervised 미확정).

**영향**: ledger/handoff 갱신. 다음 후보: (a) Score/WD-aware δ* 탐색, (b) 인코더 정합 ablation, (c) 목표 포지셔닝(unsupervised-competitive 인정 vs supervised 도입) codex 위임.

---

## 2026-05-19 — per-topic δ* (v4.1.2-exp): 구현·검증 → "안전하나 무이득", default 승격 안 함

**결정**: 전역 δ* 의 v4.1.1 을 default 로 **유지**. per-topic δ*_k 는 `src/hi_ontop/sem_core_v412_exp.py`(`HiOnTopSegmenterV412Exp`) 로 **experimental 구현 + 실측 검증 완료** — codex 2026-05-19 P2 그대로, default 승격 **안 함**.

**구현 (codex 안전 수식)**: D_k = accepted within-continuation δ_eff only(boundary/fresh/same-label-restart 제외 — v3.3.9-pre mixture 붕괴 차단). n_k<N_min(6) → γ=0 → δ*_k=δ*_0 (전역과 정확 동일, graceful fallback). n_k≥N_min → r_k=Q0.80(D_k)+κ·MAD, γ=γ_max·(m+1)/(m+1+ν), δ*_k=clip((1−γ)δ*_0+γ·r_k, [0.85,1.15]δ*_0). σδ² 전역 고정 유지. prev-topic δ*_{k_prev} 로 prior-corrected fresh baseline → repeat>fresh ⟺ δ_eff<δ*_{k_prev} 동치 유지. sanity: 짧은토픽 v412exp==v411 비트동일, 긴토픽 발동 확인.

**검증 (공식 metric, 임베딩 캐시, RNN-skip)**:
- SuperDialseg test: v4.1.2-exp Score 0.463/F1 0.432/Pk 0.471/WD 0.541/predr 0.316 = **v4.1.1 과 모든 자리 동일** → 가드 통과(짧은 segment 3~4턴<N_min → 무발동). 무해 실증.
- Dialseg711 test: v4.1.2-exp Score **0.589** vs v4.1.1 0.590 (F1/Pk 동일, WD 0.415→0.416) → **이득 없음, 노이즈 내 미세 손실**. "긴 대화서 per-topic 이득" 가설 **데이터 기각**.

**원인 (구조적, 재오픈 조건 명시)**: segment ~7턴(dialseg711)이라도 within-continuation N_min=6 모이면 segment 가 거의 종료 → δ*_k 가 segment 끝 1~2턴에만 늦게 발동 → 순효과≈0. **안전 게이트(N_min, 붕괴방지 필수) vs 발동시점 근본 충돌**: N_min 낮추면 v3.3.9-pre식 noise/붕괴 위험, 안전하면 표준 DTS 대화길이엔 너무 늦어 무의미. → per-topic δ* 는 *훨씬 긴 대화(segment ≫ N_min)* 데이터에서만 재검토 가치. 현 벤치(TIAGE/superseg/dialseg711)에선 종결.

**영향**: v4.1.1 = 현 라인 유지(orchestrator default 는 여전히 v3.3.9; v4.1.1 승격은 별건). v4.1.2-exp 코드·어댑터 배선 보존(재오픈용). methodology 는 v4.1.1.md 에 본 negative result 한 줄. ledger/handoff 갱신. 깨끗한 negative result — 가설 검증·기각·메커니즘 규명 완료.

## 2026-05-19 — baseline: Def-DTS / Def-DTS-based-online 분리

**맥락**: DTS baseline 으로 Def-DTS(ElPlaguister, ACL2025) 를 Crts/gpt-4o 로 도입. Def-DTS 는 offline·whole-dialogue (대화 1개 = LLM 1콜, 전체 발화 joint intent 추론). 사용자가 Hi-OnTop(online·턴단위)과 공정 latency 비교 위해 "턴당 1콜" 개조 요구.

**결정 (codex:rescue 위임 + 후속 논의)**:
1. Def-DTS 원형은 **개조하지 않음**. 대화당 latency + amortized 턴당(=대화latency/발화수, "offline 사후 분배 추정치"로 명시) 으로만 보고. 논문 비교 가능성 보존.
2. 진짜 online 턴당 latency 가 필요하면 **별도 baseline `Def-DTS-based-online`** 신설. 정의: 턴 t 에서 `dialogue[0:t]`(미래 미관측)만 같은 Def-DTS 프롬프트에 넣고 **마지막 발화 topic_shift 만** 채택 → 턴당 1콜. **Def-DTS 결과와 절대 혼합 금지**, 이름·REPORT 분리.

**근거**: (a) "전체 대화 매턴 반복"(변형 a)은 미래 발화를 알아야 하므로 정의상 online 불가(offline 반복). (b) online 은 필연적으로 prefix-only(변형 b) → 문맥이 joint→causal 로 바뀌어 예측·점수가 원 Def-DTS 와 달라짐 → 다른 방법. 같은 프롬프트 텍스트라도 *적용 입력*이 달라 계산되는 양이 다름. 정직한 벤치마킹 = 다른 방법은 다른 이름.

**영향**: `scripts/run_defdts_online.py` 신설(전용 experiment/REPORT). 목적은 정밀 점수 아닌 데이터셋별 턴 100개 표본 평균 latency → full 불필요, 누적 발화≈100 까지 대화 표본. resume+사이드카(crash-safe), workers=1(비경합 isolated 턴 latency). Crts USD quota 복구 전 실행 불가(429) — 코드/하네스는 선구축.

## 2026-05-19 — baseline: Plain LLM prompting (online) 신설

**맥락**: 비교표(Offline CSM/Def-DTS [bi-direction] vs Online CSM/Plain LLM prompting/Ours [past-only]) 의 "Plain LLM prompting | past-only" 행 충당. 원 SuperDialseg Plain Text Prompting(`benchmarks/superdialseg/.../models/llm` ChatGPTSegmenter) 은 offline·whole-dialogue·대화당 1콜.

**결정 (codex:rescue 위임 결과, Def-DTS-online 과 동일 논리)**: 원형 개조 금지. 진짜 online 이 필요하면 **별도 baseline `Plain LLM prompting (online)`** 신설 — turn t 에 `U1..Ut`(미래 미관측)만 동일 plain 프롬프트로 1콜, Ut 에서 새 part 시작 여부 = 경계. offline 결과와 **표·집계·해석 분리**, prefix-causal 임을 REPORT 명시. parsing 규칙·실패처리 명시 필수(codex caution #4).

**산출**: `scripts/run_plainprompt_online.py` (run_defdts_online.py 동형: Crts gpt-4o, workers=1 비경합 latency, resume+사이드카, 데이터셋별 누적 발화≈100 표본). 수집 컬럼 = Pk/WD/F1/**Score(0.5·F1+0.25·(1-Pk)+0.25·(1-WD), 공식 SuperDialseg)**/턴당 latency(ms)/턴당 LLM 호출수/턴당 토큰. 전용 experiment/REPORT.

**영향**: 비교표 행 매핑 — Offline Def-DTS=run_defdts_crts.py(완료), Plain LLM prompting(online)=본 스크립트, Online CSM=별도(추후), Ours=Hi-OnTop. 모든 행 동일 metric/표본 정의로 채워야 공정.

## 2026-05-20 — `methods/` 디렉토리 신설 (offline 원본 / online 수정본 정리)

**결정(사용자)**: baseline 의 원본(offline)·Hi-OnTop 수정본(online)을 레포 홈
`methods/` 에 정리. 범위 = TextTiling, BayesSeg 둘만(추후 확장 가능).
방식 = **A(wrapper, 코드 복사 없음)**: `benchmarks/superdialseg`(read-only)
원본 무복사·무수정, `methods/<m>/offline.py` 는 원본 알고리즘 호출만,
`online.py` 는 `scripts/run_*_prefix.py`(검증본) 실행 진입점.

**근거**: CLAUDE.md `benchmarks/* 읽기전용·코드복사 금지` 준수 + offline/
online 을 동일 harness(Def-DTS 번들 데이터, autoseg Pk/WD+F1, Score=
0.5F1+0.25(1-Pk)+0.25(1-WD))로 비교. online(prefix-causal)은 codex
2026-05-20 결정대로 AUXILIARY(핵심 5행표 제외, Pk/F1 indicative,
latency 가 비교값). offline Pk/F1 은 원 SuperDialseg 논문값과 데이터·
공식 metric 차이로 정확 일치 아님(방향·정상동작 검증용).

**영향**: 신규 `methods/{texttiling,bayesseg}/{offline,online}.py`+README.
산출 `outputs/experiments/<name>/REPORT.md`. architecture 문서 반영.

## 2026-05-20 — TextTiling-online-streaming 신설 (true O(w)/turn 변형)

**맥락**: 기존 `methods/texttiling/online.py` (= `scripts/run_texttiling_prefix.py`)
는 인터페이스만 causal (past-only) 일 뿐 매 turn `nltk.TextTilingTokenizer.tokenize(utts[:t])`
를 fresh 호출 → 전체 O(n²) recompute. *online baseline 핵심 비교값 = per-turn
latency* (decision-log 2026-05-20 위 entry) 라는 정책에 맞춰, 진짜 streaming
TextTiling 을 별도 method 로 추가.

**codex:rescue 위임 결과 (한국어 답변)**:
- 옵션 1 (진짜 streaming) 채택. 단 **이름 분리** — `TextTiling-online-streaming`
  (또는 `CausalTextTiling-RS`) 로 호명. 원본 NLTK/SuperDialseg TextTiling 점수
  *재현 안 함*. running mean/std threshold + one-sided depth 라 boundary set
  자체가 NLTK 와 다름. 정직성 핵심 = 같은 표·같은 이름으로 섞지 않기.
- **유지**: lowercase/punct/stopword, block-cosine, depth-score valley 개념.
- **변형**: depth threshold = Welford running mean+c·std; close-boundary
  suppression = 최근 boundary 와 거리 < min_gap 인 causal rule.
- **폐기**: NLTK 전체 문서 재토큰화, 전체 depth 재정렬, 미래 정보 기반 global
  revision.

**결정**: `src/hi_ontop/baselines/texttiling_streaming.py` 의 `StreamingTextTiling`
class (push()/flush() API) + `methods/texttiling/online_streaming.py` runner
(tiage anno_test.json 직접 로드, segeval 직접, **Def-DTS 의존 없음**) +
`tests/test_texttiling_streaming.py` (11 테스트).

**HP default 결정**: codex 권고 w=10/k=6 은 SuperDialseg 기준. tiage (평균
대화 16발화 × ~5단어) 에선 2k pseudo-sentence 못 채워 boundary 0. runner 의
default 를 w=5/k=3/min_gap=3 으로 축소 (REPORT setup §에 근거 명시). class
default 는 w=10/k=6 유지 (Hi-OnTop 본체 통합 등 일반 사용에 fair).

**산출 (3-benchmark full test set, 2026-05-20)**:
`outputs/experiments/2026-05-20_texttiling_streaming/REPORT.md`

| dataset | n(dial/turn) | Pk | WD | F1 | Score | lat/turn mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.527 | 0.548 | 0.225 | 0.344 | 0.008ms | 0.015ms |
| dialseg711 | 711/18639 | 0.471 | 0.490 | 0.271 | 0.395 | 0.010ms | 0.020ms |
| superseg | 1322/16006 | 0.462 | 0.467 | 0.262 | 0.399 | 0.011ms | 0.026ms |

Pk/WD/F1 = INDICATIVE (codex 의 intended algorithmic difference). **per-turn
latency 모두 ms 단위 이하** — baseline 핵심 주장 검증. nltk prefix-recompute
(`run_texttiling_prefix.py`) 와의 turn-by-turn diff 는 별도 작업.

**영향**: `methods/README.md` 3종 구분 (offline / online prefix-recompute /
online streaming). `pyproject.toml` 에 `segeval` 의존 추가. `benchmarks/Def-DTS`
clone (데이터 로드 전용). SEM 계승 원칙 적용 대상 아님 (TextTiling 은 SEM
외부 baseline).

## 2026-05-21 — GreedySeg-online-delay2 신설 (BERT bounded-lookahead)

**맥락**: SuperDialseg 의 GreedySeg (Jiang et al. 2023, `models/greedyseg/
modeling_greedyseg.py`) 를 online 비교표에 추가. 원본 = offline whole-document
BERT cosine segmentation (left/center/right context, larger=max, argmin greedy
선택, 비가역).

**codex:rescue 위임 결과 (한국어)**:
- 옵션 B (**bounded-lookahead, delay=2**) 채택. 원본 score 공식·HP·argmin
  선택을 *그대로 보존*. 입력 인터페이스 streaming + boundary emit 만 right
  context (WINDOW_SIZE=2 미래 발화) 확보 후로 지연.
- **codex 2026-05-21 algorithm-integrity 검증 통과**: 강한 명명
  `GreedySeg-online-delay2` 정직 — score 공식·greedy 선택·HP·encoder 모두
  보존. **본 plan 의 3 baseline (TextTiling-streaming, GreedySeg-delay2,
  GraphSeg-window-d) 중 유일하게 "5행 핵심표 가능"** (codex 검증). 단 offline
  결과와 별도 열/블록 분리 보고 의무.
- *strict prefix-causal 은 아님* — right_sent 의존성. 이름이 강함을 정당화하는
  근거 = "same scoring/selection, delayed emission".

**device-agnostic 정책 (codex 2026-05-21 권고)**:
- `--device {auto,cuda,mps,cpu}` (default `auto`), 우선순위 cuda → mps → cpu.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` env var — torch/transformers import *이전*
  설정 필수 (runner 모듈 최상단 `os.environ.setdefault`).
- `model.to(device)` + tokenizer output 도 동일 device 이동.
- `model.eval()` + `torch.inference_mode()` 고정.
- 결정성·reproducibility: CPU > CUDA > MPS. seed·버전·device·fallback 여부
  REPORT 명기. 동일 device 반복 측정.

**결정**: `src/hi_ontop/baselines/greedyseg_delay2.py` 의 `GreedySegOnlineDelay2`
class (push()/flush()/state() API, lazy BERT load) + `src/hi_ontop/baselines/
_device.py` (`resolve_device`, `enable_mps_fallback`) + `src/hi_ontop/baselines/
_seg_utils.py` (TextTiling-streaming 과 공유: parse_defdts / bnds_to_masses /
boundary_set_f1 / latency_stats / pk_wd) + `methods/greedyseg/online_delay2.py`
runner + `tests/test_greedyseg_delay2.py` (6 ea, tiny-distilbert fixture).

**미해결 / 가정**:
- 원본 GreedySeg 의 BERT pooling 방식 (CLS vs mean) 확인 못함 — superdialseg
  install 없음. 본 구현 = **segment-concat → [CLS] embedding** default. REPORT
  한계 §에 명시.
- 원본 paper 점수와 직접 비교 불가 (데이터·metric·인터페이스 모두 차이) —
  방향성·정상동작 검증용.

**산출 (3-benchmark full test set, 2026-05-21 device=mps M4 Pro)**:
`outputs/experiments/2026-05-21_greedyseg_online_delay2/REPORT.md`

| dataset | n(dial/turn) | Pk | WD | F1 | Score | lat/turn mean | p95 | bert_fwd/utt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.537 | 0.554 | 0.142 | 0.298 | 13.8ms | 67.3ms | 2.13 |
| dialseg711 | 711/18639 | 0.416 | 0.443 | 0.412 | 0.491 | 16.3ms | 81.8ms | 2.72 |
| superseg | 1322/16006 | 0.507 | 0.511 | 0.278 | 0.384 | 9.1ms | 41.4ms | 1.91 |

Pk/WD/F1 = INDICATIVE (원본 paper 점수와 데이터·metric·인터페이스 모두 차이).
**5행 핵심표 대상** (codex 검증 통과) 이지만 *offline GreedySeg 결과와 별도
열·블록 분리 보고 의무*. lat/turn 9-16ms (BERT forward 2-3회/utt) — encoder
cost 때문에 TextTiling-streaming (0.01ms) 과 같은 latency 표 배치 금지.

**영향**: `methods/README.md` GreedySeg 행 추가 + 실행 가이드 §, `methods/
greedyseg/` 신규 디렉토리, `src/hi_ontop/baselines/{_device,_seg_utils,
greedyseg_delay2}.py` 신규. 기존 TextTiling-streaming runner 도 `_seg_utils`
공통 모듈 사용으로 refactor. SEM 계승 원칙 적용 대상 아님 (외부 baseline).

## 2026-05-21 — GraphSeg-inspired bounded-window 신설 (graph-based AUXILIARY)

**맥락**: GraphSeg (Glavaš et al. 2016, SemEval; 구현 = Dobatymo/graphseg-python)
의 sentence similarity (IC × GloVe + Hungarian) + Bron-Kerbosch maximal clique
+ sequential merge 3-phase 를 online 변형으로 추가. 원본은 전체 대화 graph
위에서 1회 수행하는 *전역 구조화*.

**codex:rescue 위임 결과 (한국어, 2026-05-20)**:
- 옵션 A (**bounded-lookahead window=d**) 채택. window 안에서만 graph 구축 +
  clique + merge 재계산.
- **codex 2026-05-21 algorithm-integrity 검증**: 전역 graph 본질이 깨짐 →
  강한 `GraphSeg-online` 명명 **금지**. 정직 명명 = `GraphSeg-inspired
  bounded-window` (paper 본문), short `GraphSeg-window-d` (file/CLI/표 헤더만).
  **AUXILIARY only** — 원본 GraphSeg paper 결과와 같은 표 등재 금지.

**유지/양보** (codex 권고):
- 보존: sentence similarity 공식 (IC × GloVe + Hungarian), Bron-Kerbosch
  maximal clique, sequential merge 3-phase, content-word POS filter.
- 양보: full-dialogue graph (→ window=d), global clique structure
  (→ window-local), single-pass global merge (→ window 마다 재계산),
  backtracking (= 불가, boundary 비가역 lag-emission).

**결정**: `src/hi_ontop/baselines/graphseg_window.py` (`GraphSegWindowD`,
push/flush/state, lazy GloVe + NLTK brown IC table) + `methods/graphseg/
online_window.py` runner + `tests/test_graphseg_window.py` (9 ea, tiny
embedding fixture). 신규 의존: `networkx`, `scipy` (uv add). 외부 download:
GloVe 6B.300d (~1GB 압축해제, `benchmarks/glove/`, gitignored).

**산출 (3-benchmark full test set, 2026-05-21 device=cpu, M4 Pro)**:
`outputs/experiments/2026-05-21_graphseg_window_d/REPORT.md`

| dataset | n(dial/turn) | Pk | WD | F1 | Score | lat/turn mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.493 | 0.516 | 0.252 | 0.374 | 0.85ms | 1.68ms |
| dialseg711 | 711/18639 | 0.449 | 0.482 | 0.366 | 0.450 | 1.15ms | 1.84ms |
| superseg | 1322/16006 | 0.539 | 0.542 | 0.164 | 0.312 | 0.62ms | 1.88ms |

cold-start = 13.66s (GloVe 400k vocab + brown IC table). Pk/WD/F1 = INDICATIVE
(원본 paper 점수와 데이터·metric·인터페이스·전역 graph 모두 차이). encoder-
light (no BERT) 라 latency 가 BERT-based GreedySeg-online-delay2 보다 1~2자릿수
빠름.

**미해결 / 가정**:
- 원본 paper 의 word frequency 출처 (Wikipedia) 대신 NLTK brown corpus 사용
  (실용적 선택, REPORT 한계 §에 명시).
- POS filter = NLTK averaged_perceptron_tagger.
- boundary lag-emission: 한 boundary 가 여러 window 평가에서 재출현 가능,
  본 구현은 *최초 발견 즉시 비가역 채택*.

**영향**: `methods/README.md` GraphSeg 행 추가 + 실행 가이드 §, `methods/
graphseg/` 신규 디렉토리, `benchmarks/glove/` gitignored. SEM 계승 원칙 적용
대상 아님 (외부 baseline).

## 2026-05-21 — `sem_core_v412_exp.py` (per-topic δ*_k) 코드 삭제

2026-05-19 검증으로 default 미승격(negative result, 무이득) 확정 후 보존만 하던 실험 코드. 사용자 정리 요청으로 `src/hi_ontop/sem_core_v412_exp.py` 및 `scripts/run_superdialseg_eval.py` 의 v4.1.2-exp 참조(import/seg_pred_v412exp/eval row) 일괄 제거. 결과·결정(2026-05-19 entry) 은 git history + 위 문서 기록으로 보존. 재오픈 필요 시 git revert 또는 history 에서 복원.

## 2026-05-20 — v4.2.2 신설: scene encoder swap (bge-base → aws-ai/dse-bert-base)

사용자 요청 "v4.1.1 의 RNN 을 `aws-ai/dse-bert-base` 로 대체한 v4.2.2"
재해석: v4.1.1 default 가 이미 RNN 미사용 (`eta_prev=1` → `_uses_rnn=False`
auto-skip) → 실질 변경점은 **scene vector encoder 교체** (`baai/bge-base-en-v1.5`
→ `aws-ai/dse-bert-base`). codex 분석 (option 1/2/3 비교) 결과 option 1 채택.

**SEM 계승 3-step 통과**: encoder 선택은 SEM2 algorithmic core 에 포함 안 됨
(SEM2 는 정규화 임베딩이면 무엇이든 수용). sCRP / Bayes / local-MAP /
prior-corrected B0 와 충돌 0 — δ_eff 수식 동일, scale 만 corpus·encoder 별
이동. retrieval § 위반 아님 (segmentation upstream).

**산출**:
- `src/hi_ontop/sem_core_v422.py` — thin subclass `HiOnTopSegmenterV422(HiOnTopSegmenterV411)`,
  `encoder_name` kwarg 로 expected (`aws-ai/dse-bert-base`) 미일치 시 warn.
- `context/methodology/v4.2.2.md` — 운영 절차 + δ\* 재calibration 필수 명시.

**필수 후속 (보고 전)**:
1. `.env`: `HIONTOP_EMBEDDING_MODEL=aws-ai/dse-bert-base` (Crts api 서빙 가능 여부
   첫 호출 확인; 실패시 `HIONTOP_EMBEDDING_BACKEND=local` fallback).
2. δ\* 재calibration: SuperDialseg/TIAGE/Dialseg711 train split, encoder=DSE-BERT.
   v4.1.1 default 0.5557 (bge+TIAGE-train) 은 DSE-BERT 환경 무효.
3. smoke 실험: train/dev 에서 v4.1.1(bge) vs v4.2.2(DSE-BERT) Score 비교.
   노이즈 밖 우위 없으면 본 doc "결과" 섹션 negative result + 본 log append.

**근거**: codex 권고 ("encoder 교체 라면 SEM 계승 3-step 통과 OK, 단 δ\* 재보정
필수"). 사용자 진행 의사 명시 (옵션 B: 곧장 v4.2.2 만들기).

**영향**: methodology 신규 doc, 코드 한 파일 추가, 실험 라인 1개 추가. 알고리즘
본체 무변경. v4.2.1 (`f0_min_starts=1`) 과 직교 — 동시 적용 가능.

## 2026-05-20 — v4.1.1.md 결과 표에 TIAGE-test row 추가

사용자 지시 — v4.2.x encoder ablation smoke 의 baseline 으로 측정된
v4.1.1 의 TIAGE-test 결과 (Score 0.468 / F1 0.410 / Pk 0.442 / WD 0.508)
를 `context/methodology/v4.1.1.md` 결과 표에 row 로 추가. ⚠ disclosure
한 줄 (calibration source = TIAGE-train 과 같은 corpus → in-domain 측정)
같이 명시. 다른 두 벤치 (superseg / dialseg711) 의 transfer 측정은 변경
없음.

**근거**: encoder ablation 비교에 TIAGE-test 가 필요했고 측정해보니 sanity
match (w_topic=1.0 결과와 일치). 측정값 자체는 valid 한 v4.1.1 성능 —
공식 doc 에 누락된 row 였을 뿐. 사용자 요청으로 정식 기록.

**영향**: v4.1.1.md 결과 표 1줄 추가 + 변경 이력 1줄. 알고리즘 / 코드 무관.

## 2026-05-20 — v4.2.3-exp 신설 → exp 보존 (default 승격 안 됨)

사용자 + codex 흐름: v4.2.2 smoke 의 mixed result (mpnet vs DSE-BERT
도메인별 우위) → "두 encoder 가 다른 dialogue 정보" 가설 → v4.2.3-exp
설계 (codex option E = calibrated energy combination).

**알고리즘**: v4.1.1 algorithm 무변경, scene vector 만 dual-channel
threshold-normalized energy `r = √(w_t·z_t² + w_f·z_f²)` 로 확장.
boundary ⇔ `r ≥ 1` (internal `δ*=1.0` trick 으로 V411 머신러리 그대로
재사용). topic 채널 = mpnet, flow 채널 = DSE-BERT. 각 채널 TIAGE-train
calibration HP 그대로.

**SEM 계승 3-step**: CONDITIONAL PASS (codex 2026-05-20) — framing 을
"SEM2 residual likelihood 의 calibrated two-feature reduction" 으로 유지.
Zacks 2007 EST 는 개념 인용만, 성능 근거로 사용 금지.

**산출**:
- `src/hi_ontop/sem_core_v423_exp.py` (HiOnTopSegmenterV423Exp(V411))
- `context/methodology/v4.2.3.md`
- `scripts/run_v423_smoke.py` (single trial, w=0.75)
- `scripts/run_v423_weight_sweep.py` (w_topic ∈ {1.0~0.5} 7-point)
- `outputs/experiments/2026-05-20_v423_smoke/REPORT.md`
- `outputs/experiments/2026-05-20_v423_weight_sweep/REPORT.md`

**결과 요약**:
- Single trial (w=0.75): TIAGE +2.29pp, Dialseg711 -2.31pp, SuperDialseg -3.01pp
- Sweep mean: w=1.0 (=v4.1.1) 가 best (0.5068). dual 어떤 weight 도 평균 못 이김.
- Per-dataset best: TIAGE w=0.5 (+2.98pp), Dialseg711 w=0.95 (+0.26pp), SuperDialseg w=1.0 (DSE 항상 해로움)
- 진단: corr(z_t, z_f) 가 dialseg711 에서 0.765 (가장 높음, only_topic 거의 0)
  → 두 encoder noise correlation 높을 때 dual 이 mpnet decision 을 희석

**결정**: v4.2.3-**exp** suffix 보존, default 미승격. v4.1.1 유지.
도메인 특화 사용 옵션 보존 (`HiOnTopSegmenterV423Exp(w_topic=0.5)` 자연 대화용).
adaptive weighting / joint calibration / 다른 flow encoder 등 v4.2.4 후보.

**근거**: dual channel 가설은 *부분 입증* (TIAGE +3pp), *부분 반증*
(정형 topic-shift 손해). single global weight 로는 v4.1.1 못 이김. codex
권고 "exp 로 시작, 성능 주장 smoke 전 금지" 그대로 지킴.

**영향**: 코드 4개 (segmenter / 2 script / methodology doc), 결과
REPORT 3개, decision-log 1줄. v4.1.1 default 무변경.

## 2026-05-21 — v4.2.4-exp 신설: DSE 가 δ_model (RNN slot) 자리 대체

사용자 + Claude 흐름: v4.2.2 (DSE 단독, mpnet 대체) / v4.2.3-exp (calibrated
dual-channel energy) 모두 single global setting 으로 v4.1.1 못 이김.
v4.2.4 는 **v4.1.1 식 구조 보존 + RNN slot 만 DSE 로 교체** 형태로 다른
각도 시도. codex 2026-05-18 P2 ("frozen-pretrained-f, C 천장측정 후 재검토")
의 직접 실현.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
  (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- DSE channel: δ_model = a_dse·δ_prev_dse + (1-a_dse)·δ_ctx_dse,
  HP = v4.2.2 best (m_dse=2, ρ_dse=0.5, a_dse=0.0)
- Blend: δ_eff² = η·δ_adj² + (1-η)·δ_model² (v4.1.1 식 그대로)
- δ* = 0.5594 (mpnet, 그대로). σδ² 도 그대로.
- use_rnn=False 강제 (RNN compute 회피).

**SEM 계승 3-step**: PASS (conditional). SEM2 의 dynamics slot 자체는 존재.
v4.2.4 는 같은 slot 의 구현을 learned-RNN → frozen-pretrained-DSE 로
교체. **새 mechanism 도입 아님**. sCRP / Bayes / local-MAP / B0 무변경.
likelihood 식 무변경. framing 은 "RNN 학습 환경 부족 → frozen 대체"
유지. Zacks EST 는 개념 인용만, 성능 근거 금지.

**산출**:
- `src/hi_ontop/sem_core_v424_exp.py` (HiOnTopSegmenterV424Exp(V411),
  assign_pair(s_topic, s_dse))
- `context/methodology/v4.2.4.md`
- `scripts/run_v424_eta_sweep.py`
- `outputs/experiments/2026-05-21_v424_eta_sweep/REPORT.md`

**결과 요약 (η sweep)**:
| η    | TIAGE  | dialseg711 | superseg | mean   |
|------|--------|------------|----------|--------|
| 1.00 | 0.4675 | 0.5897     | 0.4631   | 0.5068 (=v4.1.1 sanity ✓)|
| 0.75 | 0.4724 | 0.6254     | 0.4275   | **0.5084** ⭐|
| 0.50 | 0.4892 | **0.6480** | 0.3806   | 0.5059 |
| 0.25 | 0.4860 | 0.6455     | 0.3368   | 0.4894 |
| 0.00 | 0.4488 | 0.6352     | 0.3036   | 0.4625 |

**핵심 발견**:
1. **Dialseg711 의 천장 돌파** — v4.1.1.md 명시 oracle-δ* (0.629, test 누설)
   상한을 v4.2.4 @η=0.5 (TIAGE-cfg, test 누설 없음) 가 **0.6480** 으로 돌파.
   "유사도 천장 0.590" 가설 약화.
2. **mean Score 첫 우위** — η=0.75 mean 0.5084 가 v4.1.1 의 0.5068 보다
   +0.0016 (미세하지만 v4.2.x 라인 첫 우위 사례).
3. **SuperDialseg 일관 회귀** — DSE 가 어떤 η 든 superseg 에 해로움.
   under-segmentation 의심 (DSE δ raw scale 작아 mixed δ_eff 작아짐).

**메커니즘 caveat**: η<1 의 효과는 (a) DSE dialogue-flow 정보 추가 +
(b) δ_eff scale 축소로 boundary 완화 두 효과 섞임. δ\* re-calibration
없이는 분리 불가.

**결정**: v4.2.4-**exp** suffix 유지, v4.1.1 default 유지. 다음 step:
1. `calibrate_v424_delta_star.py` — η 별 TIAGE-train δ\* re-calibration
2. SuperDialseg-train δ\* 별도 산출 (회귀 원인 분리)
3. multi-seed variance 측정
4. v4.2.3 / v4.2.4 cross-effect 실험

**근거**: 첫 mean Score 우위 + Dialseg711 의 oracle 천장 돌파는 codex P2
가설 ("frozen-pretrained-f 가 RNN 학습 환경 부족 보완") 강한 정량 증거.
다만 mechanism 분리 (DSE 정보 vs scale 부수효과) 미해결 + SuperDialseg
회귀 + single seed → exp 보존 합리.

**영향**: code 1 + script 1 + methodology doc 1 + REPORT 1 + decision-log 1줄.
v4.1.1 default 무변경.

## 2026-05-21 — v4.3.1-exp 신설: DialoGPT-small surprisal 이 δ_model 자리

사용자 + Claude 흐름: 외부 공격 "Hi-OnTop 의 prediction-by-production 은
진짜 다음 발화 예측이 아니지 않은가?" 에 대한 *방어 강화 카드*. v4.2.4
가 cosine-based DSE 로 δ_model 자리를 채운 패턴 동형으로, v4.3.1 은
**token-level autoregressive surprisal `−log P(u_t | u_{<t})`** 을
DialoGPT-small (Zhang+ 2020) 로 직접 산출하여 같은 slot 에 주입.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
  (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- LM channel: δ_model(t) = mean-token NLL of u_t given u_{t-m_LM..t-1},
  m_LM=5, raw EOS-concat 포맷 (DialoGPT 학습 분포 보존), no normalization
- Blend: δ_eff² = η·δ_adj² + (1-η)·δ_model² (v4.1.1 식 그대로)
- δ* = 0.5594 (mpnet, raw NLL scale mismatch caveat). σδ² 그대로.
- use_rnn=False 강제.
- precompute 분리: `precompute_v431_nll.py` 가 LM forward, segmenter 는
  scalar nll_t 만 소비 (v4.2.4 의 DSE 임베딩 캐시 패턴).

**CSM (Xing & Carenini 2021; Zhou+ 2022) 와 구분**:
1. CSM 은 NSP head 를 *학습* → 그 score 가 boundary rule 자체.
2. v4.3.1 은 **frozen zero-shot** LM surprisal → SEM PE channel 로 흡수.
3. boundary 는 sCRP + local-MAP + B0 가 결정, cosine retrieval 미사용.

**SEM 계승 3-step**: PASS (conditional). v4.2.4 와 동일 논리 (SEM2
LinearEvent dynamics slot 의 frozen 대체). 새 mechanism 도입 아님.
"prediction-by-production 의 생성적 구현" 같은 강한 주장 금지, 단지
*surprisal 이 boundary 정보를 담는다는 경험 가설* 만 검증한다.

**산출**:
- `src/hi_ontop/sem_core_v431_exp.py` (HiOnTopSegmenterV431Exp(V411),
  assign_pair(s_topic, nll_t))
- `context/methodology/v4.3.1.md`
- `scripts/precompute_v431_nll.py` (DialoGPT-small NLL 캐시)
- `scripts/run_v431_smoke.py` (η sweep + REPORT)

**핵심 caveat (sweep 전 사전 명시)**:
- Raw NLL scale (~2~8 nats/token) >> δ_adj scale (~0.4~0.6) → η<1 에서
  δ_eff 가 surprisal 항에 급팽창 → 과분절 위험. v4.2.4 (DSE: 과합병)
  와 정반대 방향.
- 1차 sweep η ∈ {1.0, 0.99, 0.95, 0.9, 0.75, 0.5} (η→1 쪽 조밀).
- δ\* 재calibration 은 후속 (`calibrate_v431_*`) 보류.

**결정**: v4.3.1-**exp** suffix 유지. sweep 결과 본 후 promotion 검토.
v4.1.1 default 무변경.

**근거**: 외부 공격 (prediction-by-production 의 진정성) 에 대한
방어카드 + SEM2 frozen-f 슬롯의 또 다른 옵션 (LM-surprisal vs DSE-cosine)
. 두 방향 (cosine 응집성 vs token surprisal) 의 정량 비교로 SEM 의
dynamics slot 에 무엇이 더 informative 한지 측정.

**영향**: code 1 + scripts 2 + methodology doc 1 + decision-log 1.
v4.1.1 default 무변경.

## 2026-05-21 — v4.3.2-exp 신설: frozen ST5 + learned NextEmbedHead 가 δ_model 자리

사용자 + Claude 흐름: v4.3.1 (DialoGPT NLL) 의 자매판. 같은 외부 공격
("Hi-OnTop 의 prediction-by-production 의 진정성?") 에 대해 *embedding-space
에서 실제로 next-utterance vector 를 regression 으로 예측* 하는 답안.
PE 옵션 로드맵 (6)번 항목 — pretrained Sentence-T5 + 작은 학습 head.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
  (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- ST5+head channel: \\hat{s}_t = head(s_st5[t-m_LM..t-1]),
  δ_model = 1 − cos(\\hat{s}_t, s_st5[t]),
  m_LM=5, left-zero-pad context, NextEmbedHeadMLP(3840 → 1024 GELU → 768 → L2-norm)
- Blend: δ_eff² = η·δ_adj² + (1-η)·δ_model² (v4.1.1 식 그대로)
- δ* = 0.5594 (mpnet 그대로; ST5 cosine-distance scale ≈ 0.3~0.7 로 mpnet 과 자연 양립)
- use_rnn=False 강제

**학습 (Colab)**:
- DailyDialog (HF `daily_dialog`, 11k train + 1k valid + 1k test)
- Frozen ST5 embedding 1회 캐시 → head training (no encoder grad)
- Loss = mean(1 − cos(\\hat{s}, s_t))
- AdamW lr=1e-3, wd=1e-4, linear warmup 5%, batch 256, epoch 10
- ckpt → `outputs/runs/_misc/next_embed_head_<tag>.pt` (사용자가 Colab→로컬 복사)

**CSM (Xing+ 2021; Zhou+ 2022) 와 구분**:
1. CSM = discriminative NSP ranking (positive/negative, hinge loss).
2. v4.3.2 = **continuous next-embedding regression** (cosine loss).
   negative sampling 안 함, target = 실제 다음 발화의 ST5 vector.
3. boundary 는 sCRP + local-MAP 가 결정 (CSM 처럼 score 직접 사용 안 함).

**SEM 계승 3-step**: PASS (conditional). SEM2 ``LinearEvent`` 의
learned dynamics 와 *가장 직접 대응*. 차이: per-event learning →
corpus-wide pretrained. v4.2.4 (frozen DSE) / v4.3.1 (frozen DialoGPT)
와 같은 방향성 (frozen pretrained-f). sCRP / Bayes / local-MAP / B0
무변경. δ scale 도 cosine-distance 라 mpnet 과 자연 양립.

**산출**:
- `src/hi_ontop/sem_core_v432_exp.py` (HiOnTopSegmenterV432Exp(V411),
  assign_pair(s_topic, delta_model_t))
- `src/hi_ontop/next_embed_head.py` (NextEmbedHeadMLP + pack_causal_window)
- `scripts/train_next_embed_head.py` (Colab 권장; HF `datasets` lazy import)
- `scripts/precompute_v432_delta.py` (로컬: ST5 encode + head inference → δ 캐시)
- `scripts/run_v432_smoke.py` (η sweep + REPORT)
- `context/methodology/v4.3.2.md`

**v4.3.1 와의 차이 (PE 옵션 로드맵상 (5) vs (6))**:
| | (5) v4.3.1 (DialoGPT) | (6) v4.3.2 (ST5+head) |
|---|---|---|
| δ_model 정의 | per-token NLL | 1 − cos(pred, target) |
| Range (raw) | 2~8 nats | 0.3~0.7 |
| mpnet scale 일치 | ✗ (큰 mismatch) | ✓ (자연 양립) |
| 학습 0회? | ✓ | ✗ (head 학습) |
| 외부 corpus | 없음 | DailyDialog |

→ scale 일관성은 v4.3.2 가 우수. 학습 의존성 / OOD generalization 위험은
v4.3.2 가 추가. 두 방향의 정량 비교는 두 sweep REPORT 비교로.

**결정**: v4.3.2-**exp** suffix 유지. ckpt 받아 sweep 후 promotion 검토.
v4.1.1 default 무변경.

**Colab/Local 분리 준수 (2026-05-21 보강)**: 사용자 결정으로 학습
데이터 source 를 HF `datasets.load_dataset('daily_dialog')` → **원본
`ijcnlp_dailydialog.zip` 자동 탐색 + 재귀 추출** (colab_csm_train.ipynb
[2] 셀과 동일 패턴) 으로 변경. HF datasets 의존 자체를 제거 →
pyproject.toml 무변경, train script 도 외부 dep 추가 0.

**codex 학습 setup 검토 반영 (2026-05-21)**: 1) scheduler 첫-step lr=0
버그 수정 ((step+1)/warmup), 2) target L2-normalize guard, 3)
eval_diagnostics 추가 (mean_baseline_loss / gain_vs_mean / delta_std /
pred_norm_mean) — codex 1위 위험 "regression-to-mean / output collapse"
의 epoch 별 자동 진단 + 경고, 4) optional batch-InfoNCE (`--nce-weight`,
default 0) — codex 권고대로 monitoring 우선, collapse 확인 시 조건부.
HP (lr/batch/epochs/wd/architecture) 자체는 적정 판정으로 변경 없음.

**Colab 흐름 별도 notebook 분리 (2026-05-21)**: 사용자 결정 — Colab
에서는 Hi-OnTop repo clone 등 어떤 사전 셋업도 없이 **완전 독립** 실행.
새 self-contained notebook `colab_train_next_embed_head.ipynb` (repo
root) 추가: 11 셀 (env + dep / zip 업로드 / zip 자동 추출 / parse /
NextEmbedHeadMLP inline / HP / ST5 encode / loaders+diag / train loop /
ckpt download). colab_csm_train.ipynb 의 IS_COLAB 분기 + zip 재귀 추출
패턴 그대로. Local CLI 흐름 (`scripts/train_next_embed_head.py`) 은
그대로 유지 — 두 진입점 동등.

**근거**: v4.3.1 / v4.3.2 두 사매판으로 'next-utterance prediction' 의
*확률* vs *벡터* 두 정의를 동시 측정. SEM2 learned dynamics 슬롯에
가장 직접적인 구현 ([[v4.2.4]] 의 frozen-DSE 보다 SEM2 LinearEvent 에
더 가까움).

**영향**: code 2 + scripts 3 + methodology doc 1 + decision-log 1.
v4.1.1 default 무변경. pyproject.toml 무변경.

## 2026-05-21 — v4.1.3 신설: graded boundary score + re-entry tracking (output-only)

사용자 요청 — paper 의 "graded boundary score" (Ben-Yakov & Henson 2018
hippocampal graded profile 동형) + "non-linear topic reentry" (schema
reinstatement) contribution bullet 정량 근거 확보.

**알고리즘**: v4.1.1 무변경. **output instrumentation 만 추가**.

**노출 API**:
- `seg.last_graded_score` = `δ_eff / δ*` per turn (boundary strength scalar)
- `seg.last_is_reentry` = boundary INTO previously-visited topic
- `seg.history()` / `reentry_turns()` / `graded_scores()` / `boundary_strength()`

**SEM 계승 3-step**: PASS trivial (algorithm 무변경, side-effect attribute 만).

**산출**:
- `src/hi_ontop/sem_core_v413.py` (HiOnTopSegmenterV413(V411))
- `context/methodology/v4.1.3.md`
- `scripts/run_v413_demo.py`
- `outputs/experiments/2026-05-21_v413_demo/REPORT.md`

**핵심 발견 (v4.1.1 의 봉인된 re-entry)**:

| dataset | f0_min_starts | bnd | reentries | rate |
|---|---:|---:|---:|---:|
| tiage | 2 (v4.1.1 default) | 431 | 0 | 0.0% |
| dialseg711 | 2 (v4.1.1 default) | 5202 | 0 | 0.0% |
| tiage | 1 (v4.2.1) | 696 | 603 | **86.6%** |
| dialseg711 | 1 (v4.2.1) | 8367 | 7854 | **93.9%** |

v4.1.1 의 `f0_min_starts=2` default 가 re-entry mechanism 을 사실상 봉인
(f0 centroid 가 ≥2 episode start 모여야 활성화되는데, 매 boundary 가 새
topic_id 생성하니 영원히 못 채움). [[v4.2.1]] 의 `f0_min_starts=1` 로 풀면
schema reinstatement 정상 작동.

**시사**: v4.1.3 + v4.2.1 조합이 paper 의 "non-linear topic reentry"
정량 근거. v4.2.1 의 marginal F1 효과 (codex 2026-05-19 예측) 와 함께
보면 — re-entry 가 **양적으론 dramatic 변화 (0% → 87-94%) 지만 정량
지표상 marginal**. 즉 segmenter 의 topic 구조가 질적으로 달라지지만
boundary 위치 자체는 비슷.

**Boundary strength bands** (graded_score 기반):
- `< 0.7` very weak (downstream 보류 권고)
- `0.7-1.0` weak (repeat 우세)
- `1.0-1.3` normal
- `≥ 1.3` strong (즉시 commit)

SeCom 등 downstream uncertainty-aware consumer 가 직접 활용 가능.

**영향**: code 1 + script 1 + methodology doc 1 + REPORT 1 + decision-log
1줄. v4.1.1 default 무변경. F1/Pk/WD 무변경 (algorithm 무변경).

## 2026-05-21 — v4.1.3 default 변경: f0_min_starts=1 흡수 (v4.2.1 통합)

원래 v4.1.3 = pure output instrumentation (V411 default f0_min_starts=2
상속). 검증 결과 v4.1.1 default 의 re-entry 가 0% (segmenter 가 매번 새
topic_id 생성 → f0 centroid 가 영원히 2 episode-start 못 채움 → 봉인).
"non-linear topic reentry" paper claim 데이터 0 → v4.1.3 자체 가치 약함.

**사용자 결정 (2026-05-21)**: v4.1.3 default 를 `f0_min_starts=1` 로 바꿔
[[v4.2.1]] 의 SEM2-faithful 변경을 v4.1.3 안으로 흡수. 관찰 도구 +
관찰 대상 동시 활성화.

**결과**: TIAGE 86.6% / Dialseg711 93.9% re-entries — paper 의
"non-linear topic reentry" contribution 정량 근거 default 로 즉시 확보.
explicit `f0_min_starts=2` 전달 시 V411 default 복원 (옛 ablation 참조용).

**[[v4.2.1]] 위치 변화**: 별도 클래스로 보존, default 변경의 명시적
이름 carrier. v4.1.3 가 그 default 를 흡수했으므로 *기능상 중복*. 다만
파일·class·이력 모두 그대로 유지 — 향후 ablation / decision-log 참조용.

**영향**: src/hi_ontop/sem_core_v413.py 의 `__init__` signature 변경
(`f0_min_starts=1` 명시), v4.1.3.md 결과 / 추천 사용 패턴 갱신, demo
재실행 (TIAGE+Dialseg711 default 표 갱신). 다른 코드 / v4.2.x 라인 무관.

## 2026-05-21 — v4.2.4-exp negative result 확정 (calibration artifact)

η sweep (2026-05-21_v424_eta_sweep) 의 핵심 우위 — η=0.75 mean Score
0.5084 (+0.0016 vs v4.1.1) 와 Dialseg711 η=0.5 의 0.6480 ("oracle 천장
돌파") — 가 δ\* 재calibration 후 모두 사라짐 (`2026-05-21_v424_calib`).

**Calibration 결과** (F1-best δ\*, tiage-train + superseg-validation):
- tiage-train η=0.5: δ\*=0.5260 (mpnet 단독의 0.5594 보다 -0.033)
- tiage-train η=0.75: δ\*=0.5209
- superseg-val η=0.5: δ\*=0.3801 (mpnet 보다 매우 낮음)
- superseg-val η=0.75: δ\*=0.4529

**Re-cal 후 mean Score**:
- v4.1.1: 0.5068
- v4.2.4 @η=0.5 re-cal: 0.4970 (-0.0098)
- v4.2.4 @η=0.75 re-cal: **0.4900 (-0.0168)**

**Dialseg711 의 0.6480 → 0.5690 (-0.079)**: prior δ\*=0.5594 가 mixed
signal 에서 우연히 Score-best 근방. F1-best 로 다시 잡으니 boundary
trigger 잦아져 Pk/WD 악화 → Score 떨어짐. v4.1.1.md 가 superseg val-cal
에서 미리 경고한 "F1-best 가 과분절 유발" 패턴 정확 재현.

**결정**: v4.2.4-exp **promotion 후보 탈락**, exp suffix 유지, v4.1.1
default 유지. 코드 보존 (Score-best δ\* / dataset-aware / CSM 학습 후
CSM-as-encoder v4.2.5 등 후속 연구 출발점).

**Important lesson**: F1-best δ\* ≠ Score-best δ\*. Train calibration
정책 (F1-best) 이 metric (Score) 과 misaligned 한 점은 v4.x 라인 전체에
적용되는 caveat. 후속 변형은 Score-best δ\* 직접 sweep 옵션 검토.

**산출**: scripts/calibrate_v424_delta_star.py, outputs/experiments/
2026-05-21_v424_calib/REPORT.md, v4.2.4.md "판정" 섹션 갱신. 또한
부산물로 cache: sds_emb_tiage_train_*.pkl (×2 encoder),
sds_emb_superseg_validation_*.pkl (×2 encoder) — 영구 보존, 미래 train/val
calibration 빠르게 가능.

## 2026-05-21 — v4.1.3 sub-type 분리 + per-band precision 정량 확보

`outputs/experiments/2026-05-21_v413_reentry_analysis` 분석 결과:

**Per-band precision (graded_score band 별 boundary prediction precision)**:

| dataset | very_weak | weak | normal | strong |
|---|---:|---:|---:|---:|
| TIAGE | 0.000 | 0.238 | 0.345 | **0.520** |
| Dialseg711 | 0.000 | 0.129 | 0.383 | **0.800** |

→ graded_score 가 진짜 calibrated boundary signal. 특히 strong band
(≥1.3) 의 precision 0.52~0.80 으로 paper 의 "score > 1.3 → 즉시 commit"
downstream 정책 정량 근거.

**Re-entry sub-type 분리** (case study 보고 두 mechanism 섞임 발견):
- `same_label_restart`: V411 `_is_restart` 경로, topic_id 유지, 같은 topic 의
  새 episode
- `cross_topic_reentry`: 옛 topic 으로 복귀, 진짜 non-linear

| dataset | total re | same_label (n, prec) | cross_topic (n, prec) |
|---|---:|---|---|
| TIAGE | 603 | 473 (78%), 0.309 | 130 (22%), 0.315 |
| Dialseg711 | 7854 | 6586 (84%), 0.319 | 1268 (16%), 0.221 |

**해석**:
- 78-84% 가 same_label_restart (within-topic resumption) — paper 의
  "non-linear" claim 과는 의미 다름.
- Cross_topic_reentry 가 22% / 16% — *진짜* non-linear, paper 가 강조해야
  할 type.
- Dialseg711 의 cross_topic precision (0.221) 이 same_label (0.319) 보다
  낮음 — 옛 topic 으로 복귀 결정에서 f0 의 discrimination power 낮은 듯.

**코드 변경** (`src/hi_ontop/sem_core_v413.py`):
- `last_is_same_label_restart`, `last_is_cross_topic_reentry` attribute 추가
- `same_label_restart_turns()`, `cross_topic_reentry_turns()` 메서드 추가
- `is_reentry` = OR (backward compat 유지)
- history dict 에 두 sub-flag 추가

**Paper framing 정정**:
- 이전: "TIAGE 87%, Dialseg711 94% boundaries are re-entries" (광의)
- 수정: "Cross-topic re-entries = **22% (TIAGE) / 16% (Dialseg711) of
  boundaries**, of which 32% / 22% match human-annotated boundaries."

**미해결**: cross_topic_reentry 의 precision 0.22 가 낮은 이유 — f0 centroid
의 quality 문제일 수 있음. 후속: f0_min_starts=3,4 등 다른 값에서 sweep,
또는 f0 centroid 의 outlier 제거 변형 검토.

**영향**: src/hi_ontop/sem_core_v413.py + scripts/analyze_v413_reentry.py +
v4.1.3.md "결과 / 산출 / 변경이력" 섹션 + outputs/experiments/
2026-05-21_v413_reentry_analysis/REPORT.md. v4.1.1 default 무관, F1/Pk/WD
무변경.

## 2026-05-21 — v4.1.3 default revert: f0_min_starts=2 (Score 보존)

같은 날 revision 1 (`f0_min_starts=1` 흡수, v4.2.1 통합) 후 직접 측정
(`/tmp/test_f0_effect`):

| dataset | f0=2 baseline | f0=1 (revision 1) | Δ |
|---|---:|---:|---:|
| TIAGE | 0.4675 | 0.3987 | **-6.9pp** |
| Dialseg711 | 0.5897 | 0.3940 | **-19.6pp** |
| SuperDialseg | 0.4631 | 0.3604 | **-10.3pp** |

**원인 진단**: `is_boundary` (label-change OR same-label-restart) 와 metric 의
boundary (label-change 만) 가 다른 정의. f0=1 활성 시 segmenter 가 옛
topic_id 재사용 → label-change count 감소 (TIAGE 431→223) → recall 폭락 →
F1/Score 폭락. v4.2.1 docstring 의 codex "marginal effect" 예측이 실제론
large drop. 측정 안 했을 뿐.

**SEM2 충실 vs Score trade-off**: 사용자 질문 "SEM2 가 가장 예측오차 적은
topic 으로 재진입하는 게 맞나?" → 답: 맞음. SEM2 의 `update_f0`/
`f0_is_trained` 가 첫 등장부터 활성. v4.1.1 의 `f0_min_starts=2` 는
Hi-OnTop-only 이탈 (보수). 즉 SEM2 원리는 v4.2.1/v4.1.3-analysis mode 가
충실. 다만 대화 도메인의 f0 단일 sample noise 가 큰 게 Score 폭락의
원인 — SEM2 원리 자체의 문제가 아니라 dialogue domain 의 f0 quality 문제.

**결정 (사용자)**: v4.1.3 default 를 `f0_min_starts=2` 로 되돌림.
- Performance mode (default): Score = v4.1.1 그대로
- Analysis mode (explicit `f0_min_starts=1`): re-entry 정량 분석 가능,
  Score 폭락 감수

**Paper framing 수정**: re-entry 는 "v4.1.3 가 expose 하는 sCRP mechanism
의 behavioral readout" 으로 명시. **default 에선 mechanism dormant**,
analysis mode 에서만 활성. "94% boundaries are re-entries" 표현 X.

**산출**: src/hi_ontop/sem_core_v413.py 의 `__init__(f0_min_starts: int = 2)`
복원, v4.1.3.md 의 "한 줄"/"추천 사용 패턴"/"변경 이력" 갱신.

## 2026-05-21 — v4.1.3 final: re-entry 기능 전체 제거 (graded_score 만)

같은 날 revision 2 (default revert to f0=2) 후 사용자 결정 "토픽 회귀
하지마" → v4.1.3 에서 **re-entry 관련 전체 제거**.

**제거된 항목**:
- `last_is_reentry`, `last_is_same_label_restart`, `last_is_cross_topic_reentry` attribute
- `reentry_turns()`, `same_label_restart_turns()`, `cross_topic_reentry_turns()` method
- `is_reentry`, `is_same_label_restart`, `is_cross_topic_reentry` history fields
- `analyze_v413_reentry.py` (script 자체는 보존 — historical record)

**유지 항목**:
- `last_delta_eff`, `last_graded_score`, `last_is_boundary` attribute
- `history()`, `graded_scores()`, `boundary_strength()` method
- Ben-Yakov & Henson 2018 graded profile 매핑
- per-band precision (TIAGE strong 0.52 / Dialseg711 strong 0.80)

**Paper contribution 정리**:
- ✅ "graded boundary score (Ben-Yakov & Henson 2018 mapping)" — v4.1.3
- ❌ "non-linear topic reentry" — paper 에서 제외

**v4.2.1 위치**: ablation 코드로만 보존 (HiOnTopSegmenterV421 클래스 유지).
실험·default·paper 어느 쪽에도 사용 안 함. 향후 재검토 필요 시 참조.

**Score 영향**: 0 (algorithm 무변경, output 만 추가). v4.1.1 그대로
0.4675/0.5897/0.4631 (test 평균 0.5068).

**산출**: src/hi_ontop/sem_core_v413.py 전면 재작성, scripts/run_v413_demo.py
재작성 (per-band precision + boundary strength 만), context/methodology/
v4.1.3.md 전면 갱신 (re-entry 섹션 제거, "제거된 기능" 섹션 추가).

## 2026-05-21 — v4.3.2-exp: CLEAR NEGATIVE (single η), no promotion

[[v4.3.2]] (frozen Sentence-T5 + DailyDialog-trained NextEmbedHead) 의
Colab 학습 + 3-dataset calibrated z-blend sweep + boundary signal 진단
종합 결과: **정직 비교 (single η, no test leak) 에서 clear negative**.
exp suffix 영구 유지.

**정직 비교 (single η = mean-best, 모든 dataset 동일 적용)**:

| | best single η | mean Score | vs v4.1.1 |
|---|---:|---:|---:|
| v4.1.1 | — | 0.5068 | — |
| v4.2.4 (raw, frozen DSE) | 0.75 | 0.5084 | +0.16pp (noise 의심) |
| **v4.3.2 (calibrated, TX head)** | **0.75** | **0.5023** | **−0.45pp** |

→ v4.3.2 는 어떤 single η 로도 v4.1.1 못 이김.

**Per-dataset best (⚠ test leak, ablation 분석용만 — promotion 근거 X)**:
- TIAGE η=1: 0.4675 (head 무용)
- Dialseg711 η=0.75: 0.5997 (+1pp, v4.2.4 의 그림자)
- SuperDialseg η=1: 0.4631 (head 무용)

⚠ 사용자 정직성 지적 (2026-05-21 후속): "per-dataset best η 는 test
leak". 그 전 entry 의 "mixed result (+1pp)" 표현은 ablation 정보로 명시,
판정은 single-η 기준 clear negative 로 정정.

**학습**: 6 ckpt 비교 (MLP / Transformer × NCE 0/0.03/0.05). TX head +
NCE=0 best (valid loss 0.1323, ACCEPTABLE). Transformer 가 MLP 대비
명확 우수 (pad-mask + pos-emb + self-attn). NCE auxiliary 는 모든 조합
에서 trade-off 나쁨.

**Segmentation sweep (3 dataset)**:
1. Raw blend (TIAGE만): η<1 에서 F1=0 폭락 (δ_model scale 0.171 <<
   δ_adj 0.5 → 과합병).
2. Calibrated z-blend (사용자 처방, v4.2.3 패턴 `z = δ/δ*`, 3 dataset):
   single η=0.75 (mean best) → mean 0.5023 < v4.1.1 0.5068 (**-0.45pp**).
3. Per-dataset best (⚠ ablation only, test leak): Dialseg711 만 +1pp
   약신호, TIAGE/SuperDialseg head 무용.

**Boundary signal 진단 (사용자 명령)**: TIAGE test 의 boundary-after
turn (n=315) vs non-boundary (n=1149) 의 δ_model 분포 비교:

```
boundary:    0.1799 ± 0.0293
non:         0.1692 ± 0.0312
diff:        +0.0107  (방향성 맞음)
t-test:      t=5.46, p=5.5e-08    (통계 유의)
Cohen's d:   +0.348               (small-to-medium, 분류로는 약함)
diff/σ_nb:   0.343 σ              (두 분포 ~86% 겹침)
```

→ head 가 boundary 정보를 *조금* 학습했지만 mpnet δ_adj 보다 *훨씬*
약한 신호. blend 가 mpnet 신호 dilute → 회귀.

**진단**:
- 학습 자체 ✓ (ACCEPTABLE), Architecture (TX) ✓, Calibrated z-blend ✓
- 그러나 head 의 boundary signal 본질적으로 약함 → segmentation 으로
  안 옮겨짐 = **task misaligned** (codex/사용자 1차 우려 적중).
- "Regression-to-mean" 정량 증거: δ_std plateau 0.025-0.036, head 의
  output 분포가 학습 corpus 평균 방향으로 수렴.

**산출 (negative)**:
- code 2 변형: `sem_core_v432_exp.py` 에 `calibrated` 옵션 (z-blend +
  internal δ\*=1), `next_embed_head.py` 에 `NextEmbedHeadTransformer` +
  `make_head` factory
- script: `run_v432_smoke.py --calibrated --delta-star-model auto`
- methodology v4.3.2.md "결과 / 판정" 섹션 채움 (negative result 정직
  기록)

**학술적 가치 (정직 보고)**:
1. DailyDialog → TIAGE domain generalization 실패의 정량 데이터
2. Regression-to-mean 의 직접 증거 (δ_std plateau, boundary signal
   weak)
3. Calibrated z-blend (v4.2.3 패턴) 의 safety 재확인 (raw 폭락 회피)
4. Continuous regression (v4.3.2) vs discriminative ranking (CSM) 비교
   에서 *부정적* 데이터 — CSM 류가 보조 channel 로 더 적합할 가능성
   시사 (단 직접 비교 별도)

**결정**: v4.3.2-exp 영구 보존, v4.1.1 default 무변경. 추후 GRU
head / TIAGE-train 직접 학습 / discriminative objective 등 후속
가능하나 본 결과로는 우선순위 낮음. [[v4.3.1]] (DialoGPT NLL) sweep
은 별도 (사용자 GPU 시간 결정).

## 2026-05-21 — v4.2.5-exp 신설: CSM (finetuned BERT-base) 이 δ_model 자리

사용자 + Claude 흐름: 사용자가 Colab 에서 CSM (Coherence Model, Xing &
Carenini 2021; lxing532/Dialogue-Topic-Segmenter) 학습 완료 (`cpt_277000.pth`,
state_dict). PE 옵션 로드맵의 (4) 번 항목 — finetuned BERT-base
discriminative coherence head.

**동기**: [[v4.3.2]] (continuous regression) 의 clear negative 후,
**NSP-supervised discriminative ranking 학습** 이 regression-to-mean
한계를 회피하는지 직접 비교. 같은 DailyDialog corpus, 다른 objective.

**알고리즘**:
- mpnet channel (δ_adj = a·δ_prev + (1-a)·δ_ctx): v4.1.1 default 그대로
- CSM channel: δ_model(t) = 1 − softmax(decoder([CLS] u_{t-1} [SEP] u_t [SEP]))[0]
  = incoherence probability
- coherence_decoder = `Sequential(Linear(768,768), ReLU, Dropout(0.1), Linear(768,2))`
- BERT backbone = bert-base-uncased (train_csm_hf.py default)
- Blend: raw (v4.1.1 식) 또는 calibrated z-blend (v4.2.3 패턴, `--calibrated`)
- use_rnn=False 강제

**v4.3.2 와 직접 비교 (예정)**:
| | v4.3.2 (TX head) | v4.2.5 (CSM) |
|---|---|---|
| 학습 목적 | continuous regression (cosine) | discriminative ranking (marginal hinge) |
| Output | similarity | probability |
| Regression-to-mean | 큼 (clear negative 확정) | 원리적으로 낮음 (TBD) |

**SEM 계승 3-step**: PASS (conditional). v4.3.2 와 같은 frozen-encoder
+ learned-head 패턴. dynamics slot 채우는 frozen-f. sCRP/Bayes/B0 무변경.

**산출**:
- `src/hi_ontop/sem_core_v425_exp.py` (HiOnTopSegmenterV425Exp(V411),
  assign_pair(s_topic, delta_model_t), calibrated 옵션)
- `scripts/precompute_v425_delta.py` (CSM forward, 3 dataset δ cache)
- `scripts/run_v425_smoke.py` (η sweep, raw/calibrated, REPORT)
- `external/Dialogue-Topic-Segmenter/` (clone, gitignored 추가)
- `context/methodology/v4.2.5.md`
- `.gitignore` external/ 추가

**진행 중**: precompute background (CPU 30분 추정). sweep 끝나면
v4.2.5.md "결과 / 판정" 채우고 decision-log 정리. 핵심 비교는
v4.2.5 (CSM, discriminative) vs v4.3.2 (TX, regression) 의 same-blend
mean Score.

**결정**: v4.2.5-**exp** suffix 유지, sweep 결과 본 후 promotion 검토.
v4.1.1 default 무변경. external/ tree 는 gitignored (vendored 안 함).

---

### 2026-05-21 (저녁) — v4.2.5 raw sweep 결과 + 사용자 중단

**진행**: precompute (CPU 9h+) 가 tiage / dialseg711 까지 완료, superseg
처리 중에 사용자 중단 요청 ("v4.2.5 실행 멈추고 싹다 저장하자").

**결과 (raw blend, single-η mean-best, 2 dataset)**:

| | best η | mean Score | vs v4.1.1 |
|---|---:|---:|---:|
| v4.1.1 (η=1) | — | 0.5286 | — |
| v4.2.5 raw (CSM) | 1.00 | 0.5286 | **−0.0000** |

→ **NEGATIVE** (mean-best 기준). η<1 → mean −8.3pp 폭락.

**데이터셋별 (test-leak, 참고)**:
- tiage: η=0 (CSM only) **+2.0pp** (F1 +8.5pp, Pk/WD 무변)
- dialseg711: η=1 (CSM off) best, CSM 섞으면 **−18.6pp 붕괴**

**v4.3.2 와 직접 비교**:
| | v4.3.2 | v4.2.5 |
|---|---:|---:|
| single-η mean-best | −0.45pp | −0.00pp |
| tiage best | 무유의 | **+2.0pp** |

→ "discriminative ranking 이 regression-to-mean 회피" 가설은 **부분
지지** (in-domain 데이터셋 한정). 도메인 shift 에는 v4.3.2 보다 *더*
취약. 두 head 모두 mean-best 기준 net-neutral/negative.

**원인 (잠정)**:
1. raw scale mismatch (δ_csm ≈ binary {0,1}, mpnet ≈ continuous) — CSM
   spike 가 δ_eff² dominate. calibrated z-blend 미검증.
2. dialseg711 (Wikipedia-style) 이 DailyDialog NSP 학습 corpus 와 매우
   멀음.

**미수행 (사용자 중단)**:
- superseg precompute (CPU 비용)
- calibrated z-blend sweep (raw scale 보정 효과 미확인)
- v4.3.2 와 같은 sample/η 에서 head-only 차이 분리

**보존 자산** (회수 가능):
- `outputs/runs/_misc/sds_v425delta_tiage_test_csm_bert_base.pkl`
- `outputs/runs/_misc/sds_v425delta_dialseg711_test_csm_bert_base.pkl`
- `outputs/runs/_misc/cpt_277000.pth` (CSM ckpt)
- `src/hi_ontop/sem_core_v425_exp.py`, `scripts/precompute_v425_delta.py`,
  `scripts/run_v425_smoke.py`, `external/Dialogue-Topic-Segmenter/`

**결정**: v4.2.5-exp **종결 (net-neutral)**. promotion 안 함. v4.1.1
default 무변경. 차후 calibrated sweep + superseg 완성 + v4.3.2 와
head-only ablation 시 재오픈 가능.

**산출**: `outputs/experiments/2026-05-21_v425_csm_2ds_raw/REPORT.md`
(해석/판정 채워짐), `context/methodology/v4.2.5.md` (결과/판정 채워짐).

## 2026-05-22 — v4.1.3 dead-code audit → Hi-OnTop (reduced form) main 모델 채택

**맥락**: v4.1.3 (`HiOnTopSegmenterV413(HiOnTopSegmenterV411)`) 가 paper 의 main segmenter 인데,
SeCom-swap latency profiling 중 `assign()` 시간의 77% 가 *안 쓰는* per-topic EventRNN
모듈 생성에 쓰임을 발견 (η=1 default → RNN 비활성). lazy-init 으로 수정 후 모든 HP 를
출력 변화로 전수 audit.

**audit 결과 (실증)**: v4.1.3 의 segmentation 출력에 영향을 주는 HP 는
`delta_star` / `ctx_window` / `ctx_decay` / `ctx_blend_a` 4개뿐 (+ `beta` 극단값만 marginal).
나머지는 default/canonical setting 에서 dead — 출력 byte-identical:
- EventRNN 전체 (`eta_prev=1` → δ_model weight 0)
- f0 / restart / re-entry (`f0_min_starts≥2` circular deadlock 으로 영구 봉인)
- SEM2 variance machinery (per-topic σ²_k, scaled-inv-χ² posterior — dead f0 로만 흘러감)
- sticky-CRP `alpha` 및 canonical/default `lmda=10` — `_fresh_baseline_for_prev`
  (codex 2026-05-18) 의 prior-cancel 설계로 repeat-vs-fresh argmax 에서 상쇄.
  단 후속 검증에서 `lmda=1` 같은 낮은 stickiness 는 non-prev f0 fallback 경로와
  상호작용해 일부 데이터에서 출력 차이를 낼 수 있음이 확인됨. 따라서 `lmda` 는
  전역 dead HP 가 아니라 default/canonical reduced-form parity 조건으로만 제외한다.
- `sigma_delta_c`, `var_likelihood_weight`, `pe_prior`, `cos_threshold`, `pe_threshold`,
  `hard_pe_fallback`, `min_transitions_for_pe` — 모두 dead

**환원 결과**: v4.1.3 의 실제 작동 알고리즘 =
`δ_eff = a·(1−cos(s_{t-1},s_t)) + (1−a)·(1−cos(causal_window, s_t)); boundary ⟺ δ_eff ≥ δ*`.
4단계 SEM2 파이프라인 (sCRP prior / RNN PE / σ²_k likelihood / Bayes posterior) 은
코드로 실행되나 출력 무영향.

**결정** (codex:rescue 위임 → A안 채택): main 모델을 `src/hi_ontop/hi_ontop.py` 의
`HiOnTop` (reduced form, dead code 0, ~270줄) 로 재정의. v4.1.3 와 byte-identical
검증 완료 (TIAGE/Dialseg711/SuperSeg 38,242 turn 0 mismatch). SEM2 full form
(`sem_core_v413.py`) 은 `archive/legacy_sem_ablation/` 으로 이동 — 삭제 아님,
paper 의 audit disclosure 재현용 ablation 증거물. `src/hi_ontop/sem_core_v411.py` 등
공유 인프라는 v4.2.x/v4.3.x 가 의존하므로 src 유지.

**근거 / SEM 계승 정합성**: codex 권고 — paper 는 Hi-OnTop 를 "full SEM2 구현" 이 아니라
"SEM 의 prediction-error boundary 직관의 minimal online realization" 으로 서술하고,
"SEM2-style RNN/sticky-CRP/variance 를 구현·audit 했으나 v4.1.3 default 에서 argmax
결정에 영향 없음" 을 명시 (audit disclosure). main claim 은 graded boundary score +
online O(1) latency + SeCom LLM-backend drop-in + latency 대폭 감소로 재구성.

**biology revival 검토 후 보류**: 사용자가 신경과학(해마-mPFC-PMN) 메커니즘을 실제
작동하게 되살리는 안 (mPFC reset / 해마 snapshot / re-entry, segment_id↔event_id 분리)
을 codex 와 설계까지 했으나 — "새 메커니즘 도입 말고 dead code 제거만" 으로 최종 결정.
revival 설계는 codex 스레드에 보존, 후속 버전 후보.

**영향**: `secom_adapter.py` → HiOnTop 로 re-point (byte-identical 이라 SeCom-swap 결과 불변).
`scripts/{run_v413_demo,run_v413_hp_sweep,analyze_v413_reentry}.py` 는 archived v413
import. `scripts/secom_swap/{03b,13}` 는 HiOnTop 로 re-point. methodology 갱신 필요.

**2026-05-22 후속 — `ctx_window` default 3 → 2 정정**: parity 검증
(`scripts/verify_hiontop_parity.py`, `outputs/experiments/2026-05-22_hiontop_parity/`)
중 발견 — v4.1.x 코드 default `ctx_window=3` 이 보고 수치(TIAGE 0.4675 / Dialseg711
0.5897 / SuperSeg 0.4631)를 낸 canonical TIAGE-cfg(`m=2`)와 어긋나 있었음. `HiOnTop`
는 main 모델이므로 default 를 보고 config 에 맞춰 `2` 로 정정. output parity 는 m 과
무관하게 구조적으로 성립(매칭 config 에서 38,242 turn diff 0 재검증). 산출:
`context/methodology/hi_ontop.md` 신설, `outputs/reports/hi_ontop_algorithm_walkthrough.{md,pdf}`
신규, `scripts/verify_hiontop_parity.py` 신규.

---

## 2026-05-23 — Hi-OnTop-v2: lexical-overlap 보정 변형 도입 (검증 대기, v1 유지)

**배경**: [[hi-ontop]] (`HiOnTop`) 의 알려진 실패 모드 — wording 은 비슷한데 topic 이
바뀌는 경계(δ≈낮은데 GT 경계)를 인접 임베딩 cosine 신호로는 못 잡음. 사용자 요청:
TextTiling 식 단어-빈도 겹침(lexical overlap) 신호를 δ_eff 에 결합해 v2 를 만들고
세 벤치에서 검증. 사용자 확정 방향 = **어휘 겹침↓ → δ_eff↑ / 겹침↑ → δ_eff↓**
(어휘 응집 시 경계 억제).

**설계 (codex:rescue 위임)**: 결합 형태·인과적 lexical 신호 정의·HP·SEM 계승
정당화 모두 codex 권고 채택.
`δ_eff_v2 = clip_[0,2]( δ_base + w_lex·r_t·(lexdist − μ_lex) )`. `δ_base` = HiOnTop
δ_eff 불변. `lexdist = 1 − cos_tf(L_{t-1}, subtf(u_t))`, L = 직전 m_lex turn 의
ρ_lex-감쇠 sublinear-TF 합. `r_t` = 짧은-turn 신뢰 게이트. `μ_lex` = train median
(residual centering). 가산형 채택 — 곱셈형은 δ_base 가 낮은 바로 그 실패 모드에서
힘이 약함. lexical 신호는 zero-lag·causal — streaming TextTiling 의 right-block
닫힘 lag 회피. `w_lex=0` 시 v1 과 byte-parity (검증됨).

**SEM 계승 3-step**: lexical overlap 은 SEM 에 **없음** — SEM 은 structured scene
dynamics 의 예측가능성으로 boundary 를 설명하지 단어 표면형 cohesion 을 추적하지
않음(추상화 수준 차이, 의도적 금지 아님). 충돌은 제한적 — lexical 항은 sCRP/local-MAP
가 아니므로 SEM 원형 충실 재현과는 이질적이나, Hi-OnTop-v2 목표가 SEM-inspired online
segmentation 이므로 충돌 없음. lexical 항을 **SEM core 가 아닌 domain-specific
보조 관측 feature** 로 기록. 우선순위: SEM 계승성(online·causal·local) > 도메인
적합성 > SEM 원형 충실 재현. CLAUDE.md "Retrieval 은 importance score 만" 규칙과는
무관(segmentation 영역).

**결과** (`outputs/experiments/2026-05-23_hiontop_v2/REPORT.md`, official SuperDialseg
Score, calib=train/validation/30%-tune, w_lex sweep): mean-3 v1 0.4927 → w_lex=0.30
0.4989 (+0.0062). dialseg711 은 w_lex 단조 증가(+0.005~+0.015, robust 양),
superseg 약한 양(+0.003~0.005), tiage noise(표본 100, 부호 불안정).

**결정**: **v1 대체 승격 보류**. mean-3 +0.6 Score point 는 약하고 tiage 가 noise
라 main 모델 교체 근거 부족. 단 dialseg711 단조 추세가 lexical 신호의 원리적 유효성을
보이므로 폐기도 아님 — `HiOnTopV2` 는 검증 대기 변형으로 보존. **현 main 모델 =
`HiOnTop` 유지.** 승격 재검토 전제: (a) multi-seed 유의성, (b) lexical HP 2차 grid,
(c) v1 실패 모드 turn case-level 검증.

**산출**: `src/hi_ontop/hi_ontop_v2.py` 신규, `scripts/run_hiontop_v2.py` 신규,
`context/methodology/hi-ontop-v2.md` 신규, `context/03-architecture.md` 갱신.

---

## 2026-05-23 — DTS 표 Ours latency 정직성 정정 (host-shared → realtime per-turn)

**배경**: `outputs/reports/dts_result.md` 의 Online "Ours (p70/p75/p80/oracle)" 4 행이
Pre. 0 · Seg. 0.2 ms 로 보고돼 있었음. caption 자체에 *"host-shared accounting (encoder
forward is amortized with the surrounding pipeline's representation step)"* 라고
명시 — 즉 encoder forward 비용이 표에서 **제외**된 채 보고됨. 사용자 정의("인코딩
캐시 따로 쓰지 않고 리얼타임 설정에서 매턴마다 계산해서 합치고 턴으로 나눈 값")
와 정반대. baseline 들(GreedySeg 13 ms 등)은 encoder forward 포함이라 한 표에서
비대칭.

**조치**: `scripts/measure_hiontop_latency.py` 신규 — Def-DTS bundle test split 에서
seed=0 으로 벤치당 500-turn budget subsample, **encoder 캐시 off + batch=1 + 매 turn
perf_counter (encode + assign)**, 첫 발화 제외(baseline 들과 동일 정책). encoder
= MPNet (`multi-qa-mpnet-base-dot-v1`) CPU. 결과 (`outputs/experiments/
2026-05-23_hiontop_latency_realtime/`):

| 벤치 | n turn | Pre. mean (ms) | Pre. p50 | Seg. mean | Seg. p50 |
|---|---:|---:|---:|---:|---:|
| tiage | 468 | 1128 | 881 | 0.42 | 0.25 |
| dialseg711 | 498 | 849 | 738 | 0.34 | 0.23 |
| superseg | 468 | 932 | 665 | 0.35 | 0.24 |
| **cross-bench** | **1434** | **967** | **747** | **0.37** | **0.24** |

**결정**: `dts_result.md` 의 Ours 4 행 Pre./Seg. 셀을 0/0.2 → **967/0.37** (cross-bench
mean) 으로 교체. caption 의 *host-shared accounting* 문장 폐기, *realtime per-turn,
no cache, ΣΔt/N* 정의로 교체. §4 TODO 의 *"Latency Pre./Seg. split 정밀 측정"* 항목
체크 처리. §6 해석 문단도 갱신 (segmentation 자체는 sub-ms 로 거의 free, 우세 비용은
encoder forward).

**한계**: CPU only, MPNet encoder 한정. GPU 또는 MiniLM/MiniLM-int8 인코더로 교체
시 Pre. 한 자릿수 ms 까지 떨어질 가능성 — 별도 측정 필요. 표본은 벤치당 500-turn
budget (전수 36k turn 측정 ~6 시간 회피).

**산출**: `scripts/measure_hiontop_latency.py` 신규, `outputs/experiments/
2026-05-23_hiontop_latency_realtime/{REPORT.md,latency.json}` 신규,
`outputs/reports/dts_result.md` 갱신.

### 2026-06-08: AMI 도메인 robustness 진단 + Hi-OnTop v3(granularity-adaptive) 라인 신설 + Hi-DoTS→Hi-OnTop rename

**1) AMI(회의 음성) 도메인 진단 — turn 단위 붕괴, 원인 = granularity 불일치**

AMI scenario meetings(NXT manual annotation, manifest 12미팅)로 Hi-OnTop 도메인
robustness 검증. **turn(=AMI segment) 단위로 완전 붕괴**:
- 같은 화제 인접 발화 cosine 중앙값 **0.167** (60%가 cos<0.2), δ_eff 경계 분리
  **AUC 0.567**(≈무정보), top-level F1 0.05~0.07, δ* p80=0.88·p98=0.98(거의 전 인접쌍이 경계처럼).
- 원인: 회의 발화 31%가 1단어 backchannel(yeah/okay/mm-hmm), median 4단어로 짧고
  다화자 반응형 → "같은 화제=비슷한 발화" 라는 방법 전제 불성립. 진짜 top 경계의
  31~44%가 ≤2단어 발화 위에 있어 **후처리(짧은 발화 drop/경계금지/defer) 전부 실패**
  (실측: F1 0.05→0.02, 신호 자체가 빔).
- **고정 시간 블록(전 화자 발화 이어붙임)으로 단위 교체 시 회복**: win sweep —
  30s AUC0.669, 45s 0.689, **60s 0.664/bestF1 0.379·bestScore 0.446**, 90s 0.629.
  같은화제 cos 0.167→0.44. **60s가 sweet spot이나 = AMI에 최적화된 magic number**
  (label-free 철학 위배). 산출: `scripts/ami_topic_eval.py`(turn), `ami_topic_block_eval.py`(블록),
  `outputs/experiments/2026-06-08_ami_topic_block_w{30,45,60,90}/`.

**2) 결정 — Hi-OnTop v3 = "granularity 적응" major 라인 신설, 경쟁 후보 2종**

적응 축 정리: v1=고정 / **v2=threshold 적응**(lexical 잔차가 turn별 effective δ*) /
**v3=granularity 적응**(무엇을 분절기에 넣을지). v3 아래 같은 문제의 경쟁 해법 2종:
- **v3.1 (B)** — 명시적 단위 적응: 시간 대신 content량(~N단어/토큰) 기준 적응 블록.
  단위 선택 층을 δ_eff 앞에 둠. (TextTiling baseline 과는 *역할* 이 다름 — baseline은
  최종 경계 산출기, v3.1은 unit 형성기.)
- **v3.2 (C)** — SEM 계승형: 고정 블록 폐기, turn 유지하되 각 발화를 **현재 segment
  누적 centroid μ_k**(=현재 event model)와 비교 + 짧은 발화 down-weight + sCRP
  stickiness(α)로 단일 blip 흡수 → segment 경계가 sequential MAP에서 emerge, magic 단위 없음.
  δ_ctx(고정 window m=2)의 *segment 전체-window + 경계 reset* 일반화.

**승자 = 정식 v3 승격** (.x 후보/ablation 보존). 결정 규칙(사후 cherry-pick 방지):
(a) AMI turn 단위 AUC 0.567/F1 0.05 대비 유의 상승, (b) DTS 3벤치(TIAGE/Dialseg711/
SuperSeg) v1/v2 무회귀, (c) magic number 없이 label-free/deployable 유지(남는 HP는
인코더·모델 속성, 도메인 의존 아님). 기존 experimental→승격 패턴(v3.3.10→v4.1.1) 계승.

**SEM 계승 3-step (v3.2 핵심)**: ① backchannel 처리는 SEM/SEM2 입력 도메인(풍부한
scene/단락) 밖이라 미구현 — 의도적 배제 아님. ② 누적 centroid 비교 = SEM `f_k.update(x_t)`
근사, α의 blip 흡수 = sticky prior 본래 역할 → SEM 메커니즘 직접 구현, 철학 충돌 없음.
③ 본 엔트리로 근거 기록. **v3.1(B)의 lexical/표면 cohesion은 SEM 비계승**(sCRP 연결고리
없음) — 실용 대안으로만.

**구현 순서**: v3.2(C) 먼저(`hi_ontop_v3_event_exp.py`/`HiOnTopV3Event`) → AMI+DTS 실측 →
v3.1(B)(`hi_ontop_v3_unit_exp.py`/`HiOnTopV3Unit`) → 비교표 → 승자 v3 승격.

**3) Hi-DoTS → Hi-OnTop 전면 rename (흔적 제거)**

Hi-DoTS는 Hi-OnTop의 옛 이름. 코드는 이미 마이그레이션 완료(`hi_ontop.py`/`HiOnTop`),
잔존하던 backward-compat shim(`hi_dots.py`/`hi_dots_v2.py`)·별칭(`HiDoTS=HiOnTop`)·파일명·
디렉토리·문서 흔적을 전부 제거. shim 2개 삭제, 파일/디렉토리 44개 rename, 텍스트 67개
파일 치환(Hi-DoTS→Hi-OnTop 등). DTS(task)·Def-DTS(외부레포)는 불변. 검증: 텍스트 흔적 0,
import smoke OK, 테스트 회귀 0(267 passed; 기존 실패 4건은 STM/locomo-data 무관 이슈).
**미해결**: figure PDF/PNG 내부 렌더링 텍스트(압축 스트림)는 재생성 필요, methodology
`hi_ontop.md`/`hi-ontop.md` 중복(구 `hi_dots.md`/`hi-dots.md`) 병합 보류.

---

## 2026-06-10 — AMI 분절: V_rel 상대신호 확립 + online reset 부트스트랩 병목 (Hi-OnTop)

**배경**: AMI 139미팅에서 기존 Hi-OnTop δ_eff ewma = ±2F1 0.151 / Score 0.203. LLM
full-context 0.543/0.640 대비 큰 격차. 원인 규명 + 신호 개선 (codex gpt-5.5 위임 2회).

**결정 1 — 경계 판정을 magnitude 임계치 → "active-event 설명 실패 + background 상대거리"로.**
진단: 가장 강한 δ_eff peak가 경계 아니라 noise(화자전환·단발 이상치). magnitude 단독 분리 불가.
신호 = **V_rel = r_active − λ·r_global** (r_active=1−cos(x, active prototype EWMA);
r_global=1−cos(x, global running centroid, g_rho=0.15); λ=0.6). gold-reset clean prototype +
이 신호 → oracle ±2F1 천장 0.687(>LLM 0.543). overfit 2-fold 격차 0.000 (knob 전 split 동일).

**SEM 계승 3-step**: ① r_global(background 대비)은 SEM new-event base distribution 비교의 거리공간
근사 — codex 설계 `log p(x|new)−log p(x|active)`의 reduction. 의도적 도입, SEM에 근거 있음.
② active prototype 누적·경계 reset = SEM event 모델 갱신/신규 event 개시. sCRP/local MAP 철학과 정합.
③ 본 엔트리 기록. λ·g_rho는 calibration(인코더·도메인 속성), magic number 아님(overfit 검증 완료).

**결정 2 — 병목은 임계치가 아니라 online reset 부트스트랩 (미해결, 다음 iteration).**
격차 분해: clean(gold-reset)+단순 μ+cσ = ±2F1 0.554(LLM급). 즉 신호·임계치 충분. 그러나
detected-reset deploy = 0.15 (오염→under-seg 악순환). robust·peak·anchor·refractory·EM 반복정제·
BOCPD top-K particle filter(codex 설계) 모두 oracle에 도달 못 함. BOCPD는 개수만 잡음(Score 0.305,
±2F1 0.069). **deployable 최선 = 단순 V_rel 적응임계치 Score 0.358** (기존·TextTiling·even-spacing
oracle 모두 상회).

**결정 3 — AMI 포지셔닝 = robustness/한계 도메인 (primary 아님).** drift+sparse+gold-offset 3중
난점, even-spacing oracle이 Pk/WD 지배(content 신호와 구조적 불일치). DTS(concat-seam) primary 유지.
codex 자문 일치.

상세 + 전체 수치: `outputs/experiments/2026-06-10_ami_vrel_localmap/REPORT.md`.
신규 스크립트: `ami_vrel_eval.py`, `ami_vrel2_eval.py`, `ami_bocpd_eval.py`, `ami_localmap_eval.py`.
**미해결**: V_rel을 정식 hi_ontop 모듈/버전으로 승격할지는 deploy가 oracle 격차 메운 뒤 결정 (현재 보류).

---

## 2026-06-11 — de-neutralized prototype + run-length 적응 β (Hi-OnTop, AMI+DTS 동시)

**배경**: V_rel(=r_active−λ·r_global)이 AMI(drift)엔 강하나 DTS(concat-seam)엔 회귀. 두 도메인이
정반대 구조(DTS는 직전 점프 sharp, AMI는 prototype 필요). prototype 형태 변형(mean/nn/medoid/varnorm/
subspace/info-gate) 전부 superseg 벽(0.467) 못 넘음 — 평균은 짧은 segment에 원리적 불리.

**결정 1 — de-neutralized prototype.** prototype·발화에서 global(중립) 성분을 β만큼 제거 후 비교:
`r_active = 1 − cos(normalize(x − β(x·g)g), normalize(m − β(m·g)g))`. "화제의 변별적 방향"만 봄.
β=1(full)에서 **superseg 0.506 > 0.467 — 처음으로 벽 돌파**, DTS 3개 다 δ_eff 초과. 단 AMI 0.222(짐).
β=0 = V_rel(AMI 0.659). 두 도메인 β에 정반대.

**결정 2 — run-length 적응 β (자기검출).** `β_t = clip(A − B·log(1+l/L0), 0, 1)`, l=segment 길이.
짧은 segment(DTS)→β→1(de-neut), 긴 segment(AMI)→β낮음(V_rel). **run-length가 sharp/drift를 도메인
라벨 없이 판별** (R̄=global 집중도는 방향 반대라 실패). best (A,B)=(2.0,1.0): tiage 0.462/dialseg 0.384/
superseg 0.506/AMI 0.341 — oracle 4개 strict.

**검증**: 2-fold(도메인 내) — dialseg/superseg/AMI robust, **tiage tie**(even fold 회귀). LOO(cross-domain)
— 4개 held-out 다 PASS, (A,B) 전이됨(AMI는 grid 어떤 (A,B)든 >δ_eff). deploy(공정 calib-c, held-out test) —
adaptive-deneut Score **0.367** vs δ_eff 0.343 (**+0.024 modest**, ±2F1 동률 0.106).

**SEM 계승**: de-neut의 global 성분 제거 = "공통/background 대비 화제 변별" — SEM new-event base distribution
대비의 변형. λ·g_rho는 V_rel 결정(2026-06-10) 계승. run-length 가중은 event 지속(persistence) 기반 reliability
weighting — SEM event-model reliability 원리와 정합(codex 자문, 단 직접 메커니즘은 아니라 heuristic). [[2026-06-10]]

**판정 / 미해결**: **신호/oracle 차원 = 진짜 성과**(superseg 벽 cross-domain robust 돌파). **deploy 차원 =
modest(+0.024), localization 동률** — oracle 우위가 online으로 안 넘어옴. 원인 = **online reset 부트스트랩**
(clean prototype을 online 유지 불가; hard-reset·robust·peak·EM·BOCPD particle filter·lagged emission 모두
미흡). 정식 hi_ontop 버전 승격은 deploy가 이 격차 메운 뒤로 **보류**. (A,B) 자기보정 + predictive prototype
(transformer) 후속 후보. tiage tie 정직 인정.

상세: `handoff/HANDOFF_0609_deploy-oracle-gap.md`. 스크립트: `ami_dts_{deneut_oracle,beta_sweep,adaptive_beta,beta_overfit,
beta_loo,deploy_calib}.py`, `ami_adaptive_deneut_deploy.py`.

**2026-06-11 추가 (calibration-free 평가)**: de-neut deploy 비교 시 c(적응임계치 σ-배수)를 calib/test로 고르는
번거로움 → 불필요 검증. (1) AMI Score-vs-c: deneut Score가 *모든* c에서 δ_eff≥ → 우열 c-무관, 고정 c/Otsu면
충분 (`ami_dts_score_vs_c.py`). (2) AUC(threshold-free): ±2 AUC AMI deneut 0.621 vs δ_eff 0.497(+0.123),
tiage/superseg도 우세, dialseg711 혼재; δ_eff ±2 AUC~0.5=거의 random (`ami_dts_auc_eval.py`). 단 지표 비대칭 —
Score는 deneut 압승이나 ±2F1은 δ_eff 약간 우세(deneut Score 우위는 Pk/WD에서). 평가지표 Score 유지 시 deneut 승.

---

## 2026-06-11 — online reset 부트스트랩 돌파 방향 = commit-and-refine (Hi-OnTop, 축 2)

**배경**: 축 1(신호)은 de-neut+적응β로 oracle 천장 확보(δ_eff 초과, V_rel은 LLM 천장도 초과). 그러나 축 2
(deploy)는 **online reset 부트스트랩**에 막혀 oracle(0.55~0.69)을 deploy(±2F1 0.15)로 못 끌어올림. gap 분해로
병목이 reset임은 증명됨(clean+μcσ 0.554 vs detected-reset 0.15). 시도 전부 실패(BOCPD particle/lagged/EM/
robust-peak-anchor). 방향 결정을 codex:rescue에 위임(gpt-5.5 high effort, direct `codex exec`).

**결정 — bounded-lag commit-and-refine + local split scoring + shadow prototype 채택.**
- **핵심 원칙**: 경계 *감지*와 reset *확정* 분리. `d_t>θ_t`를 reset 확정이 아니라 **split 후보 생성**으로만 사용;
  확정은 오른쪽 후보 segment가 자기 prototype을 만들 만큼 응집(**persistence 항**)할 때만. → 악순환 뿌리인
  *불가역 hard reset* 제거.
- **알고리즘(Local Commit-and-Refine Splitter)**: split gain = SSE(a..t)−[SSE(a..b−1)+SSE(b..t)]
  +α·shock_b+β·persistence−γ·short_penalty (SSE는 Σ‖z‖²−‖Σz‖²/n으로 O(1)). persistence = mean_{k=b..t}
  [dist(z_k,p_L)−dist(z_k,p_R)] (단일 outlier vs 새 응집 덩어리 구분). 잠정 경계 emit(shadow prototype 사용,
  no-split branch 유지) → L window 내 b* 이동 정정 → 확정(right_len≥m_min, margin, h연속)/거절(buffer 병합,
  high-shock singleton outlier 처리). 실무값 L=3~5, m_min=2~3, h=1~2; ±2 tol 중시면 L=2~3 + 즉시 provisional
  emission + event-time correction.
- **탈락**: predictive prototype(무학습 제약상 다음 prototype 예측 근거 약, 보조로 prior 약조정 한계),
  전역 particle filter(soft hypothesis는 commit window *내부* 위치 정정용으로만). 4개 실패 원인 진단:
  BOCPD 위치 posterior가 본질적으로 넓음(hazard/pruning이 위치 결정, top-K가 약한 진짜 가설 죽임) / lagged는
  후보별 prototype 재계산 누락 / EM 자기강화 + 짧은 segment 고분산 → local optimum / robust류는 불가역 reset 못 되돌림.

**SEM 계승**: commit-and-refine = SEM local-MAP event 개시를 "응집 확인 후 확정"으로 보수화한 것;
persistence 항 = event-model reliability(지속) 원리와 정합. shadow prototype = 미확정 event 후보의 잠정 추정.
hard split 즉시확정 대신 bounded-lag 정정은 SEM 철학(local Bayes)과 충돌 없음.

**판정 / 미해결**: 방향만 확정, **미구현**. 다음 세션 = `scripts/`에 Local Commit-and-Refine Splitter 프로토타입 →
AMI/DTS deploy에서 oracle 격차(±2F1 0.15→0.55) 메우는지 검증. (A,B) 자기보정은 별도 후속.

상세: `handoff/HANDOFF_0609_deploy-oracle-gap.md` §2.5, codex 자문 전문 `outputs/runs/_misc/codex_bootstrap_consult.md`. [[2026-06-11]]

**2026-06-11 추가 (commit-and-refine 1차 구현·실측)**: 위 방향을 AMI deploy 에서 3단계로 구현(`scripts/
ami_commit_refine_deploy.py`). v1 `segment_cr`(persistence 게이트 확정만) → 실패(±2F1 0.11≤hard-reset, 위치
최적화 없어 localization 불변). v2 `segment_cr2`(+local split-gain b* refinement, b*=argmax[SSE(a..t)−SSE(a..b−1)
−SSE(b..t)]) → **새 deploy best Score 0.401/±2F1 0.140** (hard-reset 0.372/0.131 대비 +0.029/+0.009). v3
`segment_cr3`(shock gate 제거, split-gain sliding 상시) → 실패(raw 는 global 지배로 과분절 pred 6k~30k, de-neut 도
shock precision 없이 과분절 Score≤0.33). **결론: shock 감지(V>θ)의 precision + b* refinement 의 localization 이
sweet spot. 그러나 oracle 천장(±2F1 0.554) 격차는 미해결** — refinement 는 감지된 후보 위치만 고치고, 오염
prototype 이 놓친 경계(recall)는 못 살림(gate 풀면 precision 붕괴). 부트스트랩 본질 미해결, 정식 승격 보류 유지.
다음 후보: gate(precision)+상시 split-gain(recall) 2-stage 결합, prototype 오염 완화(robust shadow), (A,B)·c 자기보정.
상세 REPORT: `outputs/experiments/2026-06-11_ami_commit_refine/REPORT.md`. [[2026-06-11]]

**2026-06-11 추가2 (v4 2-stage 결합 — 실패)**: cr2(precision) + 긴 segment recall booster(de-neut split-gain
강제 분할) 결합. recall booster 가 ±2F1 0.140→0.182 까지 올리나 Score 0.401→0.294 붕괴(pred 521→2248 과분절,
Pk/WD 무너짐) — frontier 못 옮기고 δ_eff식 과분절 곡선으로 이동. **reset 기법 4각도(v1 확정 / v2 b* refinement /
v3 sliding / v4 2-stage) 모두 oracle 격차 못 메움 → 병목은 reset 메커니즘이 아니라 online localization 정밀도
자체.** v2(Score 0.401) 가 best balanced, 단순 reset 기법 추가 탐색은 수확체감 → 신호/정밀도 또는 prototype 오염
완화 쪽 전환 권고. REPORT 갱신: `outputs/experiments/2026-06-11_ami_commit_refine/REPORT.md`. [[2026-06-11]]

**2026-06-11 추가3 (v5 prototype 오염완화 — 실패)**: v2 base 에 outlier-gated 업데이트(V>μ+gate·σ 발화의
prototype 업데이트 rho 를 damp 축소) + recency(고정 rho) 추가(`segment_cr2` 파라미터 rho_fix/gate/damp).
**outlier-gating 거의 무효과**(g=1.5 → ±2F1 0.138/Score 0.401 ≈ v2) → prototype 오염이 국소 high-V outlier 가
아니라 **확산적/구조적**(점진 drift + 놓친 경계가 두 topic 통째 포함, 발화 단위 게이팅 복구 불가). recency 는 악화
(신호 noisy, 오염 미감소). **종합: reset 기법 4 + 오염완화 1 = 5각도 모두 oracle 격차 못 메움 → 병목은 online 운영
구조적 한계로 결론.** 단순 reset/prototype 미세조정 **종료**. 잔여 방향: (A,B)·c 자기보정(gap-closing 아닌 tuning
제거), 또는 v2(Score 0.401) ship + 격차 명시적 한계 문서화. 정식 hi_ontop 승격 보류 유지.
REPORT: `outputs/experiments/2026-06-11_ami_commit_refine/REPORT.md`. [[2026-06-11]]

## 2026-06-11 — Hi-OnTop-CR 정식 승격 (main 분절 모델)

**결정**: de-neut + run-length 적응-β 신호(universal, calibration-free) + commit-and-refine drift deploy 를
**정식 main 분절 모델로 승격** → `src/hi_ontop/hi_ontop_cr.py` `segment(emb, reset=...)`, methodology [[hi-ontop-cr]].
직전 보류([[hi-ontop]] 2026-06-10 "online deploy 격차로 승격 보류")를 **격차를 명시적 한계로 두고** 해제.

**범위 (사용자 결정 2026-06-11)**: **신호=universal**(두 도메인), **commit-and-refine=drift(AMI) deploy reset**,
**sharp(DTS)=기존 threshold hard reset** (commit-and-refine 은 짧은 segment 억제로 DTS 회귀 — `reset=` 인자로 분리).
→ DTS 회귀 없이 승격.

**근거 수치**: commit_refine AMI deploy Score 0.401 / ±2F1 0.140 (hard-reset 0.372/0.131 대비 +0.029/+0.009,
deploy 신기록). 신호는 oracle 천장에서 LLM 초과(AMI V_rel 0.687>0.543). canonical 모듈 parity 검증(research
`segment_cr2`/`segment_hard` 와 byte-identical: 521/0.140/0.401, 4105/0.131/0.282).

**정식 config (검증)**: c=1.0(calibration-free), L=8(AMI 5-fold CV L\*=8 만장일치·overfit gap 0), (A,B)=(2.0,1.0)
(LOO 검증), m_min=2(미검증 — 유일 잔여 상수).

**명시적 한계 (승격 전제)**: online deploy ±2F1 0.140 ≪ oracle 0.554. reset 기법 5각도(v1~v5) 모두 격차 못 메움
→ 병목은 online 운영 구조적 한계(놓친 경계가 prototype 오염, 발화 단위 복구 불가). "신호는 LLM급(oracle),
online 실현은 구조적 천장"이라는 정직한 스토리 위에서 승격. streaming 클래스(lagged emission) 통합은 scope 밖
(δ_eff [[hi_ontop]] `class HiOnTop` 유지). 후속: (A,B)·m_min 자기보정 / streaming 설계.

상세: [[hi-ontop-cr]], REPORT `outputs/experiments/2026-06-11_ami_commit_refine/REPORT.md`. [[2026-06-11]]

**2026-06-11 추가 (m_min 검증 — open thread 마감)**: 승격 entry 에서 '유일 미검증 상수'였던 m_min 을 AMI k-fold
CV 로 검증 — 5-fold train-선택 m\*=2 만장일치, held-out overfit gap 0, 민감도 m=1·2 ±2F1 동률(m=2 Score-best,
m≥3 회귀). → **Hi-OnTop-CR 정식 config 전 상수(c·L·A·B·m_min) 검증 완료, 미검증 0개.** 잔여 open thread = streaming
클래스(lagged emission) 통합뿐이며 이는 Hi-OnTop 파이프라인 scope-barred(현재 작업 범위 밖) — batch segment() 는 이미
online 충족, 파이프라인 재개 시 buffer-and-flush wrapper 로 부착(codex 위임). [[hi-ontop-cr]]. [[2026-06-11]]

**2026-06-12 — Hi-OnTop-CR default reset = `threshold` (0-lag) 로 변경**: 사용자 요구 = 버퍼/지연 원치 않음(다음 턴 즉시 emit). 측정(AMI 139, best-c by Score): threshold 0-lag Score 0.372 / commit_refine 의 0.401 은 **전적으로 lag 매입분**(L=4≈12s 0.379, L=8≈26s 0.401; L≤3 은 threshold 와 동급/이하). → `segment()` default 를 commit_refine→threshold 로. commit_refine 은 "버퍼(≥~12s) 허용 시 +0.029 옵션"으로 강등. DTS 는 원래부터 threshold. [[hi-ontop-cr]] §2 lag–Score 표. [[2026-06-12]]
