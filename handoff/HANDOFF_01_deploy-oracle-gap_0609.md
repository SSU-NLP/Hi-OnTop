# Handoff — AMI/DTS 화제분절 신호 탐색 (Hi-OnTop)

> **⚠ DTS 결론 무효 (2026-06-13, [[HANDOFF_04]])**: 본 handoff 의 채점 harness(`ami_dts_*.py::oc()`)에
> **경계 off-by-one 버그**(신호 스파이크 t 를 gold t 에 비교; 정상은 끝-turn 규약 t→gold t-1)가 있어,
> **DTS(축 1·2) 수치·결론은 신뢰 불가**. 공식 채점(SuperDialseg `SegmentationEvaluation`)+정상정렬에서는
> **DTS 3개 전부 δ_eff > de-neut/CR** ("superseg 벽 0.467 돌파"·"de-neut > δ_eff" 는 버그 산물).
> **AMI 는 DTS 와 규약이 반대**(2026-06-14 데이터 확정: AMI gold `bnd_top`=새 topic **시작-turn**, DTS=끝-turn).
> ⇒ AMI 신호의 off-by-one '버그'는 버그 아님 — **shift 0 이 규약정렬**(신호 첫-turn ↔ gold 첫-turn). de-neut > δ_eff
> 결론 유지 (shift0: **0.342 vs 0.214**). shift -1(0.370/0.225)은 filler-lag overfit, ±2 가 ±1 잔차 흡수. [[feedback_ami_gold_start_turn_convention]]
> 상세·재현: [[HANDOFF_04]], `scripts/{validate_official_scorer,dts_official_rescore,ami_alignment_recheck}.py`,
> decision-log 2026-06-13.

**기간**: 2026-06-09 ~ 2026-06-11 · **대상**: Hi-OnTop 화제분절 *신호/판정* 개선 (메모리 파이프라인 아님)

## 0. 한 줄 + 두 축
이 작업은 **두 개의 분리된 노력**이었다.
- **축 1 — 더 좋은 *신호* 찾기 (oracle 차원)**: gold-reset(깨끗한 prototype) 가정 하에 경계를 제일 잘 가르는
  신호. → **성공.** `de-neut + run-length 적응 β`가 두 도메인 다 δ_eff를 넘고, V_rel은 oracle 천장에서
  LLM(±2 0.543)을 넘음(0.687).
- **축 2 — 그 신호를 *online에서 실현* (deploy 차원)**: 정답 모르고도 그 oracle 성능 내기. → **실패.**
  전부 **online reset 부트스트랩**에 막혀 oracle(0.55~0.69)을 deploy(±2F1 0.15)로 못 끌어올림.

> **요약**: 좋은 신호는 찾았다(축1 ✅). 그걸 online에서 못 꺼낸다(축2 ❌). 남은 1순위 = 축 2 = reset 부트스트랩.

## 0b. 셋업
- 기존 신호 `δ_eff = a·δ_prev + (1−a)·δ_ctx` (직전 발화 + 2턴 윈도우 cosine 거리), 적응임계치 μ+cσ.
- 도메인: **AMI**(긴 회의, drift, 잡음 多, gold 극sparse) vs **DTS**(tiage/dialseg711/superseg; 짧은 대화 concat,
  sharp seam, 잡음 無). 인코더 MiniLM-int8 고정, online·0-look-ahead·무학습.
- metric: AMI=±2 tol F1 / Score(0.5F1+0.25(1-Pk)+0.25(1-WD)); DTS=exact F1. **oracle**=per-meeting 최적임계
  (+ gold-reset prototype) = 신호 천장. **deploy**=완전 online.

---

# 축 1 — 신호 발견 (oracle 차원) ✅ 성공

## 1.1 진단 (왜 기존이 약한가)
- δ_eff z-score: LLM 경계 turn에서 +0.545(랜덤 +0.021)지만 **가장 큰 spike는 경계가 아니라 noise**(화자전환·단발
  이상치) → magnitude 단독 분리 불가. top-K/z-threshold로 LLM 일치 ~0.11 고착.
- prototype "요약"은 텍스트 아님(임베딩 평균 벡터). 길게 누적하면 추임새 같은 **중립점으로 흐려짐**
  (추임새 cos 0.7, 내용 0.3). gold/LLM 없이 텍스트 요약 불가. (trace: `outputs/reports/vrel_*_trace.md`)

## 1.2 V_rel — active 대비 background 상대거리
`V_rel = r_active − λ·r_global` (r_active=1−cos(x, active prototype EWMA, 경계서 reset); r_global=1−cos(x,
global EWMA, g_rho=0.15); λ=0.6). 직관: 진짜 경계는 active서 멀고 global선 평범 → 큼; noise는 둘 다 멂 → 작음.
- **AMI oracle 천장 0.687 > LLM full-context 0.543.** overfit 2-fold 격차 0.000.
- **단 DTS에선 회귀** — concat-seam은 prototype(평균)보다 직전 점프(δ_prev)가 sharp.

## 1.3 de-neut — 중립성분 제거 (superseg 벽 돌파)
`r_active = 1−cos(normalize(x − β(x·g)g), normalize(m − β(m·g)g))`. "화제의 *변별적* 방향"만 봄.
- β=1: **superseg 0.506 > 0.467 — 아무도 못 깬 벽 첫 돌파**, DTS 3개 다 δ_eff 초과. 단 AMI 0.222(짐).
- β=0 = V_rel (AMI 0.659). 두 도메인 β에 정반대.
- prototype 형태(mean/nn/medoid/window/content-weight/robust/subspace/varnorm/info-gate)는 **전부 superseg 못
  넘음**(0.42~0.44) — 평균은 짧은 segment에 원리적 불리. de-neut만 돌파.

## 1.4 run-length 적응 β (둘 다 이기기, 자기검출)
**먼저 고정 β로 보간**: `s = a·δ_prev + (1−a)·(r_active − λ·r_global)` 등 — 두 도메인 정반대라 단일 고정값으론
strict 동시우위 어려움. **부분 de-neut 고정 β=0.70**은 4개 oracle strict(superseg 0.475)지만 **AMI를
0.659→0.290으로 크게 희생** → "고정 최적화"라 불만족. λ(global 항)이 DTS 회귀 원인 아님도 확인(λ=0도 superseg<δ_eff).
**→ 적응이 답.**
`β_t = clip(A − B·log(1+l/L0), 0, 1)`, l=segment 길이. 짧은 segment(DTS)→β→1(de-neut), 긴 segment(AMI)→β낮음
(V_rel). **run-length가 sharp/drift를 도메인 라벨 없이 판별.** (R̄=global 집중도는 방향 거꾸로라 실패 —
superseg R̄ 최고/AMI 최저.) 고정 β=0.70보다 *모든* 도메인에서 우월(superseg 0.506>0.475, AMI 0.341>0.290).
- **best (A,B)=(2.0,1.0) oracle**: tiage 0.462 / dialseg711 0.384 / superseg 0.506 / AMI 0.341 — 4개 strict
  (δ_eff .452/.313/.467/.235). β평균 DTS 1.00 / AMI 0.39.

## 1.5 검증 (cherry-pick·calibration 차단)
| 검증 | tiage | dialseg711 | superseg | AMI | 결론 |
|---|--:|--:|--:|--:|---|
| **2-fold (도메인 내)** | even **FAIL**/odd ✓ | ✓✓ | ✓✓ | ✓✓ | dialseg/superseg/AMI robust, **tiage tie** |
| **LOO (cross-domain)** | +0.010 | +0.071 | +0.039 | +0.072 | (A,B) 전이됨; AMI는 grid 어떤 (A,B)든 >δ_eff |
| **AUC (threshold-free, ±2)** | +0.047 | **−0.045** | +0.025 | **+0.123** | AMI 크게 우세, dialseg711 혼재, δ_eff ±2 AUC~0.5(random) |
| **Score-vs-c** | — | — | — | 모든 c에서 deneut≥δ_eff | **calibration 불필요** (c 안 골라도 됨) |
- **calibration-free**: deneut Score가 *모든* c에서 δ_eff 이상 → 고정 c/Otsu면 충분. (앞서 "공정 calib +0.024"는
  δ_eff가 자기 best-c=과소분절로 끌어올린 것; 고정 c에선 격차 +0.04~+0.17.)
- **★ 지표 비대칭 (반드시 인지)**: `Score`는 deneut 압승이나 `±2F1`만 보면 δ_eff 약간 우세(δ_eff ±2F1 ~0.168 vs
  deneut ~0.138). deneut의 Score 우위는 Pk/WD(개수·간격)에서 옴.
- **tiage는 정직하게 tie** (작은 데이터 noise + δ_ctx가 짧은 tiage에 잘 맞음).

## 축 1 핵심 수치
```
                          tiage  dialseg  superseg   AMI(±2)
δ_eff (baseline oracle)   0.452   0.313    0.467     0.235
adaptive-deneut (2.0,1.0) 0.462   0.384    0.506     0.341
                          tie     ✓        ✓벽깸     ✓
(V_rel AMI oracle 0.687 > LLM 0.543 / clean+μcσ deploy-천장 0.554 / LLM full-context ±2 0.543·Score 0.640)
```

---

# 축 2 — online 실현 (deploy 차원) ❌ 미해결 = reset 부트스트랩

## 2.1 reset 부트스트랩이란
좋은 분절 ← 깨끗한 prototype ← 정확한 reset ← 좋은 분절 … **닭-달걀.** online은 정답을 몰라 경계를 *추측*해서
reset → 첫 추측이 틀리면 prototype 오염 → 더 못 봄 → 더 오염 (악순환). 한번 미끄러지면 못 빠져나옴.

## 2.2 gap 분해 — 병목이 reset임을 증명 (`ami_dts_*`/decompose_gap)
```
gold-reset + per-meeting 최적임계  0.687   [상한]
gold-reset(clean) + 단순 μ+cσ      0.554   ← 임계치는 충분! 깨끗하기만 하면 LLM급
detected-reset deploy             0.15    ← 여기서 폭락
```
→ **신호·임계치 충분. 병목은 오직 "online에서 깨끗한 reset 부트스트랩".**

## 2.3 시도들 — 전부 oracle 못 끌어올림
- **V_rel 적응임계치** deploy → Score 0.358. **de-neut adaptive deploy** → Score 0.372(새 best, 그래도 ±2F1 0.115).
- **local-MAP A1~A5**(`ami_localmap_eval`): active-event LLR+sCRP prior+penalty → 과분절/실패(Score 0.305).
- **BOCPD top-K particle filter**(`ami_bocpd_eval`): 개수 맞춤 but localization 나쁨(±2F1 0.069).
- **BOCPD lagged changepoint-start emission**(`ami_bocpd_lag_eval`, codex): 0.069→0.10, 그래도 ≪ 0.554.
- **robust/peak/anchor/refractory**(`ami_vrel2_eval`): 무효(±2F1 0.15 고착).
- **EM 반복정제**(`em_refine`): 나쁜 고정점 0.13 수렴 안 함.

## 2.4 현재 deploy best
**Local Commit-and-Refine Splitter v2 (shock-gated + split-gain b\* refinement), AMI Score 0.401 / ±2F1 0.140**
(`segment_cr2`, L=8 m=2 c=1.0). 직전 hard-reset deneut(Score 0.372 / ±2F1 0.131) 대비 **+0.029 / +0.009**.
단 **oracle 천장(±2F1 0.554)과의 격차는 여전히 큼** — 부트스트랩은 눌렀을 뿐 미해결.
(이전 best: de-neut adaptive hard-reset Score ~0.37, ±2F1 ~0.13.)

## 2.5 다음 후보 (축 2 = 1순위)
1. **online reset 부트스트랩** — **codex 자문 + 1차 구현·실측 완료(2026-06-11)**. 채택 방향:
   **bounded-lag commit-and-refine + local split scoring + shadow prototype** (`codex_bootstrap_consult.md`).
   **실측 결과**(REPORT `outputs/experiments/2026-06-11_ami_commit_refine/`): v1(확정만)·v3(상시 split-gain)·
   v4(2-stage 결합) 실패, **v2(+split-gain b\* refinement) = 새 best Score 0.401/±2F1 0.140 (modest +0.029/+0.009)**.
   v4 recall booster 는 ±2F1 0.18 까지 올리나 Score 0.29 로 붕괴(과분절, frontier 못 옮김). v5(prototype 오염완화:
   outlier-gated 업데이트/recency)도 실패 — gating 무효과(오염이 국소 outlier 가 아니라 구조적), recency 악화.
   **격차(±2F1 0.14 vs oracle 0.55) 미해결 — 5각도 모두 실패 → 병목은 reset/오염원이 아니라 online 운영의 구조적
   한계(놓친 경계가 prototype 을 두 topic 으로 오염, 발화 단위 복구 불가).** reset 미세조정 종료. 잔여: (A,B)·c
   자기보정(tuning 제거 별개 win) / v2(Score 0.401) ship + 격차 명시. 정식 승격 보류.
   - **핵심 원칙**: 경계 *감지*와 reset *확정* 분리. `d_t>θ_t`는 **split 후보 생성**으로만 쓰고, 확정은
     오른쪽 후보 segment가 자기 prototype을 만들 만큼 응집(**persistence 항**)될 때만 → 불가역 hard reset 제거.
   - **알고리즘(Local Commit-and-Refine Splitter)**: split gain = `SSE(a..t)−[SSE(a..b−1)+SSE(b..t)]+α·shock_b
     +β·persistence−γ·short_penalty` (SSE는 sum/‖sum‖²/count로 O(1)). 잠정 경계 emit(shadow prototype,
     no-split branch 유지) → L window 내 b* 이동 정정 → 확정(right_len≥m_min, margin, h연속)/거절(buffer 병합,
     high-shock singleton은 outlier 처리). 실무값 `L=3~5, m_min=2~3, h=1~2`; **±2 tol 중시면 L=2~3 + 즉시
     provisional emission + event-time correction**.
   - **탈락**: predictive prototype(무학습 제약상 다음 prototype 예측 근거 약함), 전역 particle filter
     (soft hypothesis는 commit window *내부* 위치 정정용으로만). 4개 실패(BOCPD/lagged/EM/robust) 진단 → §2.3.
2. **(A,B) 자기보정** — 데이터 통계(평균 segment 길이 등)에서 유도 → tuning 제거.
3. **predictive prototype** (transformer dynamics, RNN보다 선호) — codex 자문서 주력 비추천(위 1번). 보조로
   "새 segment prior 약하게 조정" 정도가 한계. 무학습 제약 충돌·짧은 segment 데이터 부족.

## 2.6 시도 #7 — reset-as-transaction (codex, 2026-06-15) + 정보-한계 종결

**먼저 oracle 재확인 (2026-06-15)**: §2.2 의 clean-oracle 수치(0.687/0.554)가 확정 채점기(`ami_scoring`)·
**raw-EWMA prototype** 에서 재현됨 — raw r_active 0.486, V_rel per-meeting oracle **0.706**, clean+단순 μ+cσ(c2.0)
**±2F1 0.626**. (세션 중 "clean 신호도 약함(0.31)" 주장은 prototype **매-step 정규화** 재구현 버그였고 철회.
재현·대조: `scripts/ami_clean_oracle_repro.py [--norm-proto]`, decision-log 2026-06-15.) → **신호=LLM급 확정, 병목=online reset.**

**시도 #7 설계 (codex)**: 불가역 hard-reset 을 **가역 split transaction** 으로. V>L 이면 τ 에 provisional 경계를
열고, old(a_old)/new(b) dual prototype 에 대한 응집 증거 S 를 SPRT bound 로 누적 → confirm(a←b)/cancel(a←a_old
replay-merge). τ 고정(위치 정정 없음 → commit-refine 의 lag-이득과 무관). 구현: `scripts/ami_reset_transaction.py`.

**결과 — 실패 (baseline 보다 나쁨)**: confirmed deploy ±2F1 0.006~0.09 (< hard-reset 0.15). 진단(아래)으로 원인 확정.

**정보-한계 진단 (`scripts/ami_reset_discriminator_diag.py`, 확정 채점기, AMI 139)**:
1. **transaction 판별력 0**: provisional τ 가 gold(±2)인지로 가른 취소율 — **TRUE 0.82 / FALSE 0.82 (동일)**.
   dual-prototype SPRT 가 true/false 를 전혀 못 가림.
2. **판별자 AUC(true>false)** (provisional 12508, true율 5.4%): FROZEN old prototype 거리(0-lag류) **AUC 0.409(랜덤)**;
   τ-window 상호 응집 cos = **W=2 0.665 / W=3 0.666 / W=4 0.662 / W=6 0.649 / W=8 0.637 / W=12 0.614**.
   → 판별 정보는 **"스파이크 직후 1발화가 스파이크와 닮았나"(W=2)** 에 집중, 이후 희석. (codex 의 "pair 많을수록 sqrt(M) 상승"
   예측은 틀림 — pair 비독립.)  0-lag 거리는 랜덤.
   > ⚠ **정정 (2026-06-20, §"축2 재공략" B)**: 이 "W=2 0.665" 는 `load_ami` 의 미팅별 Frobenius 정규화 + cross-meeting
   > pooling 아티팩트. **미팅-내 정상 측정 시 1-lag 도 ≈0.52(랜덤)**. "1-lag 엔 정보 있음" 결론은 무효(0-lag·1-lag 모두 랜덤).
3. **최선 판별자(1-lag 응집)를 deploy** persistence-gate 로 실제 사용 → ±2F1 **0.133 (≈baseline)**. AUC 0.66 @ 5%
   base rate 는 F1 으로 전환 안 됨 (절대 임계도 calibration-free 불가: int8 임베딩 내 발화쌍 cos 이 워낙 낮음).

**종결 판정 (정보 한계)**: online reset 부트스트랩은 영리한 임계·prototype·transaction 으로 넘을 벽이 아니라
**정보적으로 막힘** — 새 화제 onset 과 화제 내 outlier 가 *첫 발화에서 거의 구분 불가*(고정 임베딩·무학습·5% base rate).
0-lag 판별 AUC~0.41, lag 을 사도 천장 AUC~0.665(commit-refine 의 +0.03 과 일치), clean(0.55) 도달 불가.
codex 결론: 넘으려면 **더 강한 임베딩 / 학습된 판별자 / lag 허용** 중 하나 필요 — 셋 다 현 제약 위반.
→ **AMI online deploy = "신호 LLM급 / online reset 판별 정보-한계로 ±2F1 ~0.15 천장"인 구조적 한계 도메인.**
논문 limit 근거 = oracle-deploy gap + 판별자 AUC 곡선(0-lag 0.41 → 1-lag 0.665) + 1-lag deploy 천장 0.133.
**reset 부트스트랩 공략 종료** (8각도: A1~A5/BOCPD×2/EM/robust/commit-refine v1~v5/transaction 전부 같은 정보 한계).

---

## 부록 A — 전체 시도 ledger (축별, 빠짐 없이)
**축 1 (신호, oracle)**: V_rel / prototype 10종(mean·centroid·nn·medoid·window·content-weight·robust·subspace·
varnorm·info-gate) / de-neut(돌파) / fixed-a combo / **adaptive a_t reliability blend**(codex, z-score blend →
어중간한 중간으로 실패) / 부분 de-neut β / **R̄ 적응 β**(방향 거꾸로 실패) / **run-length 적응 β**(성공).
**축 2 (deploy)**: V_rel 적응임계치 / de-neut deploy / local-MAP A1~A5 / BOCPD particle / BOCPD lagged /
robust·peak·anchor·refractory / EM 반복정제. **진단**: z-score / picking / **gap 분해**(reset 병목 증명).
**검증**: 2-fold / LOO / deploy calib / Score-vs-c·Otsu / AUC.

**이 handoff scope 밖 (별도 문서)**: LLM 버퍼-지연 곡선(`outputs/experiments/2026-06-09_llm_buffer_curve/
REPORT.md`, figure_R), filler forward-merge/geometry/info-content/GraphSeg(`outputs/experiments/2026-06-09_ami_
filler_prototype/`, `outputs/reports/ami_*_view.md`, `run_graphseg_ami.py`) — AMI robustness 진단 단계.

## 부록 B — 산출물
- **축 1 스크립트**: `ami_dts_{deneut_oracle,beta_sweep,adaptive_beta,beta_overfit,beta_loo,auc,score_vs_c}_eval.py`
  (일부는 `_eval` 없음), `ami_dts_deploy_calib.py`.
- **축 2 스크립트**: `ami_vrel_eval.py`,`ami_vrel2_eval.py`,`ami_localmap_eval.py`,`ami_bocpd_eval.py`,
  `ami_bocpd_lag_eval.py`,`ami_adaptive_deneut_deploy.py`.
- **trace**(`outputs/reports/`): `vrel_segment_trace.md`,`vrel_compare3_trace.md`,`vrel_summary_build_trace.md`,
  `vrel_proto_content_c2.md`. **REPORT**: `outputs/experiments/2026-06-10_ami_vrel_localmap/REPORT.md`(V_rel까지).
- **decision-log**: 2026-06-10(V_rel), 2026-06-11(de-neut/적응β + calibration-free).

## 부록 B2 — SEM 계승 (정당성)
de-neut의 r_global 항 = "공통/background 대비 화제 변별" = SEM new-event **base distribution** 대비의 거리공간
근사. active prototype reset = SEM event 모델 신규 개시. run-length 가중 = event persistence 기반 reliability
weighting (SEM event-model reliability 원리와 정합; 단 직접 메커니즘 아닌 heuristic — codex 자문, decision-log
2026-06-11 기록). λ·g_rho·(A,B)는 calibration(overfit·LOO 검증), magic number 아님.

## 부록 C — 정직한 한 줄 결론
**축 1 성공**: de-neut + run-length 적응 β로 신호가 두 도메인 다 δ_eff를 넘고(superseg 벽 cross-domain robust 돌파),
V_rel은 oracle에서 LLM 천장도 넘음. **축 2 modest**: 그 신호를 online에서 실현하는 게 reset 부트스트랩에 막혀
deploy는 modest(commit-and-refine Score 0.401/±2F1 0.140, oracle 0.554 격차 미해결).

**→ 2026-06-11 승격 (`reset 부트스트랩 해결 전제'를 격차 명시 한계로 대체)**: 신호(universal) + commit-and-refine
(drift deploy)를 **[[hi-ontop-deneut]] (`src/hi_ontop/hi_ontop_deneut.py`) 정식 main 승격**. 범위 = 신호 universal /
commit-and-refine drift(AMI) / sharp(DTS) threshold 유지(`reset=` 인자). reset 기법 5각도(v1~v5) 모두 격차 못
메움 → 병목은 online 운영 구조적 한계로 결론. δ_eff streaming `class HiOnTop`은 Hi-OnTop 파이프라인용 유지.
REPORT `outputs/experiments/2026-06-11_ami_commit_refine/`. tiage tie 인정.

---

## 축 2 재투자 — DeNeut을 DTS deploy까지 (2026-06-13, [[HANDOFF_04]] 후속)

**문제 재정의**: δ_eff 는 *비교 baseline* (채택 X). 우리 방법 = **de-neut(`hi_ontop_deneut`)**. 그런데 de-neut 이
DTS deploy 에서 baseline δ_eff 에 짐 (de-neut 0.367/0.302/0.293 vs δ_eff 0.602/0.476/0.405, 공식 per-dialogue).
목표 = de-neut 을 DTS+AMI 둘 다 되게 (universal). codex 2회 자문 + 구현·실측.

**신호 진단 (공식 per-dialogue oracle, gold-reset)**: DTS 에서 de-neut 신호 자체가 δ_eff 에 짐
(dialseg 0.504 vs 0.720). 범인: ① λ·r_global 항(상수)이 sharp seam 깎음(+0.11 회복 가능), ② de-neut(global 제거)
+평균 prototype 이 직전점프 무디게(δ_ctx 0.778 vs de-neut 0.634). `scripts/diag_deneut_dts_signal.py`,
`diag_dts_threshold_isolation.py` (threshold 종류는 주범 아님: 러닝 μ+1.0σ ≈ 고정 percentile).

**해결 방향 (codex)**: 단일 신호 `S=(1−q)·S_sharp + q·S_drift`. S_sharp=δ_eff(무상태), S_drift=de-neut+적응β+V_rel.
게이트 q 를 **segment-age 대신 무상태 burstiness**(rolling δ_prev 의 heavy-tail+spike-isolation, window 64)로 →
과분절 악순환 회피. + SQD quarantine(emit 0-lag, drift prototype 은 격리·다음턴 검증 accept/rollback).

**oracle 결과 (tuned 게이트 SLOPE=2.0/BIAS=1.0 + adaptive-β drift, gold-reset, gate leak 없음)** — `diag_gate_tune_oracle.py`:
| | dialseg711 | tiage | superseg | AMI±2 |
|---|--:|--:|--:|--:|
| tuned 게이트 신호 | **0.711** | **0.648** | **0.591** | **0.307** |
| δ_eff baseline | 0.720 | 0.598 | 0.567 | 0.225 |
| de-neut | 0.504 | 0.487 | 0.544 | 0.370 |
→ **단일 신호가 DTS 3개 δ_eff 필적/초과 AND AMI δ_eff 초과 = universal 신호 oracle 달성.** (게이트 무상태→leak 아님.)

**deploy 결과 (gated + adaptive-β + SQD quarantine, c=1.0 calibration-free)** — `deploy_gated_quarantine.py`:
| | dialseg711 | tiage | superseg | AMI±2 |
|---|--:|--:|--:|--:|
| gated+quarantine | 0.485 | 0.371 | 0.326 | 0.094 |
| (이전 gated, no-Q) | 0.480 | 0.375 | 0.345 | 0.115 |
| oracle 천장 | 0.711 | 0.648 | 0.591 | 0.307 |
| δ_eff baseline | 0.602 | 0.476 | 0.405 | — |
→ **deploy 미해결.** 게이트가 DeNeut DTS 를 0.367→0.485 로 올렸으나(악순환은 깸) δ_eff baseline·oracle 천장에 못 미침.
**quarantine 효과 없음**(no-Q 와 동급/이하 — emit 된 FP 는 출력에 남아 점수 반영, 내부 rollback 은 cascade 만 막음).

**결론**: 신호(oracle) = universal 달성. **deploy = oracle 의 ~절반에서 벽**(dialseg 0.485/0.711, AMI 0.094/0.307).
universal-blend / SQD / 무상태-burstiness-gate / quarantine 모두 **oracle→deploy 부트스트랩 벽** 못 깸 (축2 5각도에 4각도 추가, 총 9각도 음성). 0-lag·calibration-free 제약 하 online 실현은 **여전히 미해결 구조적 천장**.
스크립트: `scripts/{diag_deneut_dts_signal,diag_dts_threshold_isolation,diag_gated_oracle,diag_gate_tune_oracle,deploy_universal_blend_test,deploy_sqd_test,deploy_gated,deploy_gated_quarantine}.py`.

### 축2 마감 — LOO 검증 후 현 디폴트 유지 (2026-06-13)
stateless(no-reset, sliding-window drift) + 무상태 burstiness 게이트가 test-sweep 에선 dialseg 0.544 까지
올랐으나, **(BIAS,Wd) 상수를 LOO cross-domain 검증**(`scripts/deploy_loo_validation.py`)하니:
- honest LOO deploy: tiage 0.401 / dialseg711 0.438 / superseg 0.335 / AMI±2 0.114.
- overfit gap: tiage 0.000·superseg 0.003·AMI 0.051 (robust) / **dialseg711 0.134 (일반화 실패)** — 다른 도메인 상수가
  dialseg 에 전이 안 됨. test-sweep 0.544 는 in-domain overfit 이었음.
- **여전히 δ_eff baseline(0.476/0.602/0.405) 못 넘음.** 현 디폴트 DeNeut(0.302/0.367/0.293)보다 DTS 는 낫지만 AMI 는 0.114<0.131.
**결정**: option 3(deploy 부트스트랩) 을 universal-blend/SQD/burstiness-gate/quarantine/stateless-no-reset + LOO 까지
시도 — 모두 baseline 미달. 사용자 기준("더 이상 안 되면 현 디폴트")에 따라 **현 디폴트(de-neut threshold,
`hi_ontop_deneut.segment()`, AMI Score 0.372) 유지로 확정.** 코드 변경 없음. DTS<baseline 은 정직 disclosure.
oracle→deploy 부트스트랩은 미해결 구조적 천장으로 동결. 신호(oracle universal)는 별건 자산으로 보존(§축2 재투자).

---

## 축2 재공략 — 벽은 표현이 아니라 reset 결정 (2026-06-20)

사용자 지시로 reset 부트스트랩 벽을 다시 팜. **세 결과: (a) 벽은 인코더-독립(표현 한계 아님), (b) HANDOFF_01
진단의 "1-lag 판별자 AUC 0.665" 는 정규화 아티팩트(실제 ≈랜덤), (c) codex 방향 = LLM-judge-at-spikes.**

### A. 인코더 강도 falsification (벽이 약한 int8 아티팩트인가)
이전 정보-한계 진단은 **MiniLM-int8 하나로만** 측정됐었음. 인코더 교체는 fine-tuning 아님(허용) → 세 체급 비교.
- 공통 setup: AMI 139(`data/ami/topic`, bnd_top=start-turn, shift0), 확정 채점기 `hi_ontop.ami_scoring`(±2 tol).
  oracle = gold-reset prototype + **raw-EWMA**(사용시점만 정규화, λ=0.6 g_rho=0.15 rho_min=0.05) + per-meeting 최적 임계.
  deploy = `hi_ontop_deneut.segment()` (default c=1.0, A=2.0, B=1.0). 판별자 = spike(μ+1.5σ)에서 onset(±2) vs outlier AUC.
- 임베딩 캐시: `outputs/runs/_misc/{ami_emb(minilm-int8 384d), ami_emb_mpnet(all-mpnet-base-v2 768d),
  ami_emb_te3large(openrouter/openai/text-embedding-3-large 3072d)}`. 생성: `scripts/gen_ami_emb_encoder.py`(로컬 ST),
  `scripts/gen_ami_emb_api.py`(Crts API). 측정: `scripts/probe_encoder_wall.py <emb_subdir>`.

| 지표 | MiniLM-int8 384d | mpnet 768d | **te3-large 3072d** |
|---|--:|--:|--:|
| oracle V_rel(β=0) ±2F1 | 0.706 | 0.715 | **0.770** |
| oracle de-neut(default) ±2F1 | 0.372 | 0.373 | 0.373 |
| **deploy c=1.0 ±2F1** | 0.131 | 0.139 | **0.140** |
| deploy c=1.5 ±2F1 | 0.106 | 0.123 | 0.125 |
| **판별자 0-lag AUC** | 0.476 | 0.503 | 0.488 |
| 판별자 1-lag(W2) AUC | 0.519 | 0.510 | 0.488 |

→ **인코더를 키우면 oracle 천장은 오르나(0.706→0.770) deploy·판별자는 불변.** 강한 인코더는 oracle–deploy 격차를
오히려 **벌림(0.77 vs 0.14 = 0.63)**. **병목은 표현(embedding)이 아니라 online reset 결정** 확정. 인코더 축 종결.

### B. 정규화 아티팩트 정정 — "1-lag 판별자 AUC 0.665" 는 측정 오류
`scripts/ami_reset_transaction.py::load_ami` 의 `_nrm(e)=e/np.linalg.norm(e)` 는 2D 배열에 **per-row 단위정규화가
아니라 미팅별 Frobenius 스칼라(≈1/√n)** 로 나눔. 미팅 길이(76~774)가 제각각이라 여러 미팅 spike 를 pooling 한
AUC 에 미팅-길이 누수가 섞임. → §2.6 의 "τ-window 응집 W=2 AUC 0.665"(commit-refine lag 이득·정보 한계 결론의 핵심 기둥)는 아티팩트.

| (minilm-int8) | POOLED W2 AUC | per-meeting-mean W2 AUC |
|---|--:|--:|
| Frobenius(원본 diag2) | 0.665 | **0.526** |
| per-row(표준) | 0.519 | 0.524 |

→ 미팅 내 정상 측정 시 **1-lag 판별자도 ≈0.52(랜덤)**, 양 인코더 동일. "1-lag 엔 판별 정보 있음(0.665)" 은 거짓.
(단 clean oracle 0.69~0.77 은 유효 — gold-reset 이라 미팅-내 측정. §2.6 의 deploy 결론 자체는 불변.)

### C. 재구성 + codex 방향 (2026-06-20)
- **벽의 재구성**: "0-lag 가 경계를 못 봄" 이 아님(clean oracle 0.71~0.77 이 반증). 벽 = ① **비가역 commit/오염**
  (경계 놓치면 두 화제가 한 prototype 에 평균, 발화단위 회복 불가) + ② **첫 발화 모호성**(onset vs outlier 가 그 시점엔
  정보적 구분 불가, 지속성은 미래 필요). 둘 다 인코더 무관.
- **codex:rescue 자문 (high reasoning, 2026-06-20)**: 우선순위 **1→비파괴 상태관리→2→3→4**.
  1. **spike-gated LLM binary judge (1순위)** — cosine 판별자(랜덤 0.49) 대신 spike(~5%)에서만 LLM 에 raw text +
     최근 k턴 + active topic running summary 를 주고 `change/continuation/unsure` 판정. *전체 분절을 LLM 에 맡기지
     말고* 좁혀진 후보의 binary 만. (full-context LLM 분절 천장 ±2F1 ~0.54 = 현 0.14 의 4배.)
     **선결**: spike 후보의 gold ±2 recall 상한 먼저 측정(낮으면 임계 낮춰 10~15% 로 확대). perfect judge 상한 = 후보 recall.
  2. **★ label commit ↔ state reset 분리 (핵심 구조변경)** — 출력 label 은 LLM score 로 즉시 결정하되, 내부 memory 는
     단일 active prototype 으로 확정 금지. spike 마다 H0(continuation, 현 발화 outlier 가능→저weight/quarantine) +
     H1(현 발화 newborn seed) **이중상태 보존**, 이후 몇 턴 내부 검증(0-lag 라 과거 출력 label 은 불수정).
     → 첫발화모호성=LLM 공격, commit오염=상태관리 공격. (기존 reversible beam 실패 원인 = emission 이 여전히 embedding 거리 중심이라 같은 벽.)
  3. cross-encoder/NLI = 단독 1순위 아님(관련성≠화제전환), **cascade 용 2순위**(cheap score → 애매한 spike 만 LLM, 호출률 5%→1~2%).
  4. lag 허용/학습 판별자 = 강하나 **별도 relaxation 트랙**(main=strict 0-lag·no-train, relaxation study 로 "성능차=표현력 아닌 관측가능성" 논증).
  5. negative result 마감은 **LLM 0-lag judge 까지 실패해야** 정당(현재는 시기상조). 실패 시 결론 = "0-look-ahead onset 판정은 첫 발화서 정보부족, embedding 강도·online Bayesian 으로 회복 불가, raw-text reasoning/supervised prior/lag 필요".

### D. spike 후보 recall 상한 (2026-06-20) — LLM-judge 천장 확인
codex 선결과제: perfect judge 천장 = 후보집합 gold ±2 recall. 후보 = `hi_ontop_deneut.segment(e, c, R)` spike.
`scripts/measure_spike_recall_ceiling.py` (AMI 139, gold 938, ami_scoring shift0 ±2tol):

| (minilm-int8, R=4) | 후보율=LLM예산 | recall 상한 | perfect-judge ±2F1 | #후보 |
|--:|--:|--:|--:|--:|
| c=1.0 (현 deploy) | 7.7% | 0.364 | 0.482 | 4105 |
| **c=0.5** | **16.7%** | **0.716** | **0.809** | 10058 |
| c=0.0 | 21.1% | 0.892 | 0.936 | 13215 |
| c=−0.5 | 23.0% | 0.953 | 0.973 | 14496 |

- **현 deploy(c=1.0)가 막힌 핵심 = 후보가 sparse(recall 0.36).** 후보를 ~17%(c=0.5)로 넓히면 perfect-judge 천장
  **±2F1 0.81** (현 0.14 의 6배, full-context LLM 0.54 도 초과). → **후보생성은 병목 아님; 병목 = 넓힌 후보의 ~91% false 기각.**
- te3-large 도 거의 동일(c=0.5 recall 0.709) → **후보생성도 인코더 무관** (cheap minilm 충분).
- caveat: perfect-judge F1 은 precision→1 가정. c=0.5 후보 10058 vs gold 938 = **false:true ≈ 9:1**. 실현 F1 은 LLM judge 의 불균형 기각 정밀도가 결정.

### E. spike-gated LLM judge 프로토타입 (2026-06-20) — 0-lag 벽 실질 돌파
후보 = `hi_ontop_deneut.segment(e, c=0.5, R=4)` spike(minilm-int8). 각 후보에서 LLM 에 **과거 발화만**(0-lag) raw text
(speaker 포함) 주고 NEW/SAME binary 판정 → pred=NEW. `scripts/llm_judge_spikes.py` (Crts API, 디스크캐시+스레드,
모델/맥락 k/c 인자). 채점 `ami_scoring`(shift0 ±2tol). 크기-stratified 미팅 표본.

| 모델 / 맥락 | 표본 | cand precision(±2) | **±2F1** | Score |
|---|--:|--:|--:|--:|
| gpt-4o-mini / k=12 | 8미팅 | 0.22 | 0.156 | 0.37 |
| gpt-4o-mini / k=30·60 | 8미팅 | 0.19·0.24 | 0.13·0.15 | — |
| gpt-4o-mini / full-past | 8미팅 | 0.32 | 0.144 | 0.41 |
| gpt-4o / k=12 | 8미팅 | 0.31 | 0.237 | 0.45 |
| gpt-4o / full-past | 8미팅 | 0.43 | 0.401 | 0.60 |
| **gpt-4o / full-past** | **20미팅** | **0.34** | **0.325** | **0.540** |

- **두 레버가 결정적: 모델 강도(mini→4o) + 맥락(k짧→full-past).** 약한모델·짧은맥락은 ±2F1 0.13~0.16(임베딩 deploy 동급,
  codex 의 "0-lag 정보부재" 시나리오), **강한모델+full-past 는 ±2F1 0.33 / Score 0.54** (deploy 0.13 의 2.5×, **0-lag 유지**).
  (8미팅 0.40 은 표본운; n=20 의 0.325 가 신뢰 추정.)
- **→ 0-lag 벽은 근본 정보한계가 아니라 "약한 판별자" 한계였다.** 강한 raw-text reasoner 를 spike 에 넣으면 깨짐. Score 0.54 =
  offline full-context LLM(`ami_llm_full_139`, qwen3.5-27b, 미래 봄) 과 동급을 **0-lag 로** 달성. (단 §"인코더 falsification"
  의 "embedding online 은 표현 무관 0.14 천장" 결론은 불변 — LLM 은 *다른 관측축*.)
- 헤드룸: 후보 perfect-judge 천장 0.80, LLM 실현 0.325 = 40%. 잔여 = 프롬프트·reasoning 모델·후보예산(c↓)·label/state 분리(codex).
- **비용/한계**: full-past gpt-4o 를 후보(~17% turn)마다 호출 = 비쌈(cheap embedding 정체성과 트레이드오프). deploy 정의상
  0-lag·무학습은 충족하나 "무거운 LLM" 이라 §SEM 계승/경량 목표와 별개 트랙으로 위치 필요. n=139 full 미실행(비용).

### F. 원툴 universal LLM judge — AMI + DTS 단일 규격 (2026-06-20)
[[feedback_one_tool_no_per_dataset_tuning]]: 데이터별 튜닝 금지. **동일 config 로 4개 벤치**:
후보 = `hi_ontop_deneut.segment(emb, c=0.5)` (de-neut, 전역 c=0.5 1개), judge = gpt-4o full-past + 도메인중립 동일
프롬프트(NEW/SAME), 경계=NEW 후보. 채점만 벤치 규약(AMI shift0/±2tol `ami_scoring`, DTS shift-1/exact `dts_scoring`).
`scripts/llm_judge_universal.py --c 0.5 --n-ami 40 --n-dts 0`. (AMI 40미팅 표본, DTS 전체.)

| dataset | metric | 후보 | NEW | precision | F1 | Score | 후보천장 | deploy baseline |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| AMI | ±2tol | 2760 | 208 | 0.279 | 0.249 | 0.478 | 0.815 | 0.131 |
| tiage | exact | 135 | 40 | 0.425 | 0.072 | 0.312 | 0.098 | ~0.10 |
| dialseg711 | exact | 2649 | 809 | 0.661 | 0.263 | 0.471 | 0.276 | ~0.17 |
| superseg | exact | 1110 | 137 | 0.613 | 0.026 | 0.266 | 0.110 | ~0.09 |

- **judge 자체는 universal 하게 작동**: dialseg711 F1 0.263 ≈ 천장 0.276(**95% 실현**, precision 0.66) — deploy 0.17 대비 +0.09.
- **병목이 도메인별로 다름**: **AMI = judge-limited**(천장 0.81 의 31%만 실현; 도메인중립 프롬프트라 §E meeting프롬프트 0.325 보다↓ 0.249).
  **DTS-dense = 후보-limited**(천장 자체가 낮음). 특히 **superseg gold율 23%(조밀)인데 c=0.5 후보 6%만 발화 → exact 천장 0.11 붕괴**.
- **근본 원인 = exact-F1 + 단일 c 의 밀도 불일치 = c-tension 재현**(AMI 희소 vs DTS 조밀이 정반대 c 요구; [[HANDOFF_05]]). DTS 향상은
  judge 가 아니라 후보 recall/localization 문제인데 단일-c 원툴과 충돌. tiage/superseg 는 LLM 이 deploy baseline 도 못 넘음(천장 < baseline 영역).

**codex:rescue 방향 (gpt-5.5 high, 2026-06-20)**: 우선순위 **(c) framing 분리 → (a) 후보생성 재설계 → (b) AMI judge**.
- 핵심: **단일 c 로 경계밀도 맞추기는 무라벨 streaming 에서 원칙적 식별불가** → "더 똑똑한 threshold(Otsu/knee 재시도)"가 아니라
  **single-config adaptive proposal policy** 로. ① 고정 multi-scale 후보 union(embedding shift+prototype drift+re-entry+discourse marker,
  각 전역상수, 합집합) ② 온라인 self-calibration(데이터별 아님 — 스트림 *자기 과거* quantile/online-FDR/hazard → volatility·novelty 높은 stream 에서 후보율 자연↑)
  ③ binary spike→ranked proposal(judge 가 필터) ④ exact-localization 신호(derivative/CUSUM/abruptness — dense exact 는 seam 위치가 관건).
- framing: AMI(sparse+tolerant=judge 게임)·DTS(dense+exact=recall·localization 게임)는 operating point 정반대 → **regime 분리 보고**(단일 config 유지·명시).
- judge(2순위): compressed topic state+recent turns+rubric+reasoning model, 단 **모든 벤치 동일 프롬프트**.
- **진행중 (무료 우선)**: 단일-config multi-scale/online-quantile 후보생성기 구현 → `measure_*` 로 recall 천장 재측정(LLM 비용 0). 천장 오르면 그때 judge 재실행.
  n=139 full AMI 미실행(비용 ~$73).
스크립트(영속): `scripts/{probe_encoder_wall, gen_ami_emb_encoder, gen_ami_emb_api, reproduce_default_oracle_deploy, measure_spike_recall_ceiling, llm_judge_spikes, llm_judge_universal}.py`.

### G. 후보생성기 재설계 — 적응형 peak[δ_eff] (2026-06-20, 무료 recall-천장 단계)
codex 1순위(후보생성) 실측. `scripts/measure_candidate_generators.py` — 단일 config 생성기들의 perfect-judge F1 천장(LLM 0).

| 생성기 (단일 config, 전 벤치 동일) | AMI(±2) | tiage | dialseg711 | superseg |
|---|--:|--:|--:|--:|
| 옛 de-neut μ+cσ (c=0.5, deploy 검출기) | 0.81 | 0.10 | 0.28 | 0.11 |
| **peak[δ_eff] k=0.5 (적응형, 채택)** | **0.88** | 0.50 | **0.82** | 0.42 |
| FIX δ_eff top-q=0.25 (고정, 참조) | 0.80 | 0.60 | 0.84 | 0.50 |

- **μ+cσ 실패 진단(데이터 확정)**: dense stream(superseg gold율 23%)은 변동성↑→σ↑→임계↑→발화↓(6%) = **적응 방향이 거꾸로**.
- **채택 = `peak[δ_eff]`**: δ_eff 의 prominent peak(s_t>s_{t-1} AND s_t−med_W > k·1.4826·MAD_W, W=20 robust rolling baseline)만 후보.
  **발화율이 stream-적응(0.17~0.26), 고정 아님**(사용자 요구), 단일 config·데이터별 튜닝 0. 옛 대비 천장 전부↑(tiage 5×/dialseg 3×/superseg 4×/AMI 유지).
- **정직한 한계(codex unidentifiability 확정)**: superseg(조밀+exact, segment ~4턴)는 적응형·고정 둘 다 최약(0.42/0.50) — seam 이 국소 outlier 로 안 튐. 무라벨로 밀도 식별 불가의 실증. (옛 0.11 → 0.42 로 4× 개선이나 천장 자체가 regime 한계.)
- **다음**: 이 적응형 생성기로 LLM judge 재실행. 1차는 저비용(gpt-4o-mini, AMI n=20+DTS full, ~$1) 으로 생성기 효과 확인 → 좋으면 gpt-4o 본 수치.
스크립트 추가(영속): `scripts/measure_candidate_generators.py`, `llm_judge_universal.py --gen peak`.

### H. 적응형 생성기 + LLM judge 실측 — 새 생성기가 압도 (2026-06-20)
`llm_judge_universal.py --gen peak --peak-k 0.5 --n-ami 20 --n-dts 0 --model gpt-4o-mini` (저비용 ~$1.2).

| dataset | metric | deploy | §F 옛생성기+gpt-4o | **새 peak+gpt-4o-mini** | 후보천장 | 천장실현 |
|---|---|--:|--:|--:|--:|--:|
| AMI | ±2tol | 0.13 | 0.249 | **0.321** | 0.911 | 35% |
| tiage | exact | ~0.10 | 0.072 | **0.344** | 0.487 | 71% |
| dialseg711 | exact | ~0.17 | 0.263 | **0.747** | 0.817 | **91%** |
| superseg | exact | ~0.09 | 0.026 | **0.127** | 0.408 | 31% |

- **후보생성기 교체만으로 (같거나 싼 모델) DTS 3~4×↑**: dialseg711 0.263→**0.747**(천장 91% 실현, precision 0.84), tiage 0.072→0.344. **codex 1순위 적중.**
- **dialseg711/tiage = 사실상 judge-포화** (천장 근접) → cheap mini 로 충분. dialseg 0.747 은 offline TextTiling F1 ~0.245 도 크게 상회.
- **AMI = judge-limited** (천장 0.91 의 35%만, mini 약함) → gpt-4o 가 유일하게 가치 있는 곳.
- **superseg = hard regime** (천장 0.41 자체가 낮음, 조밀+exact unidentifiability). deploy 0.09→0.127.
- 전부 단일 config·적응형·0-lag·$1.2. → **벽(deploy ±2F1 0.13)은 two-stage(적응형 peak 후보 + LLM judge)로 실질 해소**, regime 별 도달 수준만 다름(codex framing).

**AMI gpt-4o (judge-limited 확인, `--only AMI --model gpt-4o`, n=20, ~$15)**: ±2F1 **0.363** / Score **0.503** (천장 0.91 의 40%).
mini 0.321 → gpt-4o 0.363 (**+0.04 뿐 — 큰 레버는 모델 아닌 후보생성기**). Score 0.503 ≈ offline full-context LLM 0.54 를 **0-lag 로** 달성.
AMI 는 강한 모델로도 judge-limited(40%) — 0-lag 회의 onset 판정의 본질적 난이도(offline 은 미래 봄). 잔여 헤드룸 = codex judge 개선(compressed topic state·rubric·reasoning model, 단일 프롬프트 유지).

**종합 (best config, 단일 도구 = 적응형 peak[δ_eff] 후보 + LLM judge, 0-lag, 데이터별 튜닝 0)**:
| | AMI ±2F1 | tiage exact | dialseg711 exact | superseg exact |
|---|--:|--:|--:|--:|
| embedding deploy(벽) | 0.13 | 0.10 | 0.17 | 0.09 |
| two-stage | **0.363**(4o) | **0.344**(mini) | **0.747**(mini) | **0.127**(mini) |
| 후보천장 | 0.91 | 0.49 | 0.82 | 0.41 |
regime: dialseg711=해결(천장 91%) / tiage=양호 / AMI=judge-limited(40%, judge 개선 여지) / superseg=hard(조밀+exact, 천장 자체 0.41).

### I. ★ 버퍼-LLM vs two-stage 정직 비교 (2026-06-20) — two-stage 우위는 AMI 뿐
"LLM 어차피 호출하면 버퍼와 뭐가 다른가" 검증. `scripts/compare_buffer_vs_twostage.py` (mini, AMI n=20·DTS 전체, Score 메인,
3축: Score / call-rate / 토큰 / emission-lag). 버퍼 = tumbling B-turn 윈도우(채울 때까지 lag, 윈도우 내 미래 봄).

| dataset | two-stage(0-lag) Score | buffer best Score | tok 비 | 결론 |
|---|--:|--:|--:|---|
| AMI | **0.451** | 0.280 (B=32) | two 60× 비쌈 | **two-stage 압승** (장거리 맥락 경계 — B=32 윈도우로도 0.28 천장) |
| tiage | 0.490 | **0.550** (B=8) | two 3× | 버퍼 우위 |
| dialseg711 | 0.796 | **0.886** (B=32) | two 6× | 버퍼 우위 |
| superseg | 0.326 | **0.607** (B=8) | two 2× | 버퍼 압승 (국소 seam, 후보게이트가 오히려 recall 제약) |

- **two-stage 정당화 = AMI 뿐** (sparse + 장거리 맥락 경계 + 0-lag). 버퍼는 B=32 윈도우로도 AMI 0.28 — full-past 판정만이 회의 화제구조를 봄.
- **DTS 3종은 버퍼가 Score↑·토큰↓·호출↓.** DTS 경계는 국소 sharp seam → 윈도우면 충분, 버퍼는 윈도우 내 전 턴을 봐 우리 candidate-recall 한계가 없음. **우리 후보게이트가 DTS 에선 방해.**
- **two-stage 약점 = full-past 토큰 (AMI 10558k vs buffer 165k = 60×).** 0-lag·call-rate 도 버퍼보다 불리(call/turn 0.18~0.26 vs 0.03~0.16). 유일 우위축 = 0-lag(lag 0 vs 3~15턴) + 장거리.
- **정리**: "0-lag 필요 + 장거리 경계(AMI)"에서만 two-stage 의미. lag 허용/국소 경계(DTS)면 버퍼-LLM dominates. → 후속 (a) full-past→bounded context 토큰 절감(AMI 품질 유지 검사).

### J. bounded context 토큰 절감 검토 → full-past 유지 결정 (2026-06-22)
`llm_judge_universal.py --ctx-k K` (judge 맥락 최근 K턴, 0=full-past). AMI n=20, mini:

| ctx_k | kTok | AMI F1 | AMI Score | DTS |
|---|--:|--:|--:|---|
| 0 (full) | 10188 | 0.321 | **0.451** | tiage .490 / dlseg .796 / super .326 |
| 64 | 2167 | 0.301 | 0.385 | **동일**(dialogue 짧아 k=64=full, 캐시 재사용) |
| 32 | 1182 | 0.304 | 0.379 | 동일 |
| 16 | 668 | 0.285 | 0.361 | 동일 |

- **토큰 비용 근원 = AMI 뿐**(긴 회의×full-past); DTS 는 원래 쌈(짧은 dialogue). global k 는 "짧은 대화 다 쓰고 긴 회의만 cap" = 자동 적응.
- k=64: AMI 토큰 4.7×↓ but Score 0.451→0.385(−0.066). bounded 여도 AMI 는 버퍼(0.280) 크게 상회(k=16 도 0.361).
- **결정 (사용자, Score 메인)**: **full-past 유지**(ctx_k=0 default). AMI Score 최대(0.451) 우선, 토큰 절감 보류. bounded 는 옵션으로 보존.

### K. ★ "20분 붕괴 = segment-all 태스크 결함" 확정 + 게이트 재검증 (2026-06-22)
사용자 지적: HANDOFF_03 v1/v3 버퍼 베이스라인은 매 호출 **"입력 전체를 분절(segment-all)"** 이라 잘못된 프레이밍.
올바른 = "맥락 주고 **현재 turn 이 경계인지 binary 판단**"(= 우리 two-stage judge). SeCom `baseline_segment_v1.md` 의
Context/criterion 유지 + Question 만 binary 로 최소변경 → `scripts/secom_prompts/baseline_segment_binary_v1.md`.

**실험 (AMI n=16, gpt-4o-mini, full-past, 0-lag, ami_scoring; `scripts/binary_breakpoint_fullpast.py`,
records `outputs/experiments/2026-06-22_binary_breakpoint_fullpast_ami/`)** — prefix 시간 bin 별 recall:

| prefix | 0-4분 | 4-10 | 10-20 | 20-30 | 30분+ |
|---|--:|--:|--:|--:|--:|
| recall | 0.643 | 0.889 | 0.933 | 0.941 | **1.000** |
| NEW율/turn | 0.11 | 0.17 | 0.18 | 0.28 | 0.35 |

- **★ 붕괴 없음 — recall 0.64→1.00 상승.** HANDOFF_03 §1.5 v1(segment-all): 0-4분 ~50% → 20분+ **0% 붕괴**.
  → **20분 붕괴는 긴 full-past 탓이 아니라 segment-all 태스크 결함.** binary 로 바꾸면 긴 맥락 OK.
- **단 반대편 문제 — 후반 과분절**: NEW율 0.11→0.35(맥락 길수록 "새 화제" 남발) → WD 0.781, every-turn 전체 **Score 0.358**.
  (v1 = under-predict 붕괴, binary = over-predict 과분절 — 방향만 반대인 calibration 실패.)
- **★ peak 후보게이트 재검증**: peak-gate(binary,mini) **Score 0.408 > every-turn 0.358**, **호출 4×↓**(1795 vs 7390).
  게이트가 후반 과분절 억제 → "게이트 무의미" 우려 반박, **올바른 프레이밍에서 게이트가 실제 도움.**
- 모델 주: 이번은 mini(붕괴측정 목적). best AMI Score = **0.503**(peak+gpt-4o, domain-neutral, n=20, §H). v1 0.50 은 qwen·conservative.
- **다음**: 후반 과분절(NEW율 폭증) 억제 = 남은 과제. v3(bounded)·last-64 보류(사용자).
  스크립트: `recompare_binary_judge.py`, `binary_breakpoint_fullpast.py`, `secom_prompts/baseline_segment_binary_v1.md`.

**프롬프트 분리 (2026-06-22)**: 과분절이 프롬프트 탓인지 모델 탓인지 점검. 우리 gpt-4o 캐시(domain-neutral, peak, n=20)
prefix-bin NEW율: 0.22/0.10/0.08/0.07/0.12 — **맥락 길어도 평평(상승 없음), WD 0.372 양호 → 우리 프롬프트+gpt-4o 는 과분절 안 함.**
즉 §K 과분절(NEW율→0.35, WD 0.78)은 **SeCom-binary+mini+every-turn 특유**. 단 모델(gpt-4o vs mini) confound 남음 →
**우리 프롬프트를 mini·every-turn 으로** 돌려 프롬프트 vs 모델 분리 진행중 (`binary_breakpoint_fullpast.py --prompt ours`).
부수 관찰: 우리 프롬프트는 보수적(NEW율 0.1, recall ~0.5 평평) — WD 안전하나 recall 천장(0.91) 못 채움 = 진짜 숙제.

### L. 본격 9셀 비교 (2026-06-22) — buffer-cadence가 peak 를 이김 (AMI)
k=1.0, AMI n=16, full-past, 0-lag, mini, Score 메인. **전부 캐시 재계산(LLM 호출 0)** — every-turn(ours·secom) 판정이
모든 turn 커버 → peak/buffer 는 부분집합. `scripts/compare9_k1.py`. trigger × prompt + 무-LLM floor.

| config | Score | F1 | Pk | WD | call/turn |
|---|--:|--:|--:|--:|--:|
| ours × buffer-10s | 0.462 | 0.284 | 0.317 | 0.405 | 0.254 |
| **ours × buffer-30s** | **0.461** | 0.218 | 0.276 | **0.315** | **0.106** |
| ours × peak k1.0 | 0.437 | 0.219 | 0.309 | 0.383 | 0.108 |
| ours × every-turn | 0.425 | 0.397 | 0.435 | 0.660 | 0.996 |
| secom × buffer-30s | 0.446 | 0.297 | 0.330 | 0.480 | 0.106 |
| secom × buffer-10s | 0.445 | 0.388 | 0.386 | 0.608 | 0.254 |
| secom × peak k1.0 | 0.416 | 0.246 | 0.354 | 0.475 | 0.108 |
| secom × every-turn | 0.358 | 0.356 | 0.501 | 0.781 | 0.996 |
| embedding-only k=1.5 (무-LLM) | 0.319 | 0.192 | 0.504 | 0.603 | 0.000 |

- **★ buffer-30s(0.461) > peak-k1.0(0.437), 동일 호출률(0.106≈0.108).** 고정 시간 cadence 가 적응형 peak 를 같은 비용에 이김.
  비결 = **WD 0.315(최저)**: 균등간격 예측이 AMI 의 대체로 규칙적 경계간격에 맞아 밀도 calibration 우수. (F1 은 peak 0.219 ≈ buffer 0.218 동급 → 차이 전부 Pk/WD.)
- **ours > secom 전 trigger** (프롬프트 일관 우위, SAME 기준 명시로 과분절↓).
- **embedding-only 0.319 = 바닥** → LLM judge 가 +0.10~0.14 실질 기여(raw 신호만으론 부족).
- every-turn 최악(과분절 WD 0.66~0.78).
- **함의**: AMI 에선 **적응형 peak 게이트가 최선 아님** — 단순 buffer-cadence 가 동일비용 우위(peak 의 "신호 솟는 곳" 이 경계 뭉침→WD 손해).
  ⚠ **AMI 한정·mini·n=16·±2tol**. DTS(dense+exact)는 후보 recall 이 중요해 cadence 가 exact seam 놓칠 수 있음 → 별개 검증 필요.
- 스크립트: `scripts/compare9_k1.py` (+ `binary_breakpoint_fullpast.py --prompt {ours,secom}` 가 캐시 선결).
