# Handoff — AMI/DTS 화제분절 신호 탐색 (Hi-OnTop)

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
(drift deploy)를 **[[hi-ontop-cr]] (`src/hi_ontop/hi_ontop_cr.py`) 정식 main 승격**. 범위 = 신호 universal /
commit-and-refine drift(AMI) / sharp(DTS) threshold 유지(`reset=` 인자). reset 기법 5각도(v1~v5) 모두 격차 못
메움 → 병목은 online 운영 구조적 한계로 결론. δ_eff streaming `class HiOnTop`은 Hi-OnTop 파이프라인용 유지.
REPORT `outputs/experiments/2026-06-11_ami_commit_refine/`. tiage tie 인정.
