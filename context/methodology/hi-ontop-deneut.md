# Hi-OnTop-DeNeut — de-neut + 적응-β 신호 + threshold deploy (drift/AMI 용)

`src/hi_ontop/hi_ontop_deneut.py` · `segment(emb)` · 2026-06-11 (2026-06-13 threshold-only)

## 한 줄
[[hi-ontop]] (δ_eff causal threshold) 의 후속 **drift/AMI 용** 분절 모델. **신호**를 δ_eff 에서
*de-neut + run-length 적응-β* 로 교체(calibration-free). **deploy = `threshold`(0-lag) 단독.**
신호는 AMI oracle 에서 δ_eff 초과(정상정렬 0.370 vs 0.225) — 그러나 online deploy 는 구조적 격차가
남음(정직한 한계, §5). **DTS 에서는 δ_eff(`HiOnTop`)가 우세 — CR 은 DTS 미적합** (§5 ★★, [[HANDOFF_04]]).
(`commit_refine` deploy 는 2026-06-13 폐기 → `archive/legacy_commit_refine/`, §2.)

## 1. 신호 (universal, 두 도메인, calibration-free)
발화 임베딩 `x`(인코더 고정 MiniLM-int8, 단위정규화), online·0-look-ahead·무학습.
- **active prototype** `m` = 현재 segment EWMA, **global** `g` = 전체 EWMA(g_rho=0.15).
  - **⚠ EWMA 정규화 규약 — 신호/oracle 측정 시 raw-EWMA 필수**: clean 신호·oracle 을 측정할 땐 `m`·`g` 를
    **raw EWMA** 로 누적하고 *사용 시점(cos 직전)에만* 정규화한다 — `m=(1−rho)·m+rho·x` 후 `cos(x, m/‖m‖)`.
    **매-step 정규화(`m=normalize((1−rho)·m+rho·x)`)로 재구현하면 prototype 이 중립점으로 깎여 V 가 deflate →
    clean oracle 이 ±2F1 0.69→0.32 로 붕괴**, 가짜 "신호 한계" 결론을 낳는다(2026-06-15 실제 발생한 재구현 버그).
    rho=`max(rho_min, 1/(k+1))`(k=reset 후 run length), g 는 gr=`max(g_rho, 1/(gk+1))`. 재현·대조:
    `scripts/ami_clean_oracle_repro.py [--norm-proto]`.
  - **단 deploy 코드(`_seg_threshold`, src line 66-67)는 현재 매-step 정규화를 쓴다 — 의도된 현실**: detected-reset
    오염이 표현 방식을 압도해 raw-EWMA 로 바꿔도 deploy ±2F1 은 0.14→0.16 (미미). 즉 매-step 정규화는 *clean 신호
    측정* 에선 치명적이지만 *deploy* 에선 거의 무영향(병목=reset 부트스트랩, [[HANDOFF_01]] §2.6). raw-EWMA 로의
    deploy 전환은 **소폭 open win(+0.02)** 이며 알고리즘 변경이라 codex 위임 + REPORT 대상(미적용).
- **de-neut**: `deneut(v) = normalize(v − β·(v·g)·g)` — global(중립) 성분 β 제거 → '화제의 변별적 방향'.
- **run-length 적응 β**: `β_t = clip(A − B·log(1+k/L0), 0, 1)`, k=현 segment 길이.
  짧은 segment(DTS sharp)→β→1(full de-neut), 긴 segment(AMI drift)→β↓(V_rel). **run-length 가 sharp/drift
  를 도메인 라벨 없이 자동 판별.**
- **신호**: `r_active = 1 − cos(deneut(x), deneut(m))`, `V = r_active − λ·(1 − cos(x, g))`, λ=0.6.
- **판정**: 적응 임계치 `μ + c·σ` (V 의 running mean/std, warmup=8). c 는 **calibration-free**
  (전 구간 baseline 이상; 고정 c=1.0 또는 Otsu).

근거(axis-1): V_rel/de-neut 은 **AMI** oracle 에서 δ_eff 초과(정상정렬 ±2F1 0.370 vs 0.225,
`scripts/ami_alignment_recheck.py`). ⚠ **단 HANDOFF_01 의 "superseg 벽 0.467 돌파(0.506) / 4-데이터셋
oracle strict" 등 DTS 주장은 off-by-one 버그 산물로 무효** (§5 ★★, [[HANDOFF_04]]) — 공식·정상정렬에서는
DTS 전부 δ_eff 우세.

## 2. Deploy = `threshold` (0-lag) 단독
`segment(emb)` = detected-reset hard reset: V>θ(적응 임계 μ+cσ) → 그 turn 즉시 경계 emit + active
prototype reset. **0-lag(다음 턴 즉시), 버퍼 없음.**

**AMI Score (확정 채점기, 정정 2026-06-15)**: **현 디폴트 c=1.0 = 0.282** (±2F1 0.131, pred 4105 = gold 938 의
~4배 과분절). **0.372 는 best-c=1.5** (±2F1 0.106, pred 884). 즉 이전 표기 "디폴트 0.372"는 오기 — 0.372 는
best-c 값. **c=1.0 은 AMI 비최적**(sparse → 높은 c 필요). [[HANDOFF_04]] §6.

### [DEPRECATED 2026-06-13] `commit_refine` (폐기)
bounded-lag commit-and-refine + split-gain b\* refinement. AMI Score 0.401 까지 올렸으나 **그 +0.029
우위가 전적으로 lag(L=8≈26s) 매입분**(L≤3 은 threshold 와 동급/이하)이라 0-lag 요구와 충돌 → **폐기**.

| reset | lag | AMI Score | ±2F1 | 비고 |
|---|--:|--:|--:|---|
| **`threshold`** c=1.0 (현 디폴트) | **0** | 0.282 | 0.131 | pred 4105 (4배 과분절) |
| `threshold` c=1.5 (best-c) | 0 | 0.372 | 0.106 | pred 884 ≈ gold 938 |
| ~~commit_refine L=8~~ (폐기) | ~26s | 0.401 | 0.140 | (best-c era) |

코드는 `archive/legacy_commit_refine/seg_commit_refine.py` 에 참고/재현용 보존. decision-log 2026-06-12(강등)·2026-06-13(코드 제거).

## 3. 정식 config (검증됨)
`DEFAULTS`: c=1.0, A=2.0, B=1.0, L0=8, λ=0.6, g_rho=0.15, rho_min=0.05, R=4, warmup=8
- **c**: calibration-free (전 구간 1.0~1.5 hard-reset 이상; 고정 c=1.0).
- **(A,B)=(2.0,1.0)**: 신호 단계 LOO cross-domain 검증 (handoff §1.5).
- (~~L=8, m_min=2~~ 는 폐기된 commit_refine 전용 HP — `archive/legacy_commit_refine/` 로 이동. AMI k-fold CV
  로 검증됐었으나 현행 threshold deploy 엔 미사용.)

## 4. SEM 계승
de-neut 의 r_global 항 = "공통/background 대비 화제 변별" ≈ SEM new-event **base distribution** 대비.
active prototype reset = SEM event 신규 개시. run-length 가중 = event persistence reliability.
(commit-and-refine 의 SEM local-MAP 매핑은 그 deploy 가 폐기되어 더 이상 적용 안 됨 — §2.)
SEM 철학(sCRP/Bayes/local MAP/scene dynamics)과 충돌 없음 — heuristic 근사임은 명시(decision-log 2026-06-11).

## 5. 알려진 한계 (정직)
- **★★ DTS 우위 주장은 off-by-one 버그 산물 — 무효 (2026-06-13, [[HANDOFF_04]])**: HANDOFF_01 의 DTS 축1
  결론("de-neut > δ_eff / superseg 벽 0.467 돌파")은 `ami_dts_*.py::oc()` 의 **경계 off-by-one**(신호 스파이크 t 를
  gold t 에 비교; 정상은 t→gold t-1, 끝-turn 규약) 때문이었다. **공식 채점(SuperDialseg `SegmentationEvaluation`,
  per-dialogue) + 정상정렬로 재채점하면 DTS 3개 전부 δ_eff > CR** (Score: tiage 0.465 vs 0.302, dialseg711 0.602 vs
  0.367, superseg 0.340 vs 0.293; `outputs/experiments/2026-06-13_dts_official_rescore/`). → **DTS primary 디폴트는
  δ_eff(`HiOnTop`); CR 은 DTS 미적합.** **단 AMI 우위는 실재** — 정상정렬 per-meeting oracle ±2F1 de-neut 0.370 >
  δ_eff 0.225 (shift 무관, `scripts/ami_alignment_recheck.py`). 즉 CR 의 정당 도메인은 **AMI/drift 한정**.
- **★ online deploy 격차 = 정보 한계로 종결 (2026-06-15, [[HANDOFF_01]] §2.6)**: deploy ±2F1 ~0.14~0.16 ≪ **oracle
  천장 0.55~0.69**(clean+μcσ/per-meeting, gold-reset, raw-EWMA — 확정 채점기 재현 `scripts/ami_clean_oracle_repro.py`).
  병목은 신호가 아니라(신호=LLM급) **online clean-reset 부트스트랩**. 8각도(local-MAP A1~A5 / BOCPD×2 / EM / robust /
  commit-refine v1~v5 / reset-as-transaction) 전부 실패. **원인=정보 한계**(`scripts/ami_reset_discriminator_diag.py`):
  새 화제 onset vs 화제 내 outlier 판별 AUC = 0-lag 0.41(랜덤) / 1-lag(W2 응집) 0.665 포화 → 최선조차 deploy ±2F1 0.133
  (≈baseline; AUC 0.66 @ 5% base rate → F1 전환 불가). 넘으려면 더 강한 임베딩 / 학습 판별자 / lag 허용 중 하나(현 제약 위반).
  → main 승격은 "신호는 LLM급(oracle), online 실현은 정보-한계 천장" 전제. AMI = 한계 도메인 포지셔닝.
- **DTS deploy 미정식화**: threshold 모드는 신호만 교체한 hard reset — DTS deploy 전체 비교는 별도(현재 AMI 중심).
- **streaming 클래스 미통합 (deferred)**: 본 모듈 batch `segment()` 는 **이미 online·0-look-ahead
  충족**(좌→우, bounded-lag emit) — 알고리즘 online 속성은 OK. 단 turn-단위 즉시 반환 streaming `assign()`
  API 는 미통합. 설계 질문 = commit-and-refine 이 b\* 를 lag 후 **retroactive emit** 하므로 `assign()`
  즉시-반환 API 와 불일치 → buffer-and-flush(확정 경계를 L-lag 후 방출) wrapper 필요(codex 위임 대상).
  필요 시 thin wrapper 로 붙임. [[hi_ontop]] (δ_eff streaming) 은 그대로 유지.
- **b\* 는 감지된 후보 위치만 정정**(놓친 경계 recall 은 못 살림 — 격차의 본질).

## 6. 변경 이력 / 후속 후보
- **2026-06-13 off-by-one 정정 + DTS 회귀 확정**: §5 ★★ 참조. DTS 우위 무효(δ_eff 우세), AMI 우위 유지.
  공식 채점기 = SuperDialseg `SegmentationEvaluation`([[HANDOFF_04]], `src/hi_ontop/dts_scoring.py`). decision-log 2026-06-13.
- 2026-06-11 신설·승격 (commit-and-refine, codex 자문 + v1~v5 실측). 상세 REPORT
  `outputs/experiments/2026-06-11_ami_commit_refine/REPORT.md`, decision-log 2026-06-11.
- 후속(수확체감 아닌 방향): (A,B)·m_min 자기보정(tuning 제거), streaming 클래스(lagged emission) 설계.
  단순 reset/prototype 미세조정은 종료(5각도 음성).
