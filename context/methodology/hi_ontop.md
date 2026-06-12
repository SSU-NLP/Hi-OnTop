# Hi-OnTop — Hi-OnTop Dialogue Topic Segmenter (reduced-form main model)

`src/hi_ontop/hi_ontop.py` · 2026-05-22 신설 (v4.1.x dead-code audit 결과)

## 한 줄

Hi-OnTop v4.1.x 라인([[v4.1.1]] `HiOnTopSegmenterV411` / [[v4.1.3]]
`HiOnTopSegmenterV413`)을 dead-code 감사한 끝에, **출력에 실제로 영향을 주는
부분만** 남긴 정직한 최소 online segmenter. SEM2 스타일 기계(sticky-CRP
prior, 학습형 RNN, f0/restart, per-topic σ²)가 v4.1.3 default 에서
출력상 무의미함이 증명돼 제거됐고, 그 결과 Bayesian posterior argmax 가
**causal 거리 하나에 대한 단일 임계값 판정**으로 환원된다.

## 왜 — v4.1.x 환원 증명

2026-05-22 감사: v4.1.3 default config 에서 SEM2 기계 전부가 inert.

- **학습형 per-topic RNN** — `eta_prev=1` → δ_model 가중치 정확히 0,
  RNN forward/학습 안 일어남 (객체 lazy 생성이라 만들어지지도 않음).
- **f0 / restart / 재진입 분기** — `f0_min_starts ≥ 2` 순환 교착: 매
  boundary 가 새 topic_id 를 만드니 한 topic 이 episode-start 를 2번
  겪을 일이 없어 f0 가 영영 untrained → 분기 한 번도 발화 안 함.
- **per-topic σ² (scaled-inv-χ² posterior)** — 죽은 f0 likelihood 에만
  입력 → 어디에도 안 닿음.
- **sticky-CRP prior (α/λ) + 고정 σδ²** — `_scores` 의 prior-corrected
  fresh baseline `B0_t = Lδ(δ*)+log(prior 비율)` 설계상 repeat-vs-fresh
  argmax 에서 prior 항·σδ² 가 **정확히 상쇄**.

→ posterior argmax 가 증명 가능하게 `δ_eff ≥ δ*` 로 환원. 경험적으로도
위 HP 들은 각자 전 범위에서 byte-identical segmentation 출력을 낸다.
SEM2 기계는 *틀려서*가 아니라 *이 도메인·이 설정에서 증명 가능하게
degenerate* 해서 제거 — `archive/legacy_sem_ablation/` 에 실행 가능한
감사 계보로 보존.

## 수식

L2-정규화 발화 임베딩 stream `s_1, s_2, …` 에 대해, 매 turn:

```
c_{t-1}     = normalize( Σ_{i=1..min(m,t-1)} ρ^{i-1} · s_{t-i} )  # causal, past-only
δ_prev(t)   = 1 − cos(s_{t-1}, s_t)                  # 즉각 표면 변화 (짧은 척도)
δ_ctx(t)    = 1 − cos(c_{t-1}, s_t)                  # 인과윈도우 맥락 (긴 척도)
δ_eff(t)    = a·δ_prev + (1−a)·δ_ctx                 # 두 시간 척도 PE 누적
g_t         = δ_eff(t) / δ*                          # graded boundary score
boundary(t) ⇔ g_t ≥ 1 ⇔ δ_eff(t) ≥ δ*
```

- δ_ctx 가 없으면(t<2 등) δ_eff=δ_prev, δ_prev 도 없으면(stream 시작)
  δ_eff=0.
- boundary 시 segment id +1 (단조 증가 카운터, 비가역). 아니면 현 segment
  지속.
- **online**: 과거만 사용, look-ahead 없음. turn 당 O(m) ≈ O(1).
- graded score `g_t` 는 boundary 결정에 되먹임되지 않는 순수 read-out
  (binary `is_boundary` 와 별도 노출). Ben-Yakov & Henson 2018 의
  graded hippocampal boundary response 매핑.

## State 필드

- `_prev_s` : s_{t-1} (δ_prev 용)
- `_recent` : causal window 버퍼 (최근 ≤m 발화, δ_ctx 용)
- `_topic_id` : 현 segment id (단조 증가)
- `last_delta_eff` / `last_graded_score` / `last_is_boundary` : 직전 turn
  read-out
- `_history` : turn 별 `{turn, topic_id, is_boundary, delta_eff,
  graded_score}` (`history()` / `graded_scores()` / `boundary_strength()`)

## HP default

```
delta_star  = 0.5594   # δ*: 경계 임계값. 인코더·데이터셋·(m,ρ,a) 의존 → 재calibration 필수
ctx_window  = 2        # m:  causal 맥락 발화 수 (canonical TIAGE-cfg)
ctx_decay   = 0.7      # ρ:  기하감쇠 (최근 가중↑)
ctx_blend_a = 0.5      # a:  δ_eff = a·δ_prev + (1−a)·δ_ctx
```

- `ctx_window` default = **2**. 보고 수치(아래 결과)를 낸 canonical
  TIAGE-cfg 가 m=2. v4.1.x 코드 default 는 `3` 이었으나 보고 config 와
  어긋났던 것 → Hi-OnTop 는 main 모델로서 default 를 보고 config 에 맞춤
  (2026-05-22 정정).
- `δ*` = TIAGE-train prev-cos calibration 전이값. 인코더·코퍼스·(m,ρ,a)
  바뀌면 *train* split 에서 재추정 (test leakage 금지).
- 인코더 = multi-qa-mpnet (Hi-OnTop 기본).

## SEM 계승 3-step

1. **SEM 에 있나?** — δ_prev = SEM2 cold-start identity-dynamics PE
   그 자체. causal-window 는 SEM 의 history-conditioned f(x_{1:n}) 의
   근사. graded score 는 SEM2 Gaussian residual likelihood
   `Lδ(δ)=−δ²/(2σδ²)` 가 본질적으로 graded scalar 인 것의 표면 노출.
   **새 메커니즘 도입 0** — 전부 SEM2 구성요소의 환원.
2. **충돌?** — 없음. sCRP/Bayes/local-MAP 구조를 *제거*한 게 아니라,
   prior-corrected baseline 설계 덕에 그 일반 구조가 단일 부등식으로
   *collapse* 함을 증명한 것. boundary 는 여전히 generative PE 증가 시
   발생. causal(look-ahead 없음) → online SEM 정합. 폐기된 SEM2 기계는
   archive 에 실행 가능하게 보존(제거 아님).
3. **decision-log**: 2026-05-22 (v4.1.x dead-code audit → Hi-OnTop
   reduced-form main model, commit `326b86b`).

Hi-OnTop 는 SEM 핵심 직관 — "다음 관측이 최근 사건 맥락으로 잘 예측되지
않을 때 사건 경계" — 의 최소 online 구현이지 SEM2 의 충실한 포팅이 아님.

## 결과 (공식 SuperDialseg metric, Score=0.5F1+0.25(1−Pk)+0.25(1−WD))

v4.1.3 와 byte-identical (parity 검증 — 세 벤치 38,242 turn 중 boundary
예측 diff 0, `scripts/verify_hiontop_parity.py`,
`outputs/experiments/2026-05-22_hiontop_parity/`):

| 벤치(test) | Score | F1 | Pk | WD |
|---|---:|---:|---:|---:|
| TIAGE ⚠ | 0.4675 | 0.410 | 0.442 | 0.508 |
| Dialseg711 | 0.5897 | 0.549 | 0.325 | 0.415 |
| SuperDialseg | 0.4631 | 0.432 | 0.471 | 0.541 |

⚠ TIAGE-test = calibration source(TIAGE-train)와 같은 corpus → in-domain.
Dialseg711·SuperDialseg = zero-shot transfer. 세 벤치 모두 동일 인코더
단일-임계값 prev-cos 기준선 대비 일관 우위.

### graded score per-band precision (`2026-05-21_v413_demo`)

| dataset | very_weak(<0.7) | weak(0.7~1.0) | normal(1.0~1.3) | strong(≥1.3) |
|---|---:|---:|---:|---:|
| TIAGE | 0.000 | 0.238 | 0.345 | **0.520** |
| Dialseg711 | 0.000 | 0.129 | 0.383 | **0.800** |

→ graded score 가 calibrated boundary 신호. strong band 가 downstream
"즉시 commit" 정책의 정량 근거.

## 알려진 한계 / 검증 미해결

- **δ\* 가 TIAGE-train 전이값** — SuperDialseg validation 재calibration
  미완. F1-best δ\* ≠ Score-best δ\* 확인됨(F1-best 는 과분절 유발) →
  Score/WD-aware calibration 필요.
- graded band 임계(0.7/1.0/1.3)는 휴리스틱 — Ben-Yakov & Henson 실험
  데이터 직접 fit 아님. downstream consumer 실험으로 보정 필요.
- `very_weak` band 에 stream-start turn(g=0) 섞임 → boundary 통계 시
  turn 0 필터 권고.
- 인접-cosine unsupervised 천장 벤치별 상이(SuperDialseg≈F1 0.46,
  Dialseg711≈0.54). 문헌 SOTA(0.55~0.65)는 출처·split·supervised·인코더
  미확정 → 외부 초과주장 금지. 정합 인코더 ablation 미실행.
- wording 은 비슷한데 topic 이 바뀌는 경우(δ≈낮음인데 GT 경계)는 못 잡음
  — 인접 의미거리 신호 자체의 구조적 한계. 예상 현실적 F1 천장 ≈
  0.48~0.55.

## 고려 중인 변형

- **per-topic δ\*_k** — [[v4.1.1]] 시절 `v4.1.2-exp` 로 구현·검증, 표준
  DTS 대화 길이에서 안전 게이트 발동 시점이 너무 늦어 무이득 → 미승격.
  훨씬 긴 대화(segment ≫ N_min) 데이터에서만 재검토 가치.
- **frozen-pretrained f (δ_model 채널)** — v4.2.4/v4.3.x exp 라인에서
  DSE-BERT / DialoGPT NLL / ST5+head / CSM 을 δ_model 자리에 시도, 모두
  single-η mean-best 기준 net-neutral~negative → 미승격. Hi-OnTop 는
  δ_model 항 자체를 제거(η=1 고정)했으므로 재도입은 SEM 3-step + 별
  버전.

## 변경 이력

- **2026-05-22 신설** — v4.1.x dead-code audit(commit `326b86b`)으로
  `src/hi_ontop/hi_ontop.py` `HiOnTop` 도입. v4.1.3 와 output parity 검증
  (byte-identical, 38,242 turn diff 0).
- **2026-05-22** — `ctx_window` default `3 → 2` 정정. v4.1.x 코드
  default(3)와 보고 수치 산출 config(m=2)가 어긋났던 것을 main 모델
  default 로 일치. parity 재검증 통과.
