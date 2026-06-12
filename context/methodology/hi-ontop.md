# Hi-OnTop — Hi-OnTop Dialogue Topic Segmenter (main model)

`src/hi_ontop/hi_ontop.py` · `class HiOnTop` · 2026-05-22 · current main model

## 한 줄

v4.1.3 의 dead-code 를 전부 제거한 reduced form. v4.1.3 와 **matched HP 에서
byte-identical** (TIAGE/Dialseg711/SuperSeg 38,242 turn 0 mismatch 검증).
online, O(m)/turn, past-only causal-window cosine-distance threshold segmenter.

## 왜 만들었나

v4.1.3 (`HiOnTopSegmenterV413(HiOnTopSegmenterV411)`) 의 HP audit (2026-05-22)
에서 SEM2 machinery 가 default/canonical setting 에서 출력에 0 영향임이 실증됨:

- EventRNN — `eta_prev=1` default → δ_model weight 0, 학습/forward 안 함
- f0/restart/re-entry — `f0_min_starts≥2` circular deadlock 으로 영구 봉인
- SEM2 variance (per-topic σ²_k, scaled-inv-χ² posterior) — dead f0 로만 흘러감
- sticky-CRP `alpha` 및 canonical/default `lmda=10` — `_fresh_baseline_for_prev`
  의 prior-cancel 설계로 repeat-vs-fresh argmax 에서 상쇄. 단 `lmda=1`
  같은 낮은 stickiness 는 archived full form 에서 non-prev f0 fallback 과
  상호작용해 일부 출력 차이를 낼 수 있으므로 전역 dead 로 쓰지 않는다.
- `sigma_delta_c` / `var_likelihood_weight` / `pe_prior` / `cos_threshold` /
  `pe_threshold` / `hard_pe_fallback` / `min_transitions_for_pe` — 전부 dead
- RNN 구성 HP(`rnn_hidden_dim`, `rnn_lr`, `rnn_n_epochs`,
  `rnn_ready_min_transitions`, `rnn_max_history`, `seed`) 와 SEM2 variance
  세부 HP(`pe_var_sigma0_sq`, `pe_var_df0`, `pe_var_min_sq`,
  `pe_var_max_sq`, `pe_var_window`) 도 default `eta_prev=1` /
  f0-dead 경로에서는 출력 dead

→ v4.1.3 의 4단계 SEM2 파이프라인 (sCRP prior / RNN PE / σ²_k likelihood /
Bayes posterior argmax) 이 코드로는 실행되나, 수학적으로 단일 threshold 로 환원.
Hi-OnTop = 그 환원형을 dead code 없이 정직하게 구현.

## 알고리즘

L2-normalized 발화 임베딩 스트림 `s_1, s_2, …` 에 대해:

```
c_{t-1} = normalize( Σ_{i=1..min(m,t-1)} ρ^{i-1} · s_{t-i} )   # causal window
δ_prev  = 1 − cos(s_{t-1}, s_t)
δ_ctx   = 1 − cos(c_{t-1}, s_t)
δ_eff   = a · δ_prev + (1−a) · δ_ctx
g_t     = δ_eff / δ*                          # graded boundary score
boundary(t) ⟺ g_t ≥ 1  ⟺  δ_eff ≥ δ*
```

`topic_id` = monotonic segment counter (boundary 마다 ++; 재진입 없음).

### Edge cases (v4.1.3 parity)

- turn 0: `_prev_s` 없음 → `δ_eff = 0` → graded 0, boundary False, topic 0 생성.
- turn 1: `_recent=[s_0]` → `c_0 = s_0` → `δ_ctx = δ_prev` → `δ_eff = δ_prev`.
- `_recent` 는 ctx_window 크기로 capped (FIFO).

## HP (4개 전부 live)

| HP | 의미 | default |
|---|---|---|
| `delta_star` (δ*) | boundary threshold | 0.5594 (encoder/dataset calibration 필요) |
| `ctx_window` (m) | causal window 크기 | 2 |
| `ctx_decay` (ρ) | window geometric decay | 0.7 |
| `ctx_blend_a` (a) | δ_prev vs δ_ctx blend | 0.5 |

`beta` 등 v4.1.x 의 나머지 HP 는 Hi-OnTop 에 존재하지 않음 (dead 였음).

### δ* calibration

encoder-dependent. 측정값:
- bge/mpnet 계열 TIAGE-train prev-cos: ≈ 0.5594
- SeCom-swap (multi-qa-mpnet, MTB+ δ_prev p80, ctx_window=3): 0.6194
- SeCom-swap (multi-qa-mpnet, MTB+ **δ_eff p80, ctx_window=2**): 0.5983
  — δ* 는 δ_prev 가 아니라 δ_eff 에 적용되므로 ctx_window 변경 시 δ_eff 기준
  재보정이 원칙. ctx_window 3→2 default 변경에 따라 m=2 δ_eff 로 재산출한 값.
  `02_calibrate_delta_star.py --mode delta_eff --ctx_window 2` ·
  `delta_star_calibration_hiontop_m2.json`.

HP sweep (2026-05-22, `outputs/experiments/2026-05-22_v413_hp_sweep/`):
train/val+tune split 튜닝 → held-out/test 평균에서 swept config 가 canonical
default 를 못 이김 (swept −0.006 mean-3). **default HP 가 robust** 하다는
negative result 로 보고 가능. 단 dialseg711 은 official train split 이 없어
seeded tune/held-out split 을 썼으므로 full-test literature number 로 직접
인용하지 않는다.

## graded boundary score

`graded_score = δ_eff / δ*` 를 매 turn 노출 (Ben-Yakov & Henson 2018 의
graded hippocampal boundary response 매핑). bands:

| band | graded_score | downstream 권고 |
|---|---|---|
| very_weak | < 0.7 | 보류 |
| weak | 0.7 ~ 1.0 | within-segment |
| normal | 1.0 ~ 1.3 | 경계 |
| strong | ≥ 1.3 | 즉시 commit |

## SEM 계승 (정직한 서술)

Hi-OnTop 는 **full SEM2 구현이 아니다**. SEM 의 핵심 직관 — "event boundary 는
다음 관측이 최근 event context 로 잘 예측되지 않을 때 발생" — 의 *minimal online
realization*. SEM2-style RNN dynamics / sticky-CRP prior / f0-restart / variance
calibration 은 구현·audit 했으나 v4.1.3 default 에서 argmax 결정에 영향 없음
(→ `archive/legacy_sem_ablation/sem_core_v413.py` 에 ablation 증거물로 보존).
paper 는 이 audit 을 disclosure 로 명시.

## paper main claim

graded boundary score (calibrated) + online O(m)/turn latency +
SeCom LLM-segmentation-backend drop-in 교체 + segmentation latency 대폭 감소.

## API

```python
seg = HiOnTop(dim=768, delta_star=0.5594, ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
for s in scene_vectors:
    topic_id, is_boundary = seg.assign(s)
    g = seg.last_graded_score
seg.history()            # per-turn: turn/topic_id/is_boundary/delta_eff/graded_score
seg.graded_scores()
seg.boundary_strength()  # {very_weak/weak/normal/strong: count}
```

## 한계 / 검증 미해결

- 알고리즘이 causal-window cosine-distance threshold — TextTiling 류 lexical
  heuristic 과 *구조* 가 같음. 차이는 signal (contextual embedding cosine) +
  graded score + online latency. paper 에서 이 점 정직히 서술 필요.
- 신경과학 5-요소 (예측 event model / 경계 reset / LTM snapshot / re-entry /
  adaptive timescale) 중 "예측오류→경계" 1개만, degenerate 형태로 구현.
  biology revival 설계는 codex 스레드에 보존 (후속 버전 후보).
- δ* 외 HP 가 robust 하나, 이는 *현 데이터셋* 한정 — 다른 도메인 미검증.

## SeCom-swap 인코더 backend

`HiOnTopSecomSegmenter` (`secom_adapter.py`) 의 인코더는 두 contract 를 모두
지원한다 — `sentence-transformers` (`encode(list, batch_size=, normalize_
embeddings=, ...)`) 와 `hi_ontop.embedding` 의 `QueryEncoder`/`APIEncoder`
(`encode(list) -> L2-normalized`). `_encode()` 가 TypeError fallback 으로
분기.

SeCom-swap 의 segment 단계는 default 로 **Crts `/v1/embeddings` API**
(`APIEncoder`, `03_segment_v413.py --encoder_backend api`) 를 쓴다. Crts 의
`multi-qa-mpnet-base-dot-v1` 출력은 로컬 sentence-transformers 와 bit-identical
(cos=1.0 검증) → δ* calibration·segmentation 결과 불변, 로컬 CPU forward 만
API 로 offload (encode ~0.9 s/turn CPU → ~0.11 s/turn API). `benchmarks/` 는
읽기 전용.

## AMI 도메인 + V_rel 상대신호 탐색 (2026-06-10 → 2026-06-11 승격: [[hi-ontop-cr]])

> **상태 갱신 (2026-06-11)**: 본 탐색의 신호(de-neut + run-length 적응 β)와 drift deploy
> (commit-and-refine)는 **[[hi-ontop-cr]] (`src/hi_ontop/hi_ontop_cr.py`) 로 정식 승격됨**(신호=universal
> calibration-free, commit-and-refine=drift deploy best Score 0.401/±2F1 0.140). **단 online deploy 의
> oracle 격차(±2F1 0.140 vs 0.554)는 미해결 — "신호는 LLM급(oracle), online 실현은 구조적 천장"이라는
> 정직한 한계 위에서 승격**(reset 기법 5각도 모두 격차 못 메움, decision-log 2026-06-11). 아래는 탐색 기록.
> δ_eff streaming [[hi_ontop]] (`class HiOnTop`) 은 Hi-OnTop 파이프라인용으로 그대로 유지.

drift 형 회의 코퍼스(AMI 139미팅, gold top-level 경계 미팅당 5~7개로 극sparse)에서
current main 의 δ_eff threshold 가 약함(±2 tolerance F1 0.151 / Score 0.203, LLM
full-context 0.543/0.640 대비 큰 격차). 원인 규명 + 신호 개선을 codex(gpt-5.5) 위임으로 진행.

### 진단 — magnitude 단독의 한계
- LLM 이 경계 찍은 turn 에서 δ_eff z-score = +0.545 (random +0.021) → 임베딩 신호는 진짜
  화제전환에 솟음. 그러나 *가장 큰* δ_eff peak 를 top-K/threshold 로 뽑아도 LLM 일치 ~0.11.
- 즉 **가장 큰 cosine spike 는 경계가 아니라 noise**(화자전환·단발 이상치·인용). δ_eff
  magnitude 단독으로는 boundary-spike 와 noise-spike 를 분리 못 함.

### V_rel — active-event 대비 + background 상대거리
경계 판정을 "직전 발화 대비 거리(δ_prev)" 가 아니라 **"현재 화제(active event)가 설명
못 하는 정도, 단 background 대비 상대적으로"** 로 바꿈:

```
m_t       active event prototype (EWMA, 경계서 reset)   # "지금 화제" 요약
g_t       global running centroid (EWMA, g_rho=0.15)     # "최근 전체 흐름"
r_active  = 1 − cos(s_t, m_{t-1})
r_global  = 1 − cos(s_t, g_{t-1})
V_rel     = r_active − λ·r_global        (λ = 0.6)
```

직관: 진짜 경계 = active 화제선 멀지만 global 로는 평범(r_active↑, r_global↓) → V_rel 큼.
noise = 둘 다 멂 → V_rel 작음. 이게 boundary/noise 분별의 레버.

### 검증 (gold-reset oracle 천장, ±2 F1)
| 신호 | 천장 |
|---|--:|
| δ_eff (기존, prototype 오염) | 0.226 |
| raw r_active (clean prototype) | 0.488 |
| V_rel (λ=0.6) | 0.565 |
| V_rel + global-EWMA(g_rho=0.15) | **0.687** (>LLM 0.543) |

- **overfit 아님**: 2-fold 교차 격차 0.000, (λ=0.6, g_rho=0.15) 가 A/B/manifest12/held127
  전 split 동일 best.
- **격차 분해**: clean(gold-reset) prototype + 단순 μ+cσ(c=2.0) = ±2F1 0.554 (LLM급).
  → **신호·임계치 충분.**

### 미해결 병목 — online reset 부트스트랩
clean prototype 은 *경계서 reset* 으로만 유지되는데, online 에선 경계를 몰라 추측 reset →
틀리면 prototype 오염 → r_active 작아짐 → under-seg 악순환. 시도 전부 oracle 미달:
| deploy | pred | ±2F1 | Score |
|---|--:|--:|--:|
| V_rel 적응임계치(c=2.0) | 318 | 0.056 | **0.358** (deployable 1위) |
| BOCPD top-K particle filter (codex 설계) | 1408 | 0.069 | 0.305 |
- robust 갱신·peak gating·anchor·refractory·EM 반복정제·particle filter 모두 부트스트랩
  순환 못 깸. **deployable 최선 = 단순 V_rel 적응임계치 Score 0.358** (기존 0.203·
  TextTiling 0.209·even-spacing oracle 0.282 모두 상회).

### SEM 계승
V_rel 의 r_global 항 = SEM new-event base distribution 비교의 거리공간 근사 (codex 설계
`log p(x|new) − log p(x|active)` 의 reduction). active prototype reset = event 모델 신규
개시. λ·g_rho 는 calibration(overfit 검증 완료), magic number 아님. decision-log 2026-06-10.

### 포지셔닝
AMI = drift+sparse+gold-offset 3중 난점 도메인. even-spacing oracle 이 Pk/WD 지배(content
신호와 구조적 불일치). **robustness/한계 도메인으로 보고, DTS(concat-seam) primary 유지.**

상세 수치·전 ablation: `outputs/experiments/2026-06-10_ami_vrel_localmap/REPORT.md`.
스크립트: `ami_vrel_eval.py`, `ami_vrel2_eval.py`, `ami_bocpd_eval.py`, `ami_localmap_eval.py`.
**정식 hi_ontop 버전 승격 조건**: deploy 가 oracle 격차(reset 부트스트랩)를 메운 뒤.

## 변경 이력

- **2026-06-10 V_rel 탐색**: AMI 에서 δ_eff 한계 진단 → V_rel 상대신호(oracle 천장 0.687
  >LLM, overfit·제약 OK) → online reset 부트스트랩 병목 확정(deploy 미달). 정식 승격 보류.
  위 § 참조. decision-log 2026-06-10.
- **2026-05-22 신설**: v4.1.3 dead-code audit → reduced form 추출. v4.1.3 와
  byte-identical 검증. decision-log 2026-05-22 참조.
- **2026-05-22 후속**: `ctx_window` default 를 canonical reported config 에
  맞춰 `2` 로 정정. v4.1.x archive 와의 parity 는 matched HP 기준으로 검증.
- **2026-05-22 SeCom-swap**: segment 인코더를 Crts API backend 로 전환
  (위 § 참조). ctx_window 3→2 변경에 맞춰 δ* 를 m=2 δ_eff p80 = 0.5983 으로
  재보정.
