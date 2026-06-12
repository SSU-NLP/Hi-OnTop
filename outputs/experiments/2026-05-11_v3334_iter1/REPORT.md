# Sweep: 2026-05-11_v3334_iter1

Generated: 2026-05-11T03:01:58

Path: `outputs/experiments/2026-05-11_v3334_iter1/`

n_questions: **196** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | dormant_ev_rate | top_ev_promoted | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.3-4 (v3334_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.253 | 0.134 | 0.109 | 0.061 | 0.900 | 0.036 | 0.641 | 0.535 | 0.423 | 0.0269 | 0.960 | 0.045 | 29.5 | 21.2 | 19.0 | 69 | 28 | 256.7 | 9.65 | 1.52 | 8m 13s |
| hi-ontop-full-v3.3.3-4 (v3334_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=43 | 0.257 | 0.156 | 0.107 | 0.071 | 0.900 | 0.030 | 0.628 | 0.528 | 0.399 | 0.0259 | 0.961 | 0.038 | 25.3 | 19.5 | 17.5 | 39 | 28 | 43.8 | 9.89 | 1.63 | 9m 11s |
| hi-ontop-full-v3.3.3-4 (v3334_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=44 | 0.256 | 0.135 | 0.092 | 0.076 | 0.925 | 0.027 | 0.647 | 0.538 | 0.411 | 0.0252 | 0.945 | 0.071 | 26.0 | 20.7 | 18.9 | 33 | 28 | 45.5 | 9.89 | 1.77 | 8m 49s |
| hi-ontop-full-v3.3.3 (v333_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.256 | 0.154 | 0.072 | 0.054 | 0.950 | 0.028 | 0.333 | 0.252 | 0.180 | 0.0029 | 0.927 | 0.071 | 28.9 | 21.4 | 18.3 | 48 | 32 | 88.2 | 9.77 | 1.97 | 8m 13s |
| hi-ontop-full-v3.3.3 (v333_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=43 | 0.254 | 0.138 | 0.068 | 0.042 | 0.975 | 0.025 | 0.308 | 0.230 | 0.225 | 0.0030 | 0.951 | 0.032 | 25.8 | 18.6 | 17.8 | 37 | 25 | 52.6 | 10.00 | 1.91 | 9m 15s |
| hi-ontop-full-v3.3.3 (v333_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=44 | 0.254 | 0.139 | 0.057 | 0.051 | 0.975 | 0.026 | 0.327 | 0.240 | 0.195 | 0.0029 | 0.941 | 0.058 | 25.6 | 20.8 | 18.0 | 33 | 29 | 33.3 | 9.88 | 1.98 | 8m 47s |

**best (acc)**: `v3334_r2` — 0.257

---

## 1. 실험 setup

- **목적**: v3.3.3-4 (= v3.3.3 baseline + boundary hysteresis + (topic, episode) atomic retrieval + dormant LTM safety net + promotion_threshold=0.3) 가 v3.3.3 baseline 대비 R-mh@k 회복 + acc 향상을 만드는지 sanity check (iter1, 새 시리즈).
- **데이터**: `benchmarks/locomo/data/locomo10.json`, `--limit 200 --stratify` → 196 Q × 6 runs.
- **방법**:
  - `hi-ontop-full-v3.3.3` (baseline) × 3 seeds (42/43/44)
  - `hi-ontop-full-v3.3.3-4` × 3 seeds (42/43/44)
- **HP (모든 run 공통, code default)**:
  - segmenter: `cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, lmda=10, tau=50`
  - v3.3.3-4 자동 default: `restart_p_threshold=0.5, restart_prob_margin=0.15, episode_min_span=8, f0_proto_max=1, f0_proto_weight=0.0, retrieval_mode=episode_rerank, episode_top_k=3, dormant_ltm_top_n=8, promotion_threshold=0.3, rerank_query_weight=1.0, rerank_topic_weight=0.10, rerank_episode_weight=0.35, rerank_pe_penalty=0.05, rerank_recency_weight=0.03`
- **Seed**: numpy + torch seed (LLM 은 Crts proxy, temperature stochasticity 그대로).
- **Metric**: judge-based acc, dia_id 기반 H@k/R@k/R-mh@k/P@k, dormant evidence audit (`dormant_ev_rate`, `top_ev_topic_promoted`, `n_topics_with_ev`).
- **비교 baseline**: v3.3.3 (3 seeds).

## 2. 결과 (3-seed mean ± std)

| 지표 | v3.3.3 (×3) | v3.3.3-4 (×3) | Δ | σ-effect |
|---|---:|---:|---:|---:|
| accuracy_overall | 0.2547 ± 0.0010 | 0.2551 ± 0.0020 | +0.0004 | 0.2σ (noise) |
| H@k | 0.3226 ± 0.0109 | **0.6389 ± 0.0080** | +0.316 | 29σ |
| R@k | 0.2405 ± 0.0091 | **0.5336 ± 0.0042** | +0.293 | 32σ |
| R-multi-hop@k | 0.2000 ± 0.0186 | **0.4112 ± 0.0100** | +0.211 | 11σ |
| P@k | 0.0030 ± 0.0000 | **0.0260 ± 0.0007** | +0.023 | huge (~30×) |
| dormant_ev_rate | 0.9397 ± 0.0096 | 0.9555 ± 0.0074 | +0.016 | 1.7σ (slightly worse) |
| top_ev_topic_promoted | 0.0534 ± 0.0160 | 0.0513 ± 0.0138 | -0.002 | 0.1σ (noise) |
| n_topics_with_ev | 1.872 | 1.876 | +0.004 | noise |

best (acc): v3334_r2 — 0.2573 (동질 noise 범위).
best (R-mh@k): v3334_r1 — 0.4234.

## 3. 해석

**retrieval 단**: H@k +97%, R@k +122%, R-mh@k +106%, P@k +800%. 효과 크기가 std 의 10~30× 로, noise 가 아닌 method 효과 명백. dia_id 기반 매칭이라 metric 신뢰도 높음.

**STM promotion 자체는 변화 없음**:
- `top_ev_topic_promoted`: 0.0534 → 0.0513 (사실상 동일).
- `dormant_ev_rate`: 0.9397 → 0.9555 (오히려 0.016 증가).
- promotion_threshold 0.5 → 0.3 으로 낮춰도 정답 topic 의 STM 진입 비율이 회복되지 않음. **importance score 정책 자체가 evidence-bearing topic 을 STM 못 올린다는 가설 직접 확인.**

**retrieval 향상의 원천**: STM promotion 변화 없는데 H@k 가 2 배 → 향상의 거의 전부가 **dormant LTM safety net 효과** (STM 우회 LTM 직접 cosine retrieval). episode rerank 의 기여는 STM 안에서의 turn 선택 개선 정도 (STM cap 안 turn 분포가 소폭 개선될 수 있음).

**accuracy 비변화 (0.2547 vs 0.2551)**: 정답 turn 이 prefill 에 들어왔는데 LLM 이 답을 못 냄. 가능 원인:
1. **prefill dilution**: dormant_ltm_top_n=8 + episode_top_k=3 으로 prefill 안 turn 수가 증가, LLM 이 noise turn 들 사이에서 정답을 못 골라냄.
2. **LLM 한계**: LoCoMo 의 정답 turn 만 줘도 LLM (Crts qwen 3.5-9B) 이 답을 못 내는 문제 존재 가능.
3. **multi-hop reasoning gap**: multi-hop 정답 turn 들이 prefill 에 다 있어도 chaining 을 못 함.

`accuracy_by_qtype` 보면 v3.3.3-4 의 multi-hop 0.142 vs baseline 0.144 (동일), single-hop 0.103 vs 0.066 (+56% 상대 개선) — single-hop 은 약하게 살아남.

## 4. 판정

**method 채택 X (acc 기준).** retrieval 가 폭증해도 acc 가 noise 밖으로 못 나오면 deployable 개선이 아님.

**다음 단계 (codex 판정 2026-05-11):**
1. **importance policy 재설계 우선** (broad sweep 보다 우선). 근거: promotion_threshold 0.5→0.3 완화 실패 + `compute_importance()` 가 query-time evidence relevance 신호 자체를 갖고 있지 않음.
2. **보조 sweep**: `stm_max_topics` (1순위, cap 영향 직접) + `promotion_threshold` (sanity 한정).
3. **LLM-side dilution 검증 ablation**: `episode_top_k ∈ {1,2,3}` × `dormant_ltm_top_n ∈ {4,8}`. 최소 포인트: `episode_top_k=1, dormant_ltm_top_n=4`.
4. **구현 정정 먼저**: `_episode_rerank_prefill()` 의 weighted formula 가 문서대로 다 반영되는지 확인. 코드 vs 문서 불일치 시 우선 수정.

## 5. 한계 / 검증 미해결

1. **LoCoMo 200Q stratified**: 단일 conv 의 random subset 일 가능성. 1986Q full 에서 동일 패턴 나오는지 미확인.
2. **HP 미탐색**


: `episode_top_k`, `dormant_ltm_top_n`, rerank weight 5 종은 codex 권장 default 그대로 — grid 검증 안 됨.
3. **acc 정체 원인 미분리**: prefill dilution / LLM 한계 / multi-hop reasoning gap 셋 중 어느 게 주된 원인인지 ablation 미실행.
4. **Diagnostic counter 미연결**: segmenter 의 `_n_*` (assigns / label_change / repeat_wins / restart_wins / hysteresis_blocked) 가 round summary 에 emit 안 됨. boundary 발화 통계 직접 검증 못 함.
5. **single-hop 0.103 vs 0.066 (+56%)** 상대 개선의 통계적 유의성 미확인 (n_questions 작아 std 못 봄).
6. **dormant_ev_rate 0.94+ 의 baseline value 자체**: 모든 hi-ontop-full method 에서 0.93~0.96 으로 거의 균일. importance policy 가 아닌 *segmentation 단계에서 evidence-bearing turn 을 다른 topic 에 묶어버리는 가능성* 도 배제 못 함 (segmentation level cause).
