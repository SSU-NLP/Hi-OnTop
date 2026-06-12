# Sweep: 2026-05-10_v33x_2_iter1

Generated: 2026-05-10T21:02:12

Path: `outputs/experiments/2026-05-10_v33x_2_iter1/`

n_questions: **196** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rag-observation (rag_obs_r1) | rag_k=10 | 0.266 | 0.133 | 0.089 | 0.078 | 0.975 | 0.029 | 0.647 | 0.524 | 0.422 | 0.0828 | - | - | - | - | - | - | 0.00 | 1.97 | 6m 5s |
| rag-observation (rag_obs_r2) | rag_k=10 | 0.267 | 0.136 | 0.088 | 0.083 | 0.975 | 0.030 | 0.647 | 0.524 | 0.422 | 0.0828 | - | - | - | - | - | - | 0.00 | 1.73 | 6m 11s |
| rag-observation (rag_obs_r3) | rag_k=10 | 0.267 | 0.137 | 0.097 | 0.100 | 0.950 | 0.029 | 0.647 | 0.524 | 0.422 | 0.0828 | - | - | - | - | - | - | 0.00 | 1.79 | 6m 0s |
| hi-ontop-full-v3.3.3-2 (v3332_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.250 | 0.124 | 0.058 | 0.043 | 0.975 | 0.027 | 0.333 | 0.235 | 0.185 | 0.0031 | 27.2 | 22.9 | 18.3 | 53 | 40 | 121.6 | 9.66 | 2.48 | 25m 39s |
| hi-ontop-full-v3.3.3-2 (v3332_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.260 | 0.146 | 0.059 | 0.043 | 1.000 | 0.027 | 0.314 | 0.226 | 0.177 | 0.0028 | 26.3 | 22.0 | 17.9 | 43 | 29 | 69.0 | 9.76 | 2.48 | 8m 58s |
| hi-ontop-full-v3.3.3-2 (v3332_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.254 | 0.133 | 0.064 | 0.050 | 0.975 | 0.026 | 0.321 | 0.242 | 0.193 | 0.0029 | 25.3 | 21.5 | 17.3 | 30 | 29 | 25.5 | 10.00 | 2.17 | 8m 41s |
| hi-ontop-full-v3.3.3 (v333_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.253 | 0.133 | 0.063 | 0.045 | 0.975 | 0.024 | 0.308 | 0.228 | 0.192 | 0.0029 | 25.8 | 17.9 | 17.4 | 37 | 21 | 43.5 | 10.00 | 2.62 | 7m 43s |
| hi-ontop-full-v3.3.3 (v333_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.262 | 0.151 | 0.070 | 0.044 | 1.000 | 0.024 | 0.314 | 0.229 | 0.211 | 0.0029 | 27.3 | 21.7 | 19.1 | 40 | 35 | 60.0 | 9.65 | 3.14 | 8m 6s |
| hi-ontop-full-v3.3.3 (v333_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.248 | 0.126 | 0.073 | 0.049 | 0.950 | 0.020 | 0.308 | 0.221 | 0.199 | 0.0028 | 24.1 | 18.8 | 17.4 | 32 | 25 | 39.1 | 10.00 | 2.38 | 7m 32s |
| hi-ontop-full-v3.3.4-2 (v3342_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3 | 0.254 | 0.149 | 0.032 | 0.041 | 1.000 | 0.024 | 0.250 | 0.153 | 0.194 | 0.0019 | 61.1 | 51.8 | 50.2 | 67 | 63 | 36.2 | 3.55 | 2.30 | 8m 10s |
| hi-ontop-full-v3.3.4-2 (v3342_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3 | 0.253 | 0.147 | 0.029 | 0.040 | 1.000 | 0.025 | 0.250 | 0.156 | 0.189 | 0.0020 | 60.7 | 53.9 | 50.8 | 69 | 63 | 27.5 | 3.06 | 2.80 | 8m 48s |
| hi-ontop-full-v3.3.4-2 (v3342_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3 | 0.252 | 0.163 | 0.030 | 0.042 | 0.975 | 0.024 | 0.263 | 0.161 | 0.198 | 0.0019 | 62.2 | 52.2 | 52.1 | 67 | 65 | 18.8 | 3.11 | 2.41 | 7m 48s |
| hi-ontop-full-v3.3.4 (v334_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3 | 0.246 | 0.136 | 0.034 | 0.040 | 0.975 | 0.024 | 0.276 | 0.187 | 0.208 | 0.0021 | 47.4 | 46.2 | 43.4 | 52 | 51 | 18.8 | 4.81 | 2.68 | 8m 17s |
| hi-ontop-full-v3.3.4 (v334_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3 | 0.243 | 0.130 | 0.030 | 0.031 | 0.975 | 0.026 | 0.244 | 0.154 | 0.177 | 0.0018 | 47.6 | 46.5 | 44.4 | 51 | 50 | 17.9 | 4.81 | 2.46 | 8m 22s |
| hi-ontop-full-v3.3.4 (v334_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3 | 0.246 | 0.141 | 0.030 | 0.037 | 0.975 | 0.024 | 0.244 | 0.155 | 0.183 | 0.0018 | 47.5 | 46.9 | 44.4 | 51 | 51 | 19.6 | 4.81 | 2.72 | 8m 28s |

**best (acc)**: `rag_obs_r3` — 0.267

---

## 실험 setup

- **목적**: v3.3.3-2 / v3.3.4-2 강화안이 v3.3.3 / v3.3.4 base 대비 LoCoMo 에서 retrieval / acc 향상이 있는지 1차 검증.
- **데이터**: LoCoMo `benchmarks/locomo/data/locomo10.json`. `--limit 200 --stratify` (10 conv × 5 cat × per_cat ≈ 4) → 196 question per run (uniform across runs).
- **방법 (5개 × 3 run)**: rag-observation (baseline), v3.3.3, v3.3.4, v3.3.3-2 (강화), v3.3.4-2 (강화). 각 method 3 run (`r1/r2/r3`) — LoCoMo 의 stochasticity 는 LLM temperature 만 (`run_experiment.py` 주석 참조: `data_seed=None, sampling_seed=None`). 3-run 평균은 LLM 응답 분산을 보고함.
- **HP**: 모든 v3.3.x 는 `cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3` (LoCoMo 추천 default). v3.3.4 / v3.3.4-2 는 `pe_threshold` 무시 (calibrated likelihood). v3.3.3-2 는 prototype f0 + posterior odds default (`f0_proto_max=4, restart_p_threshold=0.35, restart_pe_min=0.0`). v3.3.4-2 는 σ² shrinkage default (`pe_var_shrink_c=8.0, pe_var_robust=False`).
- **Seed / RNN init**: argparse 에 `--seed` 없음. PyTorch global RNG 가 process 시작 시점에 따라 다르므로, 같은 method 의 3 label (r1/r2/r3) 이 별도 process 로 돌면 weight init 이 다르게 됨.
- **Workers**: `--workers 50` (concurrent LLM call). gen_p50 ≈ 2.4s, latency p50 ≈ 119s/Q (queue effect). wall ≈ 8m/run.
- **LLM / Encoder**: Qwen3.5-9B (Crts proxy, `api.ssunlp.co.kr/v1`) + bge-base-en-v1.5 (api). GPU 는 Crts 서버 사용, 로컬 GPU 미사용 (driver 11040 → CPU fallback for EventRNN).

## 결과 요약 (3-run mean ± std)

| method | acc | acc_mh | H@k | R@k | R-mh@k | P@k | T1μ | n_topics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rag-observation** | **0.267 ± 0.001** | 0.135 ± 0.002 | 0.647 | 0.524 | 0.422 | 0.0828 | — | 0 |
| v3.3.3 | 0.254 ± 0.006 | 0.137 ± 0.011 | 0.310 ± 0.003 | 0.226 ± 0.004 | **0.201 ± 0.008** | 0.0029 | 25.7 | 9.88 |
| **v3.3.3-2** | 0.255 ± 0.004 | 0.134 ± 0.009 | **0.323 ± 0.008** | **0.234 ± 0.007** | 0.185 ± 0.007 | 0.0029 | 26.3 | 9.81 |
| v3.3.4 | 0.245 ± 0.001 | 0.136 ± 0.005 | 0.255 ± 0.015 | 0.165 ± 0.015 | 0.189 ± 0.013 | 0.0019 | 47.5 | 4.81 |
| **v3.3.4-2** | **0.253 ± 0.001** | **0.153 ± 0.007** | 0.254 ± 0.006 | 0.157 ± 0.003 | **0.194 ± 0.004** | 0.0019 | 61.3 | 3.24 |

**v3.3.3 → v3.3.3-2 Δ**: acc +0.001, **H@k +0.013**, R@k +0.008, R-mh@k **-0.016**, T1var 47.5 → 72.0 (분산 폭증).
**v3.3.4 → v3.3.4-2 Δ**: **acc +0.008 (+3.3%)**, **acc_mh +0.017**, H@k ≈, R@k -0.008, R-mh@k +0.005, n_topics 4.81 → 3.24 (fewer / larger topics).

## 해석

1. **v3.3.3-2 (prototype f0 + posterior odds)**:
   - 의도한 단일-label retrieval 다양화 효과는 H@k / R@k 약간 향상으로 부분 검증됨 (+0.013 / +0.008).
   - 그러나 **multi-hop strict 회수 R-mh@k 가 -0.016 회귀**. 즉 single-hop · 일반 turn 회수는 좋아졌지만 multi-hop 정답 turn 집합 전체를 잡아내는 것은 오히려 나빠짐. 가설: posterior-odds 로 same-label restart 가 자주 발화 → STM 안에 atomic episode 가 늘어남 → 한 topic 안 multi-hop evidence 가 분산됨. T1var 폭증 (47.5 → 72.0) 이 이를 뒷받침 (run 간 segmentation 결정 분산이 큼).
   - **acc 자체 변화는 noise 범위 (±0.006 std). retrieval 향상이 acc 로 옮지 못함 — v3.3.4 의 R-mh@k +20.7% 가 acc 평균 동일했던 패턴과 동일.**

2. **v3.3.4-2 (σ² shrinkage)**:
   - **acc +0.008 / acc_mh +0.017** 작지만 std (0.001) 대비 8σ 이상 → noise 아님.
   - retrieval H@k / R@k 는 v3.3.4 와 거의 같음. 즉 acc 향상은 retrieval recall 이 아니라 다른 채널 (STM topic 구조 / prefill 분포) 에서 옴.
   - **n_topics 4.81 → 3.24, T1μ 47.5 → 61.3**: σ² shrinkage 가 σ_eff² 를 σ_0² 쪽으로 안정화 → 같은 topic 으로 더 많이 묶음 → fewer/larger topic. v3.3.4 stm100 (n_topics=1.95, T1μ=47.7) 와 비슷한 방향이지만 R@k 손실 없이 acc 향상 — shrinkage 가 더 부드러운 boundary smoothing 임을 시사.

3. **공통 한계**: 둘 다 **rag-observation (0.267)** 을 못 넘음. hi-ontop 변형 끼리의 비교에서 의미 있는 강화이지 RAG 대비 우위는 아님 (v3.3.4 known 한계).

## 판정 (iter1 → iter2)

CLAUDE.md "TIAGE / LoCoMo 별로면 다시 구현" 규칙 적용:

- **v3.3.3-2**: acc 향상 없음 (+0.001) + R-mh@k 회귀 (-0.016) + T1var 폭증 → **재구현**. iter2 codex 재설계 의제: (1) R-mh@k 회귀 원인 (posterior threshold 0.35 가 너무 낮아 restart 과다 발화 의심), (2) prototype f0 의 multi-hop 회수 손실 보정 또는 제거, (3) episode 단위 retrieval atomicity 미구현 효과 (codex 1차 권장이었으나 시간상 segmentation 만 변경) 통합 검토.
- **v3.3.4-2**: acc 향상 확인 (+0.008, std 대비 유의) → **유지**. 다음 단계: full LoCoMo 1986Q × 3 runs + HP sweep (`pe_var_shrink_c ∈ {0, 4, 8, 16}`, `pe_var_sigma0_sq ∈ {0.01, 0.04, 0.10}`, `pe_var_robust ∈ {True, False}`) 로 best HP 발굴. R@k 손실 (-0.008) 도 함께 모니터.

## 한계 / 검증 미해결

1. **n=196 표본 작음** (limit 200 stratified). full LoCoMo 1986Q (10×) 가 main 신호. v3.3.4-2 의 acc +0.008 이 full set 에서도 유의한지 별도 검증 필요.
2. **acc_overall 만으로 판정 약함**: rag-observation 이 adversarial 0.96+ 로 acc 를 끌어올림. acc by qtype 별로 v3.3.x 의 multi-hop / temporal-reasoning 강도를 따로 봐야 함 (현재 표에 있지만 분석 본문에선 acc_overall 위주).
3. **TIAGE 미보강**: TIAGE iter1 결과 (`outputs/experiments/2026-05-10_tiage_iter1/REPORT.md`) 는 default HP 부적합으로 6 v3.3.x method 가 동일 corner case → method 차이 분간 못 함. iter2 에서 TIAGE-specific HP sweep 필요.
4. **LoCoMo HP sweep 미실행**: 사용자 요구한 "각 method 영향 큰 HP × 4 values × 3 runs" sweep 이 iter2 의제. iter1 에선 default HP 만 비교.
5. **wandb-tracked metric 의 일부 (T1max, T1var) 가 base v3.3.3 / v3.3.4 와 비교해 std 매우 큼**. RNN init randomness 때문 — 향후 `--seed` flag 추가가 reproducibility 에 도움.
6. **v3.3.3-2 의 R-mh@k 회귀 원인 미특정**. 가설 (restart 과다 / prototype dilution) 은 설계 직관이지 데이터 검증 안 됨. iter2 에서 restart 발화 횟수 logging + ablation (`restart_p_threshold ∈ {0.35, 0.5, 0.7}`) 으로 격리 검증.
