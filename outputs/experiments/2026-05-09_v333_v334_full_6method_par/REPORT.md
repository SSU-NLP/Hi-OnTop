# Sweep: 2026-05-09_v333_v334_full_6method_par

Generated: 2026-05-10T01:10:18

Path: `outputs/experiments/2026-05-09_v333_v334_full_6method_par/`

n_questions: **1986** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rag (rag) | rag_k=10 | 0.285 | 0.132 | 0.115 | 0.062 | 0.910 | 0.059 | 0.640 | 0.569 | 0.365 | 0.0732 | - | - | - | - | - | - | 0.00 | 1.76 | 48m 30s |
| rag-observation (rag_observation) | rag_k=10 | 0.295 | 0.153 | 0.098 | 0.084 | 0.962 | 0.055 | 0.687 | 0.607 | 0.448 | 0.0828 | - | - | - | - | - | - | 0.00 | 1.67 | 48m 52s |
| rag-summary (rag_summary) | rag_k=10 | 0.275 | 0.175 | 0.057 | 0.057 | 0.955 | 0.039 | 0.833 | 0.781 | 0.714 | 0.0054 | - | - | - | - | - | - | 0.00 | 2.01 | 47m 48s |
| hi-ontop-full-v3.3.2 (v332_best) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.258 | 0.146 | 0.076 | 0.039 | 0.877 | 0.041 | 0.421 | 0.355 | 0.251 | 0.0033 | 25.7 | 21.7 | 17.6 | 35 | 31 | 38.5 | 9.88 | 2.22 | 54m 45s |
| hi-ontop-full-v3.3.3 (v333_f0) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, restart_pe_threshold=0.5, restart_margin=0.0 | 0.256 | 0.133 | 0.074 | 0.040 | 0.879 | 0.040 | 0.412 | 0.347 | 0.232 | 0.0033 | 25.5 | 18.6 | 16.8 | 34 | 25 | 43.8 | 10.00 | 2.12 | 57m 11s |
| hi-ontop-full-v3.3.4 (v334_stm100) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3, pe_var_decay=0.95, pe_var_min_samples=5, pe_var_sigma0_sq=0.04, stm_max_turns=100 | 0.266 | 0.125 | 0.060 | 0.039 | 0.953 | 0.052 | 0.270 | 0.226 | 0.175 | 0.0031 | 47.7 | 46.1 | 28.0 | 57 | 50 | 28.4 | 1.95 | 4.46 | 58m 33s |
| hi-ontop-full-v3.3.4 (v334_var) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3, pe_var_decay=0.95, pe_var_min_samples=5, pe_var_sigma0_sq=0.04 | 0.259 | 0.136 | 0.079 | 0.047 | 0.879 | 0.028 | 0.445 | 0.386 | 0.303 | 0.0029 | 47.6 | 46.5 | 43.7 | 52 | 52 | 20.7 | 4.81 | 2.12 | 55m 13s |

**best (acc)**: `rag_observation` — 0.295
