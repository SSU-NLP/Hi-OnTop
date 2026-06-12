# Sweep: 2026-05-10_v333_iter2

Generated: 2026-05-10T23:16:00

Path: `outputs/experiments/2026-05-10_v333_iter2/`

n_questions: **196** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.3-2 (v3332_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.257 | 0.138 | 0.056 | 0.048 | 1.000 | 0.020 | 0.327 | 0.236 | 0.173 | 0.0028 | 26.8 | 21.5 | 18.6 | 48 | 29 | 78.7 | 9.65 | 2.44 | 9m 20s |
| hi-ontop-full-v3.3.3-2 (v3332_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.255 | 0.120 | 0.070 | 0.040 | 1.000 | 0.024 | 0.308 | 0.231 | 0.191 | 0.0028 | 28.6 | 19.5 | 18.5 | 44 | 27 | 62.4 | 10.00 | 2.80 | 8m 54s |
| hi-ontop-full-v3.3.3-2 (v3332_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.254 | 0.118 | 0.054 | 0.048 | 1.000 | 0.025 | 0.346 | 0.247 | 0.192 | 0.0031 | 27.0 | 20.4 | 17.6 | 36 | 29 | 38.7 | 9.89 | 2.63 | 8m 41s |
| hi-ontop-full-v3.3.3-3 (v3333_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.255 | 0.146 | 0.066 | 0.041 | 0.975 | 0.026 | 0.346 | 0.236 | 0.178 | 0.0030 | 28.9 | 22.1 | 19.4 | 36 | 31 | 54.6 | 9.65 | 2.49 | 8m 43s |
| hi-ontop-full-v3.3.3-3 (v3333_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.249 | 0.128 | 0.067 | 0.054 | 0.950 | 0.024 | 0.327 | 0.234 | 0.207 | 0.0031 | 25.2 | 20.5 | 19.2 | 34 | 32 | 57.0 | 9.65 | 2.30 | 8m 54s |
| hi-ontop-full-v3.3.3-3 (v3333_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.238 | 0.114 | 0.056 | 0.050 | 0.925 | 0.023 | 0.340 | 0.242 | 0.193 | 0.0030 | 23.0 | 18.7 | 18.4 | 33 | 26 | 42.5 | 9.88 | 2.16 | 8m 11s |
| hi-ontop-full-v3.3.3 (v333_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.260 | 0.163 | 0.062 | 0.047 | 0.975 | 0.028 | 0.359 | 0.266 | 0.231 | 0.0034 | 28.7 | 20.5 | 17.6 | 38 | 28 | 56.9 | 10.00 | 2.98 | 9m 6s |
| hi-ontop-full-v3.3.3 (v333_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.251 | 0.127 | 0.057 | 0.051 | 0.975 | 0.021 | 0.327 | 0.238 | 0.208 | 0.0028 | 25.2 | 19.5 | 17.2 | 33 | 24 | 31.5 | 10.00 | 2.66 | 8m 49s |
| hi-ontop-full-v3.3.3 (v333_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3 | 0.251 | 0.125 | 0.067 | 0.040 | 0.975 | 0.024 | 0.308 | 0.228 | 0.201 | 0.0029 | 26.6 | 18.9 | 18.4 | 34 | 27 | 44.2 | 9.89 | 2.15 | 9m 33s |

**best (acc)**: `v333_r1` — 0.260
