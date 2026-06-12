# Sweep: 2026-05-11_v3334_io_iter2

Generated: 2026-05-11T06:01:58

Path: `outputs/experiments/2026-05-11_v3334_io_iter2/`

n_questions: **196** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | dormant_ev_rate | top_ev_promoted | n_dormant_top | dormant_top_n_ev | dormant_top_share | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.3-4 (v3334io_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=42 | 0.240 | 0.112 | 0.026 | 0.016 | 1.000 | 0.024 | 0.109 | 0.047 | 0.111 | 0.0012 | 0.952 | 0.058 | 1.74 | 0.99 | 0.771 | 24.1 | 20.0 | 18.3 | 37 | 30 | 45.7 | 9.88 | 1.95 | 8m 2s |
| hi-ontop-full-v3.3.3-4 (v3334io_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=43 | 0.242 | 0.123 | 0.024 | 0.022 | 1.000 | 0.021 | 0.122 | 0.045 | 0.084 | 0.0011 | 0.955 | 0.045 | 1.74 | 1.01 | 0.760 | 25.3 | 19.5 | 17.6 | 39 | 28 | 43.8 | 9.89 | 1.94 | 8m 21s |
| hi-ontop-full-v3.3.3-4 (v3334io_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=44 | 0.239 | 0.131 | 0.027 | 0.014 | 0.975 | 0.025 | 0.115 | 0.054 | 0.123 | 0.0014 | 0.945 | 0.058 | 1.70 | 1.00 | 0.773 | 26.0 | 20.7 | 18.9 | 33 | 28 | 45.5 | 9.89 | 1.92 | 8m 24s |
| hi-ontop-full-v3.3.3 (v333_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=42 | 0.249 | 0.123 | 0.057 | 0.048 | 0.975 | 0.021 | 0.314 | 0.228 | 0.212 | 0.0028 | 0.954 | 0.077 | 1.73 | 1.02 | 0.784 | 24.5 | 20.9 | 17.6 | 37 | 28 | 43.0 | 9.88 | 3.41 | 8m 18s |
| hi-ontop-full-v3.3.3 (v333_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=43 | 0.257 | 0.133 | 0.062 | 0.043 | 1.000 | 0.024 | 0.308 | 0.230 | 0.225 | 0.0030 | 0.951 | 0.038 | 1.71 | 1.01 | 0.767 | 26.4 | 18.6 | 17.9 | 37 | 25 | 43.4 | 10.00 | 2.28 | 8m 36s |
| hi-ontop-full-v3.3.3 (v333_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=44 | 0.254 | 0.130 | 0.069 | 0.049 | 0.975 | 0.022 | 0.327 | 0.240 | 0.195 | 0.0029 | 0.941 | 0.071 | 1.71 | 1.00 | 0.754 | 25.6 | 20.8 | 18.0 | 33 | 29 | 33.3 | 9.88 | 2.12 | 8m 37s |

**best (acc)**: `v333_r2` — 0.257
