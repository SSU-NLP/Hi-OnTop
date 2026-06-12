# Sweep: 2026-05-11_imp_v2_sanity

Generated: 2026-05-11T12:34:47

Path: `outputs/experiments/2026-05-11_imp_v2_sanity/`

n_questions: **196** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | dormant_ev_rate | top_ev_promoted | n_dormant_top | dormant_top_n_ev | dormant_top_share | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.3-4 (impv2_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=42 | 0.234 | 0.132 | 0.025 | 0.016 | 0.950 | 0.025 | 0.135 | 0.065 | 0.131 | 0.0016 | 0.934 | 0.077 | 1.69 | 0.99 | 0.773 | 22.6 | 19.0 | 16.5 | 37 | 30 | 71.6 | 9.88 | 2.15 | 8m 18s |
| hi-ontop-full-v3.3.3-4 (impv2_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=43 | 0.243 | 0.127 | 0.024 | 0.021 | 1.000 | 0.020 | 0.147 | 0.065 | 0.096 | 0.0015 | 0.934 | 0.071 | 1.69 | 1.00 | 0.766 | 23.1 | 18.7 | 14.9 | 39 | 26 | 67.9 | 10.00 | 2.07 | 8m 8s |
| hi-ontop-full-v3.3.3-4 (impv2_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=44 | 0.239 | 0.137 | 0.026 | 0.016 | 0.975 | 0.019 | 0.135 | 0.071 | 0.142 | 0.0018 | 0.928 | 0.083 | 1.65 | 0.99 | 0.776 | 24.4 | 18.8 | 16.3 | 33 | 31 | 65.9 | 10.00 | 2.34 | 8m 13s |
| hi-ontop-full-v3.3.3 (v333_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=42 | 0.255 | 0.137 | 0.074 | 0.044 | 0.975 | 0.025 | 0.314 | 0.228 | 0.212 | 0.0028 | 0.954 | 0.064 | 1.73 | 1.02 | 0.784 | 24.5 | 20.9 | 17.6 | 37 | 28 | 43.0 | 9.88 | 5.54 | 8m 1s |
| hi-ontop-full-v3.3.3 (v333_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=43 | 0.248 | 0.139 | 0.060 | 0.047 | 0.950 | 0.023 | 0.308 | 0.230 | 0.225 | 0.0030 | 0.951 | 0.038 | 1.71 | 1.01 | 0.767 | 26.4 | 18.6 | 17.9 | 37 | 25 | 43.4 | 10.00 | 5.11 | 8m 35s |
| hi-ontop-full-v3.3.3 (v333_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=44 | 0.258 | 0.160 | 0.056 | 0.047 | 0.975 | 0.028 | 0.327 | 0.240 | 0.195 | 0.0029 | 0.941 | 0.058 | 1.71 | 1.00 | 0.754 | 25.6 | 20.8 | 18.0 | 33 | 29 | 33.3 | 9.88 | 3.17 | 8m 21s |

**best (acc)**: `v333_r3` — 0.258
