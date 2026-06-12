# Sweep: 2026-05-14_v3342_sigma_qwen35

Generated: 2026-05-14T21:48:20

Path: `outputs/experiments/2026-05-14_v3342_sigma_qwen35/`

n_questions: **1986** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; {mh,sh,temp,od}_topics/q — mean number of distinct topics that evidence turns of a question landed in, restricted to questions with n_evidence≥2 (smaller = segmenter co-locates evidence; computed by scripts/analyze_evidence_topics.py); T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | dormant_ev_rate | top_ev_promoted | n_dormant_top | dormant_top_n_ev | dormant_top_share | mh_topics/q | sh_topics/q | temp_topics/q | od_topics/q | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rag-obs (rag_obs_r1) | rag_k=? · override: seed=1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 0.00 | - | 10s |
| rag-obs (rag_obs_r2) | rag_k=? · override: seed=2 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 0.00 | - | 7s |
| rag-obs (rag_obs_r3) | rag_k=? · override: seed=3 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 0.00 | - | 7s |
| hi-ontop-full-v3.3.4-2 (v3342_sig_r1) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3, pe_var_sigma0_sq=0.10, pe_var_shrink_c=32, seed=1 | 0.261 | 0.146 | 0.083 | 0.045 | 0.870 | 0.041 | 0.440 | 0.377 | 0.284 | 0.0028 | 0.727 | 0.272 | 1.07 | 0.84 | 0.670 | 2.69 | 1.43 | 2.12 | 2.79 | 165.0 | 0.0 | 0.0 | 214 | 0 | 832.0 | 1.00 | 4.07 | 57m 17s |
| hi-ontop-full-v3.3.4-2 (v3342_sig_r2) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3, pe_var_sigma0_sq=0.10, pe_var_shrink_c=32, seed=2 | 0.258 | 0.135 | 0.084 | 0.051 | 0.861 | 0.038 | 0.458 | 0.394 | 0.295 | 0.0030 | 0.720 | 0.295 | 1.07 | 0.82 | 0.660 | 2.67 | 1.43 | 2.12 | 2.79 | 168.6 | 0.0 | 0.0 | 233 | 0 | 1450.9 | 1.00 | 4.19 | 51m 30s |
| hi-ontop-full-v3.3.4-2 (v3342_sig_r3) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, rnn_train_steps=3, pe_var_sigma0_sq=0.10, pe_var_shrink_c=32, seed=3 | 0.257 | 0.139 | 0.078 | 0.044 | 0.870 | 0.040 | 0.438 | 0.377 | 0.281 | 0.0031 | 0.729 | 0.272 | 1.08 | 0.83 | 0.666 | 2.69 | 1.43 | 2.12 | 2.77 | 157.8 | 0.0 | 0.0 | 214 | 0 | 1011.9 | 1.00 | 4.72 | 3h 17m 11s |

**best (acc)**: `v3342_sig_r1` — 0.261
