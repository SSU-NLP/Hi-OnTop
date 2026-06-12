# Sweep: 2026-05-11_v334_audit_full

Generated: 2026-05-11T04:11:08

Path: `outputs/experiments/2026-05-11_v334_audit_full/`

n_questions: **1986** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | dormant_ev_rate | top_ev_promoted | n_dormant_top | dormant_top_n_ev | dormant_top_share | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.4 (v334_audit) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 · override: cos_threshold=0.9, alpha=100, beta=0.25, pe_threshold=0.5, rnn_train_steps=3, seed=42 | 0.258 | 0.144 | 0.075 | 0.046 | 0.877 | 0.029 | 0.430 | 0.371 | 0.300 | 0.0029 | 0.701 | 0.307 | 1.06 | 0.79 | 0.641 | 47.9 | 47.2 | 43.2 | 57 | 57 | 14.7 | 4.69 | 1.83 | 52m 17s |

**best (acc)**: `v334_audit` — 0.258
