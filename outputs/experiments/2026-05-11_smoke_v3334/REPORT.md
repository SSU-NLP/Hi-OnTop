# Sweep: 2026-05-11_smoke_v3334

Generated: 2026-05-11T00:09:54

Path: `outputs/experiments/2026-05-11_smoke_v3334/`

n_questions: **5** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | dormant_ev_rate | top_ev_promoted | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.3-4 (smoke3334) | seg: α=1, λ=10, cos=0.7, β=0.5, rnn_train_steps=1 · mem: k_top=3, k_turn=5 | 0.059 | 0.037 | - | 0.098 | - | 0.024 | 0.400 | 0.400 | 0.000 | 0.0060 | 1.000 | 0.000 | 290.0 | 0.0 | 0.0 | 290 | 0 | 0.0 | 1.00 | 1.43 | 44s |

**best (acc)**: `smoke3334` — 0.059
