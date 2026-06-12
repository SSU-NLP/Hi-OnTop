# Sweep: 2026-05-10_smoke_v33x2

Generated: 2026-05-10T18:00:27

Path: `outputs/experiments/2026-05-10_smoke_v33x2/`

n_questions: **5** (uniform across runs)

Columns: H@k/R@k/P@k — retrieval hit/recall/precision against LoCoMo answer-evidence turn ids; R-multi-hop@k — strict all-evidence-included rate on multi-hop questions only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | H@k | R@k | R-multi-hop@k | P@k | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.3-2 (smoke332) | seg: α=1, λ=10, cos=0.7, β=0.5, rnn_train_steps=1 · mem: k_top=3, k_turn=5 | 0.000 | 0.000 | - | 0.000 | - | 0.000 | - | - | - | - | - | - | - | - | - | - | 0.00 | - | 15s |
| hi-ontop-full-v3.3.4-2 (smoke342) | seg: α=1, λ=10, cos=0.7, β=0.5, rnn_train_steps=1 · mem: k_top=3, k_turn=5 | 0.000 | 0.000 | - | 0.000 | - | 0.000 | - | - | - | - | - | - | - | - | - | - | 0.00 | - | 14s |

**best (acc)**: `smoke332` — 0.000
