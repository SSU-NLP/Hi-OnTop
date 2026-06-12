# TIAGE test — Hi-OnTop 변형 모두 비교 (v2 / v3.1.1 / v3.2.1 / v3.3.1)

`split=test` device=auto

Data: 100 dialogs / 1564 turns / 315 GT shifts (shift rate 0.215 / transition)

## Topic-shift F1 (turn-transition binary)

| Method | Precision | Recall | F1 | avg #topics | avg max-share |
|---|---|---|---|---|---|
| (a) all-boundary | 0.215 | 1.000 | 0.354 | — | — |
| (b) cosine-threshold (θ=0.450) | 0.327 | 0.635 | 0.432 | — | — |
| (c) v1 persistence (α=1, λ=10, σ²=0.01) | 0.252 | 0.654 | 0.363 | 7.5 | 0.194 |
| (d) v1 freq-shift (α=10, λ=1, σ²=0.1) | 0.235 | 0.962 | 0.377 | 13.5 | 0.150 |
| (e) v3.1.1 Bounded Cosine MAP (τ=50.0, thr=0.7) | 0.235 | 0.956 | 0.377 | 13.3 | 0.160 |
| (f) v3.2.1 sub-linear count (τ=50.0, thr=0.7, β=0.5) | 0.235 | 0.956 | 0.377 | 13.3 | 0.162 |
| (g) v3.3.1 shared-GRU PE (τ=10.0, thr=0.7, β=0.5) | 0.271 | 0.800 | 0.405 | 10.1 | 0.128 |

## Latency
- embed: 39.1s / 25.01 ms/turn
- v1 assign:    324.6ms / 0.208 ms/turn
- v3.1.1 assign:106.6ms / 0.068 ms/turn
- v3.2.1 assign:194.6ms / 0.124 ms/turn
- v3.3.1 assign:5447.1ms / 3.483 ms/turn
- 총 overhead (v1):    25.22 ms/turn
- 총 overhead (v3.1.1):25.08 ms/turn
- 총 overhead (v3.2.1):25.13 ms/turn
- 총 overhead (v3.3.1):28.49 ms/turn

