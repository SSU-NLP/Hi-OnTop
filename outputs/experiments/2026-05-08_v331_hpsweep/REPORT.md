# Sweep: 2026-05-08_v331_hpsweep

Generated: 2026-05-08T16:00:05

Path: `outputs/sweeps/2026-05-08_v331_hpsweep/`

Columns: T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock.

| method | overrides | n | acc | mh | sh | tr | adv | od | T1μ | T2μ | T3μ | T1max | T2max | T1var | n_topics | gen_p50(s) | wall(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-ontop-full-v3.3.1 (cos0p85_a100_b0p25) | — | 49 | 0.246 | 0.108 | 0.056 | 0.022 | 1.000 | 0.024 | 27.2 | 22.3 | 18.3 | 42 | 38 | 116.3 | 9.42 | 4.71 | 395.0 |
| hi-ontop-full-v3.3.1 (cos0p85_a10_b0p5) | — | 49 | 0.233 | 0.071 | 0.030 | 0.019 | 1.000 | 0.025 | 38.8 | 25.9 | 19.1 | 99 | 38 | 572.3 | 8.96 | 3.07 | 288.0 |
| hi-ontop-full-v3.3.1 (cos0p90_a100_b0p25) | — | 49 | 0.263 | 0.155 | 0.065 | 0.025 | 1.000 | 0.048 | 28.6 | 23.2 | 19.7 | 43 | 37 | 108.9 | 9.65 | 6.95 | 282.0 |
| hi-ontop-full-v3.3.1 (cos0p90_a10_b0p5) | — | 49 | 0.255 | 0.152 | 0.066 | 0.013 | 1.000 | 0.021 | 36.4 | 29.2 | 18.3 | 75 | 49 | 336.9 | 9.30 | 2.32 | 283.0 |
| hi-ontop-full-v3.3.1 (cos0p95_a100_b0p1) | — | 49 | 0.249 | 0.134 | 0.056 | 0.019 | 1.000 | 0.014 | 25.2 | 20.9 | 17.4 | 47 | 28 | 105.4 | 9.88 | 4.60 | 279.0 |

**best (acc)**: `cos0p90_a100_b0p25` — 0.263
