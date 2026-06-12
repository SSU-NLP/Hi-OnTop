# TIAGE test — full compare (12 methods × 3 seeds)

n_convs=100 · n_turns=1564 · n_shifts=315

ARI ↑: primary metric — penalises BOTH over-segmentation and collapse, collapse-immune (rows sorted by ARI). Pk / WD ↓: macro-avg over conversations, k = half mean ref segment length. collapse = fraction of conversations merged into 1 topic; † marks a degenerate method (collapse ≥ 50%) — its Pk/WD/F1 'best' is an artifact, excluded from non-ARI best.

| method | n_seeds | ARI ↑ (mean ± std) | F1 ↑ | P | R | Pk ↓ | WD ↓ | n_topics | collapse | ms/turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.3.6 | 3 | 0.359 ± 0.000 | 0.234 | 0.213 | 0.260 | 0.517 | 0.610 | 4.8 | 0% | 39.90 |
| v3.3.5 | 3 | 0.343 ± 0.000 | 0.279 | 0.230 | 0.352 | 0.497 | 0.598 | 5.8 | 0% | 11.72 |
| v3.3.7 | 3 | 0.334 ± 0.000 | 0.209 | 0.182 | 0.244 | 0.521 | 0.619 | 5.2 | 0% | 41.90 |
| v1 | 1 | 0.269 ± 0.000 | 0.363 | 0.252 | 0.654 | 0.467 | 0.721 | 7.5 | 0% | 0.49 |
| v3.1.1 | 1 | 0.111 ± 0.000 | 0.377 | 0.235 | 0.956 | 0.478 | 0.913 | 13.3 | 0% | 0.31 |
| v3.3.8† | 3 | 0.031 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.510 | 0.510 | 1.0 | 99% | 138.15 |
| v3.3.1 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.59 |
| v3.3.2 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.02 |
| v3.3.3 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.14 |
| v3.3.3-2 | 3 | 0.001 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.36 |
| v3.3.4 | 3 | 0.000 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.26 |
| v3.3.4-2 | 3 | 0.000 ± 0.000 | 0.354 | 0.215 | 1.000 | 0.487 | 0.983 | 15.6 | 0% | 1.48 |

**best ARI (primary)**: `v3.3.6` — 0.359 ± 0.000
**best F1** (non-degenerate): `v3.1.1` — 0.377 ± 0.000
**best Pk** (non-degenerate): `v1` — 0.467 ± 0.000
**best WD** (non-degenerate): `v3.3.5` — 0.598 ± 0.000
