# superseg test — official SuperDialseg Pk/WD/F1

n_dial=1322 · n_turn=17328 · n_bnd=4020 (bnd_rate=0.232)

Metric = official SuperDialseg (window='auto', verbatim formula; literature-comparable). GT = dataset segmentation_label. **Caveat**: oracle-θ / oracle-δ* are tuned ON this test set = UPPER BOUNDS, not fair external claims. v3.3.9@TIAGE-δ* = honest zero-shot transfer (no tuning here).

Score = 0.5·F1 + 0.25·(1-Pk) + 0.25·(1-WD) (official SuperDialseg aggregate).

| method | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ | pred_rate | n_dial |
|---|---:|---:|---:|---:|---:|---:|
| v3.3.9 @superseg-validation-δ*=0.520 (calibrated) | 0.451 | 0.455 | 0.470 | 0.637 | 0.500 | 1322 |
| v4.1.1 @superseg-validation-δ*=0.502 (calibrated) | 0.453 | 0.451 | 0.474 | 0.615 | 0.461 | 1322 |
| prevcos @oracleθ=0.500 (similarity ceiling) | 0.447 | 0.458 | 0.467 | 0.660 | 0.542 | 1322 |
| v3.3.9 @TIAGE-δ*=0.5557 (zero-shot) | 0.460 | 0.449 | 0.468 | 0.589 | 0.418 | 1322 |
| v3.3.9 @oracle-δ*=0.500 (upper bound) | 0.447 | 0.458 | 0.467 | 0.660 | 0.542 | 1322 |
| v4.1.1 @TIAGE-cfg(m2,ρ0.7,a0.5,δ*0.5594) (zero-shot) | 0.463 | 0.432 | 0.471 | 0.541 | 0.316 | 1322 |
| v4.1.1 @oracle-δ*=0.468 (upper bound) | 0.446 | 0.458 | 0.470 | 0.663 | 0.545 | 1322 |

GT bnd_rate = 0.232 (pred_rate 비교 기준).
