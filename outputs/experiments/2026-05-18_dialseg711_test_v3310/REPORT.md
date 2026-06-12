# dialseg711 test — official SuperDialseg Pk/WD/F1

n_dial=711 · n_turn=19350 · n_bnd=2754 (bnd_rate=0.142)

Metric = official SuperDialseg (window='auto', verbatim formula; literature-comparable). GT = dataset segmentation_label. **Caveat**: oracle-θ / oracle-δ* are tuned ON this test set = UPPER BOUNDS, not fair external claims. v3.3.9@TIAGE-δ* = honest zero-shot transfer (no tuning here).

Score = 0.5·F1 + 0.25·(1-Pk) + 0.25·(1-WD) (official SuperDialseg aggregate).

| method | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ | pred_rate | n_dial |
|---|---:|---:|---:|---:|---:|---:|
| prevcos @oracleθ=0.609 (similarity ceiling) | 0.590 | 0.536 | 0.322 | 0.389 | 0.236 | 711 |
| v3.3.9 @TIAGE-δ*=0.5557 (zero-shot) | 0.514 | 0.492 | 0.394 | 0.534 | 0.368 | 711 |
| v3.3.9 @oracle-δ*=0.609 (upper bound) | 0.590 | 0.536 | 0.322 | 0.389 | 0.236 | 711 |
| v3.3.10 @TIAGE-cfg(m2,ρ0.7,a0.5,δ*0.5594) (zero-shot) | 0.590 | 0.549 | 0.325 | 0.415 | 0.269 | 711 |
| v3.3.10 @oracle-δ*=0.613 (upper bound) | 0.629 | 0.560 | 0.285 | 0.319 | 0.154 | 711 |

GT bnd_rate = 0.142 (pred_rate 비교 기준).
