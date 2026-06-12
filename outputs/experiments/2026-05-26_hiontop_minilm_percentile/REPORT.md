# Hi-OnTop p70/p75/p80 per-metric (MiniLM, superdialseg_data)

δ\*_p = calib δ_eff p-percentile. harness = `run_encoder_comparison.py` 함수 import + cached embeddings (인코더 forward 없음).

| 벤치 | p | δ\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | calib note |
|---|---:|---:|---:|---:|---:|---:|---|
| tiage | p70 | 0.7734 | 0.4230 | 0.5160 | 0.4387 | **0.4846** | train split (calib 300 / test 100) |
| tiage | p75 | 0.7955 | 0.4196 | 0.4888 | 0.4162 | **0.4810** | train split (calib 300 / test 100) |
| tiage | p80 | 0.8192 | 0.4230 | 0.4692 | 0.3849 | **0.4694** | train split (calib 300 / test 100) |
| dialseg711 | p70 | 0.7485 | 0.3468 | 0.4595 | 0.5257 | **0.5613** | test 70:30 split (calib 498 / test 213) |
| dialseg711 | p75 | 0.7757 | 0.3208 | 0.4028 | 0.5409 | **0.5896** | test 70:30 split (calib 498 / test 213) |
| dialseg711 | p80 | 0.8011 | 0.3076 | 0.3655 | 0.5446 | **0.6040** | test 70:30 split (calib 498 / test 213) |
| superseg | p70 | 0.7812 | 0.5207 | 0.5725 | 0.3514 | **0.4024** | train split (calib 400 / test 1322) |
| superseg | p75 | 0.8058 | 0.5272 | 0.5655 | 0.3270 | **0.3903** | train split (calib 400 / test 1322) |
| superseg | p80 | 0.8352 | 0.5303 | 0.5560 | 0.2871 | **0.3720** | train split (calib 400 / test 1322) |
