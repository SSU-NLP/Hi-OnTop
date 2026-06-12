# Hi-OnTop p70/p75/p80 per-metric (MiniLM-int8, superdialseg_data)

δ\*_p = calib δ_eff p-percentile. harness = `run_encoder_comparison.py` 함수 import + cached embeddings (인코더 forward 없음).

| 벤치 | p | δ\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | calib note |
|---|---:|---:|---:|---:|---:|---:|---|
| tiage | p70 | 0.7763 | 0.4190 | 0.5067 | 0.4415 | **0.4893** | train split (calib 300 / test 100) |
| tiage | p75 | 0.7977 | 0.4195 | 0.4879 | 0.4168 | **0.4815** | train split (calib 300 / test 100) |
| tiage | p80 | 0.8223 | 0.4035 | 0.4575 | 0.4155 | **0.4925** | train split (calib 300 / test 100) |
| dialseg711 | p70 | 0.7519 | 0.3417 | 0.4494 | 0.5282 | **0.5663** | test 70:30 split (calib 498 / test 213) |
| dialseg711 | p75 | 0.7774 | 0.3202 | 0.4004 | 0.5471 | **0.5934** | test 70:30 split (calib 498 / test 213) |
| dialseg711 | p80 | 0.8029 | 0.3069 | 0.3653 | 0.5509 | **0.6074** | test 70:30 split (calib 498 / test 213) |
| superseg | p70 | 0.7839 | 0.5228 | 0.5741 | 0.3504 | **0.4010** | train split (calib 400 / test 1322) |
| superseg | p75 | 0.8121 | 0.5318 | 0.5683 | 0.3159 | **0.3829** | train split (calib 400 / test 1322) |
| superseg | p80 | 0.8409 | 0.5316 | 0.5545 | 0.2730 | **0.3650** | train split (calib 400 / test 1322) |
