# v4.1.3 demo — graded boundary score

**Algorithm = v4.1.1 identical**. v4.1.3 adds `graded_score = δ_eff / δ*`
per-turn output (Ben-Yakov & Henson 2018 hippocampal graded boundary profile mapping).

Boundary strength bands:
- `< 0.7` very weak (downstream consumer 보류 권장)
- `0.7-1.0` weak (repeat 우세)
- `1.0-1.3` normal (정상 경계)
- `≥ 1.3` strong (즉시 commit 권장)

Encoder: `sentence-transformers/multi-qa-mpnet-base-dot-v1` (TIAGE-train HP: m=2, ρ=0.7, a=0.5, δ*=0.5594).

## 데이터셋별 통계

| dataset | n_dial | n_turns | n_bnd_pred | n_bnd_gt | Score | F1 | very_weak | weak | normal | strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100 | 1564 | 431 | 315 | 0.4675 | 0.4102 | 396 | 737 | 406 | 25 |

## tiage

- **Score=0.4675** (= v4.1.1 baseline, algorithm 무변경)
- F1=0.4102, Pk=0.4421, WD=0.5082
- 총 turns: 1564
- boundary 예측: 431, GT: 315

### Per-band precision (graded score discriminator 검증)

| band | n_pred_bnd | TP | FP | precision |
|---|---:|---:|---:|---:|
| very_weak | 0 | 0 | 0 | 0.000 |
| weak | 0 | 0 | 0 | 0.000 |
| normal | 406 | 140 | 266 | 0.345 |
| strong | 25 | 13 | 12 | 0.520 |

### Example dialogs

### Dialog 3 (16 turns, GT boundaries: 3)

| turn | topic | GT | pred | graded | band |
|---:|---:|:---:|:---:|---:|---|
| 0 | 0 |   |   | 0.000 | — |
| 1 | 1 | **B** | **B** | 1.150 | normal |
| 2 | 2 | **B** | **B** | 1.117 | normal |
| 3 | 3 |   | **B** | 1.213 | normal |
| 4 | 3 | **B** |   | 0.993 | weak |
| 5 | 4 |   | **B** | 1.229 | normal |
| 6 | 5 |   | **B** | 1.196 | normal |
| 7 | 6 |   | **B** | 1.208 | normal |
| 8 | 7 |   | **B** | 1.108 | normal |
| 9 | 7 |   |   | 1.000 | weak |
| 10 | 8 |   | **B** | 1.172 | normal |
| 11 | 8 |   |   | 0.946 | weak |
| 12 | 8 |   |   | 0.898 | weak |
| 13 | 8 |   |   | 0.906 | weak |
| 14 | 9 |   | **B** | 1.107 | normal |
| 15 | 10 |   | **B** | 1.058 | normal |

### Dialog 49 (16 turns, GT boundaries: 0)

| turn | topic | GT | pred | graded | band |
|---:|---:|:---:|:---:|---:|---|
| 0 | 0 |   |   | 0.000 | — |
| 1 | 0 |   |   | 0.829 | weak |
| 2 | 1 |   | **B** | 1.242 | normal |
| 3 | 2 |   | **B** | 1.047 | normal |
| 4 | 2 |   |   | 0.929 | weak |
| 5 | 3 |   | **B** | 1.056 | normal |
| 6 | 3 |   |   | 0.914 | weak |
| 7 | 4 |   | **B** | 1.199 | normal |
| 8 | 4 |   |   | 0.547 | very_weak |
| 9 | 4 |   |   | 0.970 | weak |
| 10 | 5 |   | **B** | 1.044 | normal |
| 11 | 6 |   | **B** | 1.189 | normal |
| 12 | 7 |   | **B** | 1.022 | normal |
| 13 | 8 |   | **B** | 1.080 | normal |
| 14 | 8 |   |   | 0.941 | weak |
| 15 | 9 |   | **B** | 1.021 | normal |

