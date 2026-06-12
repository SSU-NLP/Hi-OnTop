# TopiOCQA — Hi-OnTop 변형 모두 비교 (v2 / v3.1 / v3.2)

`alpha=1.0` `lmda=10.0` `sigma0_sq=0.01` device=auto limit_convs=None

Data: 205 conv / 2514 turns / 672 GT shifts

## Topic shift F1

| Method | Precision | Recall | F1 | avg #topics | avg max-share |
|---|---|---|---|---|---|
| v2 hi-ontop-full (Gaussian) | 0.320 | 0.543 | 0.402 | 5.6 | 0.255 |
| v3.1 hi-ontop-full-v3.1 (Bounded Cosine MAP, τ=50.0, thr=0.7) | 0.311 | 0.863 | 0.458 | 8.9 | 0.276 |
| v3.2 hi-ontop-full-v3.2 (Cosine PE, τ=50.0, thr=0.7, ρ=0.5) | 0.311 | 0.866 | 0.458 | 9.0 | 0.270 |

