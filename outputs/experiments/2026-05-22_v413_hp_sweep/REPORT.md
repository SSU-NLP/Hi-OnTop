# v4.1.3 full HP sweep — segmentation Score

2-phase sweep (interacting grid + OAT), tuned on `tiage/train` + `superseg/validation` + a seeded `dialseg711` tune split.
Metric: Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD), official SuperDialseg Pk/WD. Encoder = mpnet (cached).

## Phase 1 — interacting grid (ctx_window × ctx_decay × ctx_blend_a × δ*)

Grid points: 360. **Best**: ctx_window=2, ctx_decay=0.9, ctx_blend_a=0.7, δ*=0.62 (tuning Score 0.5212)

### Top-10 grid configs

| m | ρ | a | δ* | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.9 | 0.7 | 0.62 | **0.5212** | 0.4784 | 0.4527 | 0.6325 |
| 1 | 0.5 | 0.3 | 0.62 | **0.5208** | 0.4834 | 0.4729 | 0.6062 |
| 1 | 0.5 | 0.5 | 0.62 | **0.5208** | 0.4834 | 0.4729 | 0.6062 |
| 2 | 0.7 | 0.7 | 0.62 | **0.5206** | 0.4763 | 0.4554 | 0.6302 |
| 2 | 0.5 | 0.3 | 0.5594 | **0.5204** | 0.4873 | 0.4779 | 0.5959 |
| 6 | 0.5 | 0.5 | 0.5594 | **0.5201** | 0.4819 | 0.4678 | 0.6106 |
| 3 | 0.5 | 0.5 | 0.5594 | **0.5200** | 0.4836 | 0.4759 | 0.6004 |
| 5 | 0.5 | 0.5 | 0.5594 | **0.5199** | 0.4826 | 0.4675 | 0.6096 |
| 2 | 0.5 | 0.5 | 0.62 | **0.5197** | 0.4807 | 0.4481 | 0.6303 |
| 3 | 0.5 | 0.3 | 0.5594 | **0.5196** | 0.4801 | 0.4607 | 0.6181 |

## Phase 2 — OAT (holding Phase-1 best)

### alpha → best = 0.5 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.5 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 1.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 2.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 5.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

### lmda → best = 5.0 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 1.0 | 0.5211 | 0.4784 | 0.4527 | 0.6322 |
| 5.0 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 10.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 20.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 50.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

### beta → best = 1.0 (tuning 0.5214)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.1 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.25 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.5 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 1.0 ⭐ | 0.5214 | 0.4790 | 0.4529 | 0.6324 |

### pe_threshold → best = 0.5 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.5 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.8 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 1.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

### cos_threshold → best = 0.8 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.8 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.9 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.95 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

### sigma_delta_c → best = 0.03 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.03 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.0625 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.125 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.25 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

### pe_var_sigma0_sq → best = 0.01 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.01 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.04 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 0.09 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

### pe_var_df0 → best = 0.5 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.5 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 1.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 2.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

### pe_prior → best = 0.8 (tuning 0.5212)

| value | tuning Score | tiage_train | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|
| 0.8 ⭐ | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 1.0 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |
| 1.2 | 0.5212 | 0.4784 | 0.4527 | 0.6325 |

## Final config — default vs swept (TEST sets)

| dataset | default Score | swept Score | Δ |
|---|---:|---:|---:|
| tiage | 0.4476 | 0.4306 | -0.0170 |
| dialseg711 | 0.6120 | 0.6174 | +0.0054 |
| superseg | 0.4447 | 0.4380 | -0.0067 |
| **mean-3** | 0.5014 | 0.4953 | -0.0061 |

## Final HP config

```python
alpha=1.0
lmda=10.0
beta=1.0
pe_threshold=1.0
cos_threshold=0.9
delta_star=0.62
sigma_delta_c=0.0625
ctx_window=2
ctx_decay=0.9
ctx_blend_a=0.7
pe_var_sigma0_sq=0.04
pe_var_df0=1.0
pe_prior=1.0
f0_min_starts=2
restart_pe_threshold=0.5
```

## 한계 / 검증 미해결
- Phase-2 OAT 는 interaction 무시 (Phase-1 best 고정 후 1축씩). Phase-1 의 4 HP 외 상호작용은 미탐색.
- dialseg711 은 official train split 이 없어 test dialogs 를 seeded 30% tune / 70% held-out 으로 나눔. 따라서 `dialseg711` test row 는 full official test 가 아니라 held-out split 이며, literature-comparable full-test 숫자로 직접 인용하면 안 됨.
- superseg 는 validation (1322) 으로 tune, train(6948) 미사용 (인코딩 비용). validation = test 와 동일 크기라 대표성 OK.
