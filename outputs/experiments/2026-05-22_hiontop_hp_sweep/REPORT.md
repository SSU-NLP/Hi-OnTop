# Hi-OnTop full HP sweep — all 4 HPs

Hi-OnTop has exactly 4 hyper-parameters, all live. Single grid sweep.
Tuned on tiage/train + superseg/validation + dialseg711 30%-tune; reported on the 3 held-out test sets. Metric = SuperDialseg Score.

Grid: 630 points. delta_star=[0.4, 0.45, 0.5, 0.5594, 0.62, 0.7, 0.78], ctx_window=[2, 3, 4, 5, 6, 8], ctx_decay=[0.5, 0.7, 0.9], ctx_blend_a=[0.0, 0.3, 0.5, 0.7, 1.0]

## Top-15 configs (by tuning Score)

| δ* | m | ρ | a | tuning | tiage_tr | superseg_val | dialseg711_tune |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.62 | 2 | 0.9 | 0.7 | **0.5212** | 0.4784 | 0.4527 | 0.6325 |
| 0.62 | 2 | 0.7 | 0.7 | **0.5206** | 0.4763 | 0.4554 | 0.6302 |
| 0.5594 | 2 | 0.5 | 0.3 | **0.5204** | 0.4873 | 0.4779 | 0.5959 |
| 0.5594 | 2 | 0.5 | 0.0 | **0.5202** | 0.4788 | 0.4662 | 0.6157 |
| 0.5594 | 6 | 0.5 | 0.5 | **0.5201** | 0.4819 | 0.4678 | 0.6106 |
| 0.5594 | 3 | 0.5 | 0.5 | **0.5200** | 0.4836 | 0.4759 | 0.6004 |
| 0.5594 | 5 | 0.5 | 0.5 | **0.5199** | 0.4826 | 0.4675 | 0.6096 |
| 0.62 | 2 | 0.5 | 0.5 | **0.5197** | 0.4807 | 0.4481 | 0.6303 |
| 0.5594 | 3 | 0.5 | 0.3 | **0.5196** | 0.4801 | 0.4607 | 0.6181 |
| 0.5594 | 8 | 0.5 | 0.5 | **0.5194** | 0.4809 | 0.4661 | 0.6111 |
| 0.62 | 2 | 0.5 | 0.7 | **0.5192** | 0.4764 | 0.4567 | 0.6245 |
| 0.5594 | 4 | 0.5 | 0.5 | **0.5191** | 0.4829 | 0.4701 | 0.6042 |
| 0.5594 | 5 | 0.5 | 0.3 | **0.5188** | 0.4756 | 0.4457 | 0.6351 |
| 0.5594 | 4 | 0.5 | 0.3 | **0.5187** | 0.4774 | 0.4496 | 0.6291 |
| 0.62 | 2 | 0.5 | 1.0 | **0.5187** | 0.4834 | 0.4729 | 0.5998 |

## default vs swept-best (TEST sets)

default = canonical v4.1.x config (δ*=0.5594, m=2, ρ=0.7, a=0.5)

| dataset | default Score | swept-best Score | Δ |
|---|---:|---:|---:|
| tiage | 0.4675 | 0.4301 | -0.0374 |
| dialseg711 | 0.5920 | 0.6192 | +0.0273 |
| superseg | 0.4631 | 0.4380 | -0.0251 |
| **mean-3** | 0.5075 | 0.4958 | -0.0118 |

## swept-best HP

```python
delta_star=0.62
ctx_window=2
ctx_decay=0.9
ctx_blend_a=0.7
```

## 한계 / 검증 미해결
- swept-best 는 tuning set 에 fit — test Δ 가 음수면 default 가 robust (dataset-specific tuning 불필요) 라는 negative result.
- dialseg711 은 train split 없음 → 30/70 seeded split. tune 30% 가 test 70% 와 disjoint (no leakage).
- canonical default m=2 는 v4.1.3 보고 Score 가 나온 config (decision-log 2026-05-22).
