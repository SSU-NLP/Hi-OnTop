# v4.1.2-topicctx ABLATION × 3 benches (smoke, fixed δ*)

fixed: ρ=0.7 a=0.5 δ*=0.5594 α=1 λ=10 (TIAGE-cfg) · ctx_max_len=64. **δ* 재calib 미수행** — codex 권고: 1차 효과 방향 확인만. 본평가는 재calib 후 별도.

## Score 매트릭스

| policy | tiage | dialseg711 | superseg | mean | vs window m=2 |
|---|:---:|:---:|:---:|---:|---:|
| v4.1.1 window m=2 (baseline) | 0.4675 | 0.5897 | 0.4631 | 0.5068 | +0.0000 |
| v4.1.2-topic_cur | 0.4489 | 0.5763 | 0.4413 | 0.4888 | -0.0180 |
| v4.1.2-topic_prev | 0.4375 | 0.6194 | 0.4296 | 0.4955 | -0.0113 |

## 세부 (F1/Pk/WD)

| dataset | policy | F1 | Pk | WD | Score |
|---|---|---:|---:|---:|---:|
| tiage | v4.1.1 window m=2 (baseline) | 0.4102 | 0.4421 | 0.5082 | 0.4675 |
| tiage | v4.1.2-topic_cur | 0.3851 | 0.4429 | 0.5318 | 0.4489 |
| tiage | v4.1.2-topic_prev | 0.3519 | 0.4521 | 0.5016 | 0.4375 |
| dialseg711 | v4.1.1 window m=2 (baseline) | 0.5493 | 0.3248 | 0.4151 | 0.5897 |
| dialseg711 | v4.1.2-topic_cur | 0.5229 | 0.3224 | 0.4185 | 0.5763 |
| dialseg711 | v4.1.2-topic_prev | 0.5607 | 0.2939 | 0.3500 | 0.6194 |
| superseg | v4.1.1 window m=2 (baseline) | 0.4323 | 0.4711 | 0.5410 | 0.4631 |
| superseg | v4.1.2-topic_cur | 0.4119 | 0.4795 | 0.5791 | 0.4413 |
| superseg | v4.1.2-topic_prev | 0.3786 | 0.4930 | 0.5461 | 0.4296 |

## 한계 / 정직성
- **δ* 고정 (TIAGE-train 0.5594)** — topic-aware ctx 는 δ_ctx 분포를 바꾸므로 본평가는 δ* 재calib 필요(codex 권고).
- ctx_max_len=64 안전캡 (실 대화는 거의 안 닿음).
- baseline = v4.1.1 fixed-window m=2 (현 TIAGE-cfg). m=8 (dialseg711 best) 와의 비교는 정책 선택 trade-off 평가 시 별도 필요.
- 판단 기준 (codex): tiage/superseg long-ctx 손실 회복 + dialseg711 long-ctx 이득 보존을 각각 확인 (mean 만 보지 말 것).
- ablation 위치: `v4.1.2-topicctx` 후보, default 승격은 별건.
