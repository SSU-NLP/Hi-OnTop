# v4.2.3-exp smoke — dual-channel (mpnet topic + DSE-BERT flow)

**Setup**:
- v4.1.1 algorithm + v4.2.3 dual-channel PE override.
- topic: encoder=`sentence-transformers/multi-qa-mpnet-base-dot-v1`, m=2, ρ=0.7, a=0.5, δ*=0.5594 (TIAGE-train).
- flow:  encoder=`aws-ai/dse-bert-base`, m=2, ρ=0.5, a=0.0, δ*=0.4569 (TIAGE-train, DSE-BERT).
- weights: w_topic=0.75, w_flow=0.25.
- combine: r = √(w_topic·z_topic² + w_flow·z_flow²),  boundary ⇔ r ≥ 1.
- Dialseg711 uses TIAGE-train δ\* per encoder (no train split available).
- Metric: official SuperDialseg (Pk/WD k=auto, F1 binary), Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).

## 결과 vs v4.1.1 (mpnet) / v4.2.2 (DSE-only)

| arm | dataset | **Score ↑** | F1 ↑ | Pk ↓ | WD ↓ |
|---|---|---:|---:|---:|---:|
| **v4.2.3-exp dual** | tiage | **0.4904** | 0.4419 | 0.4218 | 0.5005 |
| v4.1.1 (mpnet) | tiage | 0.4675 | 0.4102 | 0.4421 | 0.5082 |
| v4.2.2 (DSE-BERT) | tiage | 0.4747 | 0.4420 | 0.4366 | 0.5488 |
|---|---|---|---|---|---|
| **v4.2.3-exp dual** | dialseg711 | **0.5666** | 0.5528 | 0.3525 | 0.4867 |
| v4.1.1 (mpnet) | dialseg711 | 0.5897 | 0.5493 | 0.3248 | 0.4151 |
| v4.2.2 (DSE-BERT) | dialseg711 | 0.4547 | 0.4587 | 0.4472 | 0.6514 |
|---|---|---|---|---|---|
| **v4.2.3-exp dual** | superseg | **0.4330** | 0.4016 | 0.4989 | 0.5726 |
| v4.1.1 (mpnet) | superseg | 0.4631 | 0.4323 | 0.4711 | 0.5410 |
| v4.2.2 (DSE-BERT) | superseg | 0.3886 | 0.3488 | 0.5332 | 0.6098 |
|---|---|---|---|---|---|

## 진단 metric (codex risk list)

| dataset | n_trans | corr(z_t,z_f) | both_high | both_low | only_t | only_f | boundary_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 1464 | 0.605 | 0.206 | 0.496 | 0.089 | 0.210 | 0.320 |
| dialseg711 | 18639 | 0.765 | 0.252 | 0.487 | 0.027 | 0.234 | 0.354 |
| superseg | 16006 | 0.733 | 0.241 | 0.525 | 0.100 | 0.133 | 0.365 |

**해석 / 판정**: (분석 후 채울 것)

## 한계 / 검증 미해결

- 단일 trial (w_topic=0.75, w_flow=0.25 고정). sweep 없음.
- (m, ρ, a) 채널별 분리는 했지만 train calibration 은 각 채널 단독 best 그대로 —   dual 결합 시 joint optimal 일 가능성 있음.
- seed=0 단일 run, variance 미측정.
- generic opener 등 발화 유형별 bucket 분석은 본 smoke 범위 밖.
