# Hi-OnTop vs online 베이스라인 비교표 (Def-DTS 번들, 벤치별)

전부 **online (past-only)**. metric = segeval Pk/WD + boundary-set F1, Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD). 데이터 = Def-DTS 번들 test set (베이스라인과 동일).

- **Ours (p80)** : δ* = test δ_eff 80-percentile (label-free, 누수 없음). 베이스라인과 동일 데이터·무감독 → **메인 공정 비교 행**.
- **Ours (best-Score)** : δ* = labeled split 에서 Score 최대화 (supervised). **참고: supervised ceiling** (Ours 만 train 튜닝 → 베이스라인과 1:1 공정비교 아님).
- Hi-OnTop HP: m=2, ρ=0.7, a=0.5, 인코더 multi-qa-mpnet.
- 베이스라인 수치 = 각 baseline REPORT.md (TextTiling-streaming / GreedySeg-online-delay2 / GraphSeg-window-d) 그대로 인용.
- CSM-style : 별도 실행 대기 (csm_online.py, 본 표에 추후 추가).

## tiage  (test 100 dialog)

δ* — p80=0.5851 · best-Score=0.5619 (train split (300/300 dial))

| Method (전부 Online) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | Avg turn latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|
| TextTiling-style | 0.5266 | 0.5476 | 0.2252 | 0.3441 | 0.008 |
| GreedySeg-style | 0.5370 | 0.5535 | 0.1420 | 0.2984 | 13.810 |
| GraphSeg-style | 0.4925 | 0.5159 | 0.2517 | 0.3738 | 0.850 |
| CSM-style | _대기_ | _대기_ | _대기_ | _대기_ | _대기_ |
| **Ours (p80)** | 0.4436 | 0.4904 | 0.3020 | **0.4175** | 0.2676 |
| Ours (best-Score)‡ref | 0.4454 | 0.5199 | 0.3415 | 0.4294 | 0.5604 |

## dialseg711  (test 711 dialog)

δ* — p80=0.5847 · best-Score=0.5873 (‡ train split 부재 → test 보정 (in-domain upper-bound))

| Method (전부 Online) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | Avg turn latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|
| TextTiling-style | 0.4713 | 0.4896 | 0.2708 | 0.3952 | 0.010 |
| GreedySeg-style | 0.4164 | 0.4434 | 0.4120 | 0.4911 | 16.270 |
| GraphSeg-style | 0.4485 | 0.4816 | 0.3656 | 0.4503 | 1.150 |
| CSM-style | _대기_ | _대기_ | _대기_ | _대기_ | _대기_ |
| **Ours (p80)** | 0.3223 | 0.3655 | 0.5346 | **0.5953** | 0.3412 |
| Ours (best-Score)‡ref | 0.3194 | 0.3595 | 0.5333 | 0.5969 | 0.2818 |

## superseg  (test 1322 dialog)

δ* — p80=0.6148 · best-Score=0.4941 (train split (600/6948 dial subsample))

| Method (전부 Online) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | Avg turn latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|
| TextTiling-style | 0.4623 | 0.4667 | 0.2618 | 0.3987 | 0.011 |
| GreedySeg-style | 0.5072 | 0.5109 | 0.2775 | 0.3842 | 9.140 |
| GraphSeg-style | 0.5387 | 0.5423 | 0.1644 | 0.3120 | 0.620 |
| CSM-style | _대기_ | _대기_ | _대기_ | _대기_ | _대기_ |
| **Ours (p80)** | 0.5022 | 0.5261 | 0.3083 | **0.3971** | 0.0717 |
| Ours (best-Score)‡ref | 0.4773 | 0.6609 | 0.4473 | 0.4391 | 0.0346 |

## 한계 / 검증 미해결
- 데이터 = Def-DTS 번들. Hi-OnTop 의 superdialseg_data 보고 수치(0.4675 등)와는 dialseg711/superseg turn 수가 달라 직접 일치 안 함.
- dialseg711 은 번들에 train split 부재 → best-Score δ* 를 test 에서 보정(‡). 이 행은 in-domain upper-bound 로, 공정 비교는 p80 행.
- best-Score 행 전반: Ours 만 라벨로 튜닝 → 베이스라인과 비대칭. 메인 비교는 p80 행으로 읽을 것.
- CSM-style 행 미측정 (csm_online.py 별도 실행 예정).
