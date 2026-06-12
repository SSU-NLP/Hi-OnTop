# Supervised RoBERTa (SuperDialseg) — online (strict causal)

`methods/RoBERTa/online/segment.py` 산출. `methods/RoBERTa/offline/train.py` 학습 체크포인트를 **미래 발화 미관측** online 스타일로 추론.

## 1. 설정

- **체크포인트**: `/home/namchailin/Hi-OnTop/methods/RoBERTa/_roberta_unzip/roberta_seg_out/roberta_supervised/model` (offline Run-1, seed 42, 재학습 없음).
- **추론**: strict-causal. 경계 (t-1, t) 를 turn t 도착 시점의 causal 윈도우 `u_{max(0,t-20+1)..t}` 만으로 1회 결정 — look-ahead 0, 재수정 없음, turn 당 윈도우 1개.
- **offline 대비 유일한 차이**: offline 은 발화별 경계를 ~20 stride-1 윈도우(미래 최대 19발화 포함) logit 평균으로 결정. online 은 미래를 안 보는 단일 causal 윈도우. 모델·입력 인코딩·metric 동일 (offline 코드 import 재사용).
- **데이터**: `benchmarks/superdialseg_data/` 의 tiage/dialseg711/superseg test. metric = official Pk/WD (window=평균 segment 길이/2) + binary F1, `Score=(2·F1+(1−Pk)+(1−WD))/4`.
- 평균 turn 추론 latency ≈ 1487.47 ms (윈도우 1개 forward/turn).

## 2. 결과 — online vs offline(Run-1) vs 논문

| 평가셋 | online Pk↓ | online WD↓ | online F1↑ | **online Score↑** | offline Score | ΔScore (on−off) | 논문 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| superseg | 0.2275 | 0.2369 | 0.7958 | **0.7818** | 0.8158 | -0.0340 | 0.798 |
| tiage | 0.3705 | 0.3911 | 0.4786 | **0.5489** | 0.5383 | +0.0106 | 0.482 |
| dialseg711 | 0.2704 | 0.3192 | 0.6320 | **0.6686** | 0.6678 | +0.0008 | 0.702 |
| **mean-3** |  |  |  | **0.6664** | 0.6740 | -0.0075 |  |

## 3. 해석 / 판정

- online mean-3 Score = **0.6664** vs offline(Run-1) 0.6740 → ΔScore -0.0075. 최대 벤치별 |Δ| = 0.0340.
- offline 은 경계마다 ~20 윈도우 평균(미래 포함), online 은 미래를 안 보는 단일 윈도우 — 그 차이만큼의 소폭 하락은 정상. **큰 하락이면 구현 문제** (사용자 기준).
- online 은 발화 t-1 을 윈도우 끝에서 두 번째(학습 시 라벨 살아있던 non-last 위치)로 분류 → 모델 학습 분포와 정합. 구조적 mismatch 없음.

## 4. 한계 / 검증 미해결

- 단일 seed(Run-1) 체크포인트. offline 자체가 run 간 변동 있음 (특히 dialseg711) — 비교는 *동일 가중치* 기준 추론 차이만 격리.
- offline Score 는 Run-1 (`outputs/runs/_misc/...results.json`) 값 하드코딩. 다른 체크포인트로 바꾸면 offline 기준도 갱신 필요.
- per-turn latency 는 윈도우 1회 forward 기준 — batch 평균 측정값, 엄밀한 단건 측정 아님.

