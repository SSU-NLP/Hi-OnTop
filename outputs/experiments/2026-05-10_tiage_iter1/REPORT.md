# TIAGE test — full compare (8 methods × 3 seeds)

Generated: 2026-05-10 (iter1)
Script: `scripts/run_tiage_full_compare.py`
Data: `benchmarks/tiage/data/personachat/anno/test/anno_test.json`
Encoder: `QueryEncoder` (default — Qwen3-Embedding-8B Crts proxy, dim=768)

## 실험 setup

- **목적**: v3.3.3-2 / v3.3.4-2 강화안이 v3.3.3 / v3.3.4 대비 segmentation 품질 향상이 있는지 TIAGE topic-shift 검출 F1 으로 1차 검증.
- **데이터**: TIAGE PersonaChat test split — 100 conversation / 1564 turn / 315 GT shift transition.
- **방법 (8개)**: v1 / v3.1.1 / v3.3.1 / v3.3.2 / v3.3.3 / v3.3.4 / v3.3.3-2 / v3.3.4-2.
- **HP**: 모든 v3.3.x 는 LoCoMo 추천 default (`alpha=100, lmda=10, tau=50, cos_threshold=0.9, beta=0.25, pe_threshold=0.5, rnn_train_steps=3`). v1 은 persistence default (`alpha=1, lmda=10, sigma0_sq=0.01`), v3.1.1 은 cosine-MAP default (`alpha=1, lmda=10, tau=50, cos_threshold=0.7`). **TIAGE-specific tuning 없음** — 의도적으로 LoCoMo HP 그대로 적용해서 method 자체의 차이를 측정하려 함.
- **Seed**: RNN-있는 method (v3.3.x) 는 `[0, 1, 2]` (3 seed 평균). v1 / v3.1.1 은 RNN 없으므로 1 seed.
- **Metric**: turn transition binary F1 (label `'1'` = shift). precision / recall / mean n_topics_per_conv 함께 보고. timing 은 ms/turn (encoding 제외, segment 시간만).

## 결과

| method | n_seeds | F1 (mean ± std) | P (mean) | R (mean) | n_topics (mean) | ms/turn |
|---|---:|---:|---:|---:|---:|---:|
| v3.1.1 | 1 | **0.377 ± 0.000** | 0.235 | 0.956 | 13.3 | 0.09 |
| v1 | 1 | 0.363 ± 0.000 | 0.252 | 0.654 | 7.5 | 0.21 |
| v3.3.1 | 3 | 0.354 ± 0.000 | 0.215 | 1.000 | 15.6 | 1.14 |
| v3.3.2 | 3 | 0.354 ± 0.000 | 0.215 | 1.000 | 15.6 | 0.90 |
| v3.3.3 | 3 | 0.354 ± 0.000 | 0.215 | 1.000 | 15.6 | 0.99 |
| v3.3.4 | 3 | 0.354 ± 0.000 | 0.215 | 1.000 | 15.6 | 0.85 |
| v3.3.3-2 | 3 | 0.354 ± 0.000 | 0.215 | 1.000 | 15.6 | 0.98 |
| v3.3.4-2 | 3 | 0.354 ± 0.000 | 0.215 | 1.000 | 15.6 | 0.86 |

**best F1**: `v3.1.1` — 0.377 ± 0.000

## 해석

1. **v3.3.x 6개 method 가 비트 단위로 동일** (F1 / P / R / n_topics 모두 같음, std=0). 이는 method 간 차이가 사라진 것이 아니라, **default HP 가 너무 공격적이어서 모든 v3.3.x 가 "매 turn 새 topic 생성" corner case 에 빠진 결과**. 근거:
   - `R = 1.000` (모든 GT shift 를 잡음) + `P = 0.215` (15.6 topic 중 거의 모두 false positive) → 사실상 every-transition prediction.
   - `n_topics_mean = 15.6` 가 평균 dialog 길이 (1564 / 100 ≈ 15.6) 와 동일 → **거의 모든 turn 이 별개 topic** 으로 할당.
   - 이 corner case 에서는 segmenter 내부 (RNN PE / variance / f0 / posterior odds) 가 최종 prediction 에 영향을 줄 여지가 없음 → 모두 동일 결과.

2. **이는 v3.3.3-2 / v3.3.4-2 가 효과 없다는 증거가 아님**. TIAGE 짧은 dialog (~15 turn / conv) + PersonaChat 분포 + default `alpha=100, cos_threshold=0.9` 의 조합이 모든 v3.3 변형을 동일 corner case 로 밀어 넣음. 의미 있는 비교를 위해서는 **TIAGE-specific HP sweep** 이 필요 (특히 `alpha`, `cos_threshold`, `beta`).

3. v3.1.1 / v1 만 LoCoMo 와 다른 default HP 를 써서 corner case 를 피했고, 따라서 의미 있는 P/R 분포를 보임. 이 둘의 절대 F1 (0.36~0.38) 을 v3.3.x 의 baseline 으로 직접 비교하기는 어렵다 — HP 가 다르기 때문.

## 판정 (iter1)

**v3.3.3-2 / v3.3.4-2 의 TIAGE 효과 검증 실패 (HP 부적절)**. method 차이를 분간할 수 없으므로 "TIAGE 별로면 다시 구현" 규칙을 곧장 적용하지 않는다. 대신 다음을 선행한다:

1. **TIAGE-specific HP grid sweep** (다음 iteration). 각 v3.3.x method 에 대해 `alpha ∈ {1, 10, 30, 100}` × `cos_threshold ∈ {0.5, 0.7, 0.85, 0.9}` × `beta ∈ {0.25, 0.5, 1.0}` 4×4×3 grid → method 별 best HP 발견 후 재비교.
2. 또는 LoCoMo iter1 결과로 main 판정. LoCoMo 가 v3.3.3-2 / v3.3.4-2 향상 신호를 보이면 그쪽이 우선 신호.

## 한계 / 검증 미해결

1. **HP 적합성 미확인** — LoCoMo 추천 HP 가 TIAGE 환경 (짧은 dialog, dense topic shift) 에 부적절. 이로 인해 v3.3.x 6개의 method-특이적 메커니즘 (variance calibration / restart branch / prototype f0 / σ² shrinkage) 이 작동했는지 자체를 검증 못 함.
2. **Seed 영향 부재** — v3.3.x 가 모두 corner case 라 RNN init seed (0/1/2) 차이가 결과에 영향 0. 정상 HP 조건에서 seed 분산이 어떨지 별도 측정 필요.
3. **Recall=1.0 / Precision 낮음** 은 사실상 "all-boundary baseline" 과 동치. all-boundary baseline F1 (= 2·315/(315+1564) ≈ 0.336) 과 거의 같으며, v3.3.x 가 그 baseline 을 넘지 못함을 의미.
4. v3.1.1 (best, 0.377) 도 LoCoMo Phase-4 평가에서 v3.3.x 보다 retrieval / acc 가 낮았음 — TIAGE F1 만으로 LoCoMo 우열을 결론 짓지 않음.
