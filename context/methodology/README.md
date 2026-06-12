# Hi-OnTop Methodology — version index

> 이 파일 위치: **`context/methodology/README.md`**

이 디렉토리 (`context/methodology/`) 의 역할:

- 각 버전 (v1, v2, v3.1.1, …) 의 **알고리즘 수준 차이**를 한 파일씩 정리.
- 사소한 변경 (prediction caching, hyperparameter default, prior decay 등) 도 해당 버전 파일의 "고려 중인 변형" / "변경 이력" 섹션에 누적.
- 코드 위치 / 인프라 (cache, encoder lock) 는 같은 디렉토리의 [`infrastructure.md`](infrastructure.md) 별도.

## 버전 계보

```
v1   (Gaussian likelihood, flat history)
 │
 └→ v2   (= v1 segmenter 그대로; 알고리즘 변경 없음)
       │
       └→ v3.1.1   (likelihood: Gaussian → bounded cosine, centroid-only topic)
              │
              └→ v3.2.1   (sticky-CRP count: raw → sub-linear (C_k+1)^β)
                     │
                     └→ v3.3.1   (likelihood term: cos(s, μ_k) → cos(s, ŝ_k), per-topic RNN)
                            │
                            ├→ v3.3.2   (+ SEM2 surprise → hard boundary on PE spike)
                            │   │
                            │   ├→ v3.3.3   (+ SEM2 f0 / same-label restart 분기 복원)
                            │   │     │
                            │   │     ├→ v3.3.3-2   [DEPRECATED] boundary-start prototype top-M f0 + posterior-odds restart
                            │   │     ├→ v3.3.3-3   [DEPRECATED] + restart hysteresis + prototype softening
                            │   │     └→ v3.3.3-4   minimal segmenter (episode_id emit)
                            │   │
                            │   ├→ v3.3.4   (hard PE → per-topic σ²_k 로 calibrated likelihood)
                            │   │     │
                            │   │     ├→ v3.3.4-2   per-topic σ²_k Bayesian shrinkage (η_k = n_k/(n_k+c))
                            │   │     │
                            │   │     └→ v3.3.5   (SEM2 f_is_trained cold-start gating 복원)
                            │   │           └→ v3.3.6   (SEM2 persistence+history-replay dynamics + per-topic model + seed)
                            │   │                 └→ v3.3.7   (SEM2 map_variance σ², n≥2)  [#14|15 미해결 — 경험적 반증]
                            │   │                       └→ v3.3.8   (SEM2 fresh-baseline pe_prior + non-prev f0)  [pe_prior 벤치 calibration 미완]
                            │   │                             ┄→ v3.3.9 (계획: session_id 미사용 emergent SEM segmenter; time-aware prior 는 v3.3.10 후보)
```

## 한 줄 요약

| 버전 | segmenter class | 핵심 차이 |
|---|---|---|
| [v1](v1.md) | `HiOnTopSegmenter` | SEM2 본진 — Gaussian likelihood + raw sticky-CRP |
| [v2](v2.md) | `HiOnTopSegmenter` | v1 segmenter 그대로 (segmentation 알고리즘 변경 없음) |
| [v3.1.1](v3.1.1.md) | `HiOnTopSegmenterV3` | Gaussian → bounded cosine `τ·cos(s, μ_k)` |
| [v3.2.1](v3.2.1.md) | `HiOnTopSegmenterV32` | sticky-CRP count 식에 sub-linear `(C_k+1)^β` |
| [v3.3.1](v3.3.1.md) | `HiOnTopSegmenterV331` | μ_k 대신 per-topic GRU 예측 ŝ_k로 cos 점수 |
| [v3.3.2](v3.3.2.md) | `HiOnTopSegmenterV332` | + `max_k cos < 1−pe_threshold` 면 새 topic 강제 (SEM2 surprise) |
| [v3.3.3](v3.3.3.md) | `HiOnTopSegmenterV333` | + 같은 label 의 repeat / restart 분기 (`log_likelihood_f0` 복원) |
| [v3.3.3-2](v3.3.3-2.md) ⚠ DEPRECATED | `HiOnTopSegmenterV333_2` | boundary-start prototype top-M f0 + posterior-odds restart |
| [v3.3.3-3](v3.3.3-3.md) ⚠ DEPRECATED | `HiOnTopSegmenterV333_3` | + restart hysteresis + prototype softening (mean+logmeanexp) |
| [v3.3.3-4](v3.3.3-4.md) | `HiOnTopSegmenterV333_4` | minimal segmenter (episode_id emit; pivot 시 메모리-시대 항목 제거) |
| [v3.3.4](v3.3.4.md) | `HiOnTopSegmenterV334` | hard PE rule 제거, likelihood 를 `−PE²/(2σ_k²)` 로 calibrate |
| [v3.3.4-2](v3.3.4-2.md) | `HiOnTopSegmenterV334_2` | + per-topic σ²_k Bayesian shrinkage (η_k = n_k/(n_k+c)) |
| [v3.3.5](v3.3.5.md) | `HiOnTopSegmenterV335` | SEM2 `f_is_trained` gating 복원: untrained topic = fresh 와 동일 L0 (chicken-and-egg 해소) |
| [v3.3.6](v3.3.6.md) | `HiOnTopSegmenterV336` | SEM2 dynamics: untrained=persistence, per-topic model, history replay, **결정적 seed** (3턴천장·재현성 해소) |
| [v3.3.7](v3.3.7.md) | `HiOnTopSegmenterV337` | SEM2 `map_variance` posterior σ² (n≥2). ⚠ idx374 #14\|15 미해결 — 경험적 반증 기록 |
| [v3.3.8](v3.3.8.md) | `HiOnTopSegmenterV338` | SEM2 fresh-baseline `pe_prior`(cos_threshold 대체) + non-prev f0. ⚠ pe_prior 벤치 calibration 미완 (default 1.0=원칙값) |

## 버전별 hyperparameter 요약 (current sweep value)

**표 보는 법**:

- 셀 값 = **가장 최근 sweep 에서 사용한 값** (= 가장 최근 best HP).
- **★** = 해당 버전의 *코드 default 와 다른 값* (= sweep 으로 튜닝됨).
- ★ 없음 = 셀 값이 코드 default 와 동일.
- `—` = 해당 버전이 그 HP 를 사용하지 않음.

**그리스/ASCII 표기 규약** (한 번만 적음):

- `α` = `alpha`, `λ` = `lmda`, `cos` = `cos_threshold`, `β` = `beta`, `σ²₀` = `sigma0_sq`

**코드 default 위치** (`src/hi_ontop/sem_core_v*.py` 의 `__init__`):

- v1 / v3.1.1 / v3.2.1 / v3.3.1 / v3.3.2 → `alpha=1`, `cos=0.7`, `beta=0.5`, `pe_threshold=1.0`, `rnn_train_steps=1`
- v3.3.3 / v3.3.4 → `alpha=100`, `cos=0.9`, `beta=0.25`, `pe_threshold=0.5`, `rnn_train_steps=3` (codex 가 v3.3.2 best HP 를 default 로 박음)

**표에서 생략된 HP**:

- RNN 보조 (모든 v3.3.x 공통): `rnn_hidden_dim=32`, `rnn_lr=1e-3`, `rnn_max_context=8`, `rnn_min_history=2`
- 자세히는 해당 버전의 `vX.Y.Z.md` 참조.

| HP | v1 | v3.1.1 | v3.2.1 | v3.3.1 | v3.3.2 | v3.3.3 | v3.3.4 |
|---|---|---|---|---|---|---|---|
| `α` (alpha) | 1 | 10★ | 100★ | 100★ | 100★ | 100 | 100 |
| `λ` (lmda) | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| `σ²₀` (sigma0_sq) | 0.01 | — | — | — | — | — | — |
| `τ` (tau) | — | 50 | 50 | 50 | 50 | 50 | 50 |
| `cos` (cos_threshold) | — | 0.3★ | 0.9★ | 0.9★ | 0.9★ | 0.9 | 0.9 |
| `β` (beta) | — | — | 0.25★ | 0.25★ | 0.25★ | 0.25 | 0.25 |
| `pe_threshold` | — | — | — | — | 0.5★ | 0.5 | - |
| `rnn_train_steps` | — | — | — | 1 | 3★ | 3 | 3 |
| `restart_pe_threshold` | — | — | — | — | — | 0.5 | — |
| `restart_margin` | — | — | — | — | — | 0.0 | — |
| `f0_tau` | — | — | — | — | — | =τ | — |
| `f0_min_starts` | — | — | — | — | — | 2 | — |
| `pe_var_decay` | — | — | — | — | — | — | 0.95 |
| `pe_var_min_samples` | — | — | — | — | — | — | 5 |
| `pe_var_sigma0_sq` | — | — | — | — | — | — | 0.04 |
| `pe_var_min_sq` | — | — | — | — | — | — | 1e-4 |
| `pe_var_max_sq` | — | — | — | — | — | — | 0.25 |
| `var_likelihood_weight` | — | — | — | — | — | — | 1.0 |

## Cross-cutting infrastructure

버전 무관한 인프라 / 설계 결정은 [`infrastructure.md`](infrastructure.md) 별도. 다음 항목 정리:

- `EncoderCache` (segmentation 임베딩 cache)
- encoder lock / `--no-thinking` 플래그
- SEM2 cold-start gating / dynamics / σ² / fresh-baseline 계열
- 재현성 segmenter seed
- online / streaming baseline segmenter (TextTiling / GreedySeg / GraphSeg)

## 표기 규약

- `s` : 새로 들어온 turn의 L2-normalized embedding (768-D)
- `μ_k` : topic k의 centroid
- `ŝ_k` : topic k의 *예측된 다음 embedding* (v3.3.1+에서만 의미 있음)
- `C_k` : topic k의 누적 assignment 수
- `e_{n-1}` : 직전 turn의 topic id
- `α` (alpha) : sCRP 새 cluster prior weight
- `λ` (lmda) : sCRP stickiness
- `τ` (tau) : cosine likelihood temperature (v3.x)
- `β` (beta) : sub-linear count exponent (v3.2+)
- `σ²₀` (sigma0_sq) : Gaussian cold-start 분산 (v1/v2 only)
- `σ²_k` (pe_var) : v3.3.4 의 per-topic PE EMA variance

## 작성 규칙

**새 버전 추가 시 — 문서 측면**:

- `vX.Y.Z.md` 신규 작성 (직전 버전 파일을 템플릿으로).
- 이 README 갱신:
  - 계보 그림에 entry 추가
  - 한 줄 요약 표에 row 추가
  - HP 매트릭스에 column 추가
- `context/06-decision-log.md` 에 채택/폐기 사유 + 날짜 append.

**새 버전 추가 시 — 코드 측면**:

- `src/hi_ontop/sem_core_vXYZ_*.py` 신규 + topic 클래스 (필요시).
- segmenter `version == "vX.Y.Z"` dispatch 분기 + 새 HP 시그니처 추가.
- 분절 러너의 `--version` choices 에 추가.
- `tests/test_vXYZ_*.py` 단위 테스트.

**기존 버전에 마이크로 변경 적용 시**:

- 해당 `vX.Y.Z.md` 의 "변경 이력" 섹션에 추가 (날짜 + 한 줄).
- 알고리즘 의미가 바뀌면 새 버전 파일로 분리.
- 단순 성능 / 캐시 최적화는 같은 파일 안에 누적.
