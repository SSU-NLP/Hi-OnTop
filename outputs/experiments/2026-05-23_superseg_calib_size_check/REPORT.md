# SuperDialseg calib-size 민감도 — MPNet bootstrap

**질문**: 400 calib dialog 이 superseg Score 0.43 의 병목인가?

**방법**: 기존 MPNet train 캐시 400 dialog → bootstrap subsample N ∈ {50, 100, 200, 300, 400}, 각 N 에 3 seed → δ*_p80 / δ*_best-Score 계산 후 full 1322 test 에서 Score 측정.

**HP**: Hi-OnTop m=2, ρ=0.7, a=0.5. metric = 공식 SuperDialseg (0.5F1+0.25(1−Pk)+0.25(1−WD)).

## 결과 (mean ± std, 3 seeds)

| calib N | δ*_p80 | δ*_best | Score (p80) | Score (best) |
|---:|---:|---:|---:|---:|
| 50 | 0.6101 ± 0.0045 | 0.5500 ± 0.0209 | 0.4324 ± 0.0046 | 0.4612 ± 0.0016 |
| 100 | 0.6108 ± 0.0013 | 0.5331 ± 0.0220 | 0.4314 ± 0.0010 | 0.4602 ± 0.0047 |
| 200 | 0.6096 ± 0.0016 | 0.5161 ± 0.0192 | 0.4321 ± 0.0012 | 0.4571 ± 0.0051 |
| 300 | 0.6107 ± 0.0007 | 0.5500 ± 0.0048 | 0.4310 ± 0.0004 | 0.4632 ± 0.0008 |
| 400 | 0.6125 ± 0.0000 | 0.5534 ± 0.0000 | 0.4304 ± 0.0000 | 0.4626 ± 0.0000 |

## 해석

- N=50 부터 N=400 까지 Score 가 noise band 안에서 평탄 (Δp80 ≈ −0.002, Δbest ≈ +0.001 — 둘 다 3-seed σ 안).
- δ*_p80 은 N=100 부터 0.610 ± 0.001 로 안정. δ*_best 는 sweep grid (0.01 폭) 인접 후보 사이를 왔다갔다 하지만 Score 곡선이 그 영역에서 평탄해 Score-impact 미미.
- 즉 **N=50 도 이미 calib 가 천장에 도달**.

## 절대 천장 (data-snooping 검증)

test 1322 dialog 자체에서 δ* sweep (calib 와 무관한 알고리즘 천장):

- δ* = 0.5432, **Score = 0.4643**

calib 측 최고 (N=400 best-Score) = 0.4626 → **천장과 0.0017 차이**. 어떤 calib 전략을 써도 (라벨까지 다 봐도) ~+0.002 이상 못 끌어올림.

## 결론

- 400 dialog 가 superseg ~0.43–0.46 Score 천장의 원인이 **아님** — 50 도 같은 천장 도달.
- 실제 천장은 **Hi-OnTop 알고리즘의 데이터 적합도** (m=2 local context + bounded cosine 으로 doc-grounded subtopic shift 감지 한계). 다음 결정은 calib 보강이 아니라 **알고리즘 측 변경** — 예: m=3+ context, learnable scoring, doc-grounded 구조 활용.

## 한계

- bootstrap 은 same 400 base 에서 가져옴 → 400 이상의 domain coverage 증가 효과는 직접 측정 안 됨. 그러나 test-side oracle (0.4643) 이 절대 천장이라서, domain coverage 가 늘어도 +0.002 이상 불가.


### 확장 N=2000 — MINILM-INT8

| calib N | δ*_p80 | δ*_best | Score (p80) | Score (best) |
|---:|---:|---:|---:|---:|
| 400 | 0.8409 | 0.6042 | 0.3650 | 0.4336 |
| 1000 | 0.8329 | 0.6246 | 0.3688 | 0.4360 |
| 2000 | 0.8335 | 0.5941 | 0.3671 | 0.4315 |

**N=400 → N=2000 변화량**: Score(p80) +0.0022, Score(best) -0.0021.


### 확장 N=2000 — MINILM (fp32)

| calib N | δ*_p80 | δ*_best | Score (p80) | Score (best) |
|---:|---:|---:|---:|---:|
| 400 | 0.8352 | 0.6856 | 0.3720 | 0.4347 |
| 1000 | 0.8353 | 0.6042 | 0.3722 | 0.4369 |
| 2000 | 0.8342 | 0.6551 | 0.3725 | 0.4372 |

**N=400 → N=2000 변화량**: Score(p80) +0.0005, Score(best) +0.0025.



---

## 확장 검증: N=2000 (실제 domain coverage 증가)

기존 N=400 결과를 의심한 사용자 요청 → 기존 400 cache 재사용 + 추가 1600 dialog MPNet 인코딩 → 진짜 N=2000 으로 재측정.

**인덱스 결정**: `run_encoder_comparison.py` 의 mpnet 패스 RNG 상태 재현 (`rng=default_rng(0); _=rng.permutation(711); perm=rng.permutation(6948)`). N ∈ {400, 1000, 2000} 모두 perm 의 prefix 라 strict subset 관계.

| calib N | δ*_p80 | δ*_best | Score (p80) | Score (best) |
|---:|---:|---:|---:|---:|
| 400 | 0.6125 | 0.5534 | 0.4304 | 0.4626 |
| 1000 | 0.6138 | 0.5127 | 0.4283 | 0.4571 |
| 2000 | 0.6152 | 0.5127 | 0.4269 | 0.4571 |

**N=400 → N=2000 변화량**: Score(p80) -0.0035, Score(best) -0.0055.

**해석**: 5배 (400→2000) calib 증가에도 Score 변화는 0.0035/0.0055 → bootstrap 결과 (N=50 도 천장) 와 일관. domain coverage 효과도 작음 — **calib 크기는 superseg ~0.46 천장의 원인이 확정적으로 아님**.

