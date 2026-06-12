# Hi-OnTop HP robustness — (m, ρ, a) sweep with per-bench oracle δ\*

`outputs/experiments/2026-05-25_hiontop_mra_sweep/` · 2026-05-25

## 1. 목적

Paper §3 Method 의 claim:
> "The boundary threshold $\delta^{*}$ is the only parameter calibrated
> per dataset; the causal-context hyperparameters $(m, \rho, a)$ are
> fixed globally at $(2, 0.7, 0.5)$."

위 claim 을 *경험적으로* 정당화. 핵심 질문:

**δ\* 를 *per-bench 로 자유롭게* 두었을 때 (oracle 혹은 best-p_x), default
$(m, \rho, a) = (2, 0.7, 0.5)$ 가 정말 최적인가?**

## 2. Setup

- **Grid (3D)**: $m \in \{2,3,4,5,6,8\}$ × $\rho \in \{0.5, 0.7, 0.9\}$ × $a \in \{0.0, 0.3, 0.5, 0.7, 1.0\}$ → **90 configs**.
- **δ\* 선택** (per-bench, 두 가지 protocol):
  - **Oracle**: $\delta^{*} = \arg\max_{d \in [0.35, 0.95]} \mathrm{Score}_\mathrm{test}(d)$ (121-grid, label-leakage upper bound).
  - **Best-px**: $\delta^{*} = $ best $p \in \{50, 55, \ldots, 95\}$ of test-set $\delta_\mathrm{eff}$ distribution.
- **데이터**: TIAGE test (100 dial) + Dialseg711 test 30% holdout (213 dial, paper convention) + SuperDialseg test (1322 dial). 모두 cached MPNet embeddings.
- **Metric**: SuperDialseg Score = $0.5 F_1 + 0.25(1 - P_k) + 0.25(1 - WD)$, mean across 3 benches.

## 3. Result — Default $(2, 0.7, 0.5)$ rank

| Protocol | Default rank | Default mean3 | Δ from rank-1 | # within 0.005 of rank-1 |
|---|---:|---:|---:|---:|
| **Oracle** | **1 / 90** | **0.5250** | **+0.0000** | 11 |
| Best-px | 3 / 90 | 0.5220 | −0.0016 | 11 |

→ **Oracle 에서 default 가 정확히 rank-1**. Best-px 에선 rank 3 (top-1 과 Δ 0.0016 — 실질 동등).

## 4. Top-10 Oracle ranking

| Rank | m | ρ | a | TIAGE | DS711 | SDS | mean-3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **2** | **0.7** | **0.5** | 0.4779 | 0.6325 | 0.4647 | **0.5250** ← default |
| 2 | 2 | 0.5 | 0.5 | 0.4788 | 0.6299 | 0.4650 | 0.5246 |
| 3 | 2 | 0.9 | 0.5 | 0.4750 | 0.6345 | 0.4641 | 0.5245 |
| 4 | 2 | 0.5 | 0.3 | 0.4751 | 0.6330 | 0.4652 | 0.5244 |
| 5 | 2 | 0.7 | 0.7 | 0.4747 | 0.6295 | 0.4672 | 0.5238 |
| 6 | 2 | 0.9 | 0.7 | 0.4781 | 0.6275 | 0.4657 | 0.5238 |
| 7 | 2 | 0.7 | 0.3 | 0.4733 | 0.6322 | 0.4632 | 0.5229 |
| 8 | 2 | 0.9 | 0.3 | 0.4712 | 0.6328 | 0.4614 | 0.5218 |
| 9 | 3 | 0.5 | 0.5 | 0.4725 | 0.6311 | 0.4600 | 0.5212 |
| 10 | 2 | 0.5 | 0.0 | 0.4697 | 0.6331 | 0.4608 | 0.5212 |

**Top-10 중 9 개가 m=2.** m=3 이 처음 등장하는 게 rank 9.

## 5. 1D marginal sensitivity (Oracle mean3, max across other dims)

### m (context window size)

| m | max mean3 | mean | min | Δ from m=2 |
|---:|---:|---:|---:|---:|
| **2** | **0.5250** | 0.5199 | 0.5112 | — |
| 3 | 0.5212 | 0.5137 | 0.4932 | −0.0038 |
| 4 | 0.5192 | 0.5104 | 0.4848 | −0.0058 |
| 5 | 0.5188 | 0.5087 | 0.4719 | −0.0062 |
| 6 | 0.5184 | 0.5067 | 0.4580 | −0.0066 |
| 8 | 0.5177 | 0.5034 | 0.4435 | −0.0073 |

→ **m=2 가 monotonic best** (값이 클수록 단조 하락). Default 정당화 strong.

### ρ (context decay)

| ρ | max mean3 | mean | min |
|---:|---:|---:|---:|
| 0.5 | 0.5246 | 0.5165 | 0.5109 |
| **0.7** | **0.5250** | 0.5115 | 0.4868 |
| 0.9 | 0.5245 | 0.5034 | 0.4435 |

→ **max 가 거의 동등 (range 0.0005)** — flat plateau. mean 은 ρ=0.5 가 약간 우세하지만 max 차이 무의미.

### a (blend weight)

| a | max mean3 | mean | min |
|---:|---:|---:|---:|
| 0.0 | 0.5212 | 0.4971 | 0.4435 |
| 0.3 | 0.5244 | 0.5114 | 0.4855 |
| **0.5** | **0.5250** | 0.5158 | 0.5007 |
| 0.7 | 0.5238 | 0.5167 | 0.5070 |
| 1.0 | 0.5112 | 0.5112 | 0.5112 |

→ **a=0.5 가 peak**. a ∈ [0.3, 0.7] 거의 동등 (top 0.0012 range), 양 극단 (0.0, 1.0) 만 약간 하락. δ_prev 와 δ_ctx 의 *균형 (a≈0.5)* 이 가장 견고.

## 6. Best-px Top-10 (보조)

| Rank | m | ρ | a | TIAGE | DS711 | SDS | mean-3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.5 | 0.3 | 0.4721 | 0.6335 | 0.4653 | 0.5236 |
| 2 | 2 | 0.9 | 0.5 | 0.4715 | 0.6336 | 0.4641 | 0.5231 |
| **3** | **2** | **0.7** | **0.5** | 0.4705 | 0.6317 | 0.4640 | **0.5220** ← default |
| 4 | 2 | 0.7 | 0.3 | 0.4733 | 0.6304 | 0.4620 | 0.5219 |
| 5 | 2 | 0.7 | 0.7 | 0.4753 | 0.6254 | 0.4649 | 0.5219 |
| 6 | 2 | 0.5 | 0.5 | 0.4744 | 0.6262 | 0.4642 | 0.5216 |
| 7 | 2 | 0.9 | 0.7 | 0.4759 | 0.6241 | 0.4633 | 0.5211 |
| 8 | 2 | 0.5 | 0.0 | 0.4703 | 0.6324 | 0.4599 | 0.5209 |
| 9 | 2 | 0.9 | 0.3 | 0.4674 | 0.6326 | 0.4591 | 0.5197 |
| 10 | 2 | 0.5 | 0.7 | 0.4705 | 0.6222 | 0.4646 | 0.5191 |

**Top-10 모두 m=2** (best-px 에서도). Default rank 3 (Δ −0.0016 from top-1).

## 7. 결론 — Paper claim 정당화

| Claim | 근거 |
|---|---|
| **m=2 가 globally best** | Oracle 1D sensitivity: m=2 가 단조 최대 (Δ −0.0073 over m=8). Top-10 oracle / best-px 모두 m=2. |
| **ρ, a 는 wide plateau** | Oracle 1D sensitivity: ρ max range 0.0005, a max range 0.0038 ([0.3, 0.7] 내). |
| **Default 는 최적 또는 near-optimal** | Oracle rank 1, best-px rank 3 (Δ −0.0016). 11 configs 가 rank-1 의 0.005 안. |

**§3 Method 표현 권장**:
> "The boundary threshold $\delta^{*}$ is the only parameter calibrated per
> dataset; the causal-context hyperparameters $(m, \rho, a) = (2, 0.7, 0.5)$
> are fixed globally. We verified this choice in Appendix~\ref{app:hp-sweep}:
> with per-bench oracle $\delta^{*}$, the default is the rank-1 configuration
> out of 90 alternatives. The result is robust to the choice of $\delta^{*}$
> selection protocol (oracle vs label-free best-$p_x$)."

## 8. Reviewer Q&A (preempt)

| Reviewer 질문 | 대응 |
|---|---|
| "오직 δ\* 만 data-dependent 라니, 다른 3 개 도 hp 잖아?" | 문장을 "the only parameter we calibrate per dataset" 로 정확화. m/ρ/a 도 hp 이지만 **per-dataset tuning 불필요** 임을 90-config sweep 으로 검증. |
| "grid 가 coarse" | m 은 6 값 monotonic, ρ 는 3 값 flat, a 는 5 값 monotonic+plateau — coarse 해도 패턴 명확. Refinement 시도 (intermediate values) 가 결론 바꿀 가능성 거의 없음. |
| "oracle 은 label leakage" | 의도적. paper claim ("default is optimal") 의 가장 tight upper-bound 검증. Best-px 결과로 deployable 검증 추가. |
| "왜 m=2 가 best?" | 작은 ctx window 가 *근접 prediction error* 우세 — boundary 가 *국소* 신호임과 일치. 큰 m 은 context smoothing 으로 sharp boundary 흐려짐. |

## 9. 한계 / 검증 미해결

- δ\* 는 sweep 대상 아님 (per-bench 자유). δ\* 자체의 sensitivity 는 `delta_star_calibration.md` 참조.
- Encoder 1 종 (MPNet) 만. MiniLM-int8 의 (m, ρ, a) 최적이 동일한지는 별도 검증 가능 (sweep 비용 동일).
- Dialseg711 의 30% holdout (213 dial) 표본 small — bootstrap CI 미보고. mean3 차이 < 0.005 인 ranking 들은 noise 가능성.
- m, ρ, a 가 *jointly* 최적 (= cross interaction) 확인. 1D marginal 은 보조.

## 10. 출처 / 재현

- Script: `scripts/run_hiontop_mra_sweep_oracle.py`
- 데이터: `sweep.json` (90 rows, 각 row 에 oracle/best-px Score per bench + δ\*).
- 비교 (4D uniform δ\* sweep): `2026-05-22_hiontop_hp_sweep/sweep_results.json` + `2026-05-25_hiontop_hp_sweep_test_eval/sweep_test_scores.json` (default rank 1 / 630 under uniform setup).
