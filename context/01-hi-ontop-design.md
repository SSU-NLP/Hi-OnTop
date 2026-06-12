# Hi-OnTop 설계 결정사항

---

## 확정 사항

### 1. Scene 임베딩
- **인코더**: `BAAI/bge-base-en-v1.5` (768dim)
- **입력**: 쿼리만 (응답 미포함)
- **정규화**: L2 normalize 후 저장
- **근거**: no fine-tuning 제약 + 실시간 처리 요구

### 2. Prior: sticky-CRP (SEM2와 동일 수식, 하이퍼파라미터 반전)

$$\Pr(e_n = k \mid e_{1:n-1}) \propto \begin{cases} C_k + \lambda \mathbb{I}[e_{n-1}=k] & k \leq K \\ \alpha & k = K+1 \end{cases}$$

- $\alpha = 1.0$, $\lambda = 10.0$ (초기값, 벤치마크 튜닝 대상)
- **근거**: topic 수 자동 결정 + switch-to-old 자연 처리
- SEM 원본 Eq 1 그대로. Hi-OnTop에서 $s_{n-1}$ 조건 추가 (Markov 확장)는 **철회** (아래 §3 참조).

### 3. Markov 확장 철회 (2026-04-23)

초기 설계에서 $P(e_n \mid e_{n-1}, s_{n-1})$로 prior에 직전 쿼리 조건을 추가하려 했으나 **철회**:

- 사건 모델 §4가 centroid 기반 Gaussian likelihood(옵션 A)로 확정된 이후, likelihood(Eq 2)가 이미 $s_n$을 centroid 대비 평가하고 내부적으로 $\mathbf{s}_{n-1}$ 이력은 centroid 업데이트(Welford)에 반영됨.
- 여기에 prior 쪽에서 $s_{n-1}$ 의존 항을 추가하면 **double counting** (likelihood에서 이미 쓰고 있는 정보를 prior에 다시 반영).
- 해석도 혼란 — prior은 partition distribution이고 likelihood는 dynamics. 두 역할 혼합 피함.

→ **Hi-OnTop prior은 SEM 원본 Eq 1 그대로 유지.** scene-conditional 신호가 필요하면 likelihood 쪽(사건 모델)을 확장한다.

### 4. 사건 모델: 옵션 A — Centroid + diag variance (2026-04-23 확정)

$$P(\mathbf{s}_n \mid e_n = k,\, \mu_k,\, \sigma_k^2) = \mathcal{N}\big(\mathbf{s}_n;\, \mu_k,\, \mathrm{diag}(\sigma_k^2)\big)$$

- $\mu_k \in \mathbb{R}^{768}$: event k의 centroid
- $\sigma_k^2 \in \mathbb{R}^{768}_{>0}$: feature dim별 variance (SEM Eq 2의 $\beta$ 역할)
- 업데이트: Welford online (매 턴)

**선택 근거**
- 벤치마크 증거 (`outputs/benchmark-analysis.md` §4): TopiOCQA에서 **높음**, LoCoMo에서 **중**, LongMemEval에서 **낮음** (긴 content variance 예측). 전반 적합도는 낮지만 **incremental 설계 원칙**에 따라 최단순 baseline으로 출발.
- SEM2 `log_likelihood_next` / `log_likelihood_f0` 인터페이스에 정확히 대응.
- Welford `centroid/variance` 업데이트(§6의 학습 타이밍)와 일관.
- Cold start 처리 자연스러움 (§7 $\sigma_0^2$ prior로 해결).

**기각한 옵션과 사유**
- **B (Centroid + Momentum)**: 쿼리-토픽 순서 느슨한 LoCoMo/LongMemEval에서 효과 의문. 추가 복잡성 정당화 부족.
- **C (Centroid + Entity set)**: TopiOCQA에선 유효하나 Claude-유사 대화(LongMemEval)에서 엔티티 sparse. `handoff.md` 경고(TopiOCQA bias)에도 저촉.
- **D (Multi-signal ensemble)**: 종합 적합도는 최고지만 가중치 튜닝 근거가 Phase 0 시점에 없음. Phase 4 실험 후 확장 대상.
- **E (Small linear predictor)**: 작은 topic에서 과적합, cold start 어려움.

**예상 한계 (Phase 4 실험에서 재검토)**
- LongMemEval 긴 content → centroid variance 크게 튈 수 있음. diag $\sigma_k^2$가 흡수 못 하면 boundary score 노이즈 증가.
- TopiOCQA section shift(7.7/conv) > topic shift(3.3/conv) → $\lambda$ 과소면 section 수준 과분할. $\lambda$ 민감도 test 필수.

### 5. 추론 방식: Local MAP approximation

$$\hat{e}_n = \arg\max_{e_n}\, \Pr(e_n \mid \mathbf{s}_{1:n}, \hat{e}_{1:n-1})$$

SEM2 `run()` 루프와 동일한 관점. TF/GRU 의존성만 제거. `_calculate_unnormed_sCRP` + 식 (6) log-lik = −PE² 동일 구조.

### 6. Gibbs reconstruction 폐기
SEM2 의 Gibbs sampling 기반 재구성 불필요 — online local-MAP 으로 turn 단위 즉시 할당.

### 7. 학습 타이밍
- 매 턴 (online): 통계량 업데이트 (Welford)
- 매 라운드 (비동기): refinement, merge 검사

### 8. Cold Start
새 topic은 prior variance $\sigma_0^2$로 시작, $n_e \geq 3$부터 running variance로 전환.

---

## 추후 재검토 대상 (분절 HP / 확장)
- 옵션 D(multi-signal)로 확장 여부
- $\alpha, \lambda$ 최적 값, regime split (persistence α=1,λ=10,σ₀²=0.01 vs freq-shift α=10,λ=1,σ₀²=0.1)
- $\sigma_0^2$ cold start prior 민감도
- Restart-vs-repeat 분기(SEM2 `run()`의 `lmda/2` halving 포함) 포팅 여부

이후 실제 진행·확정은 `context/methodology/` (버전별 알고리즘) 와 `context/06-decision-log.md` 참조.

---

## 변경 이력
`context/06-decision-log.md` 참조.
