# SEM 논문 정리

Franklin, Norman, Ranganath, Zacks, Gershman (2020).
"Structured Event Memory: A Neuro-Symbolic Model of Event Cognition", *Psychological Review*, 127(3), 327–361.
doi: 10.1037/rev0000177.

> 식 1~24의 **원본 LaTeX 정의 + 페이지 참조 + Hi-OnTop 처리 매핑**은 별도 reference 파일 [`context/sem-equations.md`](sem-equations.md)에 영구 보존. 본 파일은 **Hi-OnTop 관점 정리**(계승/폐기 매핑, SEM2 코드 분석, 검증 미해결 지점)에 집중.

참조 자료
- `SEM-paper.pdf` — 논문 본문 (35쪽)
- `sem.txt` — `pdftotext -layout` 추출본 (gitignored)
- 수식 검증: 수식 밀집 페이지 (6, 7, 8, 9, 10, 11, 19, 20, 34, 35)를 `pdftoppm -r 200`으로 이미지 렌더링 후 직독
- 상세 노트: `/tmp/sem_notes.md` (작업 보조, 세션 전용)
- `SEM/` — **SEM2** (`nicktfranklin/SEM2`) current working build. 참조 전용.
  원본 논문 시뮬레이션은 `ProjectSEM/SEM` (archival)에 있지만 Hi-OnTop은 SEM2를 본다.

---

## 1. 모델이 풀려는 문제

SEM은 인간 사건 인지의 다섯 desiderata를 하나의 확률적 생성 모형으로 통합 설명한다:
1. **Segmentation** — 연속 감각 스트림에서 사건 경계 탐지
2. **Learning** — 사건 내부 구조 학습
3. **Inference** — 관측되지 않은 장면 추론
4. **Prediction** — 다음 장면 예측
5. **Memory** — 과거 경험 재구성

이론적 기반은 **Event Segmentation Theory (EST)**: 활성 event model을 유지하며 예측하고, 예측 오차가 경계 신호가 된다.

### 6가지 가정 (p.332)
1. 사건 종류 수 $K$는 unknown → 데이터에서 추정
2. 기존 event model 재사용 선호 (Ockham 편향)
3. 사건은 시간적으로 persistent
4. 사건은 dynamical system
5. 사건은 latent structure 공유
6. 사건 지식이 기억 재구성의 regularizer

---

## 2. 생성 모형 — 모든 수식 (논문 기준 검증본)

### 식 (1) — sticky-CRP Prior (p.332)
$$\Pr(e_n = k \mid e_{1:n-1}) \propto \begin{cases} C_k + \lambda\,\mathbb{I}[e_{n-1}=k] & \text{if } k \le K \\ \alpha & \text{if } k = K+1 \end{cases}$$
- $C_k$: $e_{1:n-1}$에서 event $k$ 할당 횟수
- $\alpha > 0$: concentration — 작으면 새 event 억제 (가정 2: simplicity bias)
- $\lambda \ge 0$: stickiness — 크면 직전 event 유지 (가정 3)
- SEM2 기본값 `alfa=10.0, lmda=1.0`. **Hi-OnTop은 반전 `α=1.0, λ=10.0`** — 대화는 topic persistence가 기본이라 stickiness를 강하게 둔 설계 선택.

### 식 (2) — Scene Dynamics (p.334)
$$\Pr(\mathbf{x}_{n+1} \mid \mathbf{x}_{1:n}, e) = \mathcal{N}\big(\mathbf{x}_{n+1};\, f(\mathbf{x}_{1:n}; \theta_e),\, \mathrm{diag}(\beta)\big)$$
- $f$: smooth function, 각 event별 $\theta_e$
- $\beta$: **feature dimension별 variance 벡터**, **diag covariance** (spherical 아님, full 아님 — 중간)
- SEM은 $f$를 4-layer GRU로 구현: leaky ReLU (α=0.3), 50% dropout, Keras+Adam
- $\beta$ 추정: MAP + inverse-χ² prior (dof $\nu$, scale $s_0^2$)
- **[검증 미해결]** 식 (2)의 `diag(β)`와 식 (11)의 `τI` (spherical)이 서로 다른 형태를 쓰는 수학적 유도는 논문에서 명시되지 않음. scene dynamics(학습 가능)와 memory corruption(단일 lossy process) 목적 차이로 추정되나 확정 근거 부족. Hi-OnTop 포팅엔 영향 없음.

### 식 (3) — Initial condition $f_0$ (p.335)
$$\Pr(s_{n+1} \mid s_n, e_{n+1}\neq e_n) = \int \mathcal{N}\big(\mathbf{x}_{n+1};\, f_0,\, \mathrm{diag}(\beta)\big)\,\Pr(f_0)\,df_0$$
- $f_0$: 함수 $f$의 초기 조건, **uniform prior**
- 신규 event $e_{\text{new}}$: 식 (3)이 **상수로 축약** → 경계 판정 가능
- 경험된 event: $f_0$ point estimate 사용 (적분 skip)

### 식 (4) — Bayes Posterior (p.335)
$$\Pr(\mathbf{e}\mid \mathbf{s}) = \tfrac{1}{Z}\,\Pr(\mathbf{s}\mid \mathbf{e})\,\Pr(\mathbf{e})$$
- $\Pr(\mathbf{e})$: sticky-CRP (식 1)
- $Z$: normalizing constant

### 식 (5) — Likelihood 분해 (p.335)
$$\Pr(\mathbf{s}\mid\mathbf{e}) = \prod_{n=1}^{N}\Pr(s_{n+1}\mid s_n, e_n)$$
Markov 가정 하 각 전이의 곱. 각 항 = 식 (2).

### 식 (6) — Log-likelihood = −(PE)² (p.336)
$$\log\Pr(s_{n+1}\mid s_n, e_n) = -\tfrac{1}{2\beta}\big\|\mathbf{x}_{n+1} - f(\mathbf{x}_{1:n};\theta_e)\big\|^2 + \text{const.}$$
- **핵심**: log-lik가 Euclidean prediction error의 음수 스케일
- low likelihood ⇔ high PE ⇒ boundary 가능성 ↑
- Footnote 2: cosine 등 다른 metric 허용 (axiomatic은 아님)

### 식 (7) — Full posterior marginalizing past (p.336, intractable)
$$\Pr(e_n\mid s_{1:n}) = \sum_{e_{1:n-1}}\Pr(e_n\mid s_{1:n}, e_{1:n-1})$$
과거 partition 전체 합산 → 조합 폭발, 계산 불가.

### 식 (8) — Local MAP 근사 (p.336)
$$\Pr(e_n\mid s_{1:n}) \approx \Pr(e_n\mid s_{1:n}, \hat e_{1:n-1})$$
과거를 single high-probability 가설로 고정 → online 처리 가능.

### 식 (9) — 재귀 MAP 정의 (p.336)
$$\hat e_n = \arg\max_{e_n}\Pr(e_n\mid s_{1:n}, \hat e_{1:n-1})$$
매 scene마다 prior(식 1) × likelihood(식 2)의 argmax. Single forward sweep, 과거 re-evaluation 없음.

### 메모리 corruption (식 10–15, p.336–337)

| Eq | 대상 | 형태 |
|---|---|---|
| 10 | joint | $\Pr(\tilde y\mid y) = \Pr(\tilde x\mid x)\Pr(\tilde e\mid e)\Pr(\tilde n\mid n)$ (독립 factorize) |
| 11 | feature | $\Pr(\tilde x\mid x) = \mathcal{N}(\tilde x; x, \tau I)$ — **spherical** isotropic (식 2의 diag $\beta$와 다름) |
| 12 | event label | Z-channel: $\epsilon_e$ 확률로 보존, $1-\epsilon_e$ 확률로 **null label $e_0$로 소실** (다른 label로 swap 안 함) |
| 13 | time index | $\tilde n\mid n \sim U[n-b,\, n+b]$ uniform discrete |
| 14 | full gen. | $\Pr(\tilde y_i\mid f,\theta) = \sum_e\int_x \Pr(\tilde y_i\mid x,e)\Pr(x\mid e,f,\theta)\Pr(e)\,dx$ |
| 15 | forgetting | $\Pr(y_0\mid y) \propto \zeta$ — null memory 상수 |

### Gibbs sampling (식 18–23, Appendix C p.360)

3-step alternating:
- **(a) feature 샘플**: 식 18–19. 특히 식 19가 핵심:
  $$\bar x = u\,f(x_{1:n-1},e_t,\theta) + (1-u)\,\tilde x_n,\qquad u = \tfrac{b}{b+\tau}$$
  재구성된 feature = (event model 예측) × (corrupted trace)의 **precision-weighted 가중평균**. $b$ 커지면 event schema 쪽으로 regularize.
- **(b) memory trace 샘플**: 식 20–21, 식 11–13의 역변환.
- **(c) event 샘플**: 식 22 = **식 4와 동형** (prior × likelihood). 식 23은 label recall fallback.

### 시뮬 전용 (식 16, 17, 24) — 본 모델 공식이 아님
- 식 16–17: Bower 1979 script false memory의 recognition prob 차이 + MC 근사
- 식 24: DuBrow & Davachi sensitivity 로지스틱 회귀

---

## 3. Scene 표현: HRR (p.333)

$$\mathbf{x} = \text{dog}\circledast\text{agent} + \text{chase}\circledast\text{verb} + \text{cat}\circledast\text{patient}$$
- $\circledast$: circular convolution (binding)
- `+`: vector addition (superposition)
- role-filler binding을 고정 dim vector에 encode
- 유사 symbol은 공유 feature로 자동 similarity 가짐
- 실험에서 영상 input은 **VAE로 100차원 압축** 후 SEM 투입

---

## 4. SEM2 코드에서 참조할 부분

`SEM/sem/sem.py` (SEM2, 2025년 기준) — **실제 코드 직접 검증**:

### `_calculate_unnormed_sCRP(prev_cluster=None)` — p.144–159
식 1 그대로 구현. 구조:
```python
prior = self.c.copy()                          # C_k (각 event 할당 횟수)
idx = len(np.nonzero(self.c)[0])               # 지금까지 본 distinct event 수
if idx <= self.k:
    prior[idx] += self.alfa                    # 새 cluster 확률 = α
if prev_cluster is not None:
    prior[prev_cluster] += self.lmda           # stickiness λ를 직전 cluster에만
return prior                                    # unnormalized
```
- **확인:** 식 (1) $C_k + \lambda\mathbb{I}[e_{n-1}=k]$ for $k \le K$, $\alpha$ for $k=K+1$ 정확히 구현됨.
- Hi-OnTop에서 그대로 포팅 가능. $\alpha/\lambda$만 반전해서 설정.

### `run(x, k, ...)` — p.161~ (온라인 MAP 루프, 식 8–9)
매 scene $x_{\text{curr}}$마다:

1. **Prior 계산**: `prior = _calculate_unnormed_sCRP(k_prev)`
2. **Likelihood**: 각 active event $k_0$에 대해
   - `current_event = (k0 == k_prev)`인 경우:
     - `x_hat_active, lik[k0] = model.log_likelihood_next(x_prev, x_curr)` — 2-tuple 반환 (예측 vector + log-lik)
     - `lik_restart_event = model.log_likelihood_f0(x_curr)` — 동일 event 재시작 가능성 (식 3 기반)
   - 그 외: `lik[k0] = model.log_likelihood_f0(x_curr)` — 신규/경험된 다른 event
3. **Restart-vs-repeat 분기** (k_prev != None일 때):
   - `restart_prob = lik_restart_event + log(prior[k_prev] - lmda)` — λ를 빼서 "새 토큰으로서의 prev event" 계산
   - `repeat_prob = _post[k_prev]` (= `log(prior[k_prev]) + lik[k_prev]`)
   - `_post[k_prev] = max(repeat_prob, restart_prob)` — MAP 선택
4. **MAP cluster**: `k = argmax(_post)`
5. **경계 판정**:
   `event_boundary = (k != k_prev) or ((k == k_prev) and (restart_prob > repeat_prob))`
6. **Boundary log-probability**:
   `log_boundary_probability = logsumexp(_post) - logsumexp(concat([_post, [repeat_prob]]))`
7. **Label posterior 계산** (boundary를 ignore한 event label 확률):
   - `_post[k_prev] = logsumexp([restart_prob, repeat_prob])` — restart/repeat을 OR (sum)으로 합침
   - **`prior[k_prev] -= lmda / 2.`** — stickiness 절반 차감 (label posterior엔 stickiness 덜 반영)
   - `lik[k_prev] = logsumexp([lik[k_prev], lik_restart_event])`
   - 최종 posterior 정규화
   - **[검증 미해결]** `lmda / 2` halving은 논문에서 explicit 유도 없음. 추정: restart 쪽은 이미 `prior[k_prev] - lmda`로 차감했지만 repeat 쪽엔 full λ가 있으므로, 두 경로를 합칠 때 stickiness 과대 반영 방지용 heuristic. Hi-OnTop 포팅 시 맹목 복사 금지 — sanity test로 λ 민감도를 확인한 후 채택 여부 결정.
8. **Prediction error**: `pe = ||x_curr - x_hat_active||` (Euclidean, 식 6의 기반)
9. **Event model 업데이트**:
   - boundary 아님 → `event_models[k].update(x_prev, x_curr)`
   - boundary → `event_models[k].new_token(); update_f0(x_curr)`
10. **Bayesian surprise** (post-hoc): `surprise = concat([[0], logsumexp(log_post + log_like[1:,:], axis=1)])`

### 나머지 모듈
- `event_models.py` — `GRUEvent` (식 2의 $f$ 구현). `log_likelihood_next`, `log_likelihood_f0`, `update`, `update_f0`, `new_token` 메소드. **Hi-OnTop 대체 대상.**
- `memory.py` — Gibbs reconstruction (식 18–23). **Hi-OnTop 폐기.**
- `hrr.py` — HRR binding (circular convolution). **Hi-OnTop은 bge 임베딩으로 대체.**
- `utils.py` — delete_object_attributes 등 유틸.

TF session, Keras backend, pretrain, `run_w_boundaries`는 참조하지 않는다.

### Hi-OnTop 포팅 시 주의 (검증된 구현 세부)
- **2-tuple 반환**: `log_likelihood_next`가 `(x_hat, lik)`를 반환. PyTorch 구현 시 동일 인터페이스로 설계 가능.
- **λ 절반 차감** (step 7): label posterior 계산 시만 `prior[k_prev] -= lmda/2`. segmentation posterior(step 3의 `_post`)엔 full λ 유지. 이유는 restart/repeat을 OR 합쳤기 때문 — Hi-OnTop에서도 유지해야 수치 일관성.
- **식 3의 역할**: `log_likelihood_f0`은 경계 직후 likelihood + restart case 동시에 쓰임. Hi-OnTop의 centroid 기반 사건 모델에서는 "cold start 분포"로 해석 가능.

---

## 5. Hi-OnTop이 계승/대체/폐기한 것

| SEM 식/구성요소 | Hi-OnTop 처리 | 근거 |
|---|---|---|
| 식 1 sticky-CRP | **유지**, 수식 동일, $\alpha/\lambda$ 반전 | topic 수 자동 결정, 대화는 persistence 기본 |
| 식 2 $f$ (GRU) | **대체 대상 미확정** (옵션 A~F) | no fine-tuning 제약 → RNN 학습 불가 |
| 식 2 $\beta$ diag covariance | 개념 유지, Welford online update로 단순화 | 실시간 제약 |
| 식 3 $f_0$ | 사건 모델 형태에 따라 결정 | 식 2와 함께 |
| 식 4 Bayes posterior | **유지** | 구조 자체는 변경 불필요 |
| 식 5 Likelihood 분해 | **유지** | Markov 구조 그대로 |
| 식 6 log-lik = −PE² | 유지, metric은 사건 모델에 따라 (cosine 가능) | Footnote 2 |
| 식 7 Full marginal | **폐기** | intractable, 애초에 SEM도 안 씀 |
| 식 8–9 Local MAP | **유지** | online 처리 핵심 |
| 식 10–15 Memory corruption | **폐기** | 기억 왜곡 시뮬은 segmentation 목적에 불필요. 원문 그대로 사용. |
| 식 18–23 Gibbs reconstruction | **폐기** | 위와 동일. 단 식 22 ≡ 식 4라는 구조 관찰은 향후 구현에서 유용 |
| 식 16, 17, 24 | N/A | 특정 실험 예측 도구 |
| HRR scene 임베딩 | **폐기** → `bge-base-en-v1.5` L2-normalize | no fine-tuning + 실시간 + 텍스트 입력 |
| VAE scene encoder | 불필요 | 입력이 토큰 |
| TensorFlow/Keras | **폐기** | Hi-OnTop은 PyTorch only |
| Pretrain 단계 | **폐기** (cold start) | sticky-CRP sparsity로 첫 턴 자동 새 topic |
| Markov 1차 $P(e_n\mid e_{n-1})$ | **확장**: $P(e_n\mid e_{n-1}, s_{n-1})$ | 대화는 직전 쿼리에 따라 급변 |
| Restart-vs-repeat 이원 분기 | **미확정** — 그대로 포팅 vs 단순화 | Phase 0-3 완료 후 결정 |

---

## 6. Hi-OnTop 관점 열린 질문

- **식 2의 $f$ 대체 형태**: Centroid-only / +Momentum / +Entity set / Multi-signal / Small linear / 새 제안. 벤치마크 분석(`04-benchmarks.md`, Step 0-2) 후 `01-hi-ontop-design.md §A`에서 결정.
- 대화에서 **prediction**이 실제 의미 있는가, 아니면 centroid-distance만으로 경계 탐지 충분한가
- $\beta$ (noise variance)를 Welford online update로 유지할 때 cold start 처리 — `01-hi-ontop-design.md §7`의 $\sigma_0^2$ prior
- sCRP $\lambda$의 실제 민감도 — 너무 크면 shift 놓침, 너무 작으면 진동. segmentation 벤치마크 (TIAGE / TopiOCQA / dialseg711 등) 로 검증 필요.
- 식 22 ≡ 식 4 관찰: 향후 memory reconstruction을 추가할 필요가 생기면 segmentation 코드를 재사용 가능.

---

## 7. 검증 미해결 지점 (정독 자기검증 결과)

SEM 논문 전 35페이지 + `SEM/sem/sem.py` 실제 코드 검증 후 남은 이론적 gap:

1. **Eq (2) `diag(β)` vs Eq (11) `τI` covariance 차이의 유도** — 논문 p.334에서 `diag(β)` 선택은 spherical/full 대비 justify하지만, memory corruption이 왜 spherical `τI`로 돌아가는지 따로 유도 없음. Hi-OnTop 포팅에는 영향 없음 (memory corruption 자체를 폐기).

2. **`prior[k_prev] -= lmda / 2.`의 이론적 유도** — SEM2 `run()` step 7에서 label posterior 계산 시 stickiness를 절반 차감. 논문에 explicit 유도 없음. heuristic으로 추정. Hi-OnTop 포팅 시 맹목 복사 대신 sanity test 후 결정.

3. **Markov 확장 $P(e_n \mid e_{n-1}, s_{n-1})$의 정확한 형태** — `context/01-hi-ontop-design.md` §3에 선언만 되어 있고 구체 수식 미확정. 두 방향 가능:
   - (a) prior 자체를 $s_{n-1}$로 조건화: $\Pr(e_n=k \mid e_{1:n-1}, s_{n-1}) \propto (C_k + \lambda\mathbb{I}[e_{n-1}=k])\cdot g(s_{n-1}, k)$
   - (b) 현행 Eq 1 유지 + likelihood(Eq 2)가 이미 $s_{n-1}$ 반영 — 확장 불필요 주장
   - 선택은 Phase 0-3 (사건 모델 확정)과 연동.

4. **정독 skip 섹션** — Hi-OnTop 진행에 필수적이지 않다고 판단:
   - Appendix A (VAE): scene encoder는 bge로 대체
   - Appendix B (HRR 정리): HRR 폐기
   - Appendix D (DuBrow sensitivity 세부): 인지실험 재현 목적이라 skip
   - Simulations 섹션 실험별 세부 파라미터: Table 2 요약만 확인, 재현 안 함

---

## 8. 변경 이력

`context/06-decision-log.md` 참조.
