# SEM 원본 수식 (식 1~24) — 전체 reference

**출처**: Franklin, Norman, Ranganath, Zacks, Gershman (2020). *Structured Event Memory: A Neuro-Symbolic Model of Event Cognition.* Psychological Review, 127(3), 327–361.

**작성 경위**: Phase 0 Step 0-1 SEM 논문 정독 시 `pdftotext` 추출본 + `pdftoppm` 이미지 직독 (수식 페이지 6, 7, 8, 9, 10, 11, 19, 20, 34, 35) + `SEM/sem/sem.py` 코드 검증으로 확정. `/tmp/sem_notes.md`에 작업 보조본을 두었으나 OS cleanup으로 소실 → **본 파일이 영구 reference**.

`context/00-sem-paper.md`는 Hi-OnTop 관점 정리(계승/폐기 매핑 + 코드 분석)이고, 본 파일은 **원본 수식 정확성** 자체를 보존한다.

---

## 표기 일러두기

- $\mathbf{s}_n, \mathbf{x}_n \in \mathbb{R}^d$: scene n번째 vector embedding (논문은 두 표기 혼용)
- $e_n \in \mathbb{Z}$: event label at time n
- $C_k$: 지금까지 event $k$에 할당된 time point 수
- $K$: 지금까지 등장한 distinct event 수
- $\theta_e$: event $e$의 dynamics parameter (RNN weights)
- $f(\cdot; \theta_e)$: scene dynamics 함수 (RNN)
- $f_0$: 새 event 시작점 (initial condition)
- $\tilde{\cdot}$: corrupted memory trace
- $y_0$: null memory token (forgetting)
- $e_0$: null event label (memory loss)

---

## §1. Event Generative 모형 (식 1~6)

### 식 (1) — Sticky Chinese Restaurant Process Prior   *(p.332)*

$$\Pr(e_n = k \mid e_{1:n-1}) \propto
\begin{cases}
C_k + \lambda\,\mathbb{I}[e_{n-1}=k] & \text{if } k \le K \\
\alpha & \text{if } k = K+1
\end{cases}$$

- $\alpha > 0$: concentration. 작으면 simplicity bias 강(새 클러스터 회피).
- $\lambda \ge 0$: stickiness. 크면 temporal autocorrelation 강.
- 새 cluster 슬롯에는 **α만 더해지고 λ는 안 더해진다** (코드 검증).
- SEM2 코드 기본값: `alfa=10.0, lmda=1.0`. Hi-OnTop persistence regime 기본값은 반전: $\alpha=1.0, \lambda=10.0$.

### 식 (2) — Scene Dynamics (Gaussian noise, diag covariance)   *(p.334)*

$$\Pr(\mathbf{x}_{n+1} \mid \mathbf{x}_{1:n}, e) = \mathcal{N}\!\big(\mathbf{x}_{n+1};\; f(\mathbf{x}_{1:n};\theta_e),\; \mathrm{diag}(\beta)\big)$$

- $\beta$: feature dimension별 variance **벡터** (spherical 아님, full covariance도 아님).
  - 논문 인용 (p.334): "spherical은 single dim이 segmentation을 주도할 위험, full은 high-dim curse of dimensionality로 추정 불안정."
- $f$: SEM 원본 = 4-layer GRU + leaky ReLU (α=0.3) + 50% dropout, Keras+Adam.

### 식 (3) — Initial Condition (event boundary 직후)   *(p.335)*

$$\Pr(\mathbf{x}_{n+1} \mid \mathbf{x}_n, e_{n+1} \ne e_n) = \int \mathcal{N}\!\big(\mathbf{x}_{n+1};\; f_0,\; \mathrm{diag}(\beta)\big)\,\Pr(f_0)\,df_0$$

- $f_0$: 새 event의 시작점. uniform prior $\Pr(f_0)$ 가정 → 신규 event는 적분이 **상수**로 축약 (segmentation 판정에서 threshold 역할).
- 경험된 event에 대해선 $f_0$의 point estimate 사용.

### 식 (4) — Bayes Posterior   *(p.335)*

$$\Pr(\mathbf{e} \mid \mathbf{s}) = \frac{1}{Z}\,\Pr(\mathbf{s} \mid \mathbf{e})\,\Pr(\mathbf{e})$$

- $\Pr(\mathbf{e})$: prior (식 1).
- $\Pr(\mathbf{s} \mid \mathbf{e})$: likelihood (식 5).
- $Z$: normalizing constant.

### 식 (5) — Likelihood Factorization   *(p.335)*

$$\Pr(\mathbf{s} \mid \mathbf{e}) = \prod_{n=1}^{N} \Pr(\mathbf{s}_{n+1} \mid \mathbf{s}_n, e_n)$$

- Markov 가정 1차 (only $s_{n-1}$ 의존).
- 각 항은 식 (2)로 주어짐.

### 식 (6) — Log-Likelihood = −Prediction Error²   *(p.336)*

$$\log \Pr(\mathbf{s}_{n+1} \mid \mathbf{s}_n, e_n) = -\frac{1}{2\beta}\,\|\mathbf{x}_{n+1} - f(\mathbf{x}_{1:n};\theta_e)\|^2 + \text{const}$$

- log-likelihood ∝ −(Euclidean PE)².
- 즉 작은 PE → 높은 likelihood → 같은 event에 머무름.
- 큰 PE → 낮은 likelihood → event 경계 가능성 ↑.
- *Footnote 2*: cosine 등 다른 similarity metric도 가능 (negative correlation with log-lik 유지하면 OK), 다만 Gaussian의 axiomatic property는 깨짐.

---

## §2. Inference (식 7~9)

### 식 (7) — Full Posterior (Marginalizing past partitions)   *(p.336, intractable)*

$$\Pr(e_{n+1} \mid \mathbf{s}_{1:n}) = \sum_{e_{1:n-1}} \Pr(e_n \mid \mathbf{s}_{1:n}, e_{1:n-1})$$

- $e_{1:n-1}$ partition 공간이 Bell number $B_n$ 만큼 폭발 (n=10 → 115,975).
- 직접 계산 불가능.

### 식 (8) — Local MAP 근사   *(p.336)*

$$\Pr(e_{n+1} \mid \mathbf{s}_{1:n}) \approx \Pr(e_n \mid \mathbf{s}_{1:n},\, \hat{e}_{1:n-1})$$

- 과거 segmentation을 single point estimate $\hat{e}_{1:n-1}$로 고정.
- 정당성: Wang & Dunson 2011 등 DP mixture clustering에서 MAP-vs-full 차이 작음.
- 한계 (논문 명시): retrospective re-evaluation 불가 — 과거 경계 사후 수정 안 됨. Particle smoothing은 가능.

### 식 (9) — 재귀 MAP 정의   *(p.336)*

$$\hat{e}_n = \arg\max_{e_n}\, \Pr(e_n \mid \mathbf{s}_{1:n},\, \hat{e}_{1:n-1})$$

- Online (single forward sweep).
- SEM2 `run()` 메서드의 핵심 루프.

---

## §3. Memory Encoding (식 10~13)

### 식 (10) — Factorized Corruption Process   *(p.336)*

$$\Pr(\tilde{\mathbf{y}} \mid \mathbf{y}) = \Pr(\tilde{\mathbf{x}} \mid \mathbf{x})\,\Pr(\tilde{e} \mid e)\,\Pr(\tilde{n} \mid n)$$

- Memory trace $\mathbf{y}_i = (\mathbf{x}_i, e_i, n_i)$ — feature/event-label/time-index 3-tuple.
- 각 component 독립적으로 corruption (단순화 가정).

### 식 (11) — Feature Corruption (Spherical Gaussian)   *(p.337)*

$$\Pr(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}\!\big(\tilde{\mathbf{x}};\; \mathbf{x},\; \tau I\big)$$

- $\tau$: scalar isotropic noise.
- 식 (2)의 `diag(β)` (vector, 학습 가능)와 다름. **이 차이의 이론적 유도는 논문에 명시 부재** (Hi-OnTop `context/00-sem-paper.md §7` 검증 미해결 1).

### 식 (12) — Event Label Corruption (Z-channel)   *(p.337)*

$$\Pr(\tilde{e} \mid e) =
\begin{cases}
\epsilon_e & \text{if } \tilde{e} = e \\
1 - \epsilon_e & \text{if } \tilde{e} = e_0 \\
0 & \text{otherwise}
\end{cases}$$

- $\epsilon_e$: 라벨 보존 확률.
- $e_0$: null label (라벨 완전 망각).
- Asymmetric: 라벨이 다른 event로 잘못 바뀌는 경우는 0 (보존되거나 완전 망각만).

### 식 (13) — Time Index Corruption (Uniform)   *(p.337)*

$$\tilde{n} \mid n \sim U[\,n - b,\; n + b\,]$$

- $b$: 작게 가정 (인접 turn 간 swap 정도, 큰 jump 방지).
- 결과: 같은 event 내 turn 순서 corruption + 경계 위치 corruption.

---

## §4. Memory Reconstruction (식 14~15)

### 식 (14) — Full Generative Process for Memory   *(p.337)*

$$\Pr(\tilde{\mathbf{y}}_i \mid f, \theta) = \sum_e \int_{\mathbf{x}}\Pr(\tilde{\mathbf{y}}_i \mid \mathbf{x}, e)\,\Pr(\mathbf{x} \mid e, f, \theta)\,\Pr(e)\,d\mathbf{x}$$

- $\mathbf{x} = \mathbf{x}_{1:n}$, $\mathbf{e} = e_{1:n}$.
- $\theta = \{\theta_e, \theta_{e'}, \dots\}$: 모든 event의 RNN parameters.
- 세 분포 (encoding · transition · prior) 결합.

### 식 (15) — Null Memory (Forgetting)   *(p.337)*

$$\Pr(y_0 \mid y) \propto \zeta$$

- $\zeta$: forgetting parameter (free).
- $y_0$: null memory token. 여러 번 등장 가능 (특정 trace에 대응 안 함).
- 연속 적분 $\int_y \Pr(\tilde{y}_i \mid y)\,dy \propto \zeta$로 균등 prior $\Pr(x) \propto 1$ 가정.

---

## §5. Bower (1979) 시뮬레이션 (식 16~17, 평가 보조)

### 식 (16) — Recognition Probability Difference   *(p.345)*

$$\mathbb{E}[\Pr(A \mid \tilde{y}) - \Pr(B \mid \tilde{y})]$$

- $A, B$: memory probe items.
- 인간 false-memory 패턴 vs 모델 예측치 비교.

### 식 (17) — Monte Carlo Approximation of (16)   *(p.346)*

$$\mathbb{E}[\Pr(A\mid \tilde{y}) - \Pr(B\mid \tilde{y})] \approx \frac{1}{N_s}\sum_{i=1}^{N_s} \mathbb{I}\!\big[g(\mathbf{x}_A, \hat{\mathbf{x}}) > g(\mathbf{x}_B, \hat{\mathbf{x}})\big]$$

where $g(\mathbf{x}, \hat{\mathbf{x}}) = \exp(-\gamma\|\mathbf{x} - \hat{\mathbf{x}}\|^2)$, $\gamma = 2.5$.

- $\hat{\mathbf{x}}$: 식 (18)로 sample된 reconstruction.
- $N_s$: number of MC samples.

---

## §6. Gibbs Sampler 상세 (Appendix C, 식 18~23)

본 절의 식들은 식 (14)의 사후확률을 sampling하기 위한 Gibbs 단계들.

### 식 (18) — Gibbs Feature Posterior   *(p.360, Appendix C)*

$$\mathbf{x}_n \mid \mathbf{x}_{1:n-1},\, \tilde{x}_n,\, e,\, \theta \;\sim\; \mathcal{N}(\bar{x},\, \lambda I)$$

- Conditional Gaussian for one scene given memory trace + event.

### 식 (19) — Precision-Weighted Mean for (18)   *(p.360)*

$$\bar{x} = u\,f(\mathbf{x}_{1:n-1}, e_t, \theta) + (1 - u)\,\tilde{x}_n$$

where $u = b / (b + t)$, $\lambda = 1 / (b^{-1} + t^{-1})$.

- $b$: model의 inverse variance (precision); $t$: encoding noise inverse variance.
- 모델 예측 $f(\cdot)$과 noisy memory $\tilde{x}_n$의 precision-가중 평균.

### 식 (20) — Gibbs Memory-Trace Posterior   *(p.360)*

$$\Pr(\tilde{y}_j \mid \mathbf{x}, e) =
\begin{cases}
0 & \text{if } \tilde{e}' \notin \{e_t, e_0\} \\
\zeta/Z & \text{if } \tilde{y}_j = y_0 \\
\mathbb{I}_{|\tilde{n}-n|<b}\,\mathcal{N}(\tilde{x}';\, \mathbf{x}_n,\, \tau I) / Z & \text{otherwise}
\end{cases}$$

where $\tilde{y}'_j = (\tilde{\mathbf{x}}', \tilde{e}', \tilde{n})$.

### 식 (21) — Normalizing Constant for (20)   *(p.360)*

$$Z = \zeta + \sum_n \mathbb{I}_{|\tilde{n}-n|<b}\,\mathcal{N}(\tilde{x}';\, \mathbf{x}_n,\, \tau I)$$

### 식 (22) — Gibbs Event-Label Posterior   *(p.360)*

$$\Pr(e \mid \mathbf{x}, e_{1:n-1}) \propto \Pr(\mathbf{x}_t \mid \mathbf{x}_{1:n-1}, e)\,\Pr(e \mid e_{1:n-1})$$

- **식 (4)와 동형** — segmentation의 Bayes 형태와 같음.
- Hi-OnTop에서는 segmentation 루프 자체와 코드 재사용 가능 (옵션 D 확장 시 유효).

### 식 (23) — Label Recall with Fallback   *(p.360)*

$$\Pr(\tilde{e}'_j \mid \mathbf{x}, e_{1:n-1}) =
\begin{cases}
\Pr(e \mid \mathbf{x}_{1:n-1}, e_{1:n-1}) & \text{if } \tilde{e}' = e_0 \text{ or } \tilde{y}_j = y_0 \\
1 & \text{otherwise}
\end{cases}$$

- null event 또는 null memory면 generative process(식 22) 재호출.
- 그 외엔 메모리 라벨 그대로 사용.

---

## §7. DuBrow & Davachi 민감도 (Appendix D, 식 24)

### 식 (24) — Bayesian Logistic Regression for Accuracy   *(p.361)*

$$\Pr(\text{Accuracy} = 1) = s(b\,\beta_b + \tau\,\beta_\tau)$$

where $s(x) = 1/(1 + e^{-x})$.

- $\beta_b, \beta_\tau$: regression weights.
- $b, \tau$: DuBrow & Davachi 2013/2016 실험의 hyperparameter sensitivity 분석에 사용.
- **코어 모델 식 아님** — sensitivity 도구.

---

## 부록: Hi-OnTop 처리 매핑

| 식 | 카테고리 | Hi-OnTop 처리 | 비고 |
|---|---|---|---|
| 1 | Event Prior | **유지** | α/λ만 regime별 분기 |
| 2 | Scene Dynamics | 대체 | $f$ → centroid Gaussian (옵션 A) |
| 3 | Initial Condition | 대체 | $f_0$ → centroid (옵션 A) |
| 4 | Bayes Posterior | **유지** | segmentation 사후확률 |
| 5 | Likelihood 분해 | **유지** | Markov 1차 분해 |
| 6 | log-lik = −PE² | **유지** | metric은 변경 가능 (footnote 2) |
| 7 | Full posterior | 폐기 | intractable |
| 8 | Local MAP 근사 | **유지** | 핵심 — Hi-OnTop online 루프 |
| 9 | 재귀 MAP | **유지** | `assign()` 구현 기반 |
| 10–13 | Memory corruption | 폐기 | segmentation 목적에 불필요 (원문 그대로 사용) |
| 14, 15 | Memory generative | 폐기 | reconstruction 사용 안 함 |
| 16, 17 | Bower 시뮬 | N/A | 인지 실험 전용 |
| 18–23 | Gibbs sampler | 폐기 | reconstruction 폐기에 따라 |
| 22 | Gibbs event posterior | (참고) | 식 4와 동형 — 옵션 D 확장 시 코드 재사용 가능 |
| 24 | DuBrow sensitivity | N/A | sensitivity 도구 전용 |

**Hi-OnTop이 실질적으로 계승**: 식 1, 4, 5, 6, 8, 9 (sticky-CRP prior + Bayes + factorization + log-lik=−PE² + local MAP).

**Hi-OnTop이 대체**: 식 2, 3 ($f$ form). 옵션 A로 centroid Gaussian.

**Hi-OnTop이 폐기**: 식 7 (intractable) + 식 10–24 중 코어 인지 모델·재구성·시뮬레이션 부분.

---

## 부록: 검증 미해결 (`context/00-sem-paper.md §7`와 동일)

1. **식 (2) `diag(β)` vs 식 (11) `τI`** covariance 형태 차이의 이론적 유도 부재.
2. **SEM2 코드 `prior[k_prev] -= λ/2` halving** 논문에 explicit 유도 없음. 옵션 A 미포팅.
3. ~~Markov 확장 형태~~ — Hi-OnTop Step 0-3에서 철회 (옵션 A에서 double counting).

---

## 참고

- 본 파일은 식 정의의 **원본 충실성** 자체에 집중. Hi-OnTop 적용 논의·코드 분석·검증 결정은 `context/00-sem-paper.md` 참조.
- SEM2 구현 verbatim pseudocode는 `context/00-sem-paper.md §4` 참조.
- 결정 이력은 `context/06-decision-log.md` (특히 2026-04-23 entries) 참조.
