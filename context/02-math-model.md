# Hi-OnTop 수학 모형

---

## 확정 수식

### 쿼리 임베딩
$$\mathbf{s}_n = \text{normalize}(\text{encoder}(\text{query}_n))$$

$\text{encoder} = $ `BAAI/bge-base-en-v1.5`, 차원 $d = 768$, L2 normalize.

### 생성 모형 (Hi-OnTop 최종형)

Markov 확장($P(e_n\mid e_{n-1}, s_{n-1})$) 철회 후 SEM 원본 구조 그대로:

$$P(\mathbf{s}_{1:N}, e_{1:N}) = \prod_{n=1}^{N} P(e_n \mid e_{1:n-1}) \cdot P(\mathbf{s}_n \mid e_n)$$

각 항은 아래 sCRP prior + centroid likelihood로 구체화.

### sCRP Prior (SEM2와 동일 수식)

$$\Pr(e_n = k \mid e_{1:n-1}) \propto \begin{cases} C_k + \lambda \mathbb{I}[e_{n-1}=k] & k \leq K \\ \alpha & k = K+1 \end{cases}$$

### Topic 동역학 — 옵션 A 확정 (2026-04-23)

$$P(\mathbf{s}_n \mid e_n = k) = \mathcal{N}\big(\mathbf{s}_n;\, \mu_k,\, \mathrm{diag}(\sigma_k^2)\big)$$

- $\mu_k \in \mathbb{R}^{d}$: event k의 centroid
- $\sigma_k^2 \in \mathbb{R}^{d}_{>0}$: feature dim별 variance 벡터 (diag covariance, SEM Eq 2의 $\beta$ 역할)
- Cold start: $n_e < 3$ 구간에서 $\sigma_k^2 = \sigma_0^2 \mathbf{1}$ 사용

**log-likelihood (SEM 식 6과 동형):**

$$\log P(\mathbf{s}_n \mid e_n=k) = -\tfrac{1}{2}\sum_{j=1}^{d}\left[\frac{(s_{n,j} - \mu_{k,j})^2}{\sigma_{k,j}^2} + \log(2\pi\sigma_{k,j}^2)\right]$$

### Topic 배정 (MAP, SEM Eq 8~9)

$$\hat{e}_n = \arg\max_k\, \log \Pr(e_n = k \mid e_{1:n-1}) + \log P(\mathbf{s}_n \mid e_n = k)$$

- prior (sCRP) + likelihood (centroid Gaussian)의 argmax
- SEM2 `run()`의 restart-vs-repeat 분기(같은 $k_{\text{prev}}$의 `log_likelihood_f0` 케이스) **포팅 여부는 Phase 4 실험 후 결정** (`context/00-sem-paper.md` §7 검증 미해결 2번 참조)

### Prediction Error / Boundary Score

옵션 A 하에서 자연스러운 PE는 Mahalanobis-form 거리:

$$\mathrm{PE}_n(k) = \sum_{j=1}^{d}\frac{(s_{n,j} - \mu_{k,j})^2}{\sigma_{k,j}^2}$$

- log-lik의 $-2\log P$ 항과 동치(상수 제외).
- Boundary score = $\mathrm{PE}_n(\hat e_{n-1})$; 임계값은 Phase 4에서 튜닝.

### Centroid / Variance 업데이트 (Welford, 매 턴 online)

```
n_e += 1
delta = s_n - mu_e
mu_e += delta / n_e
M2_e += delta * (s_n - mu_e)
sigma2_e = M2_e / n_e    # n_e >= 3 이후. n_e < 3은 sigma_0^2 유지
```

$\sigma^2$ floor: $\sigma_{k,j}^2 \gets \max(\sigma_{k,j}^2, \sigma_{\min}^2)$ — 작은 topic에서 variance 0 방지. $\sigma_{\min}^2$ 는 경험적으로 결정.

---

## 확정 하이퍼파라미터 (persistence regime 기본값)

| 파라미터 | 값 | 역할 |
|---|---|---|
| $\alpha$ | 1.0 | sCRP concentration (Hi-OnTop persistence 기본) |
| $\lambda$ | 10.0 | sCRP stickiness (Hi-OnTop persistence 기본) |
| $\sigma_0^2$ | 0.01 | cold start variance prior |

**주**: 위 값은 Hi-OnTop이 가정하는 "대화 topic persistence 우세" 상황의 기본값. Phase 1-3 TopiOCQA 실측에서는 **SEM2 원본 defaults(α=10, λ=1, σ₀²=0.1)**가 F1=0.471로 우세 — TopiOCQA shift rate 28%/transition이 frequent-shift regime이라 반대 profile 필요. 벤치마크별 최적값:

| Regime | 예시 benchmark | 권장 $\alpha$, $\lambda$, $\sigma_0^2$ |
|---|---|---|
| Persistence (few shifts) | 긴 persistence 대화 | $\alpha$=1.0, $\lambda$=10.0, $\sigma_0^2$=0.01 (초기 Hi-OnTop) |
| Frequent shift (factoid QA) | TopiOCQA | $\alpha$=10.0, $\lambda$=1.0, $\sigma_0^2$=0.1 (SEM2 defaults) |

TIAGE / dialseg711 / SuperDialseg 평가에서 regime 구분 재검증.

## 실험 후 경험적 결정 대상

- $\alpha, \lambda$ 최적값 (TopiOCQA section 과분할 회피 기준)
- $\sigma_0^2$ 민감도
- $\sigma_{\min}^2$ floor 값
- Boundary score 임계치
- Topic merge threshold (centroid cosine 등)
- SEM `lmda/2` halving 포팅 여부 (`context/00-sem-paper.md` §7)
