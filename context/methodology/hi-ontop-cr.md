# Hi-OnTop-CR — de-neut + 적응-β 신호 + commit-and-refine deploy

`src/hi_ontop/hi_ontop_cr.py` · `segment(emb, reset=...)` · 2026-06-11 · **promoted main 분절 모델**

## 한 줄
[[hi-ontop]] (δ_eff causal threshold) 의 후속 main 모델. **신호**를 δ_eff 에서 *de-neut + run-length
적응-β* 로 교체(두 도메인 universal, calibration-free). **deploy reset 기본 = `threshold`(0-lag, 즉시)**;
`commit_refine` 은 버퍼(~26s) 허용 시 +0.029 Score 사는 옵션. 신호는 oracle 천장에서 LLM 초과 — 그러나
online deploy 는 구조적 격차가 남음(정직한 한계, §5).

## 1. 신호 (universal, 두 도메인, calibration-free)
발화 임베딩 `x`(인코더 고정 MiniLM-int8, 단위정규화), online·0-look-ahead·무학습.
- **active prototype** `m` = 현재 segment EWMA, **global** `g` = 전체 EWMA(g_rho=0.15).
- **de-neut**: `deneut(v) = normalize(v − β·(v·g)·g)` — global(중립) 성분 β 제거 → '화제의 변별적 방향'.
- **run-length 적응 β**: `β_t = clip(A − B·log(1+k/L0), 0, 1)`, k=현 segment 길이.
  짧은 segment(DTS sharp)→β→1(full de-neut), 긴 segment(AMI drift)→β↓(V_rel). **run-length 가 sharp/drift
  를 도메인 라벨 없이 자동 판별.**
- **신호**: `r_active = 1 − cos(deneut(x), deneut(m))`, `V = r_active − λ·(1 − cos(x, g))`, λ=0.6.
- **판정**: 적응 임계치 `μ + c·σ` (V 의 running mean/std, warmup=8). c 는 **calibration-free**
  (전 구간 baseline 이상; 고정 c=1.0 또는 Otsu).

근거(axis-1, [[hi-ontop]] §2026-06-10 의 정식화): de-neut 가 superseg 벽(0.467) 첫 돌파(0.506), 적응 β 로
4개 데이터셋 oracle strict(δ_eff 초과). V_rel 은 AMI oracle 0.687 > LLM 0.543. 검증: 2-fold/LOO/AUC/
Score-vs-c (handoff §1.5).

## 2. Deploy reset (`reset=` 인자)
신호는 동일, **reset 메커니즘만 선택**. **default = `threshold` (0-lag).**

### ★ lag–Score 트레이드오프 (2026-06-12, AMI 139 best-c by Score)
| reset | lag | Score | ±2F1 |
|---|--:|--:|--:|
| **`threshold`** (default, 즉시) | **0 (다음 턴)** | 0.372 | 0.106 |
| `commit_refine` L=2 | ~6s | 0.364 | 0.103 |
| `commit_refine` L=4 | ~12s | 0.379 | 0.118 |
| `commit_refine` L=8 | ~26s | **0.401** | 0.140 |
- **commit_refine 의 +0.029 Score 우위는 전적으로 lag 매입분** — L≤3 에선 threshold 와 동급/이하. 0-lag(버퍼
  불가) 요구면 **`threshold` 가 best**(de-neut+적응β 즉시 hard reset). 버퍼(≥~12s) 허용 시에만 commit_refine.
- sharp-seam(DTS)은 commit-and-refine 이 짧은 segment 억제로 회귀 → DTS 는 항상 `threshold`.

### 2a. `commit_refine` (drift / AMI) — deploy best
hard reset(즉시·불가역)을 **bounded-lag commit-and-refine + local split-gain b\* refinement** 로 교체.
- 경계 *감지*(V>θ)는 **split 후보 생성**으로만 — 즉시 reset 안 함.
- shock 으로 armed → 오른쪽이 m_min 차거나 윈도우 L 소진 시, 윈도우 후보 split 위치 b 별
  **split gain** `= SSE(a..t) − [SSE(a..b−1) + SSE(b..t)]` (SSE=cnt−‖Σz‖²/cnt) 의 **argmax b\* 에 경계 emit**
  (위치 정정 → localization).
- 확정 시 왼쪽 prototype 을 b\* 오른쪽 발화로 reset(clean), 임계치 stats 재초기화.
- **효과**: ±2F1 0.131→0.140, Score 0.372→0.401 (hard-reset 대비). b\* refinement 가 핵심
  (확정-only v1 은 위치 안 고쳐 무효).

### 2b. `threshold` (sharp / DTS)
기존 detected-reset hard reset (V>θ → 즉시 reset). DTS concat-seam 은 직전 점프가 sharp 해
commit-and-refine 의 응집 대기가 짧은 segment 를 억제·회귀시킴 → sharp 도메인은 hard reset 유지.

## 3. 정식 config (검증됨)
`DEFAULTS`: c=1.0, A=2.0, B=1.0, L0=8, λ=0.6, g_rho=0.15, rho_min=0.05, R=4, warmup=8 / commit_refine: L=8, m_min=2, Mc=0.0
- **c**: calibration-free (전 구간 1.0~1.5 hard-reset 이상; 고정 c=1.0).
- **L=8**: AMI k-fold CV 검증 — 5-fold 에서 train-선택 L\*=8 **만장일치(5/5)**, held-out overfit gap 0,
  Score plateau L=6~32(완만). overfit 아님.
- **(A,B)=(2.0,1.0)**: 신호 단계 LOO cross-domain 검증 (handoff §1.5).
- **m_min=2**: AMI k-fold CV 검증 — 5-fold m\*=2 **만장일치(5/5)**, held-out overfit gap 0
  (m=1·2 ±2F1 동률 0.141/0.140, m=2 Score-best 0.401, m≥3 회귀). → **정식 config 전 상수 검증 완료, 미검증 0개.**

## 4. SEM 계승
de-neut 의 r_global 항 = "공통/background 대비 화제 변별" ≈ SEM new-event **base distribution** 대비.
active prototype reset = SEM event 신규 개시. run-length 가중 = event persistence reliability.
commit-and-refine = SEM local-MAP event 개시를 "응집 확인 후 확정"으로 보수화(bounded lag);
split-gain b\* = event 시작점 추정; shadow/오른쪽 prototype = 미확정 event 후보 잠정 추정.
SEM 철학(sCRP/Bayes/local MAP/scene dynamics)과 충돌 없음 — heuristic 근사임은 명시(decision-log 2026-06-11).

## 5. 알려진 한계 (정직)
- **★ online deploy 격차 미해결**: commit_refine AMI ±2F1 0.140 ≪ **oracle 천장 0.554**(clean+μcσ, gold-reset).
  online 운영의 구조적 한계 — 놓친 경계가 prototype 을 두 topic 으로 오염, 발화 단위 복구 불가.
  reset 기법 5각도(v1 확정 / v2 b\* refinement / v3 sliding / v4 2-stage / v5 오염완화) 모두 격차 못 메움
  ([[hi-ontop]] 후속 REPORT). → main 승격은 "신호는 LLM급(oracle), online 실현은 구조적 천장" 전제.
- **DTS deploy 미정식화**: threshold 모드는 신호만 교체한 hard reset — DTS deploy 전체 비교는 별도(현재 AMI 중심).
- **streaming 클래스 미통합 (deferred)**: 본 모듈 batch `segment()` 는 **이미 online·0-look-ahead
  충족**(좌→우, bounded-lag emit) — 알고리즘 online 속성은 OK. 단 turn-단위 즉시 반환 streaming `assign()`
  API 는 미통합. 설계 질문 = commit-and-refine 이 b\* 를 lag 후 **retroactive emit** 하므로 `assign()`
  즉시-반환 API 와 불일치 → buffer-and-flush(확정 경계를 L-lag 후 방출) wrapper 필요(codex 위임 대상).
  필요 시 thin wrapper 로 붙임. [[hi_ontop]] (δ_eff streaming) 은 그대로 유지.
- **b\* 는 감지된 후보 위치만 정정**(놓친 경계 recall 은 못 살림 — 격차의 본질).

## 6. 변경 이력 / 후속 후보
- 2026-06-11 신설·승격 (commit-and-refine, codex 자문 + v1~v5 실측). 상세 REPORT
  `outputs/experiments/2026-06-11_ami_commit_refine/REPORT.md`, decision-log 2026-06-11.
- 후속(수확체감 아닌 방향): (A,B)·m_min 자기보정(tuning 제거), streaming 클래스(lagged emission) 설계.
  단순 reset/prototype 미세조정은 종료(5각도 음성).
