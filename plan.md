# 구현 로드맵

> **Step 완료 규칙:** 각 Step의 하위 항목을 `[x]`로 표시하기 전,
> `python scripts/check_step_done.py`가 exit 0을 반환해야 한다.
> FAIL이 나오면 원인을 수정하고 재실행한다. 통과할 때까지 반복.
> 3-angle self-audit(구조/동작/설계)은 스크립트 실행 전 필수. 세부는 `CLAUDE.md` 참조.

## Phase 0: 자료 분석 및 설계 확정
### 0-1. SEM 논문 정독
- [x] `SEM-paper.pdf` 전체 정독 (pdftotext 산문 + 수식 10페이지 `pdftoppm` 이미지 직독)
- [x] 핵심 수식 1~24 이해 및 Hi-OnTop 계승/대체/폐기 분류 (context/00-sem-paper.md §2, §5)
- [x] `SEM/sem/sem.py` 실제 코드 검증 (`_calculate_unnormed_sCRP` L.144-159, `run()` L.161-354) — SEM2 기준 verbatim pseudocode 기록 (§4)

### 0-2. 벤치마크 데이터 분석
- [x] LoCoMo clone (`benchmarks/locomo/`)
- [x] TopiOCQA clone (`benchmarks/topiocqa/`)
- [x] LongMemEval clone (`benchmarks/LongMemEval/`)
- [x] LoCoMo `data/locomo10.json` 구조 분석 (10 conv × 27 sess × 22 turns, topic annotation 없음)
- [x] TopiOCQA 데이터 다운로드 및 분석 (dev 2514 turns, 205 conv, Wiki Topic ground truth)
- [x] LongMemEval 데이터 다운로드 및 분석 (oracle 500 Q, 6 question_types, 긴 chat)
- [x] `outputs/benchmark-analysis.md` 작성 (8239자, 3 벤치마크 + 옵션 A~F 증거 매트릭스 + Step 0-3 입력)

### 0-3. 사건 모델 설계 판단
- [x] 벤치마크 분석 결과에 근거해 사건 모델 형태 결정 (옵션 A — Centroid + diag variance)
- [x] `context/01-hi-ontop-design.md`의 미확정 섹션 채우기 (§4 옵션 A 확정, §3 Markov 확장 철회, B~E는 Phase 1/2로 위임)
- [x] `context/02-math-model.md`의 수식 확정 ($P(s_n|e_n=k)=\mathcal{N}(\mu_k, \mathrm{diag}(\sigma_k^2))$, PE = Mahalanobis)
- [x] `context/06-decision-log.md`에 판단 근거 기록 (옵션 A 선택 + Markov 철회 + 기각 옵션별 사유)

---

## Phase 1: Topic 경계 감지 코어 + 최소 동작 sanity check (현재 단계)

**목적**: 옵션 A 기반 segmentation이 최소 벤치마크(TopiOCQA)에서 쓸만한지 먼저 확인. 이 gate를 통과해야 Phase 2(메모리 계층) 설계가 허구 위에 쌓이지 않는다.

**Gate 의미**: TopiOCQA는 Hi-OnTop 주 타깃(Claude-유사 대화)과 다르므로 "최소 동작 sanity check"로 이해. PASS ≠ 최종 성공, FAIL = 확실한 재설계 필요 (비대칭).

### 1-1. 최소 실행 가능 코어 (`src/hi_ontop/`)
- [x] `embedding.py` — `bge-base-en-v1.5` L2-norm wrapper (768dim, lazy torch import)
- [x] `topic.py` — Topic 클래스 (centroid μ + diag σ² + Welford + cold start $\sigma_0^2$ + $\sigma_\min^2$ floor)
- [x] `scrp.py` — `sticky_crp_unnormed(counts, prev_k, alpha, lmda)` (SEM2 동치)
- [x] `sem_core.py` — `HiOnTopSegmenter.assign(s)` → `(k, is_boundary)`. 옵션 A centroid independence 근거로 SEM2 restart-vs-repeat 분기와 λ/2 halving은 **미포팅** (docstring 명시)

### 1-2. 단위 테스트 (18/18 passing, 0.89s)
- [x] `tests/test_scrp.py` — 7 tests: 첫 턴, stickiness, Hi-OnTop α/λ 반전, SEM2 default, 용량 가득, 입력 불변
- [x] `tests/test_topic.py` — 6 tests: Welford vs `np.mean`/`np.var`, cold start 복귀값, variance floor, log-lik 최대, PE zero@centroid, 입력 불변
- [x] `tests/test_sem_core.py` — 5 tests: 첫 턴 topic 0, stickiness 유지, 3 cluster 복원, boundary flag, 재현성

### 1-3. TopiOCQA dev 측정
- [x] `scripts/run_topiocqa_segmentation.py` — dev 2514 turns / 205 conv 전체 예측
- [x] **Metric: topic shift F1** — Ground truth `Topic` 필드 변화만, `Topic_section`은 noise(FP)
- [x] **Baseline 3종 비교**: (a) all-boundary / (b) cosine threshold sweep / (c) Hi-OnTop
- [x] **Latency 측정**: 턴당 추가 시간
- [x] 결과 기록: `outputs/phase-1-topiocqa.md`
- [x] **한계 명시**: TopiOCQA 12턴 평균 → variance 학습 기회 거의 없음, centroid 부분만 실측

**실측 결과 (HP 튜닝 후)**:
- all-boundary F1 0.451 / cosine(θ=0.70) F1 0.467 / **Hi-OnTop F1 0.471** (α=10, λ=1, σ₀²=0.1)
- Latency: 19.60 (embed) + 0.29 (assign) ≈ 19.89 ms/turn
- 탐색 이력 3 iter (`scripts/run_topiocqa_{sweep,variants,multisignal}.py`) → `outputs/phase-1-topiocqa.md §탐색 이력`
- **발견**: TopiOCQA는 frequent-shift regime → SEM2 defaults 유리. Hi-OnTop persistence 기본값은 LongMemEval 등에서 유지, regime-split은 `context/02-math-model.md`에 기록.

### 1-4. Gate 판정 + 분기
- [x] **PASS 조건**: Hi-OnTop F1 > cosine (0.471 > 0.467) AND F1 > 0.4 AND overhead ≤ 200ms
- [x] **Gate 결과: PASS (marginal)** — TopiOCQA는 frequent-shift regime, Hi-OnTop 주 타깃 아님
- [x] hyperparam regime split 확정 → `06-decision-log.md` 2026-04-24 entry

### 1-5. TIAGE dev/test topic-shift 평가 (Phase 1-3 확장)

**근거**: LongMemEval는 turn-level topic label이 없어 topic 경계 감지 평가엔 부적합 (04-benchmarks.md 참조). TopiOCQA는 factoid QA로 편향. **PersonaChat 기반 TIAGE**를 추가 평가해 Hi-OnTop segmentation의 chit-chat 대화에서의 성능을 독립 벤치마크로 검증한다.

- [x] `scripts/run_tiage_segmentation.py` — test 100 conv / 1564 turns 예측
- [x] **Metric: topic shift F1** — label `'1'` = shift, `'0'` = continue (인간 annotated, Cohen's Kappa 0.48)
- [x] **Baseline 3종 비교**: (a) all-boundary / (b) cosine threshold sweep / (c) Hi-OnTop
- [x] **HP 두 regime 병행**: persistence (α=1, λ=10, σ₀²=0.01) vs freq-shift (α=10, λ=1, σ₀²=0.1)
- [x] 결과 기록: `outputs/phase-1-tiage.md`
- [x] **Gate 조건**: TopiOCQA와 동일 (baseline 대비 우위 AND F1 > 0.4 AND latency +20% 이내)

**실측 결과 (2026-04-25, Colab A100)**:
- all-boundary F1=0.354 / **cosine(θ=0.525) F1=0.421** / Hi-OnTop persistence F1=**0.317** / Hi-OnTop freq-shift F1=**0.377**
- Latency: 0.73 ms/turn (overhead PASS)
- **Gate FAIL**: Hi-OnTop 두 HP 모두 cosine baseline에 명확히 패배 (Hi-OnTop F1 < 0.421, F1 < 0.4)

### 1-6. Phase 1 종합 Gate

- [x] TopiOCQA gate: 1-4 기준 PASS (marginal, F1=0.471 vs cosine 0.467)
- [x] TIAGE gate: 1-5 기준 **FAIL** (Hi-OnTop ≤ cosine, F1 < 0.4)
- [ ] **두 gate 모두 PASS** → Phase 2 진입 — **불충족**
- [x] 둘 중 하나라도 FAIL → `06-decision-log.md` append, 옵션 A 재검토 or regime-split HP 확장 — **현재 결정 대기 (report.md §7 옵션 A~E 참조)**

**종합 Gate: FAIL.** 후보 5종(`report.md §12`) 중 #1 종료, 권장 경로 = #5 → #3:
1. [x] TIAGE HP sweep (108 configs, 2026-04-25) — **best F1=0.383 (α=10, λ=3, σ₀²=0.1) < cosine 0.421, 두 Gate 조건 모두 FAIL**. → "어떤 HP로도 못 넘는다" 결정적 증거. `outputs/phase-1-tiage-sweep.json`
2. [ ] Hi-OnTop likelihood를 cosine-vs-last-turn으로 교체 (옵션 A 변형) — 보류
3. [ ] Phase 2 reframing 진입 — boundary F1 ≠ Hi-OnTop 핵심 가치, downstream QA로 정직 검증 ⭐
4. [ ] 옵션 D escalation (multi-signal 재설계) — TopiOCQA 전례상 효과 약함, 보류
5. [x] Hi-OnTop clustering 품질 측정 (V-measure / NMI / ARI, 2026-04-25) — **가설 반박**: 모든 metric에서 cosine 우위. **새 발견**: Boundary F1 ↔ ARI trade-off — persistence HP (α=1) ARI=0.398/0.397 > freq-shift HP (α=10) ARI=0.187/0.314. **메모리 시스템 관점에선 persistence HP 적합** → Phase 2 HP 선택 근거 확보. `outputs/phase-1-clustering-quality.md`


---

## 이후 — 독립 화제분절 프로젝트(Hi-OnTop)로 전환

Phase 1 의 boundary-F1 gate 는 marginal/FAIL 이었으나, 이 분석은 **Hi-OnTop 을
독립적인 실시간 화제분절기**로 발전시키는 출발점이 되었다 (downstream QA 의
하위 모듈이 아니라 분절 그 자체를 목표로 재정의).

현재 로드맵·문제·시도 이력은 더 이상 이 파일이 아니라 다음에서 관리한다:

- `handoff/` — 추적 중인 문제별 ledger (정의 + 모든 시도 + 결과)
- `context/methodology/` — 버전별 분절 알고리즘 정의 (v1 → v4.x) + infrastructure
- `context/06-decision-log.md` — 설계 결정 이력 (append-only)

(구 메모리 계층 Phase 2~5 — LTM / Memory window / orchestration / QA 평가 —
설계는 본 분절 레포 scope 밖이며 제거되었다.)
