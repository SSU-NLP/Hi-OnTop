# HANDOFF — Hi-OnTop vs LLM 실시간 화제분절 비교 (streaming quality-latency-cost)

**지정일**: 2026-06-12 · **상태**: 설계 확정 단계, sweep 미실행 · **데이터**: AMI 139 (subset N=37)

## 0. 문제 정의
- **배경**: Hi-OnTop(임베딩 기반 0-lag 화제분절기)이 LLM 대비 *실시간 스트리밍*에서 갖는 우위를 정량화.
- **목표/주장 (codex 자문 반영, 좁게)**: "**동일한 스트리밍 제약**(제한된 look-ahead·decision latency·호출비용)
  하에서 0-buffer 임베딩 분절기가 버퍼드 LLM 대비 우월한 **quality–latency–cost tradeoff**." full-context LLM 은
  online baseline 아님 → **offline oracle / upper-bound** 로 명명.
- **성공기준**: (a) buffer↓일수록 LLM 품질 열화 곡선, (b) Hi-OnTop 0-lag 가 streaming-buffer LLM 과 동급 품질을
  0 호출·~ms latency 로 달성, (c) full-context LLM 만 이기되 무한 look-ahead 비용을 치름 — 을 지표로 입증.

## 1. 비교 설계
- **Hi-OnTop**: MiniLM-int8(인코더 고정) + de-neut V + 적응임계 μ+cσ. 0-look-ahead, LLM 0콜.
  `src/hi_ontop/hi_ontop_cr.py` `segment(reset=...)`: **default `threshold`(0-lag, Score 0.372/139)**,
  `commit_refine`(L=8≈26s lag, Score 0.401/139) 옵션.
- **LLM baseline 8모델**: `openrouter/` 경유 Crts — openai/gpt-4o, gpt-4o:nitro, gpt-4o-mini,
  anthropic/claude-haiku-4.5, qwen/qwen3.5-27b, mistralai/mistral-small-3.1-24b-instruct,
  google/gemma-3-12b-it, google/gemma-3n-e4b-it. (Microsoft phi-4/mini 는 사용자 제외.)
  - buffer B초 {0,5,10,15,30,60,120,∞} × chunk offset 평균 {0,B/4,B/2,3B/4}. no-thinking, temp 0.
- **subset(sweep용) N=37**: 2D 층화 — 시리즈(ES/IS/TS/IB) 비례 × 시리즈내 길이median×density median 4사분면
  + 최장회의 force-include. 검증: series%/n_turns[76,1194]/n_bnd[1,19]/density 모두 full 과 일치(seed-free
  등간격 결정론). → `scripts/select_ami_subset.py`, `outputs/runs/_misc/ami_subset.json`.
- **확정 비교**: 139 전수 4메서드만 (Hi-OnTop / best buffered LLM / best cheap LLM / full-context oracle).
  "exploratory subset sweep + confirmatory full-set" 분리 보고.

## 2. 지표 (사용자 확정: F1·Pk·WD·Score + codex 권고)
- **분절 품질**: ±2-tol F1(primary), Pk, WindowDiff, Score(=0.5F1+0.25(1−Pk)+0.25(1−WD)), precision/recall,
  pred/gold ratio, mean segment length, per-meeting 분포, ES/IS/TS subgroup, matched-boundary timing offset,
  **계층-level 분석**(AMI top-level vs all-level(sub-topic) 정합 + 예측경계 top-hit/sub-only-hit/false).
- **latency 3분할**: algorithmic look-ahead(Hi-OnTop=0) / decision-delay(gold경계→emit, p50/p95) /
  compute latency(p50/p95).
- **비용·실시간**: RTF, model-call-count(/meeting,/hour), peak VRAM/RAM, token/cost, failure/invalid-output rate,
  wall-clock. paired bootstrap meeting-level CI.

## 3. 시도 ledger (조건 + 결과)

### 3a. Hi-OnTop live-streaming latency (subset 37미팅 17434턴, batch=1, no cache) ✅
- 조건: `scripts/latency_streaming.py`. 발화 도착→live encode(MiniLM-int8, batch=1, 캐시 없음)+segment 1-step. 부하 0(로컬 CPU).
- **cold-start**: 모델 로드 8.4s + 첫 임베딩 11.6ms = ~8.4s (일회성).
- **warm steady-state turn당**: encode p50 1.53 / p95 3.83 / p99 6.22 / max 33.3 ms. segment p50 0.03ms(무시).
  **총 p50 1.56 / p95 3.86 / p99 6.26 / max 33.4 ms**.
- **RTF**: median 0.00049 / p95 0.00064 / max 0.0008 → 실시간 ~2050배(절대 안 밀림). audio=start-span 근사.
- (구 pilot: ES2002a warm median 1.54ms — 위가 정식판: tail p99 + cold/warm + RTF 포함.)
- TODO(미측정): concurrency{1,4,8,16} p95/p99+memory, peak RSS, decision-delay(threshold=0).

### 3b. LLM per-call latency (236턴 고정 transcript, full-context 1콜, ⚠️동시부하 하=상한)
| 모델 | median(s) | min(s) |
|---|--:|--:|
| phi-4-mini(제외) | 1.03 | 0.96 |
| mistral-small-24b | 1.93 | 1.81 |
| qwen3.5-27b | 3.38 | 2.02 |
| phi-4(제외) | 4.25 | 2.93 |
| gemma-3-12b | 7.09 | 6.43 |
| gemma-3n-e4b | 11.50 | 9.28 |
→ Hi-OnTop(0.36s/미팅) 대비 최速 mistral도 ~5×. **clean(부하0) 재측정 + gpt-4o/4o:nitro/4o-mini/haiku 추가 TODO.**

### 3c. LLM full-context Score (⚠️12미팅 ES2002–2004 편향 subset — N=37로 재실행 TODO)
| 모델 | Score | ±2F1 | pred/gold | 비고 |
|---|--:|--:|--:|---|
| qwen3.5-27b | 0.647 | 0.557 | 97/84 | 강함(figure_R 0.640 재현) |
| gpt-4o | 0.612 | 0.504 | 92/84 | |
| claude-haiku-4.5 | 0.609 | 0.549 | 132/84 | |
| gpt-4o-mini | 0.567 | 0.464 | 110/84 | |
| gemma-3n-e4b | 0.349 | 0.074 | 63/84 | under-seg |
| mistral-small-24b | 0.338 | 0.000 | **0**/84 | ⚠️ pred 0 = 출력/파싱 깨짐 |
| gemma-3-12b | 0.269 | 0.254 | 375/84 | 과분절 |
- 해석: full-context면 frontier LLM(qwen/gpt-4o/haiku ~0.61–0.65)이 Hi-OnTop(0.40) 능가. 작은 gemma 류는 mediocre.

### 3d. LLM buffer ±2F1 (figure_R legacy, 12미팅, Qwen, Score 미측정)
- 10s 0.256(907콜) / 30s 0.237(585) / 60s 0.254(340) / 120s 0.372(182) / ∞ 0.526(12). → streaming 버퍼면 LLM 열화.
- **Score(Pk/WD) 미측정 → `ami_llm_buffer_eval.py`(Score 배선 완료)로 N=37 재실행 TODO.**

### 3e. 계층-level 분석 (Hi-OnTop threshold, 139) ✅
- top-level 경계 938 / all-level 1831 / 계층 미팅 125/139.
- ±2F1: vs top 0.131 / **vs all 0.206**(더 높음 → 세밀 sub-topic도 잡음). 예측경계: top-hit 8%/sub-only 8%/false 84%.

### 3f. Hi-OnTop deploy Score (139, 참조)
- threshold(0-lag) 0.372 / commit-refine(L8) 0.401 / δ_eff 0.203 / oracle 천장 ±2F1 0.671.

## 3z. 확정 실험 사양 (codex 종합감사 2026-06-12 반영 — `codex_audit.md`)

### A. positioning / 통계 규율
- 주장 = **"동일 online decision budget 에서 Pareto frontier 어느 지점이 우월한가"** (단순 "임베딩이 LLM 보다 쌈" ❌).
  보고는 single best-table 대신 **Pareto 곡선**: Score vs decision-delay / vs cost-per-hour / vs p95 latency.
- **dev/test 분리 (must, selection bias 방지)**: subset N=37 = **dev 전용**(buffer×model sweep·HP·대표모델 선택).
  최종 139 = **locked test**, dev 에서 고른 설정만 **한 번** 평가. best buffered LLM 을 test 에서 고르지 않는다.
- paired bootstrap meeting-level CI. 주요 B{0,15,60,∞}는 139 전체 sanity-check.

### B. 공정 비교 조건 (must)
- **latency 는 캐시·전처리 금지, live per-turn 임베딩.** Hi-OnTop = 발화 도착→그 자리서 encode(batch=1, no cache)
  →segment. 임베딩 시간을 online 비용에 포함(전처리로 숨기지 않음). (품질/Score 측정은 캐시 OK — 값 동일.)
- **decision time 정보표**: 메서드별 "판정 시점에 가용한 정보" 명시 — Hi-OnTop threshold(과거만, 0 look-ahead) /
  commit-refine(최대 8턴 revision, 평균·최대 lag) / buffered LLM(B초 future, **emit 은 buffer close 이후**) /
  full-context(offline oracle, online 표에서 분리).
- **LLM decision-delay**: gold 경계 발생 시각 → 그 경계를 *emit 가능한* 시각(= 해당 buffer 가 닫힌 뒤). 단순 compute 아님.
- **입력 전처리 동일화**: speaker label/timestamp/punctuation/turn 단위 제공 여부를 전 메서드 동일.
- **boundary-count hint**: gold count 힌트 = oracle(금지). online 은 과거 추정 count/density prior 만 허용.
- **공정성 핵심 방어 = local quantized LLM** 도 같은 머신에서 측정(= "API 라서 느린 것"이 아님을 입증).

### C. 지표 전체 (primary/secondary)
- **품질 primary**: ±2-tol F1 + **Pk + WD (raw)**. **Score 는 summary 로 강등**(가중치 sensitivity table appendix).
  - **tolerance curve ±0/±1/±2/±3/±5 F1** (must — tolerance cherry-pick 방어). Pk/WD window-size sensitivity.
- **품질 secondary**: precision/recall, pred/gold ratio, mean segment length, per-meeting 분포, ES/IS/TS subgroup,
  boundary displacement(matched offset mean/median/MAE, early/late bias), **계층 top-hit/sub-only-hit/false**(강점).
- **latency (must)**: 3분할 = algorithmic look-ahead(0) / decision-delay / compute. 각 **p50/p95/p99** +
  **cold-start vs warm steady-state 분리** + timeout/retry/invalid rate 포함 end-to-end. **RTF**(처리시간/audio길이).
- **throughput/concurrency (must, online 주장)**: streams/sec, turns/sec, 동시 회의 {1,4,8,16}에서 p95/p99 + memory.
- **memory (공정 보고)**: 표 둘로 — client-observed(peak/steady RSS, CPU thread, wall, network) vs provider/local
  (local LLM 만 VRAM/KV/quant/batch). **API LLM = "N/A remote"(0 아님)**. Hi-OnTop = peak RSS+model size+thread+cache 공개.
- **cost normalization**: cost/meeting·/audio-hour·/1k-turns·/predicted-boundary·/matched-boundary·**cost-at-target(Score≥0.37)**
  + invalid/retry 비용.
- nice-to-have: energy(J/Wh), p999, calibration(reliability plot), robustness(short/dense/sparse 회의별).

### D. baseline 확장 (near must)
TextTiling/lexical, fixed-threshold embedding, **pretrained neural segmenter(CrossEncoder/BERT)**, random/density-matched,
oracle-boundary-count(offline upper-bound only), **local quantized LLM**. nice: 다른 encoder(mpnet/e5/bge-small),
sentence vs turn-level.

### E. subset 추가 층화축 (codex) — 다음 재선정 시 반영
현재(길이×density) + **화자 수 / audio 시간 / turn 길이 mean·var / n_topic_top·all·계층깊이 / scenario vs IB(비시나리오)**.

### F. AMI gold 처리
"정답 복원" ❌ → **"AMI annotation convention 에 대한 alignment"** 로 표현 낮춤. tolerance curve + top/all-level +
near-miss + meeting bootstrap CI + (가능시) IAA/human upper bound 인용 + disagreement 큰 회의 error analysis.

## 4. 미해결 / 다음 (TODO)
- ✅ Hi-OnTop live-streaming latency(cold/warm·p99·RTF) 측정 §3a. ✅ buffer eval `--subset/--offsets` 배선·검증.
- ✅ **비용절감(codex `codex_cost.md`)**: concurrency 가 압도적 1순위. buffer eval 에 ThreadPool(`--workers`)+retry/backoff
  +prompt 캐시(`llm_cache_*.jsonl`, 재실행 skip)+dedup 구현. 검증: workers 8 → 54콜 18s, **0 errors**(Crts 동시호출 허용).
- 🔄 **Qwen3.5-27b dev sweep 진행중**(N=37, B{10,30,60,120}, offset{0,0.5}, **workers 12 → ~45-60분**, `sweep_qwen.log`).
1. **mistral-small-24b pred 0** — 출력 포맷/파싱 깨짐. JSON schema/strict + retry, failure-rate 지표로 보고.
2. **나머지 모델 buffer sweep** (대표 gpt-4o/gpt-4o-mini/claude-haiku; 비용상 B 줄여 scoped).
3. **clean latency LLM 8모델**(부하 0) + **decision-delay**(buffer close emit) + concurrency{1,4,8,16} + peak RSS.
4. **확정 139 전수 4메서드** (dev=N37 에서 선택 → test=139 한 번만).
5. **미측정 지표**: tolerance curve ±0~5, cost normalization, paired bootstrap CI, memory 2-표.
6. **baseline 확장**: TextTiling/fixed-thr/CrossEncoder/random + **local quantized LLM**(공정성 핵심).
7. 12-ES pilot 폐기(편향). subset=dev / 139=locked test 규율.

## 5. 산출물
- 스크립트: `ami_llm_buffer_eval.py`(Score·CF헤더 배선), `ami_llm_segment_eval.py`(Score 배선),
  `select_ami_subset.py`(N=37 2D 층화), `plot_score_latency.py`(figure_S, LLM 자리 비움),
  `ami_deploy_failure_anatomy.py`, `ami_commit_refine_deploy.py`.
- codex 자문: `outputs/runs/_misc/codex_expdesign.md`(지표 1차), `codex_audit.md`(종합감사, 진행중).
- subset: `outputs/runs/_misc/ami_subset.json`. figure: `outputs/figures/figure_S_score_latency.{pdf,png}`.
- 인프라 caveat: Crts = Cloudflare Access 게이트웨이 뒤 → `.env` 에 `CF_ACCESS_CLIENT_ID/SECRET` 필요(인가됨).
  모델 id 는 `openrouter/<provider>/<model>` (models.list 미등록 slug 도 라우팅됨).
