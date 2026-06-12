# HANDOFF — Hi-OnTop vs LLM 실시간 화제분절 비교 (streaming quality-latency-cost)

**지정일**: 2026-06-12 · **상태**: 설계 확정, **새 7모델·139 전수 실행 대기** · **데이터**: **AMI 139 전수**

> ## ★ 핵심 평가 기준 (사용자 확정 2026-06-12, 최우선)
> - **품질 Score = 처음부터 전체 84-그리드를 AMI 139 전수**(7모델 × {seg full/120/60/30/10 + incremental} × {baseline,
>   verbatim}). **ⓑ 확정**(37-dev 탐색 단계 생략). 전 조건을 139에 보고 → selection 없음 = selection-bias 없음.
> - **Latency = Score 우수 config 만** 측정(전 그리드 아님). 이유: latency 는 **N=1 순차 fresh 라 가장 비싼 작업**(모델 내부
>   병렬 불가). Score@139 에서 잘 나온 모델·버퍼·**우승 프롬프트**만 골라 **18-미팅 latency subset** 에서 측정. 선정은
>   Score 결과 나온 뒤(어떤 config 가 우수한지). 비-LLM baseline 도 동일 머신 latency 포함.
> - **latency subset N=18** (content-independent → 전수 불필요): 크기·density 극단 포함 + 모든 버퍼 window-크기 분포를
>   **전체 139 와 KS 재현**(D≤0.12, p≥0.94)하는 **최소** 미팅. `select_latency_subset.py`→`ami_latency_subset.json`, `ami_full139.json`. KS 표 appendix.
> - Pareto: **Score(139) vs latency(18-subset, 우수 config)** — latency 는 메서드·입력크기의 속성이라 미팅셋이 달라도 무방.
> - **프롬프트**: baseline vs verbatim 은 **품질(Score) 비교만**. **latency 는 Score 우승 프롬프트 1개로만** 측정(우승본=보고 baseline).
> - **offset {0,0.5} 폐기**: 구 A-mode(독립 window) 정렬 artifact 회피용. 채택한 **B-mode(누적)는 offset 미사용**(단일 누적 pass).
**DTS 범위**: 본 streaming 비교는 **AMI 전용**(`data/ami`만 보유). DTS(tiage/dialseg711/superseg)는 신호발견·인코더
ablation(§3i)·자매 handoff 0609 의 평가축이며, LLM 프롬프트는 DTS+AMI 공통(turn기반)으로 일반화해 둠(향후 DTS 확장 대비).

> ## 📦 데이터·gold 정의 (📎 논문 data 섹션·재현)
> - **AMI 139 미팅** (`data/ami/topic/<mid>.json`). 시리즈: **ES 57 / IS 38 / TS 40 / IB 4**(IB=비시나리오).
>   총 **62,678 턴**, 미팅당 turn [76, 1194] median 409.
> - **turn 단위** = `{start(초, 발화 시작 timestamp), speaker, text}`. start 는 **버퍼 windowing + RTF audio 길이 근사**에 사용.
> - **gold 경계 2종**: `bnd_top`(per-turn 0/1, **top-level 화제경계 = primary gold**, 총 **938**) / `bnd_all`(sub-topic 포함
>   all-level, 총 **1831**). `n_topic_top`/`n_topic_all`/`topic_levels`(계층깊이) 메타 포함. **마지막 turn 경계는 제외**(평가 시).
> - 계층-level 분석(§3e/§3z F): top vs all 양쪽 정합 보고 — Hi-OnTop 강점(세밀 sub-topic 도 포착).

> ## 📟 측정 환경 (하드웨어 — 📎 논문 reproducibility, latency 는 머신 의존)
> - **GPU: 2장 붙어있으나(/dev/nvidia0,1, 드라이버 580.82.07) NVML 초기화 막힘** ("Failed to initialize NVML: Unknown Error").
>   디바이스·드라이버·libnvidia-ml 버전 다 일치하는데 NVML 만 죽음 = **호스트 cgroup/런타임 이슈**(흔히 컨테이너 시작 후
>   호스트 daemon-reload 가 GPU cgroup 권한 박탈). 컨테이너 내부 해결 불가 → **호스트에서 `--gpus all` 로 재기동** 필요.
>   현재 onnxruntime providers = **CPU 만**(CUDAExecutionProvider 없음). → **모든 latency 는 CPU 측정**(GPU 살면 재측정·별도 표기).
> - **CPU: AMD Ryzen Threadripper PRO 7965WX** (24코어 / 48스레드, x86_64). **RAM 251 GB**.
> - 인코더(MiniLM int8 ONNX) CPU 추론, Hi-OnTop·비-LLM baseline latency 전부 이 머신 CPU. API LLM 은 remote.
> - ⚠️ latency 는 머신 의존 → fair-table 은 **전 메서드 동일 머신**(현 CPU)에서 재측정. GPU 가용 시 encode 한 자릿수 ms 가능(별도 표기).

## 0. 문제 정의
- **배경**: Hi-OnTop(임베딩 기반 0-lag 화제분절기)이 LLM 대비 *실시간 스트리밍*에서 갖는 우위를 정량화.
- **목표/주장 (codex 자문 반영, 좁게)**: "**동일한 스트리밍 제약**(제한된 look-ahead·decision latency·호출비용)
  하에서 0-buffer 임베딩 분절기가 버퍼드 LLM 대비 우월한 **quality–latency–cost tradeoff**." full-context LLM 은
  online baseline 아님 → **LLM-offline (offline oracle / upper-bound)** 로 명명(§1B).
- **성공기준**: (a) buffer↓일수록 LLM 품질 열화 곡선, (b) Hi-OnTop 0-lag 가 streaming-buffer LLM 과 동급 품질을
  0 호출·~ms latency 로 달성, (c) full-context LLM 만 이기되 무한 look-ahead 비용을 치름 — 을 지표로 입증.

## 1. 비교 설계
> **교차참조 (중복 금지)**: Hi-OnTop **알고리즘/수식 스펙** = `context/methodology/hi-ontop-cr.md`(de-neut V·적응-β·
> commit-refine 정식화, 단일 진실원천). **신호 발견·oracle 천장·deploy reset 부트스트랩 스토리** = 자매 handoff
> `HANDOFF_0609_deploy-oracle-gap.md`(축1 신호 성공 / 축2 deploy 미해결). 본 handoff 는 그 위에서의 **LLM 비교 실험**만.
- **Hi-OnTop**: MiniLM-int8(인코더 고정) + de-neut V + 적응임계 μ+cσ. 0-look-ahead, LLM 0콜.
  **본 비교 = `threshold`(0-lag, Score 0.372/139) 단일** (`src/hi_ontop/hi_ontop_cr.py` `segment(reset='threshold')`).
  (`commit_refine`(L=8 lag, 0.401)는 코드에 남아있으나 **본 비교 미사용 — 버퍼/lag 원치 않아 폐기**.)
  - 📎 **재현 config (appendix, 본 비교에서 고정)**: 인코더 `sentence-transformers/all-MiniLM-L6-v2`, **8비트 양자화됨**
    (ONNX quint8 = fp32→uint8 동적양자화, `onnx/model_quint8_avx2.onnx`, SentenceTransformer ONNX backend, CPU/AVX2), 단위정규화.
    HP `DEFAULTS`(`hi_ontop_cr.py`, threshold 경로): c=1.0, A=2.0, B=1.0, L0=8, λ=0.6, g_rho=0.15, rho_min=0.05,
    R=4, warmup=8, m_min=2. c·δ **calibration-free**(고정 c=1.0), seed=0.
- **LLM baseline 7모델 (확정 2026-06-12, 전 실험 공통)**: `openrouter/` 경유 Crts —
  openai/gpt-5-mini, openai/gpt-5-nano, qwen/qwen3.5-27b, anthropic/claude-haiku-4.5,
  mistralai/mistral-small-3.1-24b-instruct, google/gemma-3-12b-it, google/gemma-3n-e4b-it.
  (변경 2026-06-12: gpt-4o:nitro·gpt-4o-mini 제외 → gpt-5-mini·gpt-5-nano·gemma-3n-e4b-it 추가. phi-4류[Microsoft] 계속 제외.)
  → `secom_sweep.sh`/`secom_latency_full.py` MODELS 동기화. 겹치는 모델(qwen/haiku/mistral/gemma-3-12b) 캐시 재사용.
  **프롬프트 = SeCom 정식 instruction**(home-made 폐기), baseline(우리 적응본)+verbatim(SeCom 원본) 둘 다 — 상세 **§1A**.
- **latency**: 입력버킷 샘플 ❌ → 실제 워크로드 콜 N=1 fresh, **latency 전용 18-미팅 subset 內 전수**(content-independence
  근거, §3z C·★핵심). 단 **전 그리드 아님 — Score@139 우수 config(모델·버퍼·우승 프롬프트 1개)만** 측정(latency 가 가장 비싼
  작업). buffer {full,120,60,30,10}+incremental, **B-mode(누적, offset 미사용)**, no-thinking, temp 0. 출력예산 incremental
  16tok / segment 1024tok.
- **subset N=37 (구 dev — 현재 quality 미사용)**: 2D 층화(시리즈 비례 × 길이median×density median 4사분면 + 최장회의).
  `select_ami_subset.py`, `ami_subset.json`. ⓑ(139 전수)로 결정되며 quality 탐색용 역할 폐기 — latency-18 만 139 분포 재현용 별도 파생.
- **비교**: 전체 84-그리드를 **139 전수**(ⓑ). 메서드 = Hi-OnTop / LLM-Bsec / LLM-offline / LLM-incremental + 비-LLM
  (TextTiling/GraphSeg/GreedySeg/CSM). 명명 §1B. (37-dev 탐색·dev/test 분리 없음 — §3z A.)

## 1A. LLM 프롬프트 명세 (📎 appendix 수록 — 어떤 프롬프트로 분절했는지 전량 공개)

LLM baseline 분절은 **home-made 프롬프트 폐기** → **SeCom(Microsoft, ICLR2025) 정식 instruction** 채택.
두 계열을 **둘 다** 돌려 비교(`--prompt baseline|verbatim`). 모든 프롬프트 파일은 `scripts/secom_prompts/` 에 vendor.
appendix 에는 아래 4개 파일 **전문(全文)** 을 그대로 싣는다.

| 계열 | 모드 | 파일 (`scripts/secom_prompts/`) | look-ahead | 출처/성격 |
|---|---|---|---|---|
| **baseline**(우리 적응본, 기본) | offline/buffer 분절 | `baseline_segment.md` (3118자) | 청크 전체 | SeCom `segment_turn` 구조를 **turn 기반·multi-party 일반화**(DTS 2자 + AMI 멀티파티 공통) |
| **baseline** | online incremental | `baseline_incremental.md` (1538자) | **0** | 위 구조를 **per-turn Yes/No** 결정으로 확장(우리 작성) |
| **verbatim**(SeCom 원본) | offline/buffer 분절 | `segment_exchange.md` (3275자) | 청크 전체 | SeCom 원본 `segment_with_exchange_number`(user-bot exchange 번호) **그대로** |
| **verbatim** | online incremental | `segment_incremental.md` (318자, 5줄) | **0** | SeCom 레포 원본 incremental Yes/No **그대로** |

- **두 모드 정의** (`scripts/secom_llm_eval.py`):
  - `segment` (exchange/turn-number JSONL): `--buffer full`=회의 전체 1콜=**LLM-offline**(upper-bound). `--buffer B`초=**LLM-Bsec**
    = **B-mode 누적**(`--context B`, §1B): B초 checkpoint 마다 **누적 prefix [0,k_i] 전체** 재분절, **새 구간 [k_prev,k_i) 경계만**
    채택. 호출(=checkpoint별 prefix)은 독립 → 병렬. 출력 `{...,start_turn/exchange_number,...}` → start>0 가 경계.
  - `incremental` (LLM-incremental, online 0-look-ahead): 턴마다 "직전 open segment + 새 turn = 같은 topic? **Yes/No**". No→경계, 리셋.
    미팅 내부는 **순차**(prev_session 이 과거 결정에 의존), 미팅·모델 간 병렬.
- **입력 포맷 동일화**(전 메서드·전 모델 공통): turn 단위, `[Turn j]: (Speaker) text` (speaker tag 포함, timestamp 미제공).
  verbatim 은 LABEL=`Exchange`, baseline 은 LABEL=`Turn`. AMI 멀티파티 turn 을 SeCom "exchange" 슬롯에 매핑(도메인 적응).
- **⚠️ SeCom-incremental disclosure (논문 정직성 — reviewer 방어)**: SeCom 의 **정식/헤드라인 분절 방법은 offline `segment_with_exchange_number`**
  (대화 전체 1콜, GPT-4-0125). 논문은 iterative/streaming 분절을 *latency 때문에 일부러 회피*한다고 명시(LumberChunker 대비).
  `segment_incremental.md` 는 SeCom 레포에 **존재하나 본 방법이 아닌 secondary variant** — 그래서 우리 online baseline 으로 쓰되
  "SeCom 의 정식 방법 아님, 레포의 부차 variant" 라고 **명시 표기**. 메인 비교축 = **LLM-offline(상한) ↔
  LLM-Bsec(B곡선) ↔ Hi-OnTop / LLM-incremental(둘 다 0-lookahead)** (명명 §1B).
- 실행: `scripts/secom_sweep.sh [SUBSET_JSON]` (resumable, 결과태그에 subset 포함→37/139 비충돌) = **7모델** ×
  {seg: full,120,60,30,10 / incremental} × {baseline, verbatim}, **segment=B-mode(누적, `--context B`)**. 모델별 응답캐시
  `secom_cache_*.jsonl`(품질용 캐시 OK), 결과 `secom_sweep_results.tsv`. 기본 subset=37 dev; 139=`ami_full139.json` 지정.

## 1B. 버퍼드 LLM 컨텍스트 = 누적 prefix + 비교 메서드 명명 (사용자 확정 2026-06-12)

**문제 발견**: 기존 버퍼 모드는 각 B초 window 를 **맥락 없이 독립 분절** → 짧은 슬라이스를 잘게 쪼개 **경계 남발**
(seg-120 nitro pred **2656**/260, Score 0.214) + LLM 을 맥락-굶긴 **불공정 baseline**. prompt 로는 안 풀리는 구조 문제.

| 안 | 버퍼드 LLM 입력 | 과예측 | 비용 | 공정성 |
|---|---|---|---|---|
| A. 버퍼만(구) | B초 슬라이스만, window 독립 | 심함 | 쌈 | 약함(굶김) |
| **B. 누적 전체 ✅채택** | **0~현재까지 전부 + 새 B** | 적음 | **폭증**(컨텍스트↑) | **강함** |
| C. 요약+버퍼 | 직전 요약 + 새 B | 중간 | 유계 | 중간 |

- **채택 = B(누적)**. 공정성(LLM 에 최대 맥락) + 과예측 해소. 비용 폭증(누적 prefix → 입력 토큰 ~quadratic↑)은 오히려
  우리 **cost-latency 우위** 주장을 강화(SeCom 이 latency 때문에 피한 그 지점).

**비교 3-메서드 명명 (확정)**:
| 이름 | 정의 | look-ahead |
|---|---|---|
| **Hi-OnTop** | 임베딩 0-lag 온라인 분절기 (우리 방법) | 0 |
| **LLM-Bsec** | 버퍼드 LLM online baseline(B-mode) — B초 checkpoint 마다 **누적 prefix [0,k_i] 전체** 재분절, **새 구간 [k_prev,k_i) 경계만** 채택, 합집합=최종. **B초 buffered(0-lag 아님)** | B초 |
| **LLM-offline** | 전체 회의 1콜 full-context = offline upper-bound (`--buffer full`) | 무한 |

- **operationalization (구현됨 — `secom_llm_eval.py` run_segment `--context B` 기본; `seg_checkpoints`)**: LLM-Bsec 위 정의대로.
  decision-delay = 0~B(평균 B/2)+inference. 이름에 B 명시(예 **LLM-Bsec-10s**) — 0-lag 아닌 **B초 delayed online** 으로 정직 표기.
  콜 수=checkpoint 수, **콜당 입력=누적 prefix → 토큰·latency·cost ~quadratic↑**(cost 지표 정직 반영). 구 A(독립 window)는 `--context A` 참고용.
  offline 검증(mock): checkpoint 연속·[0,n] 커버·새구간 채택 정상. (codex gpt-5.5 자문 `codex_llm_buffered_design.md` 반영:
  main online=LLM-Bsec, offline upper-bound=LLM-offline. LLM-Bsec 은 0-lag 아닌 B초 delayed online 으로 명명, 비용 quadratic.)
- incremental(턴마다 0-lookahead Yes/No)은 별개 online LLM 포인트(**LLM-incremental**)로 유지 — 3-메서드 명명과 별도.
- **비-LLM 스트리밍 baseline 도 비교에 정식 포함**(§3h): **TextTiling / GraphSeg / GreedySeg / CSM**. 모두 online/streaming.
  현재 DTS latency 만 측정됨 → **AMI Score + 현 머신(CPU) latency 재측정**해 Pareto 에 함께 올림. 전체 비교 메서드 집합 =
  {Hi-OnTop, LLM-Bsec, LLM-offline, LLM-incremental, TextTiling, GraphSeg, GreedySeg, CSM}.

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

### 3b. LLM per-call latency (⚠️LEGACY: 236턴 고정, 동시부하 상한, 구모델 — `secom_latency_full.py` 가 정식 대체)
| 모델 | median(s) | min(s) |
|---|--:|--:|
| phi-4-mini(제외) | 1.03 | 0.96 |
| mistral-small-24b | 1.93 | 1.81 |
| qwen3.5-27b | 3.38 | 2.02 |
| phi-4(제외) | 4.25 | 2.93 |
| gemma-3-12b | 7.09 | 6.43 |
| gemma-3n-e4b | 11.50 | 9.28 |
→ Hi-OnTop(0.36s/미팅) 대비 최速 mistral도 ~5×. **정식 = 신 7모델·실워크로드 fresh·N=1 `secom_latency_full.py`로 재측정**(이 표 폐기).

### 3c. LLM full-context Score (⚠️LEGACY: 12미팅 편향 subset·구모델 — 신 7모델·139 로 §3g 가 대체)
| 모델 | Score | ±2F1 | pred/gold | 비고 |
|---|--:|--:|--:|---|
| qwen3.5-27b | 0.647 | 0.557 | 97/84 | 강함(figure_R 0.640 재현) |
| gpt-4o | 0.612 | 0.504 | 92/84 | |
| claude-haiku-4.5 | 0.609 | 0.549 | 132/84 | |
| gpt-4o-mini | 0.567 | 0.464 | 110/84 | |
| gemma-3n-e4b | 0.349 | 0.074 | 63/84 | under-seg |
| mistral-small-24b | 0.338 | 0.000 | **0**/84 | ⚠️ pred 0 = 출력/파싱 깨짐 |
| gemma-3-12b | 0.269 | 0.254 | 375/84 | 과분절 |
- 해석: full-context면 frontier LLM(qwen/gpt-4o/haiku ~0.61–0.65)이 Hi-OnTop(threshold 0.372) 능가. 작은 gemma 류는 mediocre.

### 3d. LLM buffer ±2F1 (⚠️LEGACY: figure_R, 12미팅, Qwen, Score 미측정 — §3g 가 대체)
- 10s 0.256(907콜) / 30s 0.237(585) / 60s 0.254(340) / 120s 0.372(182) / ∞ 0.526(12). → streaming 버퍼면 LLM 열화(정성).
- Score(Pk/WD) 미측정·구프롬프트 → **신 7모델·139·SeCom 프롬프트 §3g 가 정식**. 이 표는 정성 경향 참고만.

### 3g. SeCom/baseline 프롬프트 풀 스윕 (⏸ 중단·모델셋 교체로 재실행 대기 2026-06-12) — §1A 참조
- 조건: `scripts/secom_sweep.sh [SUBSET]` → **7모델** × {segment full/120/60/30/10 + incremental} × {baseline, verbatim} =
  **84 run/subset**. segment=B-mode(누적, offset 미사용), temp 0, no-thinking. 결과 `secom_sweep_results.tsv`, 로그 `secom_sweep_logs/`.
  resumable + 모델별 응답캐시 → 중단·재개 안전. 싼 segment 먼저, incremental 마지막.
- **상태**: 구 6모델·37 로 일부 진행(baseline seg-full 6 + seg-120 1) 후 **모델셋 교체로 중단**. 구 결과(gpt-4o 포함)는
  폐기, 겹치는 모델(qwen/haiku/mistral/gemma-3-12b) **응답캐시는 보존·재사용**. 새 7모델·139 로 재실행 예정.
- **확인된 findings(구 37 dev, baseline, 폐기 전 관측)**: ⓐ **mistral-small pred 0 — SeCom baseline 프롬프트로도 여전**
  (err 0인데 분절 0개) → 프롬프트가 아니라 **모델별 출력포맷 깨짐** 확정, failure-rate 로 보고. ⓑ segment 모드 **작은
  버퍼=경계 남발**: seg-120 nitro pred 2656/260(10× 과예측)·Score 0.214 → 독립 window 분절의 streaming 열화 전형.
  ⓒ seg-full(offline) baseline: qwen 0.603 / haiku 0.543 / gpt-4o:nitro 0.534 / gpt-4o-mini 0.418 / gemma-3-12b 0.419(과분절).
- 워크로드: incremental run 당 ≈17,397콜(37)·≈62,539콜(139), 미팅 순차. segment 는 window 병렬로 빠름.

### 3e. 계층-level 분석 (Hi-OnTop threshold, 139) ✅
- top-level 경계 938 / all-level 1831 / 계층 미팅 125/139.
- ±2F1: vs top 0.131 / **vs all 0.206**(더 높음 → 세밀 sub-topic도 잡음). 예측경계: top-hit 8%/sub-only 8%/false 84%.

### 3f. Hi-OnTop deploy Score (139, 참조)
- **threshold(0-lag) 0.372 = 본 비교 채택** / δ_eff 0.203 / oracle 천장 ±2F1 0.671.
- (commit-refine(L8) 0.401 = **폐기**, 버퍼/lag 원치 않음 — 참조 수치만.)

### 3h. 비-LLM 스트리밍 baseline — 구현+per-turn latency 기측정 (📎 논문 latency·baseline 표) ✅구현/⚠️재측정
DTS 벤치(tiage/dialseg711/superseg, 500턴/벤치, CPU, 첫발화 제외, cold-start 분리)에서 **이미 구현·측정**.
모두 online/streaming 모드. **per-turn = Pre.(encode/feature) + Seg.(boundary 결정)**, cross-bench mean(ms):

| 메서드 | Pre. mean / p50 | Seg. mean / p50 | **Total mean** | cold-start | 인코더/특징 | 측정 script |
|---|---|---|--:|--:|---|---|
| **TextTiling** (lexical) | 0.020 / 0.016 | 0.013 / 0.001 | **0.033 ms** | 없음 | BoW, no neural | `measure_texttiling_latency.py` |
| **GraphSeg** (window) | 4.82 / 2.13 | 1.31 / ~0 | **6.13 ms** | 103s(GloVe) | GloVe+graph | `measure_graphseg_latency.py` |
| **Hi-OnTop**(int8, 구머신) | 11.70 / 10.01 | 0.246 / 0.202 | **11.95 ms** | 34s(model) | MiniLM-int8 | `2026-05-24_hiontop_latency_int8` |
| **GreedySeg** (delay2) | 232.2 / 0.00 | 2.61 / 0.013 | **234.8 ms** | 25s(BERT) | BERT-base CLS | `measure_greedyseg_latency.py` |
| **CSM** (delay2) | 283.2 / 252.2 | 0.037 / 0.032 | **283.3 ms** | 30s(BERT) | BERT+CoherenceNet | `measure_csm_latency.py` |

- 해석: lexical(TextTiling) 0.03ms <<< Hi-OnTop 12ms << neural-pair baseline(GreedySeg/CSM) 235–283ms.
  **neural cross-encoder류는 per-turn 수백 ms = streaming 부적합**, Hi-OnTop 는 single-utt 임베딩이라 한 자릿~십 ms.
- ⚠️ **fair-table 재측정 필요**: 위는 **DTS 벤치·구 머신**(같은 구 머신 내부 상대비교만 유효). 본 비교의 §3a Hi-OnTop
  live(현 머신, total p50 **1.56ms**)와 직접 비교 불가 → baseline 들을 **현 머신·AMI 윈도우에서 동일 정의로 재측정**해야 함.
  **품질(Score)은 AMI 미측정**(위는 latency 전용) → AMI subset 에서 Score 도 측정해야 baseline 표 완성. → §3z D 갱신.

### 3i. 인코더 선택 근거 — mpnet/MiniLM/int8 기측정 (📎 appendix, 인코더 ablation) ✅
DTS 벤치(tiage/dialseg711/superseg) Score(`run_encoder_comparison.py`, `2026-05-23_encoder_comparison`, 구 Hi-OnTop HP m=2·ρ=0.7·a=0.5):

| 인코더 | mean-3 Score (p80 / best-δ) | per-utt latency mean(ms) PyTorch CPU |
|---|---|--:|
| mpnet | 0.493 / 0.515 | **600** (p50 365) |
| MiniLM(fp32) | 0.482 / 0.506 | **111** (p50 67) |
| **MiniLM-int8** | 0.488 / 0.511 | int8 realtime ≈ 12 (§3h) |

- 결론: **int8 양자화는 품질 손실 거의 0**(fp32 0.482/0.506 ≈ int8 0.488/0.511, mpnet 보다 ~0.004↓에 불과)이면서
  latency 는 mpnet 대비 **~50×↓**. → **MiniLM-int8 채택 정당화**(품질-latency 최적). δ* 는 인코더별 재보정 필요(한계).
- ⚠️ 구 Hi-OnTop HP(de-neut 이전) 기준 — de-neut-cr 에서 인코더 재비교는 미실시(품질 결론은 인코더 상대순위로 전이 가정).

## 3z. 확정 실험 사양 (codex 종합감사 2026-06-12 반영 — `codex_audit.md`)

### A. positioning / 통계 규율
- 주장 = **"동일 online decision budget 에서 Pareto frontier 어느 지점이 우월한가"** (단순 "임베딩이 LLM 보다 쌈" ❌).
  보고는 single best-table 대신 **Pareto 곡선**: Score vs decision-delay / vs cost-per-hour / vs p95 latency.
- **Score = 전체 84-그리드 139 전수(ⓑ 확정)**: selection 단계가 없으므로 selection-bias 문제 자체가 없음(전 조건을
  139에 그대로 보고). 별도 dev/test 분리 불필요. (구 37-dev subset 은 **quality 에 미사용**; latency-18 subset 만
  139 분포 재현용으로 파생.) latency 는 Score 결과 나온 뒤 **우수 config 만** 18-subset 에서 측정(★핵심).
- paired bootstrap meeting-level CI. 버퍼 {10,30,60,120,full} 전부 139.

### B. 공정 비교 조건 (must)
- **latency 는 캐시·전처리 금지, live per-turn 임베딩.** Hi-OnTop = 발화 도착→그 자리서 encode(batch=1, no cache)
  →segment. 임베딩 시간을 online 비용에 포함(전처리로 숨기지 않음). (품질/Score 측정은 캐시 OK — 값 동일.)
- **decision time 정보표**: 메서드별 "판정 시점에 가용한 정보" 명시 — Hi-OnTop(threshold, 과거만, **0 look-ahead**) /
  LLM-Bsec(B초 buffered, **emit 은 buffer close 이후**) / LLM-incremental(0 look-ahead) /
  LLM-offline(offline oracle, online 표에서 분리).
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
  - **LLM latency = 실제 워크로드 fresh·N=1/모델(동시부하0, 순차)·캐시금지** + **모드별 정확 출력예산**(incremental
    16tok / segment 1024tok). home-made WIN_PROMPT ❌ → SeCom 프롬프트 그대로. `secom_latency_full.py`(RESUME).
  - 📎 **latency 전용 미팅 subset (N=18, 기준 도출 — 논문 서술용)**: latency 는 content-independent(콜시간≈f(window크기,
    출력크기,모델)) → 139 전수(모델당 ~124k콜) 대신, **크기·density 극단 포함 + 모든 버퍼 window-크기 분포를 전체 139 와
    KS 로 재현(D≤0.12, p≥0.94)하는 최소 18미팅**(`select_latency_subset.py --subset ami_full139.json`→`ami_latency_subset.json`).
    콜 13%, 분포·극단(n_turns[76,1194]·density[0.0022,0.054] 정확 일치) 보존. **입력버킷 샘플 아님**(실제 미팅·실제 콜
    전수, 미팅 수만↓). 품질 Score 는 139 — Pareto 의 latency 축만 18미팅(메서드·크기 속성이라 미팅셋 달라도 무방).
    N=18 은 사전지정 아닌 기준의 결과(KS 표 appendix 근거).
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
- ✅ **이미 구현+latency 측정** (§3h): TextTiling / GraphSeg / GreedySeg(BERT) / CSM(CoherenceNet) streaming 구현 +
  per-turn latency 프로파일 존재. **남은 일 = (1) AMI subset 에서 Score 측정, (2) 현 머신 동일정의 latency 재측정**(fair-table).
- 인코더 ablation(mpnet/MiniLM/int8) 도 ✅측정(§3i). random/density-matched, fixed-threshold, oracle-count 는 미구현.

### E. subset 추가 층화축 (codex) — 다음 재선정 시 반영
현재(길이×density) + **화자 수 / audio 시간 / turn 길이 mean·var / n_topic_top·all·계층깊이 / scenario vs IB(비시나리오)**.

### F. AMI gold 처리
"정답 복원" ❌ → **"AMI annotation convention 에 대한 alignment"** 로 표현 낮춤. tolerance curve + top/all-level +
near-miss + meeting bootstrap CI + (가능시) IAA/human upper bound 인용 + disagreement 큰 회의 error analysis.

## 4. 미해결 / 다음 (TODO)
- ✅ Hi-OnTop live-streaming latency(cold/warm·p99·RTF) 측정 §3a. ✅ buffer eval `--subset/--offsets` 배선·검증.
- ✅ **비용절감(codex `codex_cost.md`)**: concurrency 가 압도적 1순위. buffer eval 에 ThreadPool(`--workers`)+retry/backoff
  +prompt 캐시(`llm_cache_*.jsonl`, 재실행 skip)+dedup 구현. 검증: workers 8 → 54콜 18s, **0 errors**(Crts 동시호출 허용).
- ✅ **[결정] Score@139 = ⓑ 전체그리드 139 전수** (★핵심박스). latency = 그 중 우수 config 만.
- ⏸ **품질 sweep 실행 대기**(`secom_sweep.sh ami_full139.json`): **신 7모델 × 84조합 × 139**. 구 6모델·37 결과 폐기
  (겹치는 모델 캐시 보존). home-made Qwen sweep 폐기. segment=B-mode(누적).
- ⏸ **LLM latency 실행 대기**(`secom_latency_full.py`): Score 나온 뒤 **우수 config + 우승 프롬프트**만, 실제 워크로드
  fresh·N=1/모델 순차·캐시금지, **18-미팅 subset**. **품질 sweep 종료 후** 단독 실행(동시부하 오염 방지). 구 `llm_latency_full.py` 폐기.
1. **mistral-small pred 0 확정**(§3g): SeCom 프롬프트로도 깨짐=모델별 출력포맷 문제 → failure-rate 보고 + (가능시) JSON strict.
3. **decision-delay**(buffer close emit, 139 계산) + concurrency throughput{1,4,8,16} + peak RSS + Hi-OnTop latency 도 18-subset/139 로 정렬.
4. **139 비교 Pareto** 구성: Score(139) × latency(18-subset, 우수 config). 메서드 = Hi-OnTop / LLM-Bsec / LLM-offline /
   LLM-incremental + 비-LLM(TextTiling/GraphSeg/GreedySeg/CSM). (dev/test 분리 없음 — ⓑ.)
5. **미측정 지표**: tolerance curve ±0~5, cost normalization, paired bootstrap CI, memory 2-표.
6. **baseline 확장**: §3h 구현분 AMI Score + 현머신 latency 재측정 + fixed-thr/random + **local quantized LLM**(공정성 핵심).
7. 12-ES pilot 폐기(편향). **Score=139 전수 보고**(37 subset quality 미사용, latency-18 만 139 파생).

## 5. 산출물
- **LLM 분절 (현행)**: `secom_llm_eval.py`(SeCom incremental/segment, `--prompt baseline|verbatim`, 캐시·retry·concurrency),
  `secom_sweep.sh`(84-run/subset resumable 드라이버, `--context B`), `secom_prompts/{baseline_segment,baseline_incremental,segment_exchange,segment_incremental}.md`.
  → 결과 `outputs/runs/_misc/secom_sweep_results.tsv`, 응답캐시 `secom_cache_*.jsonl`, 로그 `secom_sweep_logs/`.
- **LLM latency (현행)**: `secom_latency_full.py`(실제 워크로드 fresh·N=1·모드별 정확 출력예산·RESUME
  `secom_latency_full.jsonl`, `--report`), `select_latency_subset.py`(**N=18** 139기준 KS 도출→`ami_latency_subset.json`),
  `ami_full139.json`(139 전체 id). 품질 sweep 종료 후 단독 실행(겹치면 동시부하 오염). ⚠️ 구 `llm_latency_full.py` **폐기**.
- **비-LLM baseline (구현+latency, §3h)**: `measure_{texttiling,graphseg,greedyseg,csm}_latency.py`,
  `run_encoder_comparison.py`(인코더 ablation §3i), `run_graphseg_ami.py`, `run_texttiling_prefix.py`,
  `train_csm_hf.py`. 측정결과 = `outputs/experiments/2026-05-2{3,4}_*_latency_split/`, `_encoder_comparison/`.
- 스크립트(legacy/보조): `ami_llm_buffer_eval.py`(Score·CF헤더 배선), `ami_llm_segment_eval.py`(Score 배선),
  `select_ami_subset.py`(N=37 2D 층화), `plot_score_latency.py`(figure_S, LLM 자리 비움),
  `ami_deploy_failure_anatomy.py`, `ami_commit_refine_deploy.py`, `latency_streaming.py`(Hi-OnTop live).
- **method 스펙**: `context/methodology/hi-ontop-cr.md`(canonical). **신호/oracle**: `handoff/HANDOFF_0609_deploy-oracle-gap.md`.
- codex 자문: `outputs/runs/_misc/codex_expdesign.md`(지표 1차), `codex_audit.md`(종합감사, 반영완료 §3z).
- subset: `outputs/runs/_misc/ami_subset.json`. figure: `outputs/figures/figure_S_score_latency.{pdf,png}`.
- 인프라 caveat: Crts = Cloudflare Access 게이트웨이 뒤 → `.env` 에 `CF_ACCESS_CLIENT_ID/SECRET` 필요(인가됨).
  모델 id 는 `openrouter/<provider>/<model>` (models.list 미등록 slug 도 라우팅됨).
