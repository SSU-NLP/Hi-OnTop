# HANDOFF_03 — v2 요약-carryover streaming LLM 분절 프롬프트 개선

**지정일**: 2026-06-13 · **상태**: 문제 정의, 개선 시도 시작 · **부모 컨텍스트**: `HANDOFF_02_realtime-llm-comparison_0612.md` §1B/§3j

## 0. 문제 정의

### 배경
- HANDOFF_02 에서 LLM 분절 baseline 프롬프트를 **v1**(B-mode 누적, `baseline_segment_v1.md`)로 확정. **v2**(요약-carryover streaming,
  `baseline_segment_v2.md`)는 **채택 보류 → 개선 후보**(codex 권고: 별도 low-latency variant + 개선 후보).
- v2 동기: 긴 미팅(400~1194 turn)에서 B-mode·full-context 가 **입력 길이 때문에 후반부 붕괴**(후반부 gold ~89% 미검출).
  v2 는 입력을 짧게(직전 segment 요약 + 새 청크)로 유지해 이를 완화 + latency·cost 유계.
- **v2 측정 결과**(HANDOFF_02 §3j, qwen-27b·18-subset·N=1 직렬, err0):
  - **장점**: F1 경쟁력(60 에서 v2 0.328 > v1 0.288 > verbatim 0.274) + **latency 2~3× 빠름**(콜당 mean ~5s vs B-mode ~12s,
    in_max 3× 작음 20~24k vs 69k char) + 입력 bounded.
  - **단점**: **Pk(0.42~0.43)·WD(0.51~0.53)가 v1/verbatim(Pk ~0.29~0.32, WD ~0.34~0.39)보다 크게 나빠 Score 최저**(120 0.426 / 60 0.430 vs v1 0.500 / 0.483).

### 근본 원인 (코드 인스펙션 확정, 2026-06-13)
- v2 carryover 구현(`scripts/measure_streaming_table.py::csec_meeting`)이 **naive append**: `prior.extend(새 청크 segment 요약들)`.
  → 한 화제가 여러 청크에 걸쳐 이어져도(`continues_previous=True`) 이어진 부분을 **새 요약 entry 로 또 append** →
  **같은 화제 요약이 청크마다 중복·파편화**, merge/dedup 없음. `prior` 가 부풀고 모델이 "다 다른 화제?" 혼동 → 경계 개수·간격 calibration 깨짐 → **Pk/WD 악화**.
- 추가로 요약은 turn-level boundary 단서를 잃는 lossy 압축 → 경계 위치 정밀도↓.
- (codex 진단 "요약 누적 topic drift + granularity mismatch"와 일치.)
- ※ **재귀적 재요약(t 가 t-1·t-2 요약을 또 요약)은 아님** — 옛 요약은 verbatim 유지. 문제는 **중복 누적**.
- ⚠️ **이 진단은 §2 시도1(v3)로 부분 반박됨**: 요약을 전부 폐기한 raw 윈도우(v3)도 Pk/WD 가 v2 와 동급으로 나빴음 →
  주범은 "요약 단서소실"이 아니라 **입력을 bounded 윈도우로 자르는 것 자체**(전역 calibration 불가). §2 참조.

### 목표
- v2 carryover 를 개선해 **Pk/WD(→Score)를 v1/verbatim 수준으로 끌어올리되, latency·bounded-입력 우위는 유지** → low-latency streaming baseline 으로 viable.

### 성공기준
- **18-subset(qwen-27b, 가능시 +nitro) 120/60 에서 v2-개선 Score ≥ v1**(=확정 baseline) — 최소 verbatim 동등 이상 —
  AND **latency 우위 유지**(콜당 ~수s, 입력 bounded). 즉 HANDOFF_02 §3j ②(>v1)·③(≥verbatim)를 Score 기준 충족.

## 1. 개선 방향 (codex 자문 `outputs/runs/_misc/codex_baseline_decision.md` + 코드 진단)
1. **continuation merge**: 이어지는 화제는 직전 요약을 **update/merge**(새 entry append ❌). 또는 active topic 1개 유지.
2. **요약 + 최근 raw turns(30~80)** 병행 carry → boundary 근처 evidence 보존(WD/Pk 직접 개선 기대).
3. **segment 개수/rate prior** + "약한 shift 면 유지/merge" 규칙 프롬프트 명시(과/과소분절 보정).
4. **overlap window + reconcile** (latency 약간↑).
5. "사람용 요약" 대신 **segmentation state**(active topic·start/end·경계사유·open items) carry.
6. 후처리 length calibration(merge/split).

## 1.5 v1 붕괴선 — 윈도우 크기 근거 (2026-06-13, 데이터 확정)

**목적**: v3 슬라이딩-윈도우 크기를 데이터로 정당화 — v1(B-mode 누적)이 **누적 입력 몇 분째부터 깨지는지** 측정.
**측정**: v1(`baseline_segment_v1.md`, B-mode), **qwen-27b, AMI 18-subset(17/18 — TS3007a는 timeout 누락), buffer 60s, N=1 직렬(공정 latency)**.
각 체크포인트마다 (prefix turn수·시간, 그 콜의 latency, 새 구간 예측경계, 새 구간 gold) 저장. 524 체크포인트.

**📂 저장 위치 (세부, 재현용 — 결과가 정확히 어디에 있는가)**:
- **디렉토리**: `outputs/experiments/2026-06-13_v1-breakpoint-60_ami18/`
  - **`checkpoint_records.json`** — 524 체크포인트 raw(list of dict). 각 dict 필드:
    `mid`(미팅 id) · `kp`,`ki`(새 구간 `[kp,ki)`) · `prefix_n`(=ki, prefix turn 수) · `prefix_time_min`(prefix 끝 turn 시각, 분) ·
    `latency_s`(그 콜의 N=1 직렬 wall-clock 초) · `preds`(새 구간 예측 경계, global turn idx 리스트) · `region_gold`(새 구간 gold 경계).
    **17/18 미팅**(TS3007a 누락). 증분 저장(매 콜 후 덮어쓰기)이라 중단돼도 보존.
  - **`breakpoint_bins.json`** — 2분 bin 집계 `{"bin시작분": [gold수, ±2잡힘수, latency합, 콜수]}` (위 recall+latency 표의 원천).
  - **`run.log`** — **0바이트**(timeout 9000s 가 마지막 미팅 직전 자르며 자동 분석표 print 미실행 → bins 는 `checkpoint_records.json` 으로 **재계산**함).
- **스크립트**: `scripts/analyze_breakpoint_60.py` (⚠ 재실행 시 디렉토리 먼저 `mkdir -p` 필요 — shell 리다이렉트가 dir 선생성 요구).
- **입력**: subset `outputs/runs/_misc/ami_latency_subset.json`(18미팅) · gold `data/ami/topic/<mid>.json` 의 `bnd_top` ·
  LLM = qwen-27b **fresh 콜(N=1 latency 위해 직접 `client.create`, secom_cache 에 미저장)**. 인코더 무관(LLM-only).

**결과 — prefix 누적시간별 recall(score) + per-call latency** (gold 경계를 그 시점 prefix로 판단):

| prefix 시간 | recall(±2 잡힘) | mean latency |
|---|--:|--:|
| 0~4분 | **~50%** | **5~9s** |
| 4~10분 | 20~40% | 9~12s |
| 10~20분 | 14~40% | 17~18s |
| **20분~** | **0%** (붕괴) | 19~30s |

- **latency = prefix 단조 증가** (5s→9s→18s→30s): 입력 길수록 느림.
- **recall = 작은 prefix 최고(~50% @0-4분), 4~20분 mediocre, ~20분부터 0% 붕괴.**
- **★ sweet spot = 앞 ~4분**: recall 최고 + latency 최저. 그 뒤 latency↑·recall↓→20분 붕괴.
- **결론(v3 정당화)**: v1은 누적 ~20분에서 붕괴 → **각 LLM 콜 입력을 ~20분 한참 아래로 유지**해야 함. v3 윈도우(≤3분 맥락+청크 ≈ ~5분 입력)는 sweet spot 안 → score·latency 둘 다 최적. **"왜 3분"의 데이터 근거 = ~20분 상한 + 앞 ~4분 sweet spot.**
- 한계: 단일 모델(qwen)·17/18 미팅·recall bin noisy. 추세(단조 latency↑, ~20분 붕괴)는 견고.

## 1.6 v3 설계 확정 (2026-06-13, 사용자) — raw 슬라이딩 ≤4분 윈도우

v2(naive 요약-carryover)의 파편화·단서소실(§0 근본원인) 회피 → **요약 폐기, raw 슬라이딩 윈도우** 채택.

- **입력 = (현재 청크 직전 누적 ≤4분(240초)의 raw turns)  +  (현재 청크 = "현재 발화" raw turns)**.
  - 직전 4분 turns = context(이미 분절됨, 재분절 안 함), 현재 청크만 새로 분절.
  - **요약·merge·segmentation-state 누적 전부 없음** (v2 우려한 "이어지는 화제 merge로 경계 억제" 구조적 회피).
- **윈도우 4분 근거 = §1.5 sweet spot**: 누적 입력 **앞 ~4분 = recall 최고(~50%) + latency 최저(5~9s)**, ~20분 붕괴 상한 한참 아래.
  → 각 콜 입력을 4분 맥락 + 청크(≈5분)로 bounded → score·latency 둘 다 최적.
- **라벨** = 글로벌 turn 인덱스(`fmt_global`). 모델은 4분 맥락 보고 현재 청크의 새 경계만 출력(continues_previous 자체 판단).
- **partition 규율 유지**: v1(SeCom)의 no-missing/no-overlap/accurate-counting + genuine-shift-only (과분절 방지). carryover 부분만 raw window 로 교체.
- **프롬프트 파일 = `scripts/secom_prompts/baseline_segment_v3.md`** (v2 안 덮어씀 — 별도 신설). 러너 = `measure_streaming_table.py` 의 carryover 로직을 4분 raw window 로 교체.
- **측정**: 18-subset qwen 120/60, **N=1 + latency + per-meeting 예측 저장**([[feedback_handoff03_n1_latency_predictions]] 원칙) → **v1 / v2(naive) / v3 비교** (F1/Pk/WD/Score + latency).

## 2. 시도 ledger
*(논문/최종 비교에 들어갈 개선 시도만 — 조건 전부 + 결과 기록. dead-end·탐색은 제외.)*

### 시도 1 — v3 (raw 슬라이딩 ≤4분 윈도우) : **실패 (Score ≥ v1 미달, v2 와 동급)**

**설정 (전부, 재현용)**:
- 프롬프트 `scripts/secom_prompts/baseline_segment_v3.md` (요약 없음, 직전 ≤240초 raw 맥락 + 현재 청크, partition 규율 유지 = §1.6).
- 러너 `scripts/measure_v3.py` (자립형, `fmt_global` 글로벌 인덱스). 데이터 AMI 18-subset(`ami_latency_subset.json`, 8134턴),
  모델 qwen3.5-27b(Crts), 버퍼 120/60, **N=1 직렬(latency 공정) + per-call latency + raw 응답·체크포인트 전량 저장**.
- 채점 `E.metrics` = ±2-tol F1 + nltk Pk/WD, **PRED_SHIFT=0** (AMI gold=시작-turn 규약정렬, [[feedback_ami_gold_start_turn_convention]]).
- 콜수 120=282 / 60=556 (합 838), err0.

**결과** (v1/v2/verbatim = HANDOFF_02 §3j 동일 채점):

| buffer | 방법 | ±2F1 | Pk↓ | WD↓ | **Score** | lat p50/p95(s) | in_max(char) |
|---|---|--:|--:|--:|--:|--:|--:|
| 120 | **v1**(누적) | 0.356 | **0.321** | **0.390** | **0.500** | 10.2/27.7 | 69,021 |
| 120 | verbatim | 0.310 | 0.306 | 0.362 | 0.488 | 9.7/23.9 | 73,954 |
| 120 | v2(요약) | 0.329 | 0.429 | 0.528 | 0.426 | 4.3/10.8 | 20,682 |
| 120 | **v3**(raw4분) | 0.328 | 0.431 | 0.533 | **0.423** | **3.2/9.2** | **14,241** |
| 60 | **v1** | 0.288 | **0.292** | **0.352** | **0.483** | 9.8/28.3 | 69,021 |
| 60 | verbatim | 0.274 | 0.293 | 0.337 | 0.480 | 11.9/42.7 | 73,954 |
| 60 | v2 | 0.328 | 0.424 | 0.511 | 0.430 | 3.8/10.1 | 24,357 |
| 60 | **v3** | 0.305 | 0.424 | 0.509 | **0.419** | **3.0/10.0** | **12,561** |

**판정**: **v3 ≈ v2 < v1** — §1.6 성공기준(120/60 Score ≥ v1) **미달**(0.423<0.500, 0.419<0.483). verbatim 에도 못 미침.

**근본 원인 (진단 수정)**: v3 의 Pk/WD(~0.43 / ~0.51~0.53)가 **v2 와 사실상 동일하게 나쁨** → §0 의 "요약 단서소실/중복누적"이
주범이 아니었음. 요약을 전부 폐기하고 raw 윈도우로 바꿔도 Pk/WD 안 고쳐짐 ⇒ **진짜 원인 = 입력을 bounded 로 자르는 것 자체**
(윈도우면 요약이든 raw든 전역 경계 개수·간격 calibration 불가). v1(full 누적)만 전체 맥락으로 calibration 가능.

**남는 의미**: v3 강점 = latency p50 3.0~3.2s(v1 1/3) + 입력 12~14k char(v1 1/5, v2 보다도 작음). **품질 baseline = v1 유지**,
v2/v3 는 "latency 가 hard constraint 일 때만 쓰는 별도 low-latency variant" 로만 의미. → low-latency streaming 의 본질적
trade-off(입력 bounded → latency↓·cost↓ ↔ Pk/WD↓·Score↓) 실증.

**📂 저장 위치 (재현)**: `outputs/experiments/2026-06-13_v3_carryover_ami18/`
- `v3_summary.json`(버퍼별 집계 + `lat_all` 전량) · `v3_{120,60}_per_meeting.json`(미팅별 preds/gold/F1/Pk/WD/Score/lat/inch) ·
  `v3_{120,60}_raw.jsonl`(체크포인트별 `mid·kp·ki·in_len·latency_s·out·segs` 838줄 — 재채점·붕괴선 재분석 시 API 재실행 0) ·
  `v3_run_config.json`(모델·윈도우·subset·scorer) · `run.log`.

**후속 방향 (시도 2 후보, 미실행)**: bounded 윈도우의 Pk/WD 손상은 **후처리 전역 calibration**(merge/split, 경계 rate prior)으로만
완화 가능할 것 — 입력은 짧게 두되 출력 경계를 전역 통계로 보정. 단 v1 대비 이득 불확실 → 우선순위 낮음.

## 3. 측정 / 산출물 (재현)
- **프롬프트**: `scripts/secom_prompts/baseline_segment_v2.md`(현재 naive) → 개선판(버전 표기).
- **러너**: `scripts/measure_streaming_table.py`(`csec_meeting`/`parse_csec`/`fmt_global`) — carryover 로직 수정 대상.
  - 재현 주의: 청크 글로벌 인덱스 필수(`fmt_global`, `E.fmt_exchanges`는 로컬 0-기반이라 F1 깨짐 — HANDOFF_02 §3j).
- **데이터**: AMI 18-subset(`ami_latency_subset.json`, 18미팅 8134턴), qwen-27b(+nitro), 버퍼 120/60.
- **비교축**: 현 v2(naive, baseline) vs v2-개선 vs v1 — F1/Pk/WD/Score + per-call latency.
- **비용 참고**: v2(120+60) N=1 직렬 = qwen 18미팅 ~68분(120 24분 + 60 44분, 콜당 ~5s). 동시면 더 짧음(미팅 내부는 순차).
- **참조**: HANDOFF_02 §3j(결과·판정), `codex_baseline_decision.md`(개선방향).
