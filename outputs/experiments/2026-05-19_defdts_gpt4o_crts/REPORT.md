# Def-DTS @ openai/gpt-4o (Crts) — tiage / dialseg711 / superseg

## 1. Setup

- **Method**: Def-DTS (ElPlaguister/Def-DTS, ACL 2025) — LLM deductive-reasoning prompting, DTS as utterance-level intent classification. No checkpoint / no training: 1 LLM call/dialogue.
- **LLM**: `openai/gpt-4o` via **Crts** proxy (`api.ssunlp.co.kr/v1`, OpenAI-compatible). Single project endpoint (not OpenAI-direct / not OpenRouter). temperature=0.0.
- **Data**: bundled `benchmarks/Def-DTS/data/DTS_session_datasets/<ds>_test.jsonl`. Prompts: `prompts/defdts_{tiage,711,superseg}.prompt` (full Def-DTS template).
- **Metric**: repo-verbatim `segeval` Pk/WD + sklearn F1 (`autoseg.compute_metrics`), kept unmodified for literature comparability. tiage uses change_speaker=True (main.ipynb).
- **Code**: `benchmarks/Def-DTS` untouched (read-only); behaviour fixes are runtime monkeypatches in `scripts/run_defdts_crts.py`.

**실행 상태**: tiage·dialseg711 = 전체 완료·저장. **superseg = 풀 점수 포기(사용자 결정 2026-05-19)** — gpt-4o×1322 대화 비용 과다 + Crts USD quota 가 작은 증분($91.9→$106.9)이라 풀 완성에 ~$45-50·재시도 3-4회 필요. resume 로 **425/1322 부분 저장**되어 있으나 점수 산출 안 함(편향 표본). tiage·dialseg711 수치는 eval-only(추가 LLM 콜 0, 저장된 result json 재채점)로 산출.

## 2. Results

| dataset | n (scored/total) | Pk ↓ | WD ↓ | F1 ↑ |
|---|---:|---:|---:|---:|
| tiage | 100/100 (drop 0) | 0.2629 | 0.2938 | 0.6600 |
| dialseg711 | 711/711 (drop 0) | 0.0186 | 0.0360 | 0.9578 |
| superseg | — (풀 점수 포기; 425/1322 부분 저장, 미채점) | — | — | — |

## 3. Latency

실제 full run = **workers=100** 동시 실행. 따라서 아래 per-call 통계는 **"100-way 동시 부하 하의 대화별 콜 latency"** — isolated(단독) latency 가 아님. 동시 100요청이 Crts 를 함께 때리므로 server-side 큐잉/경합이 각 콜 latency 에 포함됨(그래서 dialseg711 mean 17.9s 대비 p95 30.2s 로 꼬리 inflate). 순수 latency 가 필요하면 workers=1 소표본 측정 별도 필요.

p50/max/sum 은 crash 한 run 이 persist 하지 않아 복구 불가(— 표기). mean/p95/wall 은 run 로그에서 파싱. end-to-end wall = 해당 데이터셋 전체 벽시계(동시성으로 sum≪이 아니라 wall≪sum).

| dataset | n_calls | mean (s) | p50 (s) | p95 (s) | max (s) | sum (s) | end-to-end wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100 | 21.33 | — | 24.94 | — | — | 35 |
| dialseg711 | 711 | 17.91 | — | 30.18 | — | — | 152 |

해석: workers=100 으로 711 대화를 152s 만에 처리(순차 환산 ≈ 711×17.9 ≈ 3.5h 대비 ~83× 단축). 대화당 콜 ~18–21s 는 Def-DTS 의 긴 deductive reasoning 출력(수천 토큰) 때문.

## 4. 해석

- **dialseg711 F1=0.958 / Pk=0.019**: Def-DTS 논문 보고치(gpt-4o 로 dialseg711 near-SOTA, Pk≈0.02 대) 와 정합. dialseg711 은 topic shift 가 명확·인위적(문서 결합형)이라 LLM reasoning 이 거의 완벽히 잡음.
- **tiage F1=0.66 / Pk=0.263**: 같은 방법·LLM 인데 dialseg711 보다 훨씬 낮음. tiage 는 open-domain chit-chat 으로 topic 경계가 모호 → 논문에서도 가장 어려운 셋. 절대 난도 차이지 구현 문제 아님(drop=0, 포맷·파싱 정상).
- **drop=0 (양 셋)**: gpt-4o 가 Def-DTS XML 출력 포맷을 100% 준수 → compute_metrics 의 sum-mismatch 제외가 발동 안 함. 즉 이 두 수치는 표본 누락 없는 정직한 값.
- Hi-OnTop 과의 비교는 superseg 결과까지 나와야 의미. 현재는 tiage/dialseg711 에서 Def-DTS@gpt-4o 의 강한 baseline 만 확인.

## 5. 한계 / 검증 미해결

- **superseg 풀 점수 부재(의도적 포기)**: gpt-4o×1322 비용 과다 + Crts quota 소액 증분으로 풀 완성 비현실적 → 사용자 결정으로 중단. 425/1322 부분 result 는 디스크 보존(편향 표본이라 미채점). 3-셋 점수 비교는 tiage·dialseg711 한정.
- **LLM 조건 불일치**: Def-DTS=gpt-4o, Hi-OnTop=Qwen3.5-9B. 같은 Crts 엔드포인트지만 LLM 이 달라 "Def-DTS@gpt-4o(논문 설정)" 대 "Hi-OnTop@Qwen" 비교 — 동일-LLM 비교 아님. 동일 조건 비교를 원하면 Def-DTS 를 Qwen 으로도 1회 별도 필요.
- **latency 가 isolated 아님**: §3 — workers=100 경합 포함값. 순수 단일 latency 미측정. p50/max/sum 미persist 로 복구 불가.
- **compute_metrics 의 관용성**: `sum(pred)!=sum(label)` 대화는 점수에서 제외(drop)되는 구조. 이번엔 drop=0 이라 무영향이나, 다른 LLM/셋에선 반드시 drop 률 동반 보고.
- **비용/usage**: 스모크에서 usage 반환 확인(tiage 3콜 ≈ $0.125 → gpt-4o full 3-셋 추정이 quota $91.9 를 초과시킨 원인). 정확 누적 비용은 Crts 측 기준.
- **재현성**: temperature=0 이나 gpt-4o 비결정성 잔존 가능, API seed 미지원, 1-run.
- 데이터: 번들 test jsonl 줄 수 tiage 99 / dialseg711 710 / superseg 1321 이나 HF datasets 로딩 시 100 / 711 / 1322 (trailing-newline). 채점 대상 = 저장된 100 / 711.
