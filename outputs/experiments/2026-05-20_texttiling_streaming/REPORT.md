# TextTiling-online-streaming — AUXILIARY baseline (3-benchmark, per-turn latency)

> **Paper-ready honest naming (codex 2026-05-21)**: `Streaming-TT-inspired` (or `CausalTextTiling-Streaming`). 원본 NLTK/SuperDialseg TextTiling 의 결정 규칙 (global mean−std/2 threshold, bilateral depth, paragraph min_gap) 모두 변경 → running mean+c·std, one-sided depth, utterance min_gap. **TextTiling 의 online 변형이 아니라 *TextTiling-inspired 별도 method***. 원본 paper 결과와 같은 표 등재 금지.
> AUXILIARY (codex:rescue 2026-05-20/21 위임 권고). 핵심 비교값 = **per-turn latency (ms)**.

## 1. Setup
- **Method**: `hi_ontop.baselines.StreamingTextTiling` (block-cosine incremental, Welford running threshold).
  w=5, k=3, c=0.5 (cutoff=mean+c·std of depth), min_gap=3, warmup_gaps=3. ※ short-dialogue (tiage ~16 발화) 대응 runner default; class default 는 NLTK 호환 w=10/k=6.
- **데이터**: Def-DTS 번들 (`benchmarks/Def-DTS/data/DTS_session_datasets/*_test.jsonl`) 의 3 dataset. 데이터 로드 외 Def-DTS 의존 없음.
- **표본 정의**: dataset 별 전체 test set (전체 대화 사용).
- **metric**: `segeval` Pk/WD (per-dialogue → macro mean), self-implemented boundary-set F1 (per-dialogue → macro mean). Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).
- **latency**: 매 `push()` 호출 perf_counter (CPU only, calls/turn=0, tokens/turn=0). 첫 발화는 표본 제외.

## 2. 보조표 (TextTiling-online-streaming | past-only, non-LLM)

| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/turn(ms) ↓ | pred bs | gold bs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.5266 | 0.5476 | 0.2252 | 0.3441 | 0.008 | 305 | 315 |
| dialseg711 | 711/18639 | 0.4713 | 0.4896 | 0.2708 | 0.3952 | 0.010 | 4210 | 2743 |
| superseg | 1322/16006 | 0.4623 | 0.4667 | 0.2618 | 0.3987 | 0.011 | 3229 | 4017 |

## 3. latency 분포 (ms/turn)

| dataset | n | mean | std | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 1464 | 0.008 | 0.005 | 0.007 | 0.013 | 0.014 | 0.017 | 0.113 |
| dialseg711 | 17928 | 0.010 | 0.006 | 0.008 | 0.018 | 0.020 | 0.026 | 0.189 |
| superseg | 14684 | 0.011 | 0.009 | 0.008 | 0.022 | 0.027 | 0.040 | 0.216 |

## 4. 해석
- streaming 의 per-turn O(w) 비용은 nltk prefix-recompute 의 O(t) 비용보다 *원리적으로* 작음 (긴 대화일수록 격차 확대). 세 dataset 모두 ms 단위 이하의 latency 가 나오면 baseline 의 핵심 주장 (낮은 per-turn latency) 이 검증됨.
- Pk/WD/F1 = INDICATIVE. running threshold 가 미래를 모르므로 NLTK 원본의 global threshold 와 다른 boundary set 을 만든다. 이 격차 자체가 *의도된 algorithmic 차이*. 같은 이름 대신 `TextTiling-online-streaming` 을 쓰는 이유.
- dataset 별 분포 차이: tiage (~16 발화) 짧음 → 2k pseudo-sentence 채우기 빠듯 (under-seg 편향), dialseg711·superseg (수십~수백 발화) 에선 running threshold 가 안정화됨.
- causal lag: gap score 는 right block (k pseudo-sentence) 이 닫혀야 계산되므로, boundary 채택 시점은 *대상 발화 인덱스보다 늦은 push() 호출 안* 에서 발생. metric 계산에는 영향 없음 (대화 종료시 boundary set 동일).

## 5. 한계 / 검증 미해결
- 표본: 전체 test set (전체 대화 사용). seed/반복 없음 (알고리즘은 결정적이라 seed 무관).
- one-sided depth (right_peak 미사용) → 원본 양방향 depth 와 boundary criterion 자체 다름.
- c=0.5, min_gap=3 모두 dev-set sweep 없이 codex 권고값 (class default c=0.5, min_gap=4; runner default 는 짧은 tiage 대화에 맞춰 조정). tuning 여지 있음.
- utterance ↔ pseudo-sentence 매핑: pseudo-sentence 가 발화 경계를 넘을 수 있어 detected boundary 의 utterance 귀속이 ±몇 발화 단위로 미세하게 shift 됨. Pk/WD 의 window-tolerant 성격으로 흡수되지만 F1 (정확 일치) 은 그만큼 낮을 수 있음.
- NLTK 원본 / `run_texttiling_prefix.py` (prefix-recompute) 와의 boundary diff 비교는 별도 작업.
