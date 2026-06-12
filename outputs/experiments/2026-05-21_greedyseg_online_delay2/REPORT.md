# GreedySeg-online-delay2 — bounded-lookahead online baseline

> **Honest naming (codex 2026-05-21)**: same scoring/selection, delayed emission (delay=2 — right context window_size=2). 원본 SuperDialseg GreedySegmenter 의 score 공식·HP·argmin greedy 선택 그대로 보존. **5행 핵심표 가능** (본 plan 의 baseline 중 유일).

## 1. Setup
- **Method**: `hi_ontop.baselines.GreedySegOnlineDelay2`. backbone=`bert-base-uncased`, window_size=2, jump_step=2, max_seg_round=8, sim_threshold=0.6, max_seq_length=50.
- **Device**: `mps` (auto-resolved; cuda→mps→cpu 우선순위). PYTORCH_ENABLE_MPS_FALLBACK=1.
- **데이터**: Def-DTS 번들 (`benchmarks/Def-DTS/data/DTS_session_datasets/*_test.jsonl`) 의 3 dataset. 전체 test set (전체 대화 사용).
- **metric**: `segeval` Pk/WD (per-dialogue → macro mean), self-implemented boundary-set F1 (per-dialogue → macro mean). Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).
- **latency**: 매 `push()` 호출 perf_counter (BERT forward 포함). 첫 발화는 표본 제외. cold-start (BERT load) 분리.
- **cold-start**: BERT/tokenizer load 2.05s (per-turn latency 와 분리).
- **Environment**: python=3.13.7, torch=2.11.0, transformers=4.49.0, platform=macOS-26.3.1-arm64-arm-64bit-Mach-O, commit=cd870da.

## 2. 결과표 (GreedySeg-online-delay2 | bounded-lookahead, BERT)

| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/turn(ms) ↓ | bert_fwd/utt | pred bs | gold bs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.5370 | 0.5535 | 0.1420 | 0.2984 | 13.81 | 2.13 | 230 | 315 |
| dialseg711 | 711/18639 | 0.4164 | 0.4434 | 0.4120 | 0.4911 | 16.27 | 2.72 | 3511 | 2743 |
| superseg | 1322/16006 | 0.5072 | 0.5109 | 0.2775 | 0.3842 | 9.14 | 1.91 | 2449 | 4017 |

## 3. latency 분포 (ms/turn, BERT forward 포함)

| dataset | n | mean | std | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 1464 | 13.81 | 22.32 | 0.00 | 36.27 | 67.34 | 101.55 | 124.66 |
| dialseg711 | 17928 | 16.27 | 25.80 | 0.00 | 52.12 | 81.76 | 106.68 | 166.52 |
| superseg | 14684 | 9.14 | 16.56 | 0.00 | 23.58 | 41.40 | 84.26 | 125.79 |

## 4. 해석
- 원본 GreedySeg 의 *알고리즘 본질 (BERT cosine score · argmin greedy selection · HP)* 가 그대로 보존됨. delay=2 는 right context (window_size=2) 가 buffer 에 도착해야 score 계산 가능하기 때문 — *boundary 채택* 자체는 비가역 greedy 그대로.
- 실제 emit 시점은 segment 안 `max_seg_round` 후보가 모두 평가된 후 (= cut_index + 7 + window_size 발화 후)이므로, boundary index 가 가리키는 utterance 와 push() 시점 간 차이는 *최대* 9 발화. interactive 사용 시 이 lag 명시 필요.
- **TextTiling-streaming (encoder-free, ~0.01ms) 과 같은 latency 표에 섞지 않음**: encoder cost (BERT forward) 차원이 다름.
- 결정성: device 마다 결정성 보장 다름 (CPU > CUDA > MPS). 본 표 수치는 device=`mps` 1회 측정.

## 5. 한계 / 검증 미해결
- 원본 GreedySeg paper 점수와 직접 비교 불가: 데이터 (Def-DTS 번들 vs 원 SuperDialseg) + metric (segeval direct vs autoseg) + bounded-lookahead 인터페이스 차이. *방향성·정상동작* 검증용 보조.
- HP 미튜닝: 원본 default (`window_size=2, jump_step=2, max_seg_round=8, sim_threshold=0.6, max_seq_length=50`) 그대로. dev-set sweep 없음.
- **`segment-concat → [CLS] embedding`** 가정 — 원본 코드의 정확한 pooling 방식 (CLS vs mean) 확인 못함 (superdialseg 로컬 install 없음). 다른 pooling 사용 시 점수 변동 가능.
- BERT 추론 결정성: torch/transformers 버전 동일 + 동일 device 반복 측정해야 reproducibility 확보. 다른 device 간 직접 비교 금지.
- delay 의 정확한 의미: `delay=2` 는 codex 의 right-context lag 표현. 실제 boundary emit 시점은 segment 길이 + window_size 만큼 lag.
- 표본은 dataset 별 전체 test set (target-turns=0). seed 없음 (BERT 결정성 + 알고리즘 결정성 → seed 무관).
