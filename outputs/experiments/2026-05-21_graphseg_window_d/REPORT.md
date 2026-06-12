# GraphSeg-inspired bounded-window (short: `GraphSeg-window-d`) — AUXILIARY

> **Honest naming (codex 2026-05-21)**: *original global graph mechanism is not preserved*. GraphSeg (Glavaš et al. 2016) 의 sentence similarity (IC × GloVe + Hungarian) + Bron-Kerbosch maximal clique + sequential merge 3-phase 를 **window 안에서만** 적용. 강한 `-online` 명명 금지.
> **AUXILIARY only** — 원본 GraphSeg paper 결과와 같은 표 등재 금지.

## 1. Setup
- **Method**: `hi_ontop.baselines.GraphSegWindowD`. window_d=10, sim_threshold (τ)=0.25, min_seg_size=3, freq_source=`brown`, use_pos_filter=True.
- **Embedding**: GloVe 6B.300d (vocab=400000). IC weighting from NLTK brown corpus.
- **데이터**: Def-DTS 번들 (`benchmarks/Def-DTS/data/DTS_session_datasets/*_test.jsonl`). 전체 test set (전체 대화 사용).
- **metric**: `segeval` Pk/WD (per-dialogue → macro mean) + boundary-set F1. Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).
- **cold-start**: GloVe + IC table load 13.66s (per-turn latency 와 분리).
- **Environment**: python=3.13.7, platform=macOS-26.3.1-arm64-arm-64bit-Mach-O, commit=f6b4c82. GraphSeg-window-d 는 PyTorch 무관 (numpy/scipy/networkx).

## 2. 결과표 (GraphSeg-inspired bounded-window | window-local clique)

| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/turn(ms) ↓ | pred bs | gold bs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100/1564 | 0.4925 | 0.5159 | 0.2517 | 0.3738 | 0.85 | 257 | 315 |
| dialseg711 | 711/18639 | 0.4485 | 0.4816 | 0.3656 | 0.4503 | 1.15 | 3968 | 2743 |
| superseg | 1322/16006 | 0.5387 | 0.5423 | 0.1644 | 0.3120 | 0.62 | 1545 | 4017 |

## 3. latency 분포 (ms/turn, Bron-Kerbosch + Hungarian 포함)

| dataset | n | mean | std | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 1464 | 0.85 | 2.14 | 0.29 | 1.61 | 1.68 | 1.84 | 78.53 |
| dialseg711 | 17928 | 1.15 | 0.67 | 1.48 | 1.75 | 1.84 | 2.01 | 7.94 |
| superseg | 14684 | 0.62 | 0.69 | 0.24 | 1.73 | 1.88 | 2.18 | 8.38 |

## 4. 해석
- **원본 GraphSeg 의 어떤 본질이 보존되고 어느 본질이 양보됐는가** (codex 2026-05-21):
  - 보존: sentence similarity 공식 (IC × GloVe + Hungarian), Bron-Kerbosch maximal clique, sequential merge 3-phase, content-word POS filter.
  - 양보: full-dialogue graph (→ window=d), global clique structure (→ window-local), single-pass global merge (→ sliding window 마다 재계산), backtracking (= 불가, boundary 비가역).
- 따라서 본 결과는 *GraphSeg 원본 점수와 직접 비교 불가*. AUXILIARY online auxiliary baseline.
- **TextTiling-streaming (encoder-free, ~0.01ms) / GreedySeg-online-delay2 (BERT, ~10ms) 와 같은 latency 표에 직접 비교 금지** — encoder/연산 카테고리가 다름. 비교 시 *어느 본질이 보존되고 어느 본질이 양보됐는지* 열 분리 (codex 권고).

## 5. 한계 / 검증 미해결
- 원본 GraphSeg paper 점수와 직접 비교 불가: 데이터 (Def-DTS vs 원 SemEval 2016 dialogue datasets), metric (segeval direct vs 원 paper), 그리고 window-local maximal clique 자체가 algorithmic 차이.
- HP 미튜닝: τ=0.25, window_d=10, min_seg_size=3 모두 codex 권고 default. dev-set sweep 없음.
- **word frequency 출처** = NLTK brown corpus (Wikipedia 보다 작은 domain). 원본은 더 큰 corpus. IC 가중치 분포 차이가 결과에 영향 가능.
- **POS filter** = NLTK averaged_perceptron_tagger. 원본 paper 의 POS-filter 와 정확히 같지 않을 수 있음.
- **boundary lag-emission**: 한 boundary 가 여러 window 평가에 걸쳐 재출현 가능. 본 구현은 *최초 발견 즉시 비가역 채택* (codex 'graph 범위 변경' 의 자연스러운 부산물).
- 결정성: numpy/scipy/networkx 모두 결정적 → seed 무관, byte-identical 결과.
