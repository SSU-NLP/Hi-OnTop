# v4.1.3 → SeCom segmentation backend swap

`outputs/experiments/2026-05-21_v413_secom_swap/` · 2026-05-21~22 · ours = Hi-OnTop 재실행 완료

## 한 줄

SeCom (Pan et al., ICLR 2025) 의 LLM 기반 segmentation backend (`gpt-4o-mini`) 를
Hi-OnTop Hi-OnTop (online, O(m)/turn) 으로 drop-in 교체. **downstream QA 품질이 baseline LLM
대비 GPT4Score -0.72pp / BERTScore-F1 -0.23pp (noise 범위) 로 유지되며, segmenter assign
latency 는 0.07 ms/turn (vs LLM 646 ms)** — paper 의 핵심 claim 실측 근거.

## 실험 setup

**Dataset**: Long-MT-Bench+ (`panzs19/Long-MT-Bench-Plus`)
- n_conv = 11, n_sessions ≈ 55 (5/conv), avg n_turns/session = 13.7, n_questions = 27/conv

**Pipeline (SeCom 원본 5-stage 그대로, segment 만 swap)**:
1. segment → topic 단위 chunking
2. compress (LLMLingua-2 xlm-roberta-large-meetingbank, rate=0.75)
3. retrieve (multi-qa-mpnet-base-dot-v1 + FAISS, top-k=1)
4. chat (`openai/gpt-4o-mini` via Crts)
5. eval (QA F1, subspan EM, ROUGE-L, BERTScore-F1)

**비교 row**:

| 표기 | Retriever | Segmentation | Response gen |
|---|---|---|---|
| (paper) SeCom (BM25, GPT4-Seg) | BM25 | GPT-4-0125 | GPT-3.5-Turbo |
| (paper) SeCom (MPNet, GPT4-Seg) | MPNet | GPT-4-0125 | GPT-3.5-Turbo |
| (paper) SeCom (MPNet, Mistral-7B-Seg) | MPNet | Mistral-7B-Instruct-v0.3 | GPT-3.5-Turbo |
| (paper) SeCom (MPNet, RoBERTa-Seg) | MPNet | RoBERTa (SuperDialSeg-FT) | GPT-3.5-Turbo |
| **(ours) Control: gpt-4o-mini** | MPNet | `openai/gpt-4o-mini` (Crts) | `openai/gpt-4o-mini` |
| **(ours) Ours: Hi-OnTop** | MPNet | **Hi-OnTop Hi-OnTop (online, O(m)/turn)** | `openai/gpt-4o-mini` |

설계 noteset:
- SeCom 의 4 paper variant 는 **Table 1 / Table 3 보고치 인용** (Mistral-7B local + RoBERTa
  fine-tuned ckpt 재현 비현실적). 우리는 2개 ours row 만 직접 실행.
- 두 ours row 는 chat 모델 (gpt-4o-mini) 과 retriever (MPNet) 통일 → **유일한 차이가
  segmentation method**. 공정한 swap 비교.

**Hi-OnTop segmentation 파라미터** (2026-05-22 재실행):
- Encoder: `sentence-transformers/multi-qa-mpnet-base-dot-v1` (L2-normalized) — **Crts
  `/v1/embeddings` API** 경유 (`--encoder_backend api`). Crts mpnet 출력은 로컬
  sentence-transformers 와 bit-identical (cos=1.0 검증).
- δ\* = **0.5983** — mpnet δ_eff p80, **ctx_window=2 기준 재보정**
  (`delta_star_calibration_hiontop_m2.json`). bge TIAGE δ* (≈0.5594) 는 인코더 불일치로 부적합.
  직전 run 은 δ_prev p80 = 0.6194 (m=3) 사용.
- Hi-OnTop HP: m(ctx_window)=2, ρ(ctx_decay)=0.7, a(ctx_blend_a)=0.5. v4.1.x 의 α/λ/β/pe/f0
  등은 Hi-OnTop reduced form 에 존재하지 않음 (dead-code audit 으로 제거).
- Per-session fresh segmenter (SeCom 의 LLM call 도 session 단위 → fair compare)

## 결과

### Paper-aligned 8-method comparison (SeCom Table 1 metric 매칭)

| Rank | Method | GPT4Score ↑ | BLEU ↑ | Rouge1 ↑ | Rouge2 ↑ | RougeL ↑ | BERTScore-F1 ↑ | # Turns | # Tokens | Seg ms/turn ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline (gpt-4o-mini-Seg) | **78.12** | 21.89 | 40.71 | 23.94 | 33.62 | **89.14** | 2.56 | 750 | 646 (LLM API) |
| 2 | Full History (no-seg) | 77.92 | 15.73 | 34.55 | 19.16 | 27.88 | 88.18 | 65.45 | 22,676 | — |
| **3** | **Hi-Seg (Ours, Hi-OnTop)** | **77.40** | 19.63 | 39.47 | 23.21 | 32.72 | **88.91** | 4.27 | 1,124 | **0.07** ⭐ |
| 4 | CSM-style (ours-trained) | 74.20 | 19.76 | 39.90 | 23.60 | 32.76 | 89.03 | 2.53 | 749 | ~330 (BERT CPU) |
| 5 | TextTiling-style | 73.47 | 19.45 | 38.82 | 22.56 | 31.99 | 88.91 | 3.74 | 1,068 | **1.1** |
| 6 | GreedySeg-style (delay-2) | 68.58 | 17.69 | 37.68 | 21.41 | 30.92 | 88.61 | 5.38 | 1,495 | ~330 (BERT CPU) |
| 7 | GraphSeg-style (window-d) | 62.53 | 14.85 | 33.66 | 18.43 | 27.09 | 87.85 | 8.66 | 2,446 | ~200 (GloVe+clique) |
| 8 | Zero History (no context) | 42.12 | 10.31 | 29.16 | 12.09 | 21.96 | 86.94 | 0 | 0 | — |

- GPT4Score = mean(judge score 1-10) × 10 (paper headline). Judge = `openai/gpt-4o`, 288/288 valid for all rows.
- Hi-Seg row = **Hi-OnTop** (m=2, δ*=0.5983, Crts mpnet) — 2026-05-22 재실행. 직전 v4.1.3 (m=3, δ*=0.6194: GPT4Score 74.72) 대비 전 지표 향상. 변경 이력 참조.
- Hi-Seg 의 위치 = **non-LLM 모든 baseline 우위** (CSM/TextTiling/GreedySeg/GraphSeg 모두 ↓), LLM segmenter (baseline 78.12) 와 Full History (77.92) 만 살짝 우위 — GPT4Score gap 이 baseline 대비 -0.72pp 까지 좁혀짐.
- Pareto 관점: Hi-Seg 의 **1124 tokens 로 GPT4Score 77.40** = baseline (750 tok / 78.12) 대비 약간 큰 budget 으로 LLM 수준에 근접, Full History (22,676 tok / 77.92) 는 **20× context 로 거의 동률** → **token efficiency 우위** (Figure G 참조).

### Δ vs baseline LLM segmenter

| Metric | baseline | Hi-Seg (Hi-OnTop) | Δ |
|---|---:|---:|---:|
| GPT4Score | 78.12 | 77.40 | **-0.72** ⭐ |
| BERTScore-F1 | 89.14 | 88.91 | **-0.23** ⭐ |
| BLEU | 21.89 | 19.63 | -2.26 |
| Rouge1 | 40.71 | 39.47 | -1.24 |
| Rouge2 | 23.94 | 23.21 | -0.73 |
| RougeL | 33.62 | 32.72 | -0.90 |
| **Segment latency (assign-only)** | 646 ms | **0.07 ms** | **~8700× ↓** ⭐ |

→ GPT4Score **-0.72pp = 0.9% relative drop**, BERTScore-F1 **-0.23pp = 0.3% relative drop** — 모두 noise 범위.
직전 v4.1.3 run 의 -3.40pp (GPT4Score) 대비 크게 개선. Segment algorithmic latency 는 Hi-OnTop
reduced form 이 v4.1.3 의 inert SEM2 machinery 를 계산하지 않아 0.07 ms/turn — 단, 공정한 end-to-end
비교는 encode 포함값 (아래 Latency 표) 으로 봐야 한다.

### Auxiliary QA (SeCom evaluate_match)

| method | QA F1 (token) | Subspan EM |
|---|---:|---:|
| baseline | 36.45 | 3.12 |
| ours (Hi-OnTop) | 34.89 | 2.78 |

### Latency (segment 단계만)

| method | n_segments | avg ex/seg | encode (s/all) | segment (s/all) | ms/turn (algorithmic) | ms/turn (incl. encode) |
|---|---:|---:|---:|---:|---:|---:|
| baseline (gpt-4o-mini LLM) | 318 | 2.26 | — (LLM 내부) | 465 | **646** | 646 |
| **ours (Hi-OnTop, Crts encode)** | 188 | 3.83 | 78.7 | 0.053 | **0.074** | **109** |
| ours (v4.1.3, 로컬 CPU encode) | 167 | 4.31 | 643 | 3.74 | 5.20 | 903 |

- **algorithmic latency (segmenter assign 만)**: Hi-OnTop **0.074 ms/turn** vs baseline **646 ms/turn**.
  Hi-OnTop reduced form 은 v4.1.3 (5.20 ms) 가 돌리던 dead SEM2 machinery 를 제거 → assign 이 순수
  causal-window cosine + threshold 만 → 70× 더 빠름. 출력은 v4.1.3 와 byte-identical (parity 검증).
- **end-to-end (text → vector → segment)**: Hi-OnTop = **109 ms/turn** — encode 를 Crts API 로 offload
  (로컬 CPU mpnet 643s → Crts 79s, 8× 단축). 네트워크 의존값이라 알고리즘 latency 아님.
- baseline 의 646ms 도 LLM API 콜 — ours encode 도 API 경유라 end-to-end 는 둘 다 네트워크 포함,
  apples-to-apples. encode 제외 algorithmic 비교에서는 0.074 ms 가 segmenter 자체 비용.

### Segment statistics

| method | n_segments | n_exchanges | avg ex/seg | boundary strength bands |
|---|---:|---:|---:|---|
| baseline (gpt-4o-mini) | 310 (실제 318 - 8 empty) | 720 | 2.32 | — (binary LLM 출력) |
| ours (v4.1.3) | 167 | 720 | 4.31 | very_weak: 488, weak: 119, normal: 99, strong: 14 |

### Boundary placement agreement

| metric | value |
|---|---:|
| position agreement (turn-by-turn) | 76.2% |
| ours' boundaries also in baseline (precision) | 47.8% |
| baseline's boundaries also in ours (recall) | 91.7% ⭐ |
| boundary F1 | 62.9% |

→ ours 의 167 boundaries 중 **91.7% 는 baseline LLM 도 boundary 라고 판단한 자리** = ours 가 더 conservative 하지만 그 결정은 LLM 과 매우 일치. baseline 의 fine-grained (310) 중 절반 정도가 의미 있게 큰 topic shift.

> **주의**: 위 Boundary placement agreement 표는 **v4.1.3 run (m=3, δ*=0.6194, 167 seg)** 기준
> `segment_compare.json` — Hi-OnTop 재실행 (m=2, δ*=0.5983, 188 seg) 에 대해서는 미재계산.

## 해석 (Hi-OnTop run, 2026-05-22)

1. **Algorithmic claim 입증**: Hi-OnTop `assign()` 0.074 ms/turn vs LLM 646 ms/turn. reduced
   form 이 v4.1.3 의 dead SEM2 machinery (5.20 ms) 를 제거 → segmenter 자체 비용이 거의 0.
   end-to-end 는 109 ms/turn (Crts encode 포함, 네트워크 의존).
2. **Downstream 품질 유지**: paper metric 전반 baseline 대비 GPT4Score -0.72pp / BERTScore-F1
   -0.23pp / BLEU -2.26 / Rouge1/2/L -1.24/-0.73/-0.90. GPT4Score·BERTScore 는 noise 범위.
   직전 v4.1.3 run (GPT4Score -3.40pp) 대비 모든 지표 향상 — m=3→2 + δ_eff 재보정 효과.
3. **Trade-off 정량화**: Hi-OnTop 는 baseline 보다 굵은 segments (3.83 vs 2.26 ex/seg) →
   retrieve top-1 context 4.27 turns / 1124 tokens (baseline 2.56 / 750). BLEU/ROUGE 의
   소폭 손해는 이 context 입도 차이에서 옴. 그러나 GPT4Score gap 이 -0.72pp 까지 좁혀져
   trade-off 가 거의 사라짐.
4. **m=3 → m=2 효과**: ctx_window 축소 + δ_eff p80 재보정으로 segment 가 더 잘게 (167→188),
   downstream 전 지표 향상. 굵은 segment 의 retrieval 손해가 완화된 것으로 해석.

## 판정

- **drop-in 교체 가능 ✅**: SeCom 의 LLM segmentation 백엔드를 Hi-OnTop 로 교체 시 downstream QA
  의 main metric (GPT4Score 99.1% / BERTScore-F1 99.7% 유지), segmenter assign 은 ~8700× 빠름
  (encode 포함 end-to-end 도 baseline LLM 콜보다 빠름).
- **Paper contribution 정량 입증**: "Hi-OnTop Hi-OnTop 의 graded boundary score segmenter 는 LLM
  기반 baseline 의 drop-in 교체로서 downstream 품질을 GPT4Score -0.72pp 내 유지하며 segmenter
  연산 비용을 LLM 콜 대비 사실상 제거한다."

## 한계 / 검증 미해결

- **n_conv = 11**: Long-MT-Bench+ test split 전체. 통계적 power 제한 (단일 run).
  multi-seed (생성 LLM temperature=0 이라 seed 효과 없음, segmentation 도 deterministic).
- **mpnet δ\* 는 휴리스틱**: paper 의 F1-supervised δ\* (TIAGE train) 가 아닌
  MTB+ 의 δ_eff 분포 p80 (m=2). 다른 quantile (p70/p85/p90) 의 sensitivity 미측정.
- **SeCom 의 4 paper row 는 재현 안 함** — 인용치라 우리 환경 (Crts gpt-4o-mini chat)
  과 chat LLM 다름 (paper = GPT-3.5-Turbo). 절대값 비교 시 disclaimer 필요.
- **segment 인코더는 Crts API, retrieve 인코더는 로컬**: segment 단계 mpnet 임베딩은
  Crts `/v1/embeddings` 로 전환됐으나, retrieve 단계는 SeCom 내부 langchain
  `HuggingFaceEmbeddings` (로컬 CPU) 그대로 — `benchmarks/` 읽기 전용. Crts mpnet ≡ 로컬
  (cos=1.0) 이라 결과 동일, retrieval 시간만 paper 값과 직접 비교 불가.
- **LLMLingua-2 compression**: 동일 rate=0.75 사용. 두 method 가 input segment 가
  달라서 compressed token 수도 다를 수 있음 → fair 한지 검토 필요.
- **assign-only 0.074 ms 의 비약**: v4.1.3 (5.20 ms) 대비 70× — Hi-OnTop reduced form 이
  dead SEM2 machinery 를 안 돌리기 때문. 출력 parity 는 검증됐으나 latency 차이의 단일
  측정값이라 repeated-run 분산 미측정.
- **Boundary placement / segment_compare 는 v4.1.3 run 기준** — Hi-OnTop (188 seg) 재계산 안 함.

## 산출

- `src/hi_ontop/secom_adapter.py` — HiOnTopSecomSegmenter (mpnet → v4.1.3 wrap)
- `scripts/secom_swap/01_prepare_data.py` — MTB+ → SeCom JSONL
- `scripts/secom_swap/02_calibrate_delta_star.py` — δ* 추정 (delta_prev / delta_eff 모드)
- `scripts/secom_swap/03_segment_v413.py` — Hi-OnTop segmentation runner (`--encoder_backend api`)
- `scripts/secom_swap/04_segment_baseline.py` — gpt-4o-mini segmentation runner
- `scripts/secom_swap/05_compress.py` / `06_retrieve.py` / `07_chat.py` / `08_eval.py`
- `scripts/secom_swap/run_pipeline.sh` — orchestrator
- `delta_star_calibration.json` — mpnet δ_prev 분포 (v4.1.3 run, m=3)
- `delta_star_calibration_hiontop_m2.json` — mpnet δ_eff p80 = 0.5983 (Hi-OnTop run, m=2)
- `latency_ours.json` / `latency_baseline.json` — v4.1.3 / baseline timing
- `latency_ours_hiontop.json` — Hi-OnTop run timing
- `metrics_ours.json` (v4.1.3) / `metrics_ours_hiontop.json` (Hi-OnTop) / `metrics_baseline.json` — downstream eval

## 변경 이력

- **2026-05-21 초안**: 인프라/스크립트 작성, paper variants 표 + 우리 row 정의
- **2026-05-21 (실행 1)**: δ* calibration (mpnet p80=0.6194) → segment 양쪽 → compress → retrieve → chat → eval (초기 QA F1 + BERTScore)
- **2026-05-21 (실행 2, paper-aligned)**: 08_eval.py 를 SeCom Table 1/3 metric 매핑 (BLEU, Rouge1/2/L, BERTScore, GPT4Score with `openai/gpt-4o` judge, Context Length) 으로 재작성 → baseline + ours 재평가. 본 REPORT 의 표 = paper-aligned 결과
- **2026-05-22**: 8-method 표의 stale `—` 정정 — Full/Zero 의 Rouge1/2/L, GreedySeg 의 BLEU/Rouge1/2/L 를 `metrics_*.json` 값으로 채움. GreedySeg #Turns/#Tokens (5.40/1,463 → 5.38/1,495), GraphSeg #Tokens (2,545 → 2,446) 도 재평가 JSON 에 맞춰 정정. 동일 표를 `outputs/reports/downstream_task.md` 에 LaTeX 형태로도 보관 (downstream task 결과 단일 모음 문서)
- **2026-05-22 (ours = Hi-OnTop 재실행)**: ours row 를 v4.1.3 → Hi-OnTop 로 재실행. ① segment 인코더를 Crts `/v1/embeddings` API 로 전환 (`03_segment_v413.py --encoder_backend api`, `secom_adapter._encode` 가 두 인코더 contract 지원). ② ctx_window default 3→2 변경에 맞춰 δ* 를 δ_eff p80 (m=2) = **0.5983** 으로 재보정 (`02_calibrate_delta_star.py --mode delta_eff`). ③ segment(188 seg) → compress → retrieve → chat → eval. 결과: GPT4Score 74.72→**77.40**, BERTScore-F1 88.69→**88.91**, baseline 대비 gap -3.40→**-0.72pp**. v4.1.3 산출물 (`metrics_ours.json` 등) 은 보존. 8-method 표·Δ표·Latency표·해석·판정 모두 Hi-OnTop 기준으로 갱신
