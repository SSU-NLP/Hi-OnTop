# 4. Experiments

본 장은 (1) **DTS 벤치마크** 에서의 segmentation 품질 비교 (§4.1–§4.4) 와 (2)
**Long-MT-Bench+ (MTB+)** 다운스트림 QA 응답 품질 비교 (§4.5) 의 두 평가를
다룬다. 모든 실험은 **CPU 환경** 에서 수행 — 일부 임베딩 모델 (특히
`sentence-transformers/multi-qa-mpnet-base-dot-v1` int8 ONNX) 이 GPU API 가
없어 환경 일관성 (NO GPU) 으로 통일했다. **fine-tuned 모델 (CSM, RoBERTa) 의
결과는 시간 제약상 seed 3회 평균을 산출하지 못하고 random 1회 학습 결과로
보고**하며, 향후 작업에서 seed 평균으로 보완할 예정이다.

---

## 4.1 Datasets

### 4.1.1 DTS 벤치마크 (§4.2–§4.4 에서 사용)

세 개의 공개 dialogue topic segmentation 벤치마크를 **Def-DTS bundle**
(`benchmarks/Def-DTS/`) 형태로 통일해 평가에 사용한다.

| Benchmark | 전체 dialogue | Train (calibration) | Test | 도메인 | 주석 출처 |
|---|---:|---:|---:|---|---|
| **TIAGE** | 400 | 300 (75%) | 100 | open-domain crowdsourced topic shift | Xie et al., NAACL'21 |
| **Dialseg711** | 711 | 498 (70%) | 213 | Wikipedia 기반 인공 합성 | Xu et al., EMNLP'21 |
| **SuperDialseg** | 1722 | 400 (23%) | 1322 | document-grounded dialogue (DC-1) | Coldog2333 et al., EMNLP'23 |

- **Train split 의 역할**: 본 논문은 train 을 *모델 파라미터 학습* 이 아니라
  **δ\* calibration source** (§4.4 와 Fig. H) 또는 supervised 베이스라인
  (RoBERTa) 의 학습용으로만 사용. unsupervised 베이스라인 (TextTiling,
  GraphSeg, GreedySeg, CSM) 은 train 을 사용하지 않는다.
- **SuperDialseg 의 23% calibration ratio**: SuperDialseg 데이터셋은 1722
  dialogue 로 가장 큰데, calibration cap = **400 개** 를 적용해 TIAGE /
  Dialseg711 와 N≤300 비교 가능성을 유지했다 (나머지 1322 dialogue 는 test
  split). Figure H 의 convergence 실험은 N≤300 까지 다루므로 이 cap 이
  결과에 영향을 주지 않는다.

### 4.1.2 Long-MT-Bench+ (MTB+, §4.5 다운스트림 평가)

- **데이터셋**: `benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl`
- **n_conv = 11 dialogue**, 각 dialogue 는 평균 5 session, 총 65 turn
- **n_qa = 288 questions** (dialogue 당 평균 26 QA)
- 각 QA 는 dialogue 내부의 정보를 reference 해 답해야 하는 multi-session
  long-context QA 형태. SeCom 원논문 (Pan et al., 2024) 의 평가 설정 그대로.

---

## 4.2 Baselines

본 절에서는 비교 대상 baseline 들을 **(원본 offline 알고리즘 요약 + Hi-OnTop 의
online 변형 / 수정 사항)** 의 짝으로 정리한다. 모든 online 변형 코드는
`methods/<baseline>/online/` 에 있다.

### 4.2.1 Unsupervised baselines

#### **TextTiling** *(Hearst, 1997)*
- **Offline 원본**: 토큰화 + 불용어 제거 + bag-of-words → 슬라이딩 윈도우 간
  cosine block-similarity → depth score 계산 → global threshold (μ−σ) 적용.
  NLTK `nltk.tokenize.TextTilingTokenizer` 기본 파라미터 (w=10, k=6) 사용.
- **Hi-OnTop 변형 (online, prefix-causal)**: `methods/texttiling/online/prefix.py`
  는 prefix-recompute (매 turn 마다 NLTK 를 `u_{1..t}` 에 fresh 호출, O(t)/turn).
  `methods/texttiling/online/streaming.py` 는 자체 구현된 incremental
  block-cosine + Welford running threshold (one-sided depth, O(w)/turn) —
  본 논문 메인 비교에서 사용된 **TextTiling-Style-Seg** 가 이쪽이다.

#### **GraphSeg** *(Glavaš et al., 2016)*
- **Offline 원본**: 발화 → POS 태깅 + 불용어/GloVe 필터 + Information
  Content lookup → IC×GloVe 가중 cosine matrix → Hungarian assignment →
  유사도 graph + Bron-Kerbosch maximal clique → sequential merge.
- **Hi-OnTop 변형 (online, windowed)**: `methods/graphseg/online_window.py` —
  원본 알고리즘을 **window=d 안에서만** 적용 (메모리/지연 한정). 강한
  "online" 명명은 codex review 결과 회피하고 `GraphSeg-window-d` 로 명명.
  본 논문 표에서는 단순히 **GraphSeg-Style-Seg** 로 표기.

#### **GreedySeg** *(Xu et al., 2021)*
- **Offline 원본**: BERT-base 의 발화 임베딩 → cosine 거리 → argmin greedy
  로 가장 dissimilar 한 인접 발화 사이를 boundary 로 선언. 슬라이딩 윈도우
  내 다른 발화와의 평균 cosine 비교.
- **Hi-OnTop 변형 (online, delay=2)**: `methods/greedyseg/online/delay2.py` —
  **delay=2 bounded lookahead** (turn t 의 boundary 는 t+2 도착 시 결정).
  원본의 score 공식·HP·argmin greedy 선택을 그대로 보존하되, 미래 context
  를 2-turn 으로 제한해 streaming 호환. device-agnostic (cuda/mps/cpu).

#### **CSM** *(Xing & Carenini, 2021)*
- **Offline 원본**: lxing532 의 **CoherenceNet** 아키텍처 (BERT encoder
  → 768→768→2 decoder, softmax 2-way) — 인접 발화쌍의 coherence score
  를 marginal ranking loss (margin=1) 로 학습. inference 는 score 위에
  TextTiling-style depth + threshold 로 boundary 결정.
- **사전학습 + 파인튜닝 setup (우리 학습)**: 본 paper 는 ckpt 를 직접
  학습한다 — backbone = `bert-base-uncased` (HF pretrained, no further
  pretraining), 학습 코드 = `scripts/train_csm_hf.py` (lxing532 의 원본
  colab 학습 셀을 HuggingFace `Trainer` 로 재작성, 알고리즘·loss·optimizer
  수식 그대로). 데이터 = lxing532 의 `UtteranceDataset` (NSP-style
  triplet, DailyDialog), HP = AdamW lr=2e-5 / batch_size=32 (paper 와
  동일) / margin=1 / linear LR schedule. **random seed = 42, single
  run** (시간 제약, std 미보고). 산출 ckpt = `methods/CSM/cpt_277000.pth`
  (gstep 277,000).
- **Hi-OnTop online 변형 (delay=2)**: `methods/CSM/online/delay2.py` (wrap
  `src/hi_ontop/baselines/csm_online.py`) — streaming CoherenceNet + depth,
  **delay=2 right context**. 추가 수정:
  1. score 함수에 **`sigmoid(logits[0, 0])`** 적용 (원본 raw logit → 안정화).
  2. **alpha=1.0** (원본 default 0.0 → paper cut_rate 일치하도록 변경).
  3. **off-by-one boundary index 수정** (`owner_t = gi + 2`, push 와 flush 모두).

### 4.2.2 Supervised baseline

#### **RoBERTa** *(Coldog2333 et al., EMNLP'23, Table 3)*
- **Offline 원본**: `RobertaForTokenClassification` 위에 발화 끝 첫 `</s>`
  토큰의 binary label (boundary / non-boundary) 학습. 슬라이딩 윈도우
  |T|=20 으로 학습 + 추론, 경계 결정 시 미래 20 발화 포함 logit 평균.
- **사전학습 + 파인튜닝 setup (우리 학습)**: backbone = `roberta-base`
  (HF pretrained, no further pretraining), 학습 코드 =
  `methods/RoBERTa/offline/train.py` (논문 Appendix A.2 그대로 재현 —
  공식 repo `super_dialseg/utils/data/data_collator.py` 의 `roberta-base`
  분기 + `models/supervised.py` 를 reference 로 collator/loss/window
  재구현; 원 repo 는 평가 전용으로 학습 스크립트 미포함). 데이터 =
  SuperDialseg `superseg/train` 6,948 dialogue (모델 선택은
  `superseg/validation` 1,322 dialogue 의 val Score). HP = AdamW lr=1e-5
  / batch_size=8 / weight_decay=1e-3 / grad_clip=1.0 / 20 epochs +
  early stopping (patience=10 epoch, best val ckpt 보존 → 실제 epoch 14
  에서 중단). **random seed = 42, single run** (시간 제약, std 미보고).
  Colab GPU 학습 산출물 (체크포인트·zip) 은 `outputs/runs/_misc/` 에
  보관 (gitignored). 보고서: `outputs/experiments/2026-05-23_roberta_supervised/REPORT.md`.
- **Hi-OnTop online 변형 (strict causal)**: `methods/RoBERTa/online/segment.py`
  — 위 학습된 체크포인트를 그대로 재사용, **추론만** 변경. 경계
  (t-1, t) 를 turn t 도착 시점의 causal window `u_{..t}` 하나로 *1회*
  결정. 미래 발화 0개, 재수정 없음, O(1)/turn. 추가 학습 없음.
- **DTS 평가 제약**: CPU-only 환경에서 SuperDialseg 의 1,322 test
  dialogue × token-classification forward 가 wall-time prohibitive 라
  3 벤치 full online 평가는 수행하지 못함. 대신 각 벤치 50 dialogue
  smoke check 결과 online ≈ offline (ΔScore TIAGE +0.0015, Dialseg711
  −0.0014, SuperDialseg +0.0002) 이어서, **본 paper 표의 RoBERTa Score
  행은 offline 수치를 보고하며 online 변형의 근사로 간주한다** (DTS 표
  caption 의 `$^\sharp$` footnote 참조).

### 4.2.3 LLM-based baselines (SeCom segmenter)

SeCom (Pan et al., 2024) 의 **LLM 기반 segmenter** 를 그대로 사용. 한 session
의 모든 발화를 prompt 로 주고 LLM 이 segment boundary 를 출력하도록 함
(non-streaming, full-session-at-once). 우리는 단지 segment LLM 만 swap.

| LLM | params | Crts slug |
|---|---:|---|
| GPT-4o-mini | ~8B (est.) | `openai/gpt-4o-mini` |
| GPT-5 | (closed) | `openai/gpt-5` (reasoning_effort=minimal) |
| Qwen3.5-122B-A10B | 122B (MoE, active 10B) | `qwen/qwen3.5-122b-a10b` |
| Qwen3.5-27B | 27B | `qwen/qwen3.5-27b` |
| Qwen3.5-4B | 4B | `qwen/qwen3.5-4b` |
| Qwen3.5-2B | 2B | `qwen/qwen3.5-2b` |
| Llama3.2-3B | 3B | `meta/llama-3.2-3b-instruct` |
| Mistral3-3B | 3B | `mistralai/ministral-3b` |

모든 LLM 은 **hybrid-thinking 비활성화** 상태로 호출 (Qwen3.5 → `reasoning_effort=none`,
GPT-5 → `reasoning_effort=minimal`) — apples-to-apples 비교를 위해 reasoning
chain 사용 안 함.

### 4.2.4 Our method: **Hi-OnTop** (Hi-OnTop 의 reduced form)

본 논문의 메인 모델. **메인 비교표 (§4.4, §4.5)** 에서 6 variant 보고:
- 인코더 2종: **MPNet** (`sentence-transformers/multi-qa-mpnet-base-dot-v1`,
  768-d, fp32) / **MiniLM-int8** (`all-MiniLM-L6-v2`, 384-d, ONNX `quint8_avx2`).
- δ\* percentile 3종: **p60 / p70 / p80** (label-free 추정값).
- δ_eff 식 (§3.3): context window m=2, decay ρ=0.7, blend a=0.5.

**Figure H (calibration N convergence ablation)** 만 인코더 3종 사용 —
위 2종에 더해 **MiniLM-fp32** (`all-MiniLM-L6-v2`, 384-d, non-quantized) 를
중간 ground truth 로 포함. 메인 표에는 보고하지 않음.

---

## 4.3 Metrics

### 4.3.1 DTS metrics (§4.4)

세 가지 표준 segmentation metric 의 **mean** + **composite Score**:

- **Pk (Beeferman et al., 1999)** ↓ — segment boundary disagreement
  probability. 낮을수록 좋음.
- **WinDiff (WD)** ↓ — Pk 의 변형, boundary count 도 함께 본다. 낮을수록 좋음.
- **F1** ↑ — boundary set 의 micro-F1 (pred boundary set vs gold).
- **Score** ↑ — *composite metric*:

  $$\text{Score} = 0.5 \cdot \text{F1} + 0.25 \cdot (1 - \text{Pk}) + 0.25 \cdot (1 - \text{WD})$$

  (Methods/README.md 정의 동일.) Pk/WD/F1 의 균형을 단일 숫자로 보기 위한 도구.

평가 라이브러리는 모두 `segeval` (autoseg 의 wrapper 사용), Def-DTS bundle 데이터.
**예외: RoBERTa** — 학습 데이터가 SuperDialseg train 만 사용하므로 metric 도
원논문 5.3 의 *sliding window = avg seg length / 2* 방식 official Pk/WD 사용
(autoseg segeval 미사용). Score 공식은 동일.

### 4.3.2 Downstream QA metrics (§4.5)

LLM-based QA 응답에 대해 6 metric 보고:

- **GPT4Score** ↑ — `openai/gpt-4o` judge 가 1–10 점 부여 → ×10 정규화.
  본 논문의 *primary* metric. 출력 풍부, 의미·완전성 평가.
- **BLEU** ↑ (sacreBLEU)
- **Rouge-1 / Rouge-2 / Rouge-L** ↑
- **BERTScore F1** ↑ (`bert_score` 라이브러리 default `roberta-large`, CPU)

Context length 두 종:
- **# Turns** — retrieved 후 chat LLM 에 입력되는 turn 수
- **# Tokens** — 같은 입력의 tokenizer 토큰 수

Latency 두 종 (ms/turn):
- **Pre. (Preprocess)** — 결정 *이전* 의 표현 추출 (encoder forward + 어휘 연산)
- **Seg. (Segmentation decision)** — preprocess 출력 이후의 결정 로직만.
  LLM segmenter 의 경우 API 1회 monolithic latency.

---

## 4.4 Experimental Setup

### 4.4.1 환경 (DTS + downstream 공통)

- **CPU only**: AMD Ryzen 9 7950X · 16 logical cores · 16 GB RAM · WSL2
  Linux 5.15 (no GPU). 일부 임베딩 모델 (MiniLM-int8 ONNX) 에 GPU 백엔드
  API 가 없는 점 + 환경 일관성을 위해 모든 latency 측정 / segmentation 을
  CPU 로 통일.
- **batch_size = 1** (per-turn streaming 시뮬레이션). 단, SeCom 의
  segmenter 가 session 단위로 LLM 1회 호출하는 부분은 batch=1 이 무의미
  (한 session 의 모든 발화가 한 prompt 안에 들어감).
- **idle CPU 측정**: latency 는 다른 background job 이 없는 상태에서 측정 —
  pipeline (LLMLingua-2 compress 등) 과 동시 측정 시 CPU contention 으로
  noise 가 끼는 것을 별도 확인했다.

### 4.4.2 δ\* calibration (Hi-OnTop)

DTS 벤치마크 평가용 calibration 은 *각 벤치의 train split* 에서 δ_eff 를
모아 percentile 값을 δ\* 로 채택 — calibration 데이터셋·split 분량:

| Benchmark | Train (calibration) | 전체 train pool | 비고 |
|---|---:|---:|---|
| TIAGE | 300 | 300 (cap=전체) | dataset 자체가 400 dialogue |
| Dialseg711 | 498 | 498 (cap=전체) | dataset 711 dialogue |
| SuperDialseg | 400 | 400 (cap=400) | dataset 1722 dialogue, 400 으로 cap |

- **Layer 1 — percentile rank 선택** (p60/p70/p80): segmentation 벤치
  (TIAGE/Dialseg711/SuperDialseg) 의 F1·Score 로 선택. 다운스트림
  (Long-MT-Bench+) 의 GPT4Score 로 선택하지 **않음** — in-sample
  selection bias 회피.
- **Layer 2 — δ\* 절대값**: deploy 도메인의 *unlabeled* δ_eff 분포의
  해당 percentile (label-free, leakage 없음). **DTS 평가용** 은 각 벤치
  (TIAGE/Dialseg711/SuperDialseg) train split 에서 percentile 채택 — 벤치별로
  값 다름 (dts_result.md 의 δ\* calibration 표 참조). **다운스트림 (MTB+)
  평가용** 은 MTB+ 자체의 unlabeled δ_eff pool 에서 percentile 채택:
  MPNet δ\*=0.4799, MiniLM-int8 δ\*=0.7049 (인코더별 분포 영역이 달라 절대값
  다름; Observation 1 §3.3 참조).
- **p60 / p80 ablation**: percentile 민감도 보고용 (메인 모델은 p70).

### 4.4.3 Random seed

- Hi-OnTop 의 calibration 자체는 deterministic (전체 train pool 에서
  percentile 계산) → seed 무관.
- **CSM fine-tuning** = `scripts/train_csm_hf.py`, **seed=42 single run**,
  bert-base-uncased backbone (§4.2.1 참조). 시간 제약상 std 미보고.
- **RoBERTa fine-tuning** = `methods/RoBERTa/offline/train.py`,
  **seed=42 single run**, roberta-base backbone (§4.2.2 참조). 시간
  제약상 std 미보고.
- LLM segmenter 호출은 stochastic (temperature 1.0 기본) 이지만 1회 실행 보고.

---

## 4.5 Downstream Task — Long-MT-Bench+ Application

### 4.5.1 Experimental setup (downstream)

#### 데이터셋
- **MTB+** (Long-MT-Bench+) — 11 conv, 288 QA.
- 평가 split 은 SeCom 원논문 그대로 사용 (cross-validation 없음, 본 데이터셋이
  evaluation-only 로 설계됨).

#### Baselines
DTS 평가용 6 unsup + 1 sup + Hi-OnTop 6 variant 에 더해, SeCom 의 원래
**LLM 기반 분절 (SeCom segmenter)** 들을 함께 비교:

- GPT-4o-mini-Seg, **GPT-5-Seg**, Qwen3.5-{2B/4B/27B/**122B-A10B**}-Seg,
  Llama3.2-3B-Seg, Mistral3-3B-Seg

추가로 두 가지 *no-DTS* upper/lower bound 도 보고:
- **Zero History** — chat LLM 에 dialogue history 0 제공 (lower bound).
- **Full History** — 전체 dialogue (~65 turn / 22.7K token) 그대로 전달
  (upper bound, retrieval/segmentation 없이).

#### Metrics
§4.3.2 참조 — primary = **GPT4Score** (gpt-4o judge), secondary = BLEU /
Rouge-{1,2,L} / BERTScore F1. Context length (# Turns, # Tokens) 와 segmentation
latency 도 함께.

#### Setup (pipeline)
SeCom 표준 pipeline 의 *segmentation 단계만* swap, 나머지는 원논문 그대로:

| 단계 | 도구 | HP |
|---|---|---|
| ① Segmentation | (swap 대상) | 각 method 자체 default |
| ② Compression | LLMLingua-2 (local HF, ~2 GB CPU) | `compress_rate = 0.75` |
| ③ Retrieval | dense bi-encoder (multi-qa-mpnet) | `topk = 1` |
| ④ Chat (응답 생성) | `openai/gpt-4o-mini` | workers=8, temperature=0.0, max_tokens=512 |
| ⑤ Eval (judge) | `openai/gpt-4o` | workers=8, JUDGE_PROMPT 동일 |

- **온라인의 범위 제한**: 본 논문은 *segmentation* 만 online (causal,
  prefix-only) 으로 한정. retrieval / compression / chat 은 SeCom 원본 그대로
  offline (segment 결정 이후의 후속 단계). 다운스트림 표의 Pre./Seg. latency
  는 segmentation 단계만의 ms/turn 이다.
- **Hi-OnTop p 선택**: percentile p ∈ {60, 70, 80} 은 **DTS 벤치마크
  (§4.4) 의 F1/Score** 만 보고 선택함 — MTB+ 의 GPT4Score 를 보지 않음.
  즉 본 표의 p70 best 결과는 **out-of-distribution selection** 의 산물.
  이 selection protocol 은 §4.4.2 Layer 1 과 동일.

실제 수치 및 해석은 **§4.6~§4.9** 본문 narrative 참조 — Hi-OnTop 6 variant + algorithmic
unsup 4종 + supervised RoBERTa + LLM 8종 + history bound 2종 의 11×17 비교표.
DTS 결과는 `outputs/reports/dts_result.md`, 다운스트림은
`outputs/reports/downstream_task.md` 에 저장되어 있다.

---

# Paper §4 본문 narrative (paper-ready draft)

> 위 §4.1–§4.5 는 *spec 형식의 supporting doc* 이고, 아래는 paper 본문에 그대로
> 넣을 **narrative form** 의 draft 다. 각 paragraph 는 "결과 한 문장 → 다음
> 문장이 기여로 주워담는" 구조로 작성됐다.

## §4 Experiments — Intro paragraph

본 장은 Hi-OnTop 의 세 가지 기여를 실험으로 검증한다. **(i) 완전한 online
설정** — 미래 발화를 보지 못하는 prefix-causal 조건에서의 분절 품질과 per-turn
latency 를 §4.6 의 DTS 평가에서 보고한다. **(ii) 라벨 없는 도메인 적응 절차** —
deploy domain 의 $\delta_{\text{eff}}$ 분포만으로 $\delta^{*}$ 를 결정하는
percentile calibration 의 효율을 §4.7 에서 oracle / supervised tuning 과의
gap-vs-$N$ convergence 로 정량화한다. **(iii) downstream 플러그인 검증** —
원본 SeCom pipeline 의 segmentation 단계를 Hi-OnTop 으로 교체했을 때 응답
품질 (Long-MT-Bench+, gpt-4o judge) 과 LLM 분절기와의 boundary 일치도를
§4.8 에서 보고한다. 모든 latency 는 동일한 CPU 환경에서 측정되며, fine-tuned
모델 (CSM, RoBERTa) 은 시간 제약으로 random seed 1회 학습값을 보고한다.

## §4.6 Main Results — DTS segmentation

세 DTS 벤치 (TIAGE / Dialseg711 / SuperDialseg) 에서 *online prefix-causal*
조건의 Score 와 latency 를 Table~\ref{tab:dts} 에 보고한다. **Hi-OnTop 의
MPNet variant 는 mean-3 Score 0.497, MiniLM-int8 variant 는 0.485 로, 가장
높은 unsupervised online baseline (CSM-Style mean-3 0.450) 을 각각 0.047
및 0.035 능가**한다. Per-bench 로 보면 MPNet 은 세 벤치 모두에서 unsup
online baseline 의 최고치를 넘어서며, MiniLM-int8 은 TIAGE / Dialseg711
에서 능가하되 SuperDialseg 에서는 CSM-Style (0.443) 보다 0.042 점 낮다
(Table~\ref{tab:dts}). 이는 *learned dynamics 없이 cosine 거리만으로*
paper §3.2 의 dual-time-scale boundary distance 를 구성한 설계 (기여 (i))
가 online 조건에서 평균적으로 효과적임을 보여준다. **MiniLM-int8 은 fp32
MiniLM 과 mean-3 Δ +0.003 (사실상 동등) 이라 quantization-free deployment
가 가능**하며, 인코더 동등성 비교는 Appendix~B 에서 다룬다. 한편 *online*
변형의 효과는 baseline 별로 비대칭이다: **TextTiling 의 online streaming
변형** 은 lexical overlap (bag-of-words cosine) 만으로 boundary 를
판정하므로 표현 변동이 큰 대화에서 곡선 첨예성을 잃어 Score 가 떨어지는
경향이 있고, **GreedySeg 의 delay-2 변형** 은 BERT forward 가 단일 발화에
국한되어 offline 의 sliding window 평균이 제공하던 안정성을 일부 상실한다.
**CSM 의 delay-2 변형** 은 right context 를 2-turn 으로 잘라 running
threshold 가 천천히 수렴하므로 SuperDialseg 같은 dense-boundary 데이터에서는
offline Score 의 97% 까지 회복하지만 TIAGE 에서는 84% 에 머문다 (Table
caption 의 \texttt{$^{\ast}$} 주석 참조). **Latency 측면** 에서 Hi-OnTop
의 per-turn decision 비용 (Seg.) 은 0.056~0.069 ms 로 모든 baseline 중 가장
낮고, encoder forward (Pre.) 까지 합쳐도 MiniLM-int8 으로 11.7 ms / turn
이라 CSM-Style 의 283 ms 대비 약 24× 빠르다. 이는 *연속적 graded readout*
($g_t = \delta_{\text{eff}} / \delta^{*}$) 자체의 계산이 O(1) 임을 paper
§3.2 에서 주장한 것을 직접 확인한다.

## §4.7 도메인 적응 절차 via Approximation of Threshold Calibration

$\delta^{*}$ 를 결정하기 위해 가장 작은 calibration 풀 (TIAGE 의 train 300)
을 기준으로 $p_x \in \{50, 55, \ldots, 95\}$, supervised-best,
test-side oracle 의 세 protocol 을 모든 (인코더, 벤치) 조합에 적용한
결과, **best percentile 과 그 절대값 $\delta^{*}$ 는 인코더와 벤치 조합마다
달라진다** (Appendix~A.2, e.g., MPNet-TIAGE 의 oracle $\delta^{*}$=0.543 vs
MiniLM-int8-TIAGE 의 0.777). 이는 새로운 배포 환경에서 *best-$p$ 선택* 과
*그 best-$p$ 로 threshold 채택* 의 두 단계를 다시 수행해야 함을 의미하지만,
두 단계 모두 deploy domain 의 *unlabeled* $\delta_{\text{eff}}$ 분포만 보면
충분하므로 paper §3.3 의 label-free 도메인 적응이라는 설계 의도가 *실제로*
적용 가능함을 보여준다 (기여 (ii)).

Figure~H 는 이 도메인 적응 절차의 **수렴 속도** 를 calibration $N$ 의 함수로
보고한다. Y 축은 percentile heuristic ($p_{60}/p_{70}/p_{80}$) 의 Score 가 ref
(oracle 또는 supervised best) 에 비해 얼마나 떨어지는지를 나타내는 gap
이고, 우리는 gap 이 $N{=}300$ 값의 $\pm 0.002$ 이내로 *유지* 되는 가장
작은 $N$ 을 **knee of the convergence curve** ($N_{\text{conv}}$) 로 정의한다.
encoder-averaged oracle gap 의 경우 $N_{\text{conv}}$ 는 TIAGE 17–75,
Dialseg711 30–40, SuperDialseg 10–17 사이에서 percentile 별로 안정화되며,
**$N{=}75$ 만으로 모든 (벤치, $p_x$) 조합이 본인의 final 값 근방에 들어온다**.
즉 라벨 없는 새로운 도메인에 적응하는 데 평균 *수십 개의 dialogue* 만 있으면
충분하다. supervised gap 은 oracle gap 보다 더 느리게 수렴하고 ($N_{\text{conv}}$
50–300) 절대값도 더 작으므로, percentile 은 supervised tuning 의 성능
근방에 매우 빠르게 도달함을 확인했다.

## §4.8 Downstream Evaluation — Plug-in into SeCom

### Setup
SeCom (Pan et al., 2024) 의 표준 segmentation-compression-retrieval-chat
pipeline 의 *segmentation 단계만* Hi-OnTop 으로 교체하고 나머지 단계 (LLMLingua-2
compression, mpnet retrieval top-1, gpt-4o-mini chat, gpt-4o judge) 는
원본을 그대로 사용한다. 평가 데이터는 Long-MT-Bench+ (n_conv=11, n_qa=288)
이며, primary metric 은 GPT4Score (gpt-4o judge, 0–100 정규화), secondary 는
BLEU / Rouge / BERTScore F1 이다. 비교 baseline 은 §4.2 의 알고리즘적
unsup 4종, supervised RoBERTa, 그리고 SeCom 원본의 *LLM 기반 segmenter*
8 종 (GPT-5, GPT-4o-mini, Qwen3.5-{2B/4B/27B/122B-A10B}, Llama3.2-3B,
Mistral3-3B) 이다. Hi-OnTop variant 의 percentile $p$ 는 *§4.6 의 DTS Score*
기준으로 사전 선택 ($p_{70}$), 즉 MTB+ 의 GPT4Score 는 selection 에 사용하지
않았다 (out-of-distribution selection protocol).

### Evaluation
**Hi-OnTop ($p_{70}$, MiniLM-int8)** 은 GPT4Score **79.90** 으로,
*full history* upper bound (77.92) 와 *zero history* lower bound
(42.12) 사이에서 LLM 분절기 cluster (GPT-5 80.63 / Qwen3.5-27B 81.28 /
Qwen3.5-122B-A10B 80.83) 의 약 1~2 점 이내에 들어왔다. retrieval context
는 query 당 평균 **3.00 turn (863 token)** 으로 LLM 분절기 cluster
(평균 약 3 turn / 863~892 token) 과 동급이며, MTB+ context (긴 발화)
에서 측정된 segmentation latency 는 **45.70 ms/turn (Pre.) + 0.069 ms/turn
(Seg.)** 로 GPT-4o-mini-Seg 의 646 ms/turn 대비 약 **14× 빠르다**
(Pre. + Seg. 총합 45.77 ms vs 646 ms). 이는 segmentation 단계를 *작은
인코더 + 단순 threshold* 로 대체해도 응답 품질이 LLM 분절과 *최소한
비교 가능한 수준* 이며, 대신 latency 가 두 자릿수 ms 안으로 들어옴을
의미한다 (기여 (iii) 의 플러그인 검증).

### Results
응답 품질 비교를 *boundary 위치 합의도* 차원에서 검증하기 위해 Figure~F
는 모든 method 쌍의 pairwise boundary F1 (per-conversation 평균) 히트맵을
제공한다. boundary F1 은 두 method 의 boundary 집합 간 set-F1 을
대화별로 계산해 macro-평균한 값이며, 1.0 에 가까울수록 두 분절이 같은
turn 사이에 boundary 를 둔다. **MTB+ 에서 Hi-OnTop ($p_{70}$, MiniLM-int8)
의 boundary 양상은 LLM 상위 3 모델과의 pairwise F1 이 GPT-5 0.912,
Qwen3.5-122B-A10B 0.905, Qwen3.5-27B 0.914 로 모두 0.90 이상이고,
약체 LLM 인 GPT-4o-mini 와는 0.748 로 떨어진다** (Hi-OnTop의 다른
percentile $p_{60}$ 은 top3 와 0.85 근방, $p_{80}$ 도 0.85 근방).
우리 모델과 LLM 상위 분절기 모두 *이 데이터에서* SeCom 기반 long-context
응답 성능이 상위권에 들었으므로, 이 결과는 *이 다운스트림 데이터에 한해*
Hi-OnTop 의 boundary 결정이 강한 LLM 분절기와 유사한 위치에 수렴한다고
**보고할** 수는 있다. 단, 같은 비교를 *서사형 데이터셋 (SHARE)* 에서
수행하면 동일 percentile 의 pairwise F1 이 크게 떨어지므로 (Limitations
참조), "무라벨 도메인에 *항상* 적응 가능" 이라는 식의 일반화는 본 결과
만으로 정당화되지 않는다. 본 논문은 따라서 *이 다운스트림 (MTB+) 에
한정한* 관찰로 보고한다.

이 *boundary 합의도 기반 가정* 위에서, Figure~P 는 downstream 영역에서의
**best-$p$ calibration 의 수렴 속도** 를 보고한다. 강한 LLM 분절기
3 종 — GPT-5 / Qwen3.5-27B / Qwen3.5-122B-A10B — 을 reference 로 설정하고,
MTB+ 의 11 conversation 을 *conv-level* 로 5 train / 6 test 로 split
(SPLIT\_SEED=0 으로 고정), train conv 들의 $\delta_{\text{eff}}$ pool 에서
$N \in \{3, 10, 30, 50, 75, 100, 150, 200\}$ 발화를 sub-sampling 한다.
각 $(N, \text{seed})$ 마다 percentile sweep $p \in \{60, 61, \ldots, 80\}$
(step 1, 21 values) 위에서 test conv 위 boundary F1 을 계산해
$p_{\text{best}}^{(N, s)} = \arg\max_p F_1^{\text{test}}(p)$ 를 잡고,
Y-axis 는 $F_1^{\text{oracle}} - F_1^{p_{\text{best}}}$ 의 **50 random
seeds 의 mean $\pm$ std** 다 (즉 figure 의 error band 는 50-seed bootstrap
의 variance, $N \geq$ train pool size 인 경우는 deterministic 1 회).
3 encoder $\times$ 3 reference LLM $\times$ 8 $N$ values = 72 cell, 각 cell
당 350 subsample (7 $\times$ 50 seeds) $\times$ 21 percentile = 529K F1
evaluations 의 결과다. LLM-ref gap 은 $N_{\text{conv}}$ 가 수십~백 단위에서
안정화되어, paper §3.3 의 *얇은 calibration* 가정이 downstream 의 무라벨
영역에서도 (이 데이터에 한해) 성립함을 시사한다. 마지막으로 best-$p$
선택 + $\delta^{*}$ 계산 자체의 latency 는 한 인코더 forward 위에서
percentile sweep + lookup 이므로 encoder 비용에 *무시 가능한* 수준이
추가될 뿐이다.

## §4.9 Ablation Study

각 구성 요소의 기여를 정량화하기 위해 (a) context window 크기 $m$, (b)
context decay $\rho$, (c) blend weight $a$ 의 3 개 hyperparameter 각각을
변경한 ablation 을 수행했다. 인코더, 벤치 조합 90 종 위에서 oracle
$\delta^{*}$ 를 기준으로 본 결과, **default $(m, \rho, a) = (2, 0.7, 0.5)$ 가
oracle 기준 rank 1 / 90 (Δ mean-3 Score 0.000)** 이고 *deployable* best-$p_x$
기준 rank 3 / 90 (Δ mean-3 Score $-0.0016$) 로, paper §3.2 의 default 가
*per-dataset tuning 없이도* 90 종 중 거의 정상에 위치함을 확인했다. 자세한
설정과 분석은 Appendix~C 에 수록한다.

---

\section*{Limitations}

본 평가는 모두 *짧은 일상 대화* 벤치마크 (평균 12–26 turn) 위에서
수행되었으며, 긴 서사형 대화 (예: SHARE) 처럼 boundary 가 의미적으로 *덜
sharp* 한 데이터에서는 percentile 기반 도메인 적응의 효과가 본 결과와
다르게 나타날 수 있다 (Figure~F 와 SHARE 의 pairwise F1 비교 참조).
또한 downstream 플러그인 검증은 *Long-MT-Bench+* 한 데이터셋에 한해
수행되어, 다른 long-context QA 데이터에서의 일반화는 후속 작업의 과제이다.
