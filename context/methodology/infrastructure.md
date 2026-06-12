
# 공용 Infrastructure (cross-cutting design)

버전 간에 공유되는 *non-version-specific* 인프라/구현 설계. 사소한 cache 정책, locking, encoding 단위 같은 항목도 여기 누적.

각 항목은 다음 형식:
- 무엇 (정의)
- 어디 (코드 위치)
- 왜 (도입 동기)
- 행동 영향 (어느 method 가 받는가)
- 알려진 한계 / 변형 후보

---

## 1. `EncoderCache` — segmentation 임베딩 캐시

- **무엇**: `(sample_id, kind)` 키로 conversation 별 corpus 임베딩을 캐싱.
- **왜**:
  - 같은 대화의 turn 임베딩을 여러 번 인코딩하지 않도록. 캐시 없으면 재사용 시마다 전체 turn 을 재인코딩.
  - encoder 는 thread lock 으로 직렬화돼 있어 (`QueryEncoder.encode` 의 global Lock), worker 수를 늘려도 인코딩 비용은 그대로.
- **행동 영향**:
  - 의미 자체는 안 바꿈 — 동일한 임베딩을 한 번만 계산.
- **Locking 패턴**: per-key build lock 을 outer `_dict_lock` 안에 둠. concurrent worker 가 같은 sample 에 대해 중복 인코딩 안 하도록.
- **알려진 한계 / 변형 후보**:
  - 현재 sample 단위로 인메모리. conversation 간 공유 캐시 없음. 다른 sample 이 같은 텍스트 chunk 를 갖고 있어도 재인코딩.

---

## 3. Encoder lock (단일 thread 직렬화)

- **무엇**: `QueryEncoder.encode` 가 global `threading.Lock` 으로 인코딩 호출을 직렬화.
- **어디**: `src/hi_ontop/embedding.py` (간접 — `EncoderCache` docstring 에 기록).
- **왜**: 임베딩 모델 자체가 multi-thread 안전하지 않음 (HuggingFace tokenizer + torch model). 한 thread 만 동시에 forward.
- **행동 영향**:
  - `--workers N` 늘려도 *인코딩 단계* 는 직렬. LLM 호출만 병렬.
  - Hi-OnTop 의 segmentation 단계는 conversation 단위 직렬 → encoder 가 직렬이라도 큰 문제는 안 됨.
- **알려진 한계**:
  - 한 sample 의 인코딩 시간이 wall-clock 직선적으로 들어감.
  - GPU encoding 으로 가도 lock 은 그대로 (일관성 보장).

---

## 4. LLM `--no-thinking` (reasoning bypass)

- **무엇**: OpenAI-compatible `extra_body` 로 두 가지 키를 동시에 보냄:
  ```
  extra_body = {
      "chat_template_kwargs": {"enable_thinking": False},  # vLLM
      "reasoning": {"enabled": False},                     # Crts / OpenRouter
  }
  ```
- **어디**: LLM-baseline segmenter 의 chat 호출에서 같은 `extra_body` dict 를 LLM kwargs 에 주입.
- **왜**:
  - qwen / DeepSeek / Crts proxied 모델들은 reasoning mode 가 default ON 이면 답을 `message.reasoning` 또는 `<think>...</think>` 로 보냄.
  - LLM 어댑터는 `message.content` 만 추출 → reasoning 모델이면 `content=None` 이 와서 모든 응답이 빈 문자열.
  - 2026-05-07 사고: `--no-thinking` 빠진 채 LLM-baseline 비교를 돌려서 모든 run 의 `error_rate=1.00`. 결과 무의미.
- **행동 영향**:
  - 모든 LLM-baseline segmenter 의 LLM 호출.
  - hyperparameter 가 아니라 *환경 호환성 플래그*.
- **권장**: 외부 OpenAI-compatible endpoint (Crts / OpenRouter) 사용 시 *항상 켤 것*. 새 LLM-baseline script 작성 시 누락하지 않도록 기본 포함.

---

## 8. 산출물 디렉토리 구조

top-level 구조:

```
outputs/                  # 모든 active/historical generated
├── experiments/             # segmentation 러너 산출 (sweep / ablation / comparison). self-contained.
│   └── <name>/<run_label>/{run.log, exit_code.txt, results/...}
├── runs/                    # 단독 ad-hoc 실행 데이터 (날짜별 누적)
│   └── _misc/               # smoke / scratch
├── reports/                 # 독립 분석 MD (committed)
└── design/                  # 설계 문서 (committed)

archive/                  # 의도적으로 폐기한 것만
└── README.md                # "왜 버렸는지"
```

**원칙**:
- 새 experiment = 분절 러너 → `outputs/experiments/<name>/` (self-contained)
- standalone debug = `outputs/runs/`
- 시간 흐른 experiment 데이터 = `outputs/runs/<date>/` 에 누적 (정리 X — 자료)
- `archive/` = *의도적 폐기* 만. "오래됐다" 가 아니라 "더 이상 쓰지 않는다고 결정함" 의 표시.

git 정책: `outputs/experiments/<name>/REPORT.md`, `outputs/reports/*.md`, `outputs/design/*` = committed. `outputs/runs/`, sweep 안의 jsonl 데이터 = gitignored.

---

## 9. SEM2 cold-start gating / dynamics / σ² / fresh-baseline 계열 (v3.3.5~8, 2026-05-17)

- **무엇**: idx374 segmentation 진단에서 파생된 4개 cross-cutting 메커니즘.
  - `f_is_trained` gating (v3.3.5): untrained topic(transition_count<min_transitions_for_pe) = fresh slot 과 동일 L0 → likelihood 동률, prior(λ) 결정. SEM2 복원.
  - persistence+replay dynamics (v3.3.6, `TopicV336`): untrained=직전임베딩(identity), per-topic 독립 EventRNN, topic history 전체 replay(n_epochs), `rnn_ready` ≠ `f_is_trained`.
  - map_variance σ² (v3.3.7): `σ²=(ν₀var₀+n·v)/(ν₀+n+2)`, n≥2 즉시. `pe_var_min_samples` gate 폐기.
  - fresh-baseline `pe_prior` (v3.3.8): L0 가 cos_threshold 아닌 pe_prior(chance PE)에서 도출. non-prev topic = f0-likelihood(SEM2 `k0≠k_prev`).
- **어디**: `src/hi_ontop/sem_core_v33{5,6,7,8}.py`, `src/hi_ontop/topic_v336.py`, `version` dispatch + HP(`min_transitions_for_pe`,`rnn_n_epochs`,`rnn_ready_min_transitions`,`rnn_max_history`,`seed`,`pe_var_df0`,`pe_var_window`,`pe_prior`), `tests/test_sem_core_v33{5,6,8}.py`.
- **왜**: methodology v3.3.5~8.md + decision-log 2026-05-17. 핵심 = v3.3.4 의 young-topic centroid 처벌(chicken-and-egg) → SEM2 충실 복원 연쇄.
- **행동 영향**: v3.3.5~8 전부. v3.3.8 default `pe_prior=1.0`(원칙값)은 idx374 mega-collapse — **작동값은 벤치마크 calibration 대상, N=1 production default 금지**.
- **한계**: v3.3.7 은 #14|15 경험적 반증(보존). v3.3.8 fresh-baseline 은 embedding-공간 의존 HP(SEM2 단일상수 불가). non-prev f0 는 generic-opener 취약.

## 10. 재현성 — segmenter seed (필수, 2026-05-17)

- **무엇**: EventRNN/per-topic 모델 random init 이 v3.3.4/5 까지 unseeded → 동일 입력 다른 분절(앞선 v3.3.4 REPORT 수치 불일치의 정체). v3.3.6+ 는 `seed`(per-topic `manual_seed(seed·100003+topic_id)`, RNG snapshot/restore) 로 결정적.
- **어디**: `TopicV336.__init__`, `HiOnTopSegmenterV33{6,7,8}(seed=...)`, `seed` param(default 0).
- **왜**: CLAUDE.md "모든 randomness seed 고정". 논문 재현성 필수.
- **행동 영향**: v3.3.6~8. v3.3.4/5 는 unseeded(결과 해석 시 명시). 실험 시 seed 보고 의무.
- **한계**: v3.3.4/5 소급 적용 안 함(별 버전).

## 15. Online graph baseline — `GraphSegWindowD` (2026-05-21)

- **무엇**: `src/hi_ontop/baselines/graphseg_window.py` 의 `GraphSegWindowD`
  class. GraphSeg (Glavaš et al. 2016) 의 sentence similarity (IC × GloVe +
  Hungarian) + Bron-Kerbosch maximal clique + sequential merge 3-phase 를
  *window=d 안에서만* 적용. `push() → list[int]` (lag-emission, 1-based),
  `flush()`, `state()`.
- **어디**: `src/hi_ontop/baselines/`, runner `methods/graphseg/online_window.py`
  (3-dataset, Def-DTS 번들, segeval 직접). 의존 = numpy/scipy/networkx +
  NLTK brown (IC table) + GloVe 6B.300d (`benchmarks/glove/`, gitignored,
  외부 download).
- **왜**: Hi-OnTop 의 online 비교표에 *graph-based unsupervised* 카테고리 보강.
  codex 2026-05-21 검증: 전역 graph 본질이 깨졌으므로 **AUXILIARY only**.
- **행동 영향**: 강한 `GraphSeg-online` 명명 금지 (paper 본문 = `GraphSeg-
  inspired bounded-window`, short `GraphSeg-window-d` 는 file/CLI 만). 원본
  GraphSeg paper 결과와 같은 표 등재 금지. TextTiling-streaming / GreedySeg-
  online-delay2 와 같은 latency 표에 *직접 비교 금지* — encoder/연산 카테고리
  다름. 결정성: numpy/scipy/networkx 모두 결정적 → seed 무관.
- **한계**: word frequency 원본 = Wikipedia, 본 구현은 NLTK brown corpus
  (실용 선택). Bron-Kerbosch worst-case 지수 — window_d=10 default, edge
  density 높으면 d=5 권장. boundary lag-emission 의 비가역성 = window 평가
  결과가 후속 window 에서 바뀌어도 *최초 발견 boundary 유지*.

## 14. Online BERT baseline — `GreedySegOnlineDelay2` (2026-05-21)

- **무엇**: `src/hi_ontop/baselines/greedyseg_delay2.py` 의 `GreedySegOnlineDelay2`
  class. SuperDialseg GreedySegmenter (Jiang et al. 2023) 의 BERT cosine score
  공식·HP·argmin greedy 선택을 *그대로 보존* 한 채 streaming 입력 + boundary
  emit lag (right context window_size=2). `push() → list[int]` (1-based,
  lag-emission), `flush() → list[int]`, `state() → dict`.
- **어디**: `src/hi_ontop/baselines/`, runner `methods/greedyseg/online_delay2.py`
  (3-dataset, Def-DTS 번들, segeval 직접). device helper `src/hi_ontop/baselines/
  _device.py` (resolve_device, enable_mps_fallback) + shared utils
  `src/hi_ontop/baselines/_seg_utils.py`.
- **왜**: Hi-OnTop 의 online 비교표에 *zero-shot BERT cosine* 카테고리 보강.
  codex 2026-05-20/21 검증 = "score 공식·greedy 선택 보존 → 강한 명명 OK,
  5행 핵심표 가능 (본 plan 유일)".
- **device-agnostic**: cuda/mps/cpu 한 코드. `--device {auto,cuda,mps,cpu}`,
  default `auto` (cuda → mps → cpu). MPS 사용 시 `PYTORCH_ENABLE_MPS_FALLBACK=1`
  반드시 transformers import 이전 설정 (runner 최상단). model.to(device) +
  tokenizer output device 이동 + model.eval() + torch.inference_mode().
- **행동 영향**: 본 baseline 은 TextTiling-streaming (encoder-free, ~0.01ms)
  과 같은 latency 표에 *직접 비교 금지* — encoder cost 차원이 다름. 별도 § 또는
  열 분리. 결정성 CPU > CUDA > MPS — 논문 reproducibility 시 동일 device
  반복 측정.
- **한계**: 원본 BERT pooling 방식 (CLS vs mean) 확인 못함 (superdialseg
  install 없음), 본 구현 default = CLS. *strict prefix-causal 아님* (delay=2).
  segment finalize 의 실제 lag = max_seg_round + window_size 발화까지 가능.

## 13. Streaming baseline segmenter — `StreamingTextTiling` (2026-05-20)

- **무엇**: `src/hi_ontop/baselines/texttiling_streaming.py` 의 `StreamingTextTiling`
  class. `push(utterance) → list[int]` (이번 호출에서 새로 확정된 boundary
  utterance index 1-based), `flush() → list[int]` (대화 종료시 잔여 gap 처리).
  block-cosine + Welford running mean/std threshold + one-sided causal depth +
  min_gap suppression. per-turn 실질 O(w).
- **어디**: `src/hi_ontop/baselines/`, 진입점 runner `methods/texttiling/online_streaming.py`
  (Def-DTS 번들 jsonl 3 dataset 로드 — tiage/dialseg711/superseg, segeval 직접 호출).
- **왜**: 기존 `run_texttiling_prefix.py` 는 매턴 nltk fresh 호출 = O(t)/turn
  → online baseline 의 "latency 비교값" 정책 위배. 진짜 streaming 버전을 별도
  method 명 (`TextTiling-online-streaming`) 으로 신설 (codex:rescue 2026-05-20,
  decision-log 참조).
- **행동 영향**: TextTiling 비교 시 *3종 구분 강제* — offline (NLTK 전체), prefix-
  recompute (NLTK 매턴 fresh), streaming (자체 구현). 같은 표에 섞지 말 것.
- **한계**: 원본 NLTK TextTiling 점수 재현 안 함 (causal running threshold ≠
  offline global threshold). Pk/WD/F1 = INDICATIVE; latency 가 비교값. class
  default w=10/k=6 (NLTK 호환), runner default w=5/k=3 (tiage 짧은 대화 대응).

---

## 작성 규칙

새 인프라/cross-cutting 설계 항목 추가 시:
1. 위 형식 (무엇/어디/왜/행동영향/한계) 으로 한 섹션 추가.
2. 어느 버전이 영향받는지 명시.
3. 실험에서 이 인프라가 *결과에 영향을 줬으면* 반드시 기록 (예: `--no-thinking` 누락으로 sweep 망친 사례).
