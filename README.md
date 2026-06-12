# Hi-OnTop — Online Topic Segmentation for LLM Conversations

> 임베딩 기반 **0-lag 실시간 화제분절기**. 인지과학 Structured Event Memory
> (Franklin et al. 2020) 의 sticky-CRP prior + 온라인 local-MAP 추론을 대화
> 화제 경계 감지로 재해석한다. 인코더는 고정(fine-tuning 없음), turn 당 O(1)
> 업데이트로 look-ahead 없이 경계를 판정 — 버퍼드 LLM 분절기 대비 우월한
> **quality–latency–cost tradeoff** 를 목표로 한다.

---

## 핵심 아이디어

- **계보**: SEM / SEM2 (`nicktfranklin/SEM2`) 의 scene segmentation 을 대화
  화제분절로 계승. sticky-CRP (sCRP) prior + Gaussian likelihood + local-MAP.
- **온라인 / 0-lag**: 발화가 도착하는 대로 turn 단위로 임베딩 → 현재 화제와의
  prediction-error 신호로 경계 여부를 즉시 판정. 미래 turn look-ahead 불필요.
- **calibration-free 임계**: 코퍼스별 정답 경계 없이도 동작하도록 percentile
  기반 적응 임계(μ + cσ) 를 사용. (Score-vs-percentile 곡선으로 일반성 검증.)
- **인코더 교체 가능**: MiniLM(int8) / bge / Sentence-T5 / multilingual 등.
  기본은 경량 MiniLM-int8 (실시간 latency 목적).

현재 진행 중인 문제·설계·시도 이력은 `handoff/` 폴더, 알고리즘 정의는
`context/methodology/` 를 본다.

---

## 레포 구조

```
Hi-OnTop/
├── CLAUDE.md                 작업 규칙 (환경 분리, 커밋, Step 완료 프로토콜)
├── README.md                 이 파일
├── plan.md                   구현 로드맵 (segmentation Phase)
├── RelatedWork.md            관련 연구 정리
├── API_GUIDE.md              OpenAI-호환 LLM 엔드포인트 사용 가이드 (LLM baseline용)
├── pyproject.toml / uv.lock  project metadata + deps (uv-native, hatchling)
│
├── context/                  설계 문서
│   ├── 00-sem-paper.md           SEM 논문 정리 (계승/대체/폐기 매핑)
│   ├── sem-equations.md          SEM 식 원본 reference
│   ├── 01-hi-ontop-design.md     Hi-OnTop 분절 설계 확정본
│   ├── 02-math-model.md          수식 확정본 + HP regime
│   ├── 03-architecture.md        모듈 구조
│   ├── 04-benchmarks.md          벤치마크 메타 (토픽 경계 평가 축)
│   ├── 05-open-questions.md      열린 질문
│   ├── 06-decision-log.md        설계 결정 이력 (append-only)
│   └── methodology/              **버전별 방법론 (최우선 관리)** + infrastructure
│
├── src/hi_ontop/             코어 구현 (+ 버전 variant 누적)
│   ├── embedding.py              인코더 wrapper (local / API backend)
│   ├── topic.py                  centroid + diag σ² + Welford 온라인 업데이트
│   ├── scrp.py                   sticky_crp_unnormed (SEM 식 1)
│   ├── sem_core.py               HiOnTopSegmenter.assign() — online MAP 루프
│   ├── hi_ontop.py               HiOnTop reduced form (δ_eff streaming)
│   ├── hi_ontop_v2.py            adaptive-threshold variant
│   ├── hi_ontop_cr.py            commit-and-refine + run-length 적응 β (deploy)
│   ├── hi_ontop_lex.py           TextTiling-style lexical variant
│   ├── secom_adapter.py          SeCom 백엔드로 Hi-OnTop 분절기 주입
│   ├── next_embed_head.py        v4.3.2 δ_model (next-embedding regressor)
│   ├── event_rnn.py              v3.3.1 GRU event model
│   ├── config.py                 HP config loader (configs/hiontop.json)
│   ├── baselines/                온라인 분절 baseline (texttiling/greedyseg/graphseg/csm)
│   └── sem_core_v*.py / topic_v*.py  연구 variant 이력
│
├── methods/                  분절 baseline 원본 (TextTiling/GreedySeg/GraphSeg/CSM/RoBERTa)
├── scripts/                  실행/분석 스크립트 (run_topiocqa_*, run_tiage_*, ami_*, 등)
├── tests/                    pytest (core sCRP/topic/sem_core + variant + baseline)
├── notebooks/                얇은 wrapper (Colab 편의용, 선택적)
├── templates/                experiment-log / module 템플릿
├── archive/                  의도적으로 폐기한 산출물 (legacy_sem_ablation 등)
├── outputs/                  실험 산출물 (experiments/REPORT.md, reports/, figures/, design/)
└── SEM/                      SEM2 레포 (참조 전용)
```

> Portability 원칙: `notebooks/` 는 `scripts/*.py` 를 호출하는 얇은 wrapper.
> 통째로 삭제해도 `python scripts/X.py` 직접 실행으로 모든 실험 가능.

---

## 평가

**평가 축 = 토픽 경계 감지** (`context/04-benchmarks.md`):

| 벤치마크 | 성격 | 지표 |
|---|---|---|
| TopiOCQA | factoid topic 경계 | turn-transition F1 |
| TIAGE | chit-chat topic 경계 | F1 |
| AMI | 회의 화제 경계 (음성/streaming) | Pk / WindowDiff / F1 |
| dialseg711 / superseg | dialogue segmentation | Pk / WD / F1 |

**비교 baseline**: 고전(TextTiling / BayesSeg / GreedySeg / GraphSeg / CSM /
RoBERTa-supervised), LLM 프롬프팅(Def-DTS), 그리고 streaming-buffer LLM 분절기
(buffer↓ 품질 열화 곡선 vs Hi-OnTop 0-lag).

**downstream 적용**: SeCom-swap 으로 Long-MT-Bench+ QA 품질 비교 — 분절 결과를
compress/retrieve 파이프라인에 주입 (LLM-agnostic).

---

## 외부 레포 / 데이터 (clone 필요, gitignored)

**SEM2 — 알고리즘 참조** (이미 레포에 `SEM/` 으로 포함; 재설치 시):
```bash
git clone --depth=1 https://github.com/nicktfranklin/SEM2 SEM && rm -rf SEM/.git
```

**분절 벤치마크**:
```bash
mkdir -p benchmarks && cd benchmarks
git clone --depth=1 https://github.com/McGill-NLP/topiocqa     # topic 경계 (factoid)
git clone --depth=1 https://github.com/HuiyuanXie/tiage        # topic 경계 (chit-chat)
cd ..
# TopiOCQA dev: cd benchmarks/topiocqa && python download_data.py \
#   --resource data.topiocqa_dataset.dev --output_dir .
```

**DTS baseline 외부 레포** (`methods/`·`scripts/run_defdts_*`·`*_online`·`*_prefix` 가 참조):
```bash
# 1) Def-DTS (tiage/dialseg711/superseg jsonl + segeval metric 원천)
git clone --depth=1 https://github.com/ElPlaguister/Def-DTS.git benchmarks/Def-DTS
# 2) SuperDialseg 툴킷 (TextTiling/BayesSeg 원본 offline)
git clone --depth=1 https://github.com/Coldog2333/SuperDialseg.git benchmarks/superdialseg
```

**BayesSeg 빌드** (offline·online, 상세 = decision-log 2026-05-20):
```bash
sudo apt install -y default-jdk ant
git clone --depth=1 https://github.com/jacobeisenstein/bayes-seg /tmp/bs
B=benchmarks/superdialseg/src/super_dialseg/models/bayesseg
cp -r /tmp/bs/lib "$B/lib"; cp /tmp/bs/build.xml "$B/build.xml"; mkdir -p "$B/classes"
cd "$B" && ant build && cd -
```

**GraphSeg** 는 GloVe 6B.300d 추가 download 필요:
```bash
mkdir -p benchmarks/glove && cd benchmarks/glove && \
    curl -LO https://nlp.stanford.edu/data/glove.6B.zip && \
    unzip glove.6B.zip && rm glove.6B.zip && cd ../..
uv run python -c "import nltk; [nltk.download(p) for p in ('averaged_perceptron_tagger','averaged_perceptron_tagger_eng','brown','punkt_tab')]"
```

**CSM 학습** (DailyDialog 원본 + lxing532 코드) 은 `colab_csm_train.ipynb` 가
Colab 에서 자동 clone + 의존성 핀 + STEP-RESUME 패치.

**참고용 (clone 불필요, 코드 읽기 전용)**:

| 레포 | 용도 |
|---|---|
| https://github.com/flippedAben/texttiling | TextTiling online 구현 참고 |

---

## 빠른 시작

```bash
# 0. 환경 셋업 (uv 0.8+ 필요: https://docs.astral.sh/uv/)
uv sync                                          # .venv + deps + hi_ontop editable install
# (Colab 사용자는 setup_colab.ipynb 실행)

# 1. core 테스트
uv run python -m pytest tests/ -q

# 2. 분절 평가 예시
uv run python scripts/run_tiage_segmentation.py
uv run python scripts/run_topiocqa_segmentation.py

# 3. Step 완료 처리 (CLAUDE.md "Step 완료 프로토콜" 참조)
uv run python scripts/check_step_done.py
```

LLM baseline (Def-DTS / streaming-buffer LLM) 실행 시 OpenAI-호환 엔드포인트
설정은 `API_GUIDE.md` 와 `.env.example` 참조.

---

## 제약 (CLAUDE.md 전문 참조)

- **No fine-tuning** (외부 LLM 학습 금지; 인코더도 고정)
- **PyTorch only** (TensorFlow/Keras 금지)
- **SEM2 코드 복사 금지** (참조만 허용)
- **설계 변경은 반드시 `context/06-decision-log.md` append 후**
- **환경 분리**: `setup_colab.ipynb` 만 Colab 의존 허용, 그 외 파일은 로컬/git 기준 동작
