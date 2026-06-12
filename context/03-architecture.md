# 코드 아키텍처

## 프로젝트 루트
`/home/namchailin/Hi-OnTop`

## src/hi_ontop/ (구현 대상, 사건 모델 확정 후 최종 레이아웃 결정)

예상 구조 (사건 모델에 따라 일부 변경 가능):
src/hi_ontop/
├── __init__.py
├── config.py              # 하이퍼파라미터 loader (segmenter 섹션)
├── embedding.py           # encoder wrapper (local / API backend)
├── topic.py               # Topic 클래스 (centroid + diag σ² + Welford)
├── scrp.py                # sticky-CRP prior
├── sem_core.py            # online MAP inference 루프 (HiOnTopSegmenter)
├── sem_core_v33{1..9}*.py # 버전별 segmenter (methodology/README 계보 참조).
│                          #   v3.3.5 f_is_trained gating / v3.3.6 persistence+replay+seed
│                          #   (topic_v336.py) / v3.3.7 map_variance σ² / v3.3.8 pe_prior
├── sem_core_v4*.py        # v4.x 라인 (δ_eff / threshold / encoder swap / δ_model 탐색)
├── hi_ontop.py            # HiOnTop reduced form (δ_eff streaming, v4.1.x)
├── hi_ontop_v2.py         # adaptive μ+cσ threshold 변형
├── hi_ontop_cr.py         # ★ main 분절 모델 — commit-and-refine + run-length 적응 β
├── hi_ontop_lex.py        # lexical-overlap 보정 변형
├── next_embed_head.py     # v4.3.2 δ_model (next-embedding regressor)
├── event_rnn.py           # v3.3.1 GRU event model
├── secom_adapter.py       # SeCom 백엔드로 분절기 주입 (downstream QA용)
└── baselines/             # 온라인 분절 baseline (texttiling/greedyseg/graphseg/csm)

## scripts/ (Phase 1 현재 실재)
scripts/
├── check_step_done.py             # Step 완료 검증 게이트 (CLAUDE.md "Step 완료 프로토콜" 2단계)
├── run_topiocqa_segmentation.py   # Phase 1-3 메인 평가 (TopiOCQA dev F1)
├── run_topiocqa_sweep.py          # 108-config HP grid (α × λ × σ₀²) — Phase 1-4 best HP 탐색
├── run_topiocqa_variants.py       # 5가지 구조 변형 비교 (gauss-origin/global/self, vMF-origin/const)
├── run_topiocqa_anchors.py        # 옵션 A 변형: anchor turn 기반 likelihood
├── run_topiocqa_bigencoder.py     # bge-large 인코더 시도 (Phase 1 추가 탐색)
├── run_topiocqa_contextualized.py # contextualized embedding 시도
├── run_topiocqa_multisignal.py    # 옵션 D escalation 탐색 (multi-signal)
├── run_tiage_segmentation.py      # Phase 1-5 TIAGE test 평가 (persistence + freq-shift 두 점)
├── run_tiage_sweep.py             # Phase 1-6 TIAGE 108-config grid (TopiOCQA sweep mirror)
└── run_clustering_quality.py      # Phase 1-6 옵션 5: V-measure/NMI/ARI 측정 (cosine vs Hi-OnTop 두 HP)

# 이후 분절 러너 다수 추가 (2026-05~): run_tiage_*·run_defdts_*·ami_*·boundary comparison 등.
#   각 러너는 self-contained outputs/experiments/<name>/REPORT.md 산출.

## tests/
tests/
├── test_topic.py        # Topic 클래스 (Welford 온라인 update + Gaussian likelihood)
├── test_scrp.py         # sticky-CRP prior (SEM2 `_calculate_unnormed_sCRP` 수치 매칭)
├── test_sem_core.py     # HiOnTopSegmenter MAP 할당 루프 (prior×likelihood argmax + boundary flag)
├── test_config.py       # HP config loader (segmenter 섹션)
└── test_*_v*.py / baseline 테스트  # 분절 variant + online baseline (texttiling/greedyseg/graphseg)

## 진입점 (분절기)
```python
from hi_ontop.sem_core import HiOnTopSegmenter   # 또는 hi_ontop.hi_ontop_cr.HiOnTopCR

seg = HiOnTopSegmenter(alpha=1.0, lmda=10.0, sigma0_sq=0.01)
for emb in turn_embeddings:           # turn 임베딩이 도착하는 대로
    is_boundary = seg.assign(emb)     # O(1)/turn, look-ahead 없음
```

현재 main 분절 모델은 `hi_ontop_cr.HiOnTopCR` (2026-06-11 승격) — `context/methodology/hi-ontop-cr.md`.

---

## methods/ (2026-05-20 신설)

baseline 의 원본(offline)·Hi-OnTop 수정본(online, prefix-causal) 진입점 정리.
범위: TextTiling, BayesSeg. 방식 A(wrapper, benchmarks 무복사·read-only 유지).
- `methods/texttiling/{offline,online}.py`, `methods/bayesseg/{offline,online}.py`, `methods/README.md`
- 이후 확장: `methods/greedyseg/online_delay2.py`, `methods/graphseg/online_window.py`
- offline=전체대화(원본 알고리즘 호출), online=`scripts/run_*_prefix.py`(검증본) 실행 진입점
- 동일 harness: Def-DTS 번들 데이터 + autoseg Pk/WD/F1 + Score. online=AUXILIARY(codex). 산출 outputs/experiments/<name>/REPORT.md
- **`methods/RoBERTa/{offline/train.py, online/segment.py}`** (2026-05-23 신규)
  — supervised RoBERTa 분절기 (Coldog2333/SuperDialseg EMNLP 2023 Table 3
  `RoBERTa` 충실 재현). `offline` = 학습+평가, 경계마다 미래 포함 ~20 윈도우
  logit 평균. `online` = offline 체크포인트 재사용, 추론만 strict causal —
  경계 (t-1,t) 를 turn t 시점 causal 윈도우 하나로 1회 결정 (미래 0).
  harness 예외 — SuperDialseg 번들 데이터 + 논문 official Pk/WD metric.
  결과 → `outputs/experiments/2026-05-23_roberta_{supervised,online}/`.
decision-log 2026-05-20 참조.

## src/hi_ontop/ segmenter 모델 (2026-05-23 갱신)

```
src/hi_ontop/
├── hi_ontop.py       # HiOnTop — δ_eff streaming 모델 (v4.1.x reduced form). Hi-OnTop 파이프라인용 유지
├── hi_ontop_v2.py    # HiOnTopV2(HiOnTop) — adaptive μ+cσ threshold + adaptive_boundaries()
├── hi_ontop_cr.py    # ★ promoted main 분절 모델 (2026-06-11): de-neut+적응β 신호 + commit-and-refine
│                     #   deploy. segment(emb, reset='commit_refine'|'threshold'). methodology hi-ontop-cr.md
└── hi_ontop_lex.py    # HiOnTopLex(HiOnTop) — lexical-overlap 보정 변형 (검증 대기, 2026-05-23)
```

- `HiOnTopLex` = `HiOnTop` + TextTiling 식 단어-빈도 겹침 보정항. `w_lex=0` 시 v1 과
  byte-parity. 설계·결과 → `context/methodology/hi-ontop-lex.md`, decision-log 2026-05-23.
- 실험 entry: `scripts/run_hiontop_v2.py` → `outputs/experiments/2026-05-23_hiontop_v2/`.
- 현재 main 모델은 `HiOnTop` 유지 — `HiOnTopLex` 는 v1 대체 승격 보류 (검증 대기).
