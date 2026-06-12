# 미확정 질문

## 최우선: 사건 모델 (Phase 0 완료 후 결정)
`context/01-hi-ontop-design.md`의 "미확정 사항 A" 참조.

---

## 설계 차원

1. Multi-signal 가중치 튜닝 방법 (사건 모델에 신호 앙상블 포함 시)
   - A: grid search on dev set
   - B: 학습 가능한 linear layer
   - C: 고정값 시작 + 실험 조정

2. Entity extractor (사건 모델이 엔티티 사용 시)
   - spaCy `en_core_web_sm` (가벼움 ~5ms)
   - spaCy `en_core_web_trf` (정확 ~20ms)
   - 간단한 noun phrase chunker

---

## 평가 차원

3. Baseline 비교 대상
   - 단순 sliding window
   - TextTiling / GreedySeg / GraphSeg 등 unsupervised segmenter
   - 단순 sCRP (boundary score 없이)

---

## 마감 기한
코드 작성 전 최소 사건 모델 결정 필요.