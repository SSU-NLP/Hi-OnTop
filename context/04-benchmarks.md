# 벤치마크 정리

## 평가 축 = Topic 경계 감지

| 목적 | 벤치마크 | 메트릭 |
|---|---|---|
| **Topic 경계 감지** | **TopiOCQA**, **TIAGE** (factoid / chit-chat) | turn-transition binary F1 |
| **표준 dialogue segmentation** | **dialseg711**, **SuperDialseg** | Pk / WindowDiff / F1 |
| **회의 화제 경계 (streaming)** | **AMI** | Pk / WindowDiff / F1 |

> 표준 분절 벤치(dialseg711/SuperDialseg)·AMI 의 도입 경위와 metric 정의는
> decision-log 2026-05-18~ 및 `context/methodology/` 참조.

## Topic 경계 감지 벤치마크

### TopiOCQA
- 경로: `benchmarks/topiocqa/`
- 데이터: `python download_data.py --resource data.topiocqa_dataset.dev`
- 구조: dev 2514 turns, 205 conv, 평균 12턴, 3.3 shift/conv
- Topic 정의: Wikipedia document 경계 (명시 annotation)
- 평가: topic shift detection F1
- 레포: https://github.com/McGill-NLP/topiocqa
- **특성**: factoid QA, shift rate 28%/transition — frequent-shift regime

### TIAGE
- 경로: `benchmarks/tiage/`
- 데이터: `benchmarks/tiage/data/personachat/anno/{train,dev,test}/anno_*.json` (레포 내 포함)
- 구조: train 300 / dev 100 / test 100 conv, 평균 15.6턴, 3.15 shift/conv (test 기준)
- Turn label: `-1`(첫 턴) / `0`(continue) / `1`(shift). 인간 주석, Cohen's Kappa 0.48.
- 평가: topic shift detection F1 (turn-transition binary)
- 레포: https://github.com/HuiyuanXie/tiage
- **특성**: PersonaChat 기반 chit-chat, shift rate 20%/transition, 짧은 utterance (~50자)

---

## 데이터 준비 체크리스트
- [x] TopiOCQA clone + train/dev download (via `download_data.py`)
- [x] TIAGE clone (`benchmarks/tiage/`)
- [x] dialseg711 / SuperDialseg (Def-DTS 번들, README 참조)

---

## 데이터 분석 체크리스트 (Phase 0 완료 기준)

각 벤치마크별 필수 분석:
- [ ] 샘플 1~2개 JSON 구조 직접 확인
- [ ] 평균/중간값/최대값: 세션 수, 턴 수, 쿼리 토큰 길이
- [ ] Topic 전환 패턴 (명시 annotation 여부, 전환 빈도, 유형)
- [ ] Claude-유사 대화(코딩/글쓰기/브레인스토밍)와의 유사성 평가
- [ ] 해당 벤치마크에서 **어떤 사건 모델이 유리할지** 판단

분석 결과 → `outputs/benchmark-analysis.md`

---

## 주의: 벤치마크별 bias

각 벤치마크는 서로 다른 대화 성격을 가진다:
- **TopiOCQA**: Wiki 기반 정보 검색 QA. Named entity 풍부. frequent-shift.
- **TIAGE**: PersonaChat 기반 chit-chat. Cue phrase 많음. 짧은 utterance.
- **dialseg711 / SuperDialseg**: 표준 dialogue segmentation 벤치.
- **AMI**: 회의 전사. streaming(음성→STT) 화제 경계.

**한 벤치마크의 특성만 보고 설계를 고정하지 마라.** 여러 벤치마크에서 공통으로 유효한 신호를 찾거나, 벤치마크별 적응형 설계를 고려할 것.