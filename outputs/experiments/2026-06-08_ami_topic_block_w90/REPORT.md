# AMI 화제분절 — 60초 블록 단위 Hi-OnTop (2026-06-08_ami_topic_block_w90)

## 실험 setup
- **목적**: turn 단위로 붕괴한 Hi-OnTop(AUC 0.567)을 **90초 시간 블록** 단위로 재구성하면 화제 cohesion·경계 신호가 회복되는지 실측.
- **데이터**: AMI scenario meetings, `data/ami/topic/*.json` (NXT manual annotation). calib 6 / test 6 미팅 (deterministic split, seed=0, calib_frac=0.5).
- **블록 구성**: 각 발화의 begin_time 기준 90초 고정 bin 에 텍스트 이어붙임 → 한 블록 = 한 덩어리 텍스트. 빈 블록 제외.
- **인코더**: MiniLM-int8 (ONNX `model_quint8_avx2.onnx`), L2-normalize.
- **δ_eff**: `run_encoder_comparison.delta_eff_seq` (ctx m=2, ρ=0.7, blend a=0.5) — turn eval 과 동일.
- **δ\***: calib 미팅 블록 δ_eff 분포의 percentile (label-free). sweep p ∈ [50, 60, 70, 80, 90].
- **metric**: 블록 단위 official Pk/WD (NLTK) + F1 + Score(0.5·F1+0.25·(1−Pk)+0.25·(1−WD)).

- **블록 통계**: 평균 20.2 블록/미팅, 경계 밀도 27.8%.
- **AUC (threshold-free 분리력)**: **0.629** (turn 단위 0.567 대비).

## 결과 (test split, percentile sweep)

| p | δ\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ |
|--:|--:|--:|--:|--:|--:|
| 50 | 0.535 | 0.518 | 0.667 | 0.356 | 0.382  ← best |
| 60 | 0.572 | 0.490 | 0.593 | 0.277 | 0.368 |
| 70 | 0.651 | 0.487 | 0.545 | 0.185 | 0.334 |
| 80 | 0.715 | 0.502 | 0.502 | 0.139 | 0.318 |
| 90 | 0.770 | 0.530 | 0.530 | 0.000 | 0.235 |

## 해석
- best = **p50** (Score 0.382, F1 0.356, Pk 0.518).
- threshold-free AUC 0.629 — turn 단위(0.567)보다 분리력 상승. 단위(granularity)가 핵심이었음을 직접 확인.
- **turn 단위 결과와 절대값 직접 비교 금지**: 시퀀스 단위가 발화→블록으로 바뀌어 Pk/WD/F1 의 분모(블록 수)가 다름. 비교는 AUC·신호 회복 경향으로만.

## 판정
- turn→블록 전환으로 신호는 회복되나(AUC↑), 절대 성능은 여전히 제한적 — 회의 도메인은 ~1분 블록이 적정 단위라는 증거. DTS 3벤치(텍스트 대화)와 동급 성능은 아님.

## 한계 / 검증 미해결
- **블록 경계 해상도 ±90s**: 경계를 블록 시작 단위로만 찍어 위치 정밀도가 거침.
- **단일 인코더·단일 win**: win sweep(30/60/120/180) 은 별도 분석(2026-06-08 window 실측) 참조. 여기선 60s 고정.
- **seed 1개**: calib/test split 단일. 3-run std 미산출.
- **gold 정의**: top-level topic start 가 블록 시간구간에 들어가면 경계로 라벨 — 한 블록에 경계 2개 이상이면 1개로 합쳐짐(드묾).
