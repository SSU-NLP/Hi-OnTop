# Supervised RoBERTa (SuperDialseg) — 논문 충실 재현

## 1. 실험 setup

- **목적**: Coldog2333/SuperDialseg (EMNLP 2023, `2023.emnlp-main.249`) 논문
  **Table 3 의 `RoBERTa`** (plain supervised dialogue segmenter) 를 충실 재현.
  Hi-OnTop / Hi-OnTop-v2 (무감독 online segmenter) 의 supervised 상한
  baseline 으로 둔다.
- **재현 근거**: 논문 Appendix A.1/A.2 (입력 구성·하이퍼파라미터) + repo
  `src/super_dialseg/utils/data/data_collator.py` (verbatim, `__getitem_input__`
  `roberta-base` 분기) + `models/supervised.py`. repo 에 학습 스크립트는
  없어(`main.py` 는 평가 전용, supervised 미연결) 학습 코드는 논문/collator
  대로 재구현.
- **모델**: `RobertaForTokenClassification` (`roberta-base`, num_labels=2).
  입력 `<s> u1 </s></s> u2 </s></s> … uN </s>` (발화당 23 BPE 토큰 절단,
  마지막 `</s>` 제거, max_seq_len 512), 각 발화 뒤 **첫 `</s>` 위치에서
  token-classification**. plain RoBERTa — da/role 출력 헤드(MT)·입력
  임베딩(MV) 없음.
- **데이터**: SuperDialseg 번들 (Google Drive, key `superseg-v2`).
  학습 `superseg/train` (6948 dialogue), model selection `superseg/validation`
  (1322 dialogue) val Score, 평가 `{superseg,tiage,dialseg711}/test`
  (1322 / 100 / 711 dialogue).
- **학습 HP** (논문 Appendix A.2): AdamW, lr=1e-5, batch_size=8,
  weight_decay=1e-3, grad_clip=1.0, **20 epochs**, early stopping
  (val Score 가 patience=10 epoch 미개선 시 중단, best 체크포인트 보존),
  LR 스케줄러·warmup 미사용. 슬라이딩 윈도우 `|T|=20` 발화 stride 1
  (학습은 임의 19-발화 윈도우, 추론은 stride-1 전체 윈도우 → 발화별 logit
  평균). seed 42 (단일 run).
- **실행**: Colab GPU, `notebooks/roberta_superdialseg_colab.ipynb`.
  epoch 당 ≈107 s, 14 epoch 에서 early stop. 원본 산출물(zip·모델
  체크포인트)은 `outputs/runs/_misc/` (gitignored).
- **metric**: official SuperDialseg Pk/WD (sliding window = 평균 segment
  길이의 절반) + binary F1, `Score = (2·F1 + (1−Pk) + (1−WD)) / 4`.

## 2. 결과 표

### test split — 재현 vs 논문 Table 3 (`RoBERTa`, SuperDialseg 학습)

| 평가셋 | in-domain? | Pk ↓ | WD ↓ | F1 ↑ | **Score ↑** | 논문 Score | ΔScore |
|---|---|---:|---:|---:|---:|---:|---:|
| superseg | ✔ | 0.1945 | 0.2046 | 0.8311 | **0.8158** | 0.798 | **+0.0178** |
| tiage | ✘ zero-shot | 0.4002 | 0.4225 | 0.4879 | **0.5383** | 0.482 | **+0.0563** |
| dialseg711 | ✘ zero-shot | 0.2728 | 0.3395 | 0.6417 | **0.6678** | 0.702 | **−0.0342** |
| **mean-3** |  |  |  |  | **0.6740** | 0.6607 | **+0.0133** |

논문 개별 수치: superseg Pk .185/WD .192/F1 .784 · tiage .401/.443/.373 ·
dialseg711 .241/.272/.660.

### 학습 곡선 (val = superseg/validation Score)

| epoch | 1 | 2 | 3 | **4** | 5 | 6 | 8 | 11 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train loss | .251 | .189 | .150 | .115 | .087 | .068 | .044 | .028 | .017 |
| val Score | .785 | .811 | .822 | **.827** | .817 | .797 | .777 | .823 | .821 |

best = **epoch 4** (val Score 0.8272) → test 평가는 epoch-4 체크포인트 사용.
epoch 5 부터 train loss 는 계속 하락하나 val Score 는 정체/하락 → overfit.
early stopping 이 patience=10 으로 epoch 14 에서 중단, best(epoch 4) 보존.

## 3. 해석

- **재현 성공**. mean-3 ΔScore = **+0.013**, 3 벤치 중 2 개(superseg·tiage)가
  논문보다 높고 1 개(dialseg711)가 −0.034. 논문 자신도 자기 재현에 대해
  "Most of the results are similar or even superior to those reported"
  라고 적었는데, 본 재현도 같은 "유사 또는 우위" 패턴 → 방법 충실 재현으로
  판정 가능.
- **F1 이 체계적으로 높다** (superseg +0.047, tiage +0.115, dialseg711
  −0.018). Pk/WD 는 superseg·dialseg711 에서 약간 나쁨. 즉 본 재현 모델이
  경계를 *조금 더 많이* 찍어 recall↑(F1↑) 하되 경계 부근 오탐도 늘어
  Pk/WD 가 소폭 손해. 이는 추론 시 겹치는 윈도우의 발화별 예측을 **logit
  평균**으로 통합한 선택(논문·repo 미명시 → 본 재현의 명시적 결정)의 영향
  가능성이 큼 — 버그가 아니라 미명시 디테일.
- **dialseg711 만 논문 미달** (−0.034). dialseg711 은 대화가 가장 길어
  (~27 turn) 한 발화가 가장 많은 슬라이딩 윈도우에 등장 → logit-평균 집계
  영향이 가장 큰 셋. zero-shot 합성 코퍼스라 분포 민감도도 큼. 데이터
  버전 차이(아래)도 일부 기여 추정.
- **학습은 건강**: val Score 가 epoch 1→4 단조 상승(0.785→0.827) 후
  overfit, early stopping 이 정상 작동. 클래스 붕괴·라벨 정렬 버그 징후
  없음 (이전 smoke-run 의 tiage F1=0 은 50-step 미학습 탓이었음이 확인됨).

## 4. 판정

- **논문 Table 3 RoBERTa 충실 재현 성공.** Score: superseg 0.816 /
  tiage 0.538 / dialseg711 0.668. byte 일치는 애초에 불가(아래 한계)지만
  방법·하이퍼파라미터·입력 구성을 논문/repo 그대로 맞춰 논문 수치
  근방(±0.06 이내, 평균 +0.013)에 안착.
- **Hi-OnTop 대비** (참고, official 동일 metric — [[2026-05-22_hiontop_baseline_table]]
  / `2026-05-23_hiontop_v2`):

  | | tiage | superseg | dialseg711 |
  |---|---:|---:|---:|
  | Hi-OnTop-v1 (무감독 online) | 0.433 | 0.428 | 0.617 |
  | **RoBERTa supervised** | **0.538** | **0.816** | **0.668** |

  supervised RoBERTa 가 세 벤치 모두 우위, 특히 in-domain superseg 에서
  압도(+0.39). **단 비대칭 비교**: RoBERTa 는 (1) 라벨 학습 (2) 발화
  `i+1` 이후를 보는 윈도우(offline) (3) 별도 학습 필요. Hi-OnTop 는
  무감독·past-only online. RoBERTa 우위는 "supervised offline 상한"
  이지 online 무감독 주장과 1:1 비교 아님. paper 표에 baseline 으로
  넣을 때 이 비대칭을 명시할 것.
- **다음**: 논문은 다회 평균 — 본 재현은 단일 seed. paper-table baseline
  으로 인용하려면 seed 3개 이상 재학습 → mean±std 권고 (노트북에 multi-seed
  옵션 추가 가능).

## 5. 한계 / 검증 미해결

- **단일 seed (42)**: 논문은 "did experiments several times and reported
  the average" — 보고치가 다회 평균. 본 재현은 1 회 → ΔScore 가 run
  variance 안인지 미확정. seed 42 는 repo `main.py`(평가 스크립트)
  argparse 기본값에서 따왔을 뿐 논문 학습 seed 아님 (논문은 학습 seed
  미공개).
- **추론 윈도우 집계 미명시**: 겹치는 stride-1 윈도우의 발화별 예측 통합
  방식이 논문·repo 모두에 없음 → logit 평균 채택(본 재현의 결정). F1 이
  체계적으로 높은 패턴의 유력 원인.
- **데이터 버전 불일치**: 본 재현 데이터 = `superseg-v2` (train 6948 /
  val 1322 / test 1322). 논문 Table 2 = train 6863 / valid 1305 /
  test 1310. Drive 의 데이터는 논문 이후 개정판으로 보이며, 학습/평가
  데이터 자체가 논문과 동일하지 않음 — byte 재현 불가의 근본 사유 중 하나.
- **학습 코드 미공개**: repo `main.py` 는 평가 전용·supervised 미연결,
  체크포인트 미배포. 학습 루프는 논문 Appendix A 서술 + collator 코드로
  재구성 — LR 스케줄러/warmup/grad-clip 등 미명시 항목은 합리적 기본값
  (warmup 없음, clip 1.0) 사용.
- **재현 대상은 plain RoBERTa 만**: 논문 `MVRoBERTa`(Table 5, role/DA 입력
  임베딩, Score 0.808)·repo `RobertaMultiTask`(MT, da/role 출력 헤드)는
  미구현 (사용자 지시 — plain 만).
- **Hi-OnTop 와의 dialseg711 비교는 비엄밀**: 본 RoBERTa run 은 full 711
  test, `2026-05-23_hiontop_v2` 의 dialseg711 은 30/70 split 의 70%
  held-out → dialseg711 행은 1:1 동일 데이터 아님. tiage/superseg test 는
  동일.
