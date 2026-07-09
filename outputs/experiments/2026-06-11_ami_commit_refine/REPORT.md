# REPORT — AMI deploy: Local Commit-and-Refine Splitter (online reset 부트스트랩)

**날짜**: 2026-06-11 · **대상**: Hi-OnTop 축 2(online 실현) — reset 부트스트랩 돌파 시도
**한 줄**: codex 자문(commit-and-refine)을 3단계(v1 확정만 / v2 +split-gain b* refinement / v3 shock-gate
제거 sliding)로 구현·실측. **v2 가 새 deploy best(Score 0.401, ±2F1 0.140)** 이나 oracle 천장(±2F1 0.554)
대비 **격차는 거의 그대로** — 부트스트랩은 눌렀을 뿐 깨지 못함.

## 1. 실험 setup
- **목적**: 신호(de-neut V + run-length 적응 β)는 oracle 천장에서 충분(gap 분해: clean+μcσ ±2F1 0.554)인데,
  online detected-reset 으로는 ±2F1 0.13 대 고착. hard reset(즉시·불가역)을 commit-and-refine 으로 바꿔
  oracle 격차를 메울 수 있는지 검증. (codex 자문 2026-06-11, `outputs/runs/_misc/codex_bootstrap_consult.md`.)
- **데이터**: AMI 139 미팅, gold 경계 938 (topic 레벨). `data/ami/topic/<mid>.json` (NITE-XML manual annotation
  → `ami_topic_prep.py`), 임베딩 `outputs/runs/_misc/ami_emb/<mid>.pkl` (MiniLM-L6 int8 onnx, `gen_ami_emb.py`).
- **신호/HP (고정, 전 method 공통)**: de-neut `r_active=1−cos(deneut(x),deneut(m))`, `V=r_active−λ·r_global`,
  λ=0.6, g_rho=0.15, 적응 β=clip(2.0−1.0·log(1+k/8)), 적응 임계치 μ+cσ(warmup 8). 인코더 고정·무학습·0-look-ahead.
- **method**:
  - `δ_eff` (기존 baseline), `hard-reset deneut` (직전 deploy best = `ami_adaptive_deneut_deploy.segment`).
  - **v1 `segment_cr`**: V>θ → 잠정 경계만, persistence(오른쪽 shadow prototype 응집) 충족 시 spike 위치 b 에
    확정 / L 윈도우 소진 시 거절·흡수. **위치 최적화 없음.**
  - **v2 `segment_cr2`**: v1 + **local split-gain b* refinement**. shock 으로 armed → 윈도우 후보 split 위치 b
    별 gain=`SSE(a..t)−[SSE(a..b−1)+SSE(b..t)]` (SSE=cnt−‖Σz‖²/cnt) 의 **argmax b\* 에 경계 emit**.
  - **v3 `segment_cr3`**: shock gate 제거. split-gain 을 매 t **sliding 상시 평가**, best_g>μ+cσ 면 b* emit.
    raw / de-neut 두 변형 (de-neut: 윈도우 벡터에서 global g 제거).
- **metric**: ±2-tolerance F1(localization, 주지표) / official Pk·WD / Score=0.5F1+0.25(1−Pk)+0.25(1−WD).
  seed 없음(결정론적, 인코딩만 비결정 가능성 — 단일 캐시 고정).
- **비교 기준선**: hard-reset deneut(직전 best), oracle 천장 ±2F1 0.554(clean+μcσ, gold-reset).

## 2. 결과
| method | pred | **±2F1** | Pk | WD | **Score** |
|---|--:|--:|--:|--:|--:|
| δ_eff ewma (과분절) | 5176 | 0.154 | 0.634 | 0.861 | 0.203 |
| hard-reset deneut c=1.5 | 884 | 0.106 | 0.336 | 0.390 | **0.372** |
| hard-reset deneut c=1.0 | 4105 | 0.131 | 0.489 | 0.646 | 0.282 |
| v1 CR (L=3,m=2,c=0.8) | 1350 | 0.110 | 0.400 | 0.474 | 0.336 |
| v1 CR (L=5,m=3,c=1.2) | 502 | 0.087 | 0.333 | 0.366 | 0.369 |
| **v2 CR2 (L=8,m=2,c=1.0)** | 521 | **0.140** | 0.322 | 0.354 | **0.401** |
| v2 CR2 (L=8,m=2,c=1.2) | 371 | 0.125 | 0.310 | 0.333 | 0.401 |
| v3 raw (W=8,c=1.5) | 8761 | 0.149 | 0.651 | 0.908 | 0.184 |
| v3 deneut (W=12,c=2.0) | 2186 | 0.108 | 0.463 | 0.553 | 0.300 |
| v4 cr2+booster (Lmax=60) | 2248 | 0.182 | 0.535 | 0.652 | 0.294 |
| v4 cr2+booster (Lmax=20) | 5915 | 0.160 | 0.665 | 0.911 | 0.186 |
| v5 outlier-gate (g=1.5) | 519 | 0.138 | 0.322 | 0.354 | 0.401 |
| v5 recency (rho=0.25) | 558 | 0.125 | 0.318 | 0.357 | 0.394 |
| **oracle 천장 (clean+μcσ)** | — | **0.554** | — | — | — |

- **v2 L-saturation**: ±2F1 L=4→0.118, L=6→0.122, **L=8→0.140**, L=12→0.129, L=16→0.126, L=24→0.130,
  L=32→0.130. L=8 정점 후 포화(~0.13).
- **L=8 검증 (AMI k-fold CV, c=1.0 고정)**: 5-fold 에서 train-선택 L\*=8 **만장일치(5/5)**, held-out Score
  CV-선택 0.401 = 고정 L=8 0.401 (**overfit gap 0**, fold별 0.390~0.425). 2-fold 에서 fold0 train 이 L=12 를
  뽑아도 held-out 은 L=8(0.398)>L=12(0.389). Score plateau L=6~32 0.39~0.40(완만) → **L=8 은 overfit 아님,
  robust**. → 고정 (c=1.0, L=8, A=2.0, B=1.0, m_min=2) ship 정당.
- **m_min 검증 (AMI k-fold CV)**: 민감도 m=1:0.141/m=2:0.140/m=3:0.128/m=4:0.125 (m=1·2 ±2F1 동률, m=2
  Score-best 0.401, m≥3 회귀). 5-fold train-선택 m\*=2 **만장일치(5/5)**, held-out overfit gap 0. → m_min=2 검증.
  **정식 config 전 상수(c·L·A·B·m_min) 검증 완료, 미검증 0개.**
- v1 ±2F1 0.083~0.110 (전 grid), v3 raw pred 6k~30k(δ_eff식 spray), v3 deneut ±2F1 ≤0.118/Score ≤0.326.

## 3. 해석
- **v1 실패 원인**: persistence 게이트는 false alarm 만 거름(pred↓, Pk/WD 약간↑) — **경계 *위치* 를 안 건드려**
  localization(±2F1) 불변. codex 가 핵심으로 짚은 "b* 이동 정정"을 v1 이 빠뜨린 결과.
- **v2 가 움직인 이유**: split-gain argmax 로 spike 위치를 윈도우 내 최적 split 으로 이동 → ±2F1 0.131→0.140,
  Score 0.372→0.401. b* refinement 가 **누락분이 맞았음**을 실측 확인.
  > ⚠ 정정(2026-06-15): 이 줄의 "0.372/0.131" 짝은 표(위)의 c=1.5 Score(0.372,±2F1 0.106)와 c=1.0 ±2F1(0.131,Score 0.282)을 섞은 표기. 표가 정확. [[HANDOFF_04]] §6 / decision-log 2026-06-15.
- **v3 실패 원인 = recall 가설 반증**: shock gate 를 없애고 split-gain 상시 평가하면 recall(pred)은 늘지만
  precision 붕괴. raw 는 global 성분 지배로 항상 발화(δ_eff식 과분절), de-neut 도 shock gate 의 precision 없이
  과분절. **V>θ shock 감지가 주던 precision 이 실제로 유용** → gate 유지가 맞음.
- **격차의 본질**: v2 가 best 라도 ±2F1 0.140 ≪ oracle 0.554. ~42pt 격차 중 ~1pt 만 줄임. online 에서
  prototype 이 깨끗해지지 않는 근본 문제는 b* refinement 로 안 풀림 — refinement 는 "감지된 후보 위치"만 고치지,
  오염된 prototype 때문에 **놓친 경계(recall)** 는 못 살리고, 놓친 걸 살리려 gate 를 풀면(v3) precision 이 무너짐.
- **v4(2-stage 결합) = frontier 안 옮김**: 긴 segment recall booster 는 ±2F1 0.140→**0.182** 까지 올리지만
  Score 0.401→0.294 붕괴(pred 521→2248 과분절). 추가 경계가 충분히 정확치 않아(±2F1 0.18 도 대부분 어긋남)
  Pk/WD 가 무너짐 — **같은 precision/recall 곡선 위를 과분절 쪽으로 이동**할 뿐 더 나은 operating point 아님
  (δ_eff 과분절과 사실상 같은 frontier). recall 은 살 수 있으나 online localization 자체가 부정확해
  precision 으로 돌려막지 못함.

- **v5(prototype 오염 완화) = 실패**: v2 base 에 outlier-gated 업데이트(V>μ+gate·σ 발화 down-weight) / recency
  고정 rho 를 얹음. **outlier-gating 거의 무효과**(g=1.5 → 0.138/0.401 ≈ v2) → 오염이 "몇몇 high-V 발화" 탓이
  아니라 **확산적/구조적**(점진 drift + 놓친 경계가 두 topic 을 통째 포함, 발화 하나 게이팅으로 해결 불가).
  **recency 는 악화**(rho=0.25 → 0.125/0.394; rho=0.35 는 ±2F1 0.141 회복하나 Score 0.386 frontier 슬라이드)
  — 짧은 메모리는 신호만 noisy 하게 만들 뿐 오염 미감소.

## 4. 판정
- **향상**: v2 CR2 = 새 deploy best (**Score 0.372→0.401 +0.029, ±2F1 0.131→0.140 +0.009**). 채택 가치 有.
- **동일/회귀**: v1(확정만)·v3(상시 split-gain)·v4(2-stage 결합)·v5(오염완화) 모두 v2 대비 향상 없음 → 폐기.
  **5각도(reset 기법 4 + 오염완화 1)가 모두 oracle 격차를 못 메움** = 병목은 reset 메커니즘도 국소 오염원도
  아니라 **online 운영 자체의 구조적 한계(놓친 경계가 prototype 을 두 topic 으로 오염, 발화 단위로는 복구 불가)**.
- **다음**: 단순 reset/prototype 기법 추가 탐색은 **수확체감 확정**(5각도 음성). 잔여 후보 — (a) (A,B)·c
  **자기보정**(gap-closing 아닌 *tuning 제거* 성격의 별개 win), (b) deploy 를 **v2(Score 0.401)로 ship + 격차를
  명시적 한계**로 문서화. **정식 hi_ontop 승격은 보류**(격차 미해결). 권고: reset 미세조정 종료, 신호 자체/자기보정
  또는 ship-with-limit 으로 전환.

## 5. 한계 / 검증 미해결
- **HP**: c 는 calibration-free(전 구간 1.0~1.5 에서 hard-reset 이상, handoff §5b 정합 → 고정 c=1.0/Otsu).
  L=8·m_min=2 는 AMI k-fold CV 로 검증(위 §2, 둘 다 5-fold 만장일치·overfit gap 0). (A,B)=(2.0,1.0)은 신호
  단계 LOO 검증(handoff §1.5). → **정식 config 전 상수 검증 완료, 미검증 0개.** 잔여 open: streaming 클래스
  통합(Hi-OnTop 파이프라인 scope-barred, lagged-emission wrapper — codex 위임 대상).
- **DTS 미반영**: 본 REPORT 는 AMI 만. DTS 스모크에서 CR 은 sharp-seam 도메인에 역효과(dialseg711 ±2F1
  0.593→0.432) — CR 은 drift(AMI) 전용. 두 도메인 동시 적용은 별도 과제(CR engage 를 run-length 로 게이팅).
- **oracle 격차 미해결**: 핵심 목표(±2F1 0.14→0.55) 실패. v2 는 부분 개선일 뿐.
- **환경 재생성 caveat**: 이 컨테이너는 데이터·임베딩 부재로 재생성함 — DTS(SuperDialseg Drive), AMI(NITE-XML
  manual zip), MiniLM-int8 onnx(HF). 인코더 로드에 `torch.int4` shim 필요(`run_encoder_comparison._encoder`,
  설치된 onnxruntime/torch 버전쌍 이슈). pyproject 핀(st<4, tf<4.50)과 다른 최신 버전 설치됨 — 임베딩 수치
  동일성은 미검증(상대 비교는 단일 캐시 고정이라 유효).

## 산출물
- 스크립트: `scripts/ami_commit_refine_deploy.py` (`segment_cr`/`cr2`/`cr3`/`cr4`, baseline 포함 main grid),
  `scripts/gen_ami_emb.py` (AMI 임베딩 생성).
- codex 자문 전문: `outputs/runs/_misc/codex_bootstrap_consult.md`.
- 관련: handoff `handoff/HANDOFF_0609_deploy-oracle-gap.md` §2.4·§2.5, decision-log 2026-06-11.
