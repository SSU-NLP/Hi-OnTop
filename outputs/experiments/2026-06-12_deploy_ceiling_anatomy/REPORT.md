# REPORT — Hi-OnTop deploy 천장 해부 (왜 oracle 격차가 안 메워지나)

**날짜**: 2026-06-11~12 · **대상**: Hi-OnTop-DeNeut online deploy 의 oracle 격차 정밀 진단
**한 줄**: "정확히 왜 deploy 가 안 되나"를 계량. **(1) oracle 0.687 재현 확인**(LLM 초과 진짜), **(2) 병목 =
online prototype 오염**(분별력 AUC 0.626→0.525, −0.10, 측정확정), **(3) 그러나 오염 회복도 신호/표현/판별기
교체도 전부 실패** — per-turn 신호 margin 이 0.024 로 근본 약해, oracle(정답 엿봄)만 그 위에서 작동. deploy 천장
(Score ~0.40)은 구조적.

## 1. 실험 setup
- **질문**: oracle ±2F1 0.55~0.69 vs deploy ~0.14 격차의 *정확한* 메커니즘. (추측 아닌 측정.)
- **데이터**: AMI 139미팅(gold 938). 신호: de-neut V_rel(±β), 적응 임계치 μ+cσ. metric: ±2F1 / Score(0.5F1+
  0.25(1−Pk)+0.25(1−WD)) / 분별력 AUC(경계 turn V > 비경계 확률, threshold-free).
- **스크립트**: `ami_deploy_failure_anatomy.py`(오염 AUC), `ami_summary_proto_oracle.py`(요약 prototype),
  `ami_crossenc_deploy.py`(cross-encoder 판별기), `ami_beam_deploy.py`(beam 회복), 인라인(oracle 재현·lag·β=0).

## 2. 결과

### 2a. oracle 0.687 재현 확인 (canonical segment_vrel 신호, gold-reset + per-meeting 최적임계)
| | ±2F1 |
|---|--:|
| V_rel(β=0) EWMA prototype | **0.671** (handoff 0.687 재현 ✓) |
| V_rel true-mean prototype | 0.663 |
| adaptive-β prototype | 0.318 (handoff 0.341 재현 ✓) |
→ **깨끗한 prototype 의 V_rel 신호는 진짜로 강하다 (LLM full-context 0.543 초과).** (`ami_vrel_eval`의 "0.687"은
하드코딩 문자열이었으나, segment_vrel 신호를 gold-reset+최적임계로 직접 돌려 0.671 로 재현.)

### 2b. 병목 = online prototype 오염 (측정확정)
correct V_rel 신호, AUC(±2):
| | AUC | 비고 |
|---|--:|---|
| oracle (gold-reset, 깨끗) | **0.626** | 진짜 신호 |
| deploy (detected-reset, 오염) | **0.525** | 거의 random |
| **오염 비용** | **−0.102** | |
- deploy operating point: **recall 0.269**(경계 73% 놓침), precision 0.092. → 주범은 헛경보 아니라 **놓침**.
- 진짜 경계 vs 비경계 V margin: oracle +0.024 / deploy +0.009 (둘 다 미미; 깨끗해도 turn-level 신호 약함).
- ※ 정직 caveat: 중간에 `v_oracle` β=0 경로 **버그**로 0.259/AUC0.50(=오염 무관)이라 잘못 결론냈다가, byte-faithful
  재현(0.671)으로 **정정**. 오염이 병목이라는 원 진단이 맞다.

### 2c. 격차를 메우려는 모든 시도 — 전부 실패 (best-c by Score)
| 시도 | Score | ±2F1 | 결론 |
|---|--:|--:|---|
| threshold (commit-refine 없음) | 0.372 | 0.106 | 0-lag base |
| **commit-refine L=8** | **0.401** | 0.140 | deploy best (단 ~26s lag) |
| **요약 prototype** (gold topic label) | — | — | oracle 0.247 < mean 0.318 (레지스터 불일치, 악화) |
| **cross-encoder 판별기** (coherence) | 0.211 | 0.136 | 꼴찌, 과분절 691 |
| **β=0 (V_rel 강신호)** | 0.368 | 0.061 | adaptive-β보다 나쁨 (오염에 더 취약) |
| **beam-suffix 회복** (codex 자문) | 0.380 | 0.112 | threshold보단↑ commit-refine↓, oracle 못 메움 |
| (oracle 천장) | — | 0.671 | per-meeting 임계 + gold reset |

### 2d. lag–Score 트레이드오프 (commit-refine 의 우위 = lag 매입분)
| reset | lag | Score | ±2F1 |
|---|--:|--:|--:|
| threshold | 0 (다음턴) | 0.372 | 0.106 |
| commit-refine L=2/3 | 6~9s | 0.364/0.371 | — |
| commit-refine L=4 | 12s | 0.379 | 0.118 |
| commit-refine L=8 | 26s | 0.401 | 0.140 |
→ commit-refine 의 +0.029 는 **전적으로 lag**. L≤3 은 threshold 동급. **0-lag 요구면 threshold 가 best.**

## 3. 해석 — 정확한 "왜"
1. **깨끗하면 신호는 강하다**(AUC 0.626, oracle 0.671>LLM). 표현(평균)도 옳다 — 요약 prototype 은 *레지스터가
   달라* 오히려 악화(평균 = 발화와 같은 공간의 centroid = 옳은 객체).
2. **online 오염이 그 신호를 죽인다**(AUC 0.626→0.525, −0.10) → recall 0.27 로 붕괴. **오염이 진짜 병목.**
3. **그러나 오염을 *회복*하려면 "어디가 깨끗한 segment 인가"를 알아야 하고, 그건 경계 검출 그 자체** — beam 의
   global 목적함수가 **per-turn margin 0.024(거의 random)** 위에서 돌아 gold 분절에 못 내려앉음. 닭-달걀이 한
   겹 더 깊어진 것. β=0 으로 신호를 강화하면 오염엔 *더* 취약(deploy 0.368<0.401).
4. **oracle 0.671 의 우위 = per-meeting 최적임계 + gold reset(둘 다 정답 엿봄).** online 은 둘 다 없고, 어떤
   기법도 그 정보를 복원 못 함.

## 4. 판정
- **deploy 천장 ~Score 0.40 / ±2F1 0.14 는 구조적.** reset(5각도)·표현(요약)·판별기(cross-encoder)·신호강화
  (β=0)·오염회복(beam) — **6+ 각도 전부 수렴, oracle 0.67 도달 불가.** 근본 원인 = per-turn 신호 margin 0.024.
- **commit-refine 은 deploy best(0.401)이나 그 우위는 lag(26s) 매입분** → **default 는 `threshold`(0-lag, 0.372)**
  로 변경(decision-log 2026-06-12).
- **남은 진짜 후보**: per-turn 신호 자체의 분별력(margin 0.024)을 올리는 것 — reset/표현/회복이 아니라 **신호
  공학**. 단 무학습·인코더고정 제약상 여지 좁음. 또는 deploy 천장을 한계로 인정하고 0-lag threshold 로 ship.

## 5. 한계 / 검증 미해결
- **AMI 단일 코퍼스**: 모든 deploy 수치가 AMI(한 장르/세팅). cross-corpus drift 일반화 미검증. 신호((A,B))만
  LOO cross-domain.
- **LLM Score figure 미완**: Score 축에 LLM 버퍼곡선 추가하려면 LLM 재실행 필요한데 **Crts 가 Cloudflare Access
  403** (service token 형식은 맞으나 게이트웨이 정책 미인가 — 관리자 인가 필요). `.env` 채우고 인가되면
  `ami_llm_buffer_eval.py`(Pk/WD/Score·CF헤더 배선 완료)로 즉시 산출.
- **per-turn margin 0.024 가 인코더(MiniLM-int8) 탓인지 task 본질인지 미규명** — 더 강한 인코더로 margin 이
  오르는지는 별도(인코더 고정 제약과 충돌).

## 산출물
- 스크립트: `ami_deploy_failure_anatomy.py`, `ami_summary_proto_oracle.py`, `ami_crossenc_deploy.py`,
  `ami_beam_deploy.py`. codex 자문: `outputs/runs/_misc/codex_recoverable_proto.md`.
- 관련: [[hi-ontop-deneut]], `2026-06-11_ami_commit_refine/REPORT.md`, decision-log 2026-06-11~12.
