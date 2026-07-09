# HANDOFF_04 — DTS 채점기 단일 공식 기준 확정

**지정일**: 2026-06-13 · **상태**: 채점기 확정 + 독립 reference 검증 완료 · **부모 컨텍스트**: dts_result.md / HANDOFF_01

## 0. 문제 정의

### 배경
DTS(tiage/dialseg711/superseg) 채점이 **세 harness가 섞여** 점수 신뢰가 깨져 있었다.
같은 δ_eff 가 dialseg711 에서 0.551(우리 run_encoder_comparison) vs 0.313(HANDOFF_01) 으로
갈리는 등 비교가 불가능. 원인은 세 축의 불일치:

1. **F1 집계** — per-dialogue 평균 vs corpus 풀링(micro).
2. **Pk/WD 라이브러리** — nltk vs segeval (값 다름).
3. **경계 정렬** — gold 규약(끝-turn) 대비 off-by-one.

세 reference harness:
- **SuperDialseg** (`Coldog2333/SuperDialseg`): nltk Pk/WD, **per-dialogue binary F1**, Score 합성.
- **Def-DTS** (`ElPlaguister/Def-DTS`): segeval Pk/WD, **corpus 풀링 F1**, 개수불일치 대화 제외, Score 없음.
- **TIAGE** (`HuiyuanXie/tiage`): per-turn 분류 P/R/F1(풀링), Pk/WD 없음.

### 목표
**DTS 표는 한 채점기로 통일** + 그 채점기가 어떤 코드이고 어디 있는지 확정 + 독립 reference 로 검증.

### 성공기준
확정 채점기로 **공개된 offline baseline 점수(paper)를 재현** → 재현되면 "진짜 공식"임이 증명.

---

## 1. 확정된 채점기 (정확한 스펙)

**= SuperDialseg 공식 `SegmentationEvaluation`** (Jiang et al. 2023 의 채점기 그 자체).

| 항목 | 확정값 |
|---|---|
| 집계 단위 | **대화별 계산 후 대화 수로 평균** (per-dialogue / macro-over-dialogues) — Pk·WD·F1 전부 |
| F1 종류 | `f1_score(average='binary')` = **경계(positive=1) 클래스만** (paper: "F1(binary), not F1(macro), we only care about segmentation points") |
| Pk/WD | `nltk.metrics.{pk,windowdiff}`, window = `max(2, round(len/(sum(y_true)+1)/2))` (대화 평균 segment 길이의 절반, Fournier 2013) |
| Score | `0.5*F1(binary) + 0.25*(1-Pk) + 0.25*(1-WD)` |
| 마지막 turn | label·pred 모두 0 강제 |
| **경계 정렬** | **끝-turn 규약** — gold label=1 = segment 의 *마지막* turn (topic_id 가 다음 turn에 바뀌기 직전). |

**비-채점**: Def-DTS 의 segeval / 풀링 F1 / 개수제외, TIAGE 의 풀링 분류 F1 은 **쓰지 않는다**.

---

## 2. 코드 위치 (어디 있는가)

### 2.1 권위 출처 (레포, 읽기 전용 — import 만, 복사 금지)
- `benchmarks/superdialseg/src/super_dialseg/metrics/segmentation.py`
  - `class SegmentationEvaluation` + `compute_total_score` (Score 공식).
  - `class Pk`, `class Windowdiff` (window=auto = avg seg/2).
- `benchmarks/superdialseg/src/super_dialseg/metrics/basic.py`
  - `EvaluationBase.add` (line 56-57, `.append` → 대화별 리스트), `_compute` (line 82-89,
    대화별 metric 합산 후 `/#sample`) ⇒ **per-dialogue 평균의 근거**.
- `benchmarks/superdialseg/src/super_dialseg/metrics/classification.py`
  - `class F1Score(average='binary')`.
- `benchmarks/superdialseg/examples/reproduce/main.py` line 93/112-113/115
  - `.add` 가 대화 1개씩 호출 + 마지막 0 강제 (정렬·집계 규약의 근거).
  - ⚠ main.py 는 import 오타(`TextTilingCLSSegmenter` ≠ 실제 `TexttilingCLSSegmenter`)로
    그대로는 실행 불가 — 직접 실행 말고 metrics/segmenter 를 개별 import 할 것.

### 2.2 무거운 `__init__` 우회 import 방법 (재현용)
`super_dialseg/__init__` 가 모든 모델(gensim/torch)을 eager import 하므로 metrics 만 직접 로드:
```python
import sys, types, importlib
sys.modules.setdefault('prettytable', types.ModuleType('prettytable')).PrettyTable = object
base = "benchmarks/superdialseg/src/super_dialseg"
for nm, p in [('super_dialseg', base), ('super_dialseg.metrics', base+'/metrics')]:
    if nm not in sys.modules:
        m = types.ModuleType(nm); m.__path__ = [p]; sys.modules[nm] = m
SegEval = importlib.import_module('super_dialseg.metrics.segmentation').SegmentationEvaluation
```

### 2.3 검증 스크립트 (영속, 재현 가능)
- **`scripts/validate_official_scorer.py`** — 위 우회 import + 공식 TextTiling(nltk w=10,k=6)
  + superdialseg_data 로 paper 재현 검증. `python scripts/validate_official_scorer.py`.

### 2.4 현 프로젝트 채점기 (교정 필요)
- `scripts/run_encoder_comparison.py::score_set` (line 135-144): Pk/WD·Score·window·정렬은
  공식과 일치하나 **F1 만 풀링** (`g += yt; p += yp; f1_score(g,p)`, line 141). → 공식과
  어긋나는 유일 지점. **per-dialogue 평균으로 교정 필요** (8축 중 7축 이미 공식).
- `boundaries()` (run_encoder_comparison) 는 끝-turn 규약으로 -1 shift 적용 = **정렬 올바름**.

### 2.5 아직 없는 것
- `src/` 에 단일 채점기 모듈은 **아직 없음**. (후속: §5(a).)

---

## 3. 검증 결과 (재현)

`scripts/validate_official_scorer.py`, 공식 scorer + 공식 TextTiling + superdialseg_data:

| 데이터셋 | 공식 채점기 Score | paper TextTiling | 판정 |
|---|--:|--:|:--:|
| tiage | 0.3628 (F1 0.2043 / Pk 0.4692 / WD 0.4882) | 0.363 | ✅ |
| dialseg711 | 0.3816 (F1 0.2446 / Pk 0.4702 / WD 0.4926) | 0.382 | ✅ |
| superseg | 0.4667 (F1 0.3851 / Pk 0.4455 / WD 0.4579) | 0.471 | ✅ (Δ0.004, nltk 버전차) |

→ **소수 3째자리 일치 = 확정 채점기가 진짜 공식임 증명.** TextTiling 은 nltk 결정론적
알고리즘이라 우리 실행 = paper 실행이 동일해야 하고, 실제로 맞았다.

### 검증 범위 한계 (정직)
- baseline 중 **TextTiling 만** 검증. GreedySeg/CSM/GraphSeg/RoBERTa 는 **보류**(사용자 결정):
  이 머신 GPU(RTX PRO 6000 Blackwell, **sm_120**)를 설치된 `torch 2.5.1+cu124`(arch ≤ sm_90)가
  지원 안 해 CUDA 연산 실패 → neural baseline 은 CPU 전용(느림), RoBERTa 는 학습 ckpt 부재.
  GPU 쓰려면 `torch≥2.7 cu128`(sm_120 포함) 필요. TextTiling 만으로 채점기 확정엔 충분.

---

## 4. 경계 정렬 규약 (off-by-one — 반드시 인지)

- **⚠ 이 §4 는 DTS 전용 — AMI 는 규약이 정반대다** (2026-06-14 확정): **DTS gold = 끝-turn**, **AMI gold `bnd_top` = 시작-turn**
  (`topic_levels.start_turn`, 135/139 일치). ⇒ DTS 신호 = shift -1(규약정렬), **AMI 신호 = shift 0**(신호 첫-turn ↔ AMI gold 첫-turn).
  AMI 에 끝-turn 가정/-1 을 적용하면 오류. [[feedback_ami_gold_start_turn_convention]]
- **DTS gold 규약 = 끝-turn**: `topic_id` 가 다음 turn 에 바뀌기 *직전* turn 이 label=1.
  (tiage/dialseg711/superseg 전부 확인: label=1 위치에서 `topic[i]==topic[i-1] != topic[i+1]`, 100%.)
- **δ_eff/임베딩 신호는 새 segment *첫* turn(t)에서 솟음** → 경계(이전 segment 끝)는 t-1.
  ⇒ 채점 시 **-1 매핑** 필요. (공식 TextTiling 은 끝-turn=1 로 내므로 shift 없음.)
- **shift sweep 증거** (δ_eff, dialseg711, p80 고정, pooled F1): shift -1 = 0.537(정상),
  shift 0 = 0.088, shift +1 = 0.220.
- ⚠ **HANDOFF_01 의 `ami_dts_*.py::oc()` 는 shift 0 (off-by-one 버그)** — δ_eff 를 차별적으로
  억눌러 "de-neut 우위 / superseg 벽 0.467 돌파" 라는 착시를 만듦. (그 DTS 축1 결론은 무효.)

---

## 5. 후속 작업 (이 handoff 범위 밖, 별도 처리)

- (a) **`src/` 에 단일 공식 채점기 모듈** 신설(레포 `SegmentationEvaluation` 래핑 + bit-일치
  단위테스트) → 모든 러너가 이것만 호출.
- (b) `run_encoder_comparison.py::score_set` 의 **F1 풀링 → per-dialogue 교정** + 전 실험 재채점.
- (c) **decision-log / methodology 정정**: off-by-one 버그, CR 의 DTS 회귀(공식·정상정렬에서
  δ_eff > CR), DTS 디폴트 = δ_eff 복귀, dts_result.md 오기(offline TT dialseg711 0.482 → **0.382**).
- (d) ✅ **완료(2026-06-14)**: AMI 정렬 재확인 결과 **AMI gold=시작-turn (DTS 와 반대)** → **AMI 신호 = shift 0 이 규약정렬**.
  shift -1(0.370)은 filler-lag overfit, ±2 가 ±1 잔차 흡수 → 결론 불변(shift0: de-neut 0.342 > δ_eff 0.214). §4 상단 경고 참조.

---

## 6. AMI 공식 채점기 확정 (2026-06-15)

DTS(§1) 와 짝으로 **AMI 채점기도 단일 확정**. 코드: **`src/hi_ontop/ami_scoring.py`** (+ `tests/test_ami_scoring.py`, 6/6).
권위 출처(동치, parity Δscore=0.00000 검증): `scripts/ami_adaptive_deneut_deploy.py::{ev, tol_f1, load_ami}` — AMI 의 모든 수치를 만든 그 채점기.

| 항목 | AMI 확정값 | DTS(§1) 대비 |
|---|---|---|
| 정렬 | **shift 0** (gold=start-turn) | DTS 는 shift -1 (end-turn) |
| F1 | **±2 tolerant** boundary F1 (`tol_f1`, t=2) | DTS 는 exact binary |
| Pk/WD | nltk pk/windowdiff, window=auto(max(2,round(len/(sum+1)/2))) | **동일** |
| Score | 0.5·F1 + 0.25(1−Pk) + 0.25(1−WD) | **동일** |
| 집계 | per-meeting 평균 | **동일** (per-dialogue) |
| 마지막 turn / interior | label·pred=0 / 예측 0<p<n-1 | 동일 |

**검증**:
- **정렬 = start-turn (데이터 확정)**: `bnd_top` 경계 = depth-1 `topic_levels.start_turn`(최초 시작 turn 제외). 예 ES2002a: bnd_top=[11,20,77,94,146,202] = start_turn[1,11,20,77,94,146,202]−{1}. → 신호(새 segment 첫 turn) 와 같은 규약 → shift 0.
- **parity**: 새 `ami_scoring.score_meetings` == 레퍼런스 `ev` (현 디폴트 de-neut: F1 0.131/Pk 0.490/WD 0.646/Score **0.282**, Δ=0).

**★ 수치 정정**: 인용해온 "현 디폴트 AMI Score 0.372" 는 **best-c** 값(decision-log 2026-06-12 "best-c by Score"). **c=1.0 디폴트는 0.282.** (AMI 도 c=1.0 비최적 — c sweep 별도.)
