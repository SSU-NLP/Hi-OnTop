# CLAUDE.md — Hi-OnTop 프로젝트 규칙

## 🐳 컨테이너 환경 규칙 (최우선 규칙, 2026-06-11)

현재 작업 환경은 **Docker 컨테이너**(`/workspace` 마운트)다. 다음을 무조건 지킨다.

1. **보존할 산출물은 마운트된 `/workspace` 하위에만 저장한다.** 컨테이너는 삭제·재생성되면 마운트 밖 경로(`/tmp`, `/root/` 등)의 내용이 전부 소실된다. 스크립트·REPORT·figure·실험 결과 등 보존이 필요한 모든 것은 `/workspace` 안에 둔다. (순수 임시 중간 파일만 `/tmp` 허용.) → handoff에 남은 `/tmp/` 탐색 스크립트(`proto_beta`, `beta_loo` 등)는 소실 위험이므로 재현 가치가 있으면 즉시 `scripts/`로 승격한다.

2. **HF 캐시 마운트 필수 (연구실 공통 규칙).** 모든 docker / cuda / vllm 실행 시 아래 마운트를 반드시 추가한다. 모델은 호스트의 이 경로에 한 번만 받아 모든 컨테이너가 공유 → 재다운로드 없음.
   ```
   -v /root/.cache/huggingface:/root/.cache/huggingface
   ```
   컨테이너 안에서 모델 로드 시에도 캐시 경로는 `/root/.cache/huggingface` 그대로 쓴다. 재다운로드를 유발하는 다른 캐시 경로 지정 금지.

3. **GPU / vLLM 실행 시 필수 옵션.** GPU 컨테이너는 `--gpus all`(특정 장치는 `--gpus '"device=0,1"'`)을 붙이고, 딥러닝/추론은 `--shm-size=16g` 이상을 준다. HF 캐시 마운트(위 2번)도 항상 함께. vLLM 추론 서버 예:
   ```bash
   docker run -d --gpus all --name vllm \
     -v /root/.cache/huggingface:/root/.cache/huggingface \
     -p 8000:8000 --shm-size=16g \
     vllm/vllm-openai:latest --model <모델ID> [--tensor-parallel-size N]
   ```

## 📋 handoff 폴더 관리 (최우선 규칙, 2026-06-12)

**해결해야 할 문제(과제)는 `handoff/` 폴더에 _문제 1개 = 파일 1개_ 로 관리한다.**

1. **생성 트리거 = 사용자 지정뿐.** handoff 파일(=추적할 문제)은 **사용자가 명시적으로 지정할 때만** 만든다.
   Claude 가 _자발적으로_ "이건 문제니 handoff 만들자" 하지 않는다. 사용자가 "이 문제 handoff 에 추가" 류로
   지정하면, 그때 `handoff/<문제-slug>.md` 를 생성하고 **문제를 정의**(배경·목표·성공기준)한다.
2. **내용 = 문제 정의 + (수록할) 시도의 설정·조건 + 결과 (필수 갱신).** 지정된 문제로 작업할 때마다 그 문제의
   handoff 파일에 시도의 설정/조건/결과를 갱신한다. **단, handoff 는 "논문용 기록"이라는 점을 기준으로 선별한다.**
   - **수록 대상 선별 (논문용 기록 원칙)**: **쓸데없는 탐색 흔적·dead-end·최종 선택에서 제외된 시도**는
     *적지 않아도 된다*(노이즈 배제 — 다음 세션/협업자·논문 독자가 헷갈리지 않게). 무엇이 논문 본문/appendix·
     최종 비교에 들어갈 시도인지로 판단한다.
   - **수록하기로 한 시도는 관련 설정을 전부 기록**: HP·데이터(출처·전처리·split)·method·config·seed·
     버퍼/프롬프트/모델·평가지표 정의·스크립트 경로 등을 **하나도 빠짐없이**. (재현 가능해야 한다 — 부분 기록 금지.)
   - 각 수록 시도의 **결과**(수치·판정·실패사유)를 업데이트한다.
   - **진행사항·일시적 상태는 적지 않는다**(= ledger 아님): API 예산 소진, "지금 실행중", 프로세스 생존 같은
     ephemeral status 는 handoff 가 아니라 구두 보고로. handoff 에는 *시도·조건·결과·설계 결정* 만 남긴다.
   필수 — 지정 문제로 **수록 가치 있는** 실험/시도를 하면 *반드시* 해당 handoff 를 갱신한다(실험 결과 REPORT
   규칙과 별개로, handoff 는 그 문제의 "선별된 시도 ledger" 역할).
3. **위치/형식.** `handoff/` 폴더 1곳. 파일명 = **`HANDOFF_<MMDD>_<문제키워드>.md`**
   (예: `HANDOFF_0612_realtime-llm-comparison.md`). 날짜=문제 지정일 MMDD(연도 생략), 키워드=kebab-case 핵심어.
4. **자발적 남발 금지.** 사용자가 지정하지 않은 문제는 handoff 에 넣지 않는다. 한 문제의 시도가 다른 문제로
   번지면 **새 문제로 지정할지 사용자에게 묻고**, 지정 전엔 추가하지 않는다.

이유: 문제별로 "정의 + 모든 시도 조건 + 결과"가 한 파일에 누적되면 다음 세션/협업자가 _무엇을 어떤 조건으로
해봤고 어떻게 됐는지_ 즉시 복원. 단 사용자가 추적하기로 한 문제만 관리해 남발을 막는다.

## 중요 의사결정은 항상 codex로 답변 (최상위 규칙)

설계 옵션 선택, 수식 확정, 모듈 구조 결정, 알고리즘 트레이드오프 평가 등 **프로젝트의 향후 방향에 영향을 주는 의사결정 질문**은 Claude가 직접 답하지 않고 반드시 `codex:rescue` 서브에이전트로 위임해 한국어로 답변한다.

판단 기준 (다음 중 하나라도 해당하면 codex 위임):
- "어느 방법이 더 나은가" / "무엇을 선택해야 하나" / "이게 적절한가" 형태의 비교·판단 질문
- 수식·모델 설계 변경에 대한 권고
- mega-topic, segmentation, prior/likelihood 등 핵심 알고리즘 결정
- decision-log에 남길만한 결정

단순 코드 수정·디버그·문서 업데이트·요약은 Claude가 직접 처리해도 된다.

위임 시 사용자가 이전 codex 스레드의 컨텍스트를 이어가는 후속 질문이면 `--resume`을 붙인다.

## SEM 계승 원칙 (최상위 설계 제약)

Hi-OnTop은 SEM / SEM2 계승 모델이다. **SEM(원논문) / SEM2(`nicktfranklin/SEM2`)에 존재하지 않는 메커니즘은 합리적 근거 없이 도입하지 않는다.**

대상 예시 (SEM에 없으므로 default = 도입 안 함):
- vMF likelihood, multi-prototype topic, per-topic concentration κ 학습
- bounded cosine 외의 새로운 likelihood 함수
- 학습형 dynamics 외의 새로운 prediction error 정의
- SEM이 명시적으로 "쓰지 않는다"고 가정한 임의의 새 component

도입을 검토하려면 다음 셋을 모두 충족해야 한다:
1. **SEM에 왜 없는지** (의도적으로 빠진 건지, 단순 미구현인지) 명시한다.
2. 추가했을 때 얻는 이점이 SEM의 철학(sCRP / Bayes / local MAP / scene dynamics)과 **충돌하지 않음**을 보인다. 충돌하면 어느 쪽을 우선할지 명시한다.
3. `context/06-decision-log.md`에 근거 + 날짜를 append한다.

신규 component 제안이 들어오면 Claude의 default 응답은 **"SEM에 있나?"** — 없으면 위 세 단계의 정당화 부담을 그 제안에 얹은 채로만 검토한다.

이유: SEM 계승이라는 정체성을 잃으면 Hi-OnTop은 그저 "임의로 조합된 retrieval heuristic"이 된다. 새 메커니즘은 SEM에서 빠진 기능을 복원하거나, SEM 가정의 한계를 명시적으로 인정한 위에서만 추가한다.

## Step 완료 프로토콜 (최상위 강제 규칙)

### 1단계: 3-angle Self-Audit (check_step_done 실행 **전**)

`check_step_done.py`는 길이·키워드 수준의 피상 검증만 한다. 그 전에 반드시 다음 세 각도에서 **자기질문-자기답변**을 수행한다:

1. **구조 이해** — 결과물(논문/데이터/설계)의 **형태와 구성요소의 역할**을 스스로에게 질문하고 답할 수 있는가?
2. **동작/inference 이해** — 알고리즘·처리 흐름·검색 로직을 끊김 없이 설명할 수 있는가? 수식이라면 각 변수의 역할과 식 간 연결을 정확히 복원할 수 있는가?
3. **설계 방향 이해** — 이 결과물이 Hi-OnTop의 다음 결정(옵션 선택/수식 확정/모듈 설계)에 어떻게 연결되는가? 무엇이 열려 있는가?

**최소 3 Q&A per angle**. 답하지 못하거나 근거가 약한 항목은 **해당 결과물 파일의 "검증 미해결" 섹션에 명시**한다 (제거 금지, 솔직 기록).

Self-audit 자체는 세션 내에서 진행하고 별도 파일로 저장하지 않아도 된다. 단, **gap으로 식별된 것은 반드시 결과물에 기록**한다.

### 2단계: `check_step_done.py` 실행

```bash
python scripts/check_step_done.py
```

- **exit code 0이 나올 때까지 Step 완료 처리 금지.**
- FAIL이 남아있으면 원인을 수정하고 스크립트를 재실행한다. **통과할 때까지 반복한다.**
- WARN은 허용 가능하지만, 가능하면 해결한다.
- 검증 없이 `[x]` 처리하거나 커밋하지 않는다.
- Step에 대한 검증 로직이 `STEP_CHECKS`에 없으면 추가한 뒤 진행한다.

### 금지

- self-audit 건너뛰고 곧장 `check_step_done.py`만 돌려 통과시키기 금지. 스크립트 통과는 **필요조건이지 충분조건이 아니다.**

## 실험 결과 저장 (최상위 강제 규칙)

**어떤 벤치마크든 (TIAGE / TopiOCQA / AMI / dialseg711 / superseg / 기타) 모든 실험 결과는 `outputs/experiments/<name>/REPORT.md` 한 곳에 저장한다.** 다른 위치 (`outputs/reports/`, `outputs/` 루트, 임의 디렉토리) 에 실험 결과 REPORT 를 만들지 않는다.

### REPORT.md 필수 구성

자동 생성된 표만으로는 부족하다. 모든 REPORT.md 는 다음 섹션을 **자세하고 명확하게** 포함한다:

1. **실험 setup** — 목적 / 데이터 (path + n_conv/n_turn/n_question) / 방법 list / HP (실제 사용한 default + override) / seed / metric 정의 / 비교 baseline.
2. **결과 표** — mean ± std (3-run 이상) + 핵심 지표 + best 강조.
3. **해석** — 숫자가 의미하는 것, 왜 이런 패턴이 나오는지, 어디까지가 noise 범위 안인지.
4. **판정** — iteration 내에서 method 별 향상/동일/회귀 분류, 다음 iteration 결정 (재구현 / 다음 단계 / 보류).
5. **한계 / 검증 미해결** — HP 적합성, seed 부재, atomicity, 표본 크기 등 솔직 기록 (제거 금지).

자동 생성된 짧은 표만 남긴 REPORT 는 미완성으로 간주.

### outputs/reports/ 의 역할 (정정)

`outputs/reports/` 는 **실험 결과 저장 금지**. cross-experiment methodology 비교, design rationale doc, 일반 분석 등 **단일 실험에 귀속되지 않는 문서** 만 둔다. (이전 정책에서 "TIAGE/TopiOCQA segmentation 비교"를 `reports/` 에 두던 관행은 본 규칙으로 대체됨.)

### Single-script 실험에도 적용

`scripts/run_*.py` 형태의 단독 script (예: `run_tiage_full_compare.py`) 도 default output 을 `outputs/experiments/<name>/REPORT.md` 로 둔다. `--name` 인자 + 그 폴더 안에 self-contained.

## 산출물 / 실험 디렉토리 사용법 (canonical, 최상위 규칙)

**용어**:
- *experiment* = 이름 붙은 여러 *run* 의 집합 (sweep, ablation, comparison 모두 동일).
- *run* = method × HP 한 조합.

### 1. 새 experiment 실행 = 분절 러너 스크립트 한 entry

분절 실험은 standalone 러너로 돈다 — `scripts/run_topiocqa_segmentation.py`,
`scripts/run_tiage_segmentation.py`, `scripts/ami_*.py`, `scripts/run_*_compare.py` 등.

규칙:
- 새 sweep / ablation / comparison 마다 임시 shell 남발 **금지** — 기존 러너의 `--name` /
  HP 인자를 쓰거나, 재사용 가능하면 `scripts/` 에 러너로 승격.
- 각 러너는 `--name <date>_<descriptor>` 폴더 안에 self-contained 산출 —
  `outputs/experiments/<name>/REPORT.md` 자동 생성 (Pk / WindowDiff / boundary-F1 +
  n_topics + seg latency p50 + wall).

### 2. 단일 ad-hoc run

디버그 / smoke / 일회성 단발 실행은 `outputs/runs/` 로 보낸다. 자료가치 있으면
`outputs/runs/<date>/<exp_id>/` 에 자동 누적.

### 3. 산출물 디렉토리 구조

```
outputs/                  # 모든 active/historical generated. top-level 1개.
├── experiments/             # experiment 산출 — self-contained
│   └── <name>/<run_label>/{run.log, exit_code.txt, results/...}
├── runs/                    # 단독 ad-hoc 실행 (날짜 subdir 누적)
├── reports/                 # cross-experiment methodology/design 분석 MD only. 실험 결과 저장 금지 (위 § "실험 결과 저장" 참조).
└── design/                  # 설계 문서 (committed)

archive/                  # *의도적으로 폐기* 한 것만 (시간 흐름 무관)
├── legacy_sem_ablation/     # 폐기된 SEM ablation variant 보존
└── README.md                # 왜 버렸는지 기록
```

**원칙**:
- "오래됐다" 와 "폐기됐다" 는 **다른 것**. archive/ 는 *폐기 결정* 했을 때만. 단순히 시간 지나서 뒤로 밀린 데이터는 `outputs/runs/<date>/` 에 그대로 둔다.
- 새 dir 만들지 말고 위 4 카테고리 (`experiments/`, `runs/`, `reports/`, `design/`) 중 하나에 넣기. 의미 모호하면 `outputs/runs/_misc/`.
- archive 에 새 항목 추가할 땐 `archive/README.md` 에 *폐기 사유 + 날짜* 한 줄 append. 사유 없는 archive 추가 금지.
- `outputs/experiments/<name>/` 안의 데이터는 self-contained. 한 experiment 의 모든 정보 (config, log, hypothesis, summary, REPORT) 가 이 한 디렉토리 안에서 답이 나와야 함.

### 4. `scripts/legacy/` — 읽기 전용

옛 per-experiment shell (`run_*sweep.sh`) + 옛 aggregator (`aggregate_*.py`) 보관됨. 참고용. **새 스크립트 추가 금지**, 기존 파일도 수정하지 않는다 — 현재 유효한 entry 는 `scripts/` 의 분절 러너(`run_topiocqa_segmentation.py`, `run_tiage_segmentation.py`, `ami_*.py` 등).

### 5. git 정책

- committed: `outputs/experiments/<name>/REPORT.md`, `outputs/reports/*.md`, `outputs/design/*`, `archive/<name>/README.md`
- gitignored: `outputs/runs/`, `outputs/experiments/*/*/results/`, `outputs/experiments/*/*/run.log`, `archive/*/outputs/*.{jsonl,json,wandb-run-id}`
- `.gitignore` 변경은 cascade 검사 대상.

### 6. wandb logging (필수)

실험 러너는 wandb 로 자동 로깅한다 (project convention 2026-05-08~). 러너가 wandb run 을 시작하고, round 별 metric + final summary 를 push 한다.

**1회 setup**: `uv run wandb login` 또는 `.env` 에 `WANDB_API_KEY=...` 추가. 인증 안 돼 있으면 wandb init 가 실패 catch 되어 *no-op* 으로 fallback (실험은 진행되지만 logging 안 됨 — `[wandb] init failed` 경고).

**opt-out**: 특정 run 만 끄고 싶으면 `WANDB_MODE=disabled uv run python scripts/<runner>.py ...` 로 prefix.

**확인**: 실행 후 `outputs/experiments/<name>/<label>/run.log` 의 wandb 라인 또는 wandb 웹 UI 에서 `hi-ontop` (default `--wandb-project`) 확인.

## `context/methodology/` 가 최우선 관리 대상 (최상위 규칙)

**가장 1순위로 관리한다.** 코드 수정 / 설계 결정 / 인프라 변경 시 가장 먼저 확인·갱신해야 할 디렉토리.

이유: 버전(v1/v2/v3.1.1/v3.2.1/v3.3.1) 별 *알고리즘 수준 정의* 와 cross-cutting 인프라 결정이 모이는 단일 진실 원천. 다른 docs (handoff, plan, decision-log) 는 *현재 진행 상황* 을 기록하지만, methodology/ 는 *설계 자체* 가 무엇인지 기록한다.

기록 범위 (사소해 보여도 빠짐없이):
- 알고리즘 수식 / score 식 / update rule
- topic state 의 모든 필드, hyperparameter default
- SEM 계승 측면 (있는 것 / 없는 것 / 변형한 것)
- 알려진 한계 + 변형 후보 (적용 안 했어도 "고려 중인 변형" 섹션에 누적)
- cross-cutting 인프라 (`EncoderCache`, `HiOnTopConvCache`, encoder lock, LLM 호환 플래그(`--no-thinking`), retrieval policy, STM atomicity 등)

규칙:
- 버전마다 1 파일 (`vX.md`). 새 버전 추가 시 직전 버전 파일을 템플릿 삼아 같은 구조 유지.
- 버전 간 공유 항목은 `infrastructure.md` 에 추가. 항목 형식: 무엇 / 어디(file:line) / 왜 / 행동 영향(어느 method) / 알려진 한계.
- *알고리즘 의미가 바뀌는* 변경은 새 버전 분리. 단순 성능/캐시 최적화는 같은 파일 안 "고려 중인 변형" 또는 "변경 이력" 섹션 누적.
- 어떤 파일을 수정했든, 그 변경이 알고리즘·infrastructure 측면에서 의미가 있으면 *반드시* 해당 methodology 파일에 한 줄 이상 반영.
- 의미가 모호하면 default = 기록한다. 기록은 cheap, 망각은 expensive.

cascade 검사는 (아래 § 참조) `context/methodology/` 를 가장 먼저 확인 대상으로 둔다.

## 파일 수정 시 최신성 cascade 검사 (필수)

Claude Code가 파일 1개를 수정·생성·삭제할 때마다 **다른 파일에 영향 가능성이 있는지 즉시 검사**한다. 영향받을 가능성이 있는 파일은 **사용자에게 한꺼번에 제시하고 같이 업데이트할지 묻는다**.

확인 대상 (우선순위 순):
- `context/methodology/*` — **최우선** (알고리즘·인프라 변경 시). 위 § 참조.
- `README.md` — 디렉토리 구조, 외부 레포, gitignored 목록, 현재 상태
- `plan.md` — 체크박스, Phase 진행률, 결과 수치
- `handoff.md` — 현재 상태, 다음 할 일, 마지막 업데이트 날짜
- `context/04-benchmarks.md` — 데이터 / 평가 축 변경 시
- `context/03-architecture.md` — 모듈·파일 추가/삭제·이름 변경 시
- `context/06-decision-log.md` — 설계 결정 변경 시 (append-only)
- `context/sem-equations.md` — SEM 식 관련 작업 시
- `report.md` — Phase 결과·미해결 사항 변경 시
- `.gitignore` — 새 파일 패턴 추가/제거 시

검사 방법:
- 변경 키워드(파일명/모듈명/Step 번호/Phase 결과)를 `grep -rn` 으로 다른 docs에서 검색
- 발견된 파일 + 무엇이 stale해 보이는지 사용자에게 보고
- 사용자 응답 (전체 / 일부 / skip) 받은 후 일괄 수정

목적: docs 간 불일치 누적 방지. 한 파일만 고치고 다른 곳 잊으면 다음 세션 / 협업자가 잘못된 정보로 작업.

## 장기 실행 작업 진행 점검 (필수)

10분 초과로 예상되는 작업(실험·학습·평가 등)은 **시작 후 10분이 지나면 반드시 한 번 진행 상황을 확인**한다. background task / `run_in_background=true` / 별도 프로세스 모두 적용.

방법:
- `ScheduleWakeup` (delaySeconds=600 이하) 으로 10분 내 자가 점검 예약
- 체크 시점에 stdout 마지막, exit status, results 디렉토리 (`results/experiments/<exp-id>/checkpoints/latest.json` 또는 `summary.json`) 셋 다 확인
- 진행 정상이면 다음 점검(또 10분 이내) 예약, 정체/오류면 **즉시 사용자에게 보고**

이유: vLLM 멈춤·STM 폭주·OOM·import error 등 silent failure가 발생해도 사용자가 모르고 기다리는 일을 막는다. 한 번 시작하고 던져두지 않는다.

## Figure 저장 규칙 (최상위 강제 규칙, 2026-05-25)

논문/리뷰용으로 노출되는 모든 figure 의 *canonical* 저장 위치는
`outputs/figures/` 한 곳. 파일명은 **`figure_<알파벳>_<제목>.{pdf,png}`** 형식 통일.

- **<알파벳>**: A, B, C, ... 순서. **새 figure 추가** 시 가장 작은 미사용
  letter 부여. **기존 figure 수정·재생성** 시 동일 letter 유지 (PDF 캐시
  refresh 위해 timestamp 만 갱신, letter 안 바꿈).
- **<제목>**: snake_case, 그 figure 의 핵심 한 단어 (예: `latency_scatter`,
  `pareto_qa_context`, `delta_eff_distribution`).
- **PDF + PNG 둘 다** 저장 (PDF = paper 삽입, PNG = preview / Read tool).
- experiment 디렉토리 (`outputs/experiments/<exp-name>/plots/`) 에 원본 generation 결과를 두는 건 OK 지만, paper 에 들어갈 *최종본* 은 반드시
  `outputs/figures/` 로 copy. plot script 수정 → 재생성 → 자동으로
  `outputs/figures/` 에 sync 까지 한 번에 해 줄 것 (사용자가 따로 안 시켜도).

현재 letter assignment (참조):
- A — `band_precision` (Hi-OnTop band precision)
- B — `latency_scatter`
- C — `timeline_tape_conv3` (Long-MT-Bench+ conv 3 timeline)
- D — `boundary_density_*` (4 분량: downstream + DTS 3종)
- E — `segment_length_violin`
- F — `boundary_agreement`
- G — `pareto_qa_context`
- H — `calib_n_convergence_{main,appendix_oracle,appendix_sup}`
- I — `delta_eff_distribution`
- L — `percentile_score_curve` (DTS Score vs percentile)
- P — `distill_n_convergence_mtbp` (MTB+ LLM-distill N-convergence, p∈[5,95])
- Q — `mtbp_percentile_curve` (MTB+ F1 vs percentile, Fig L counterpart, p∈[5,95])
- J, K, M, N, O — 다른 figure 가 차지 (별도 추적). 새 figure 는 R 부터.

이유: figure 수정·인용 시 letter 가 결정적인 reference key. paper LaTeX
`\ref{fig:I}` / 본문 "Figure I" 와 파일명이 즉시 매칭되어야 함. 다른 위치
(plots/, reports/, 임시 dir) 에 흩어진 채로 paper 에 삽입 금지.

## 환경 분리

- `setup_colab.ipynb`는 항상 `.gitignore` 유지. git에 커밋하지 않는다.
- `setup_colab.ipynb`는 로컬/Colab 공용 setup notebook으로 유지한다.
- 그 외 모든 파일은 Colab 전용 의존 없이 로컬 기준으로 동작해야 한다.
  - `from google.colab import drive`, `drive.mount`, `/content/` 경로를 기본 파일에 넣지 않는다.
  - Colab 전용 코드가 꼭 필요하면 `IS_COLAB` 분기 안에만 작성한다.
- 경로는 하드코딩 금지. `git rev-parse --show-toplevel` 또는 상대경로 사용.

## Notebook 실행 정책

- **모든 실험 notebook(`notebooks/*.ipynb`)은 `setup_colab.ipynb` 선행 실행을 가정한다.**
- 실험 notebook 안에 환경 셋업(repo clone, 벤치마크 clone, 패키지 설치, 데이터 다운로드, 모델 다운로드) 로직을 **중복으로 넣지 않는다.** setup_colab이 단일 책임자.
- 실험 notebook의 첫 셀들은 `setup_colab` 사전 조건이 만족됐는지 **검증만** 하고, 부족하면 명확한 에러 메시지(`'setup_colab.ipynb 먼저 실행'`)로 실패시킨다.
- 이유: 환경 셋업 로직이 여러 notebook에 흩어지면 동기화 부담 + 혼란. setup_colab만 유지·업데이트하면 모든 실험이 따라옴.

### Notebook ↔ Script 분리 원칙 (portability)

- **모든 실험 로직은 `scripts/*.py`에 둔다.** notebook은 그 스크립트를 `subprocess.run(['python', 'scripts/X.py', ...])`로 호출하는 **얇은 wrapper**일 뿐이다.
- `notebooks/` 디렉토리 통째로 삭제해도 프로젝트가 그대로 동작해야 한다 — 로컬 GPU·다른 환경 전환 시 `python scripts/X.py` 직접 실행으로 모든 실험 가능.
- notebook이 추가로 갖는 가치: Colab kernel 연동, IPython 출력 렌더링(`display(Markdown(...))`), 셀 단위 인터랙티브 디버깅. 이 가치 외엔 `.py` 스크립트로 옮긴다.
- `setup_colab.ipynb`만 예외 — 환경 셋업은 본질적으로 노트북 형식이 자연스러워 그대로 둠 + `.gitignore`로 제외.

### Tracking 정책

- `setup_colab.ipynb`: **gitignored** (Colab 전용 환경 셋업, 일회성 도구)
- `notebooks/*.ipynb` 그 외 모두: **git tracked** (연구 기록, 협업자 공유). 단 위 portability 원칙 위반 시 무효 → script로 분리.

## 코딩 스타일

- Python 3.10+ (match statement, union types 활용)
- Type hint 필수 (`from __future__ import annotations`)
- Docstring: Google style
- Line length: 100

## 파일 조직

- 한 모듈 = 한 책임
- 순환 import 금지
- 코어 코드: `src/hi_ontop/`
- 실험/스크립트: `scripts/`
- 테스트: `tests/`

## 커밋 규칙

- **git 실행 정책 (2026-05-18 변경)**: 사용자가 명시적으로 요청하면(`commit/push/merge 해줘` 등) Claude Code 가 `git add`/`commit`/`push`/`merge` 를 **직접 실행한다**. Claude 가 *자발적으로* git 작업을 하려는 경우(사용자가 시키지 않았는데 커밋/푸시가 적절하다고 판단)는 **실행 전 한 번 확인**을 받는다. 단, `main` 등 보존 브랜치 병합처럼 전략·비가역 영향이 큰 작업은 명시 요청이어도 위험을 1줄 고지 후 진행한다.
- 한 커밋 = 한 논리적 단위
- 제목 50자 이내, 본문은 이유 중심
- prefix: `feat`, `fix`, `docs`, `refactor`, `test`, `exp`

## 문서화

- 새 모듈 추가 시 `context/03-architecture.md` 반영
- 설계 결정 시 `context/06-decision-log.md`에 근거 + 날짜 append
- 실험 시작 전 `templates/experiment-log.md` 복사해서 로그 생성

## 테스트

- pytest
- 필수 테스트: sCRP 계산, topic assignment, centroid 업데이트
- 재현성: 모든 randomness에 seed 고정

## 외부 레포 사용

- `benchmarks/*`: 읽기 전용
- `SEM/` (= SEM2, `nicktfranklin/SEM2` current build): 참조 전용, 코드 복사 금지

## 금지 사항

- 메인 LLM fine-tuning 금지
- TensorFlow/Keras 사용 금지 (PyTorch만)
- SEM2 코드 직접 복사 금지 (참조만 허용)
- 설계 문서 업데이트 없이 구조 변경 금지
- 벤치마크 분석 없이 설계 확정 금지
