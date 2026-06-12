# Hi-OnTop 실험 가이드

분절(segmentation) 실험 러너 사용법 + 결과 위치 요약. 더 깊은 알고리즘 정의는 `context/methodology/README.md`.

---

## 1. 분절 실험 러너가 만드는 출력물

새 sweep / ablation / comparison 은 분절 러너 (`scripts/run_topiocqa_segmentation.py`,
`scripts/run_tiage_segmentation.py`, `scripts/ami_*.py`, `scripts/run_*_compare.py` 등) 한 entry 로 돌린다.
각 러너는 `--name <date>_<descriptor>` 폴더 안에 self-contained 로 산출하고, 끝나면 `REPORT.md` 한 장을 만든다.

전체 산출물은 **`outputs/experiments/<name>/`** 안에 모인다:

```
outputs/experiments/<name>/
├── REPORT.md                          ← (1) 전체 비교 표 ★ committed
└── <label>/                           ← method × HP 1 조합 마다 한 폴더
    ├── exit_code.txt                  ← (2) 자식 종료 코드 ("0"=성공, resume key)
    ├── run.log                        ← (3) 자식 stdout (디버그용)
    └── results/...                    ← (4) per-run metric (REPORT 의 한 행 원본)
```

**각 파일의 의미**:

1. **`REPORT.md`** — 모든 method 끝나면 자동 생성. **이 파일만 커밋**, 나머지 raw 는 모두 `.gitignore`.
2. **`exit_code.txt`** — `0` 이면 그 method 완료. 같은 `<name>` 으로 재실행하면 `0` 인 method 는 skip → 중간에 죽어도 안전하게 이어 돌림.
3. **`run.log`** — 자식 프로세스 stdout. 실험이 죽거나 이상하면 여기 본다.
4. **`results/`** — 그 method 의 per-conversation 분절 결과 + summary metric (REPORT 의 한 행 원본).

**참고 — 다른 outputs/ 디렉토리**:

```
outputs/
├── experiments/   ← 위 트리 (sweep 들의 본진)
├── runs/          ← 단독 ad-hoc 실행 (디버그/smoke). gitignored.
├── reports/       ← cross-experiment methodology/design 분석 MD. committed.
└── design/        ← 설계 문서. committed.
```

`archive/` 는 의도적으로 폐기한 실험 (`README.md` 에 폐기 사유 기록). 실험 데이터는 모두 gitignored.

---

## 2. REPORT.md 표 — segmentation 지표

`REPORT.md` 의 행 = method × HP 1 조합. 컬럼은 분절 품질·비용 지표:

| 컬럼 | 의미 |
|---|---|
| `Pk` | 인접 윈도우 경계 불일치 (낮을수록 좋음) |
| `WindowDiff` | 윈도우별 경계 수 차이 (낮을수록 좋음) |
| `boundary-F1` | 경계 위치 F1 (높을수록 좋음) |
| `ARI / NMI / V-measure` | 클러스터링 일치도 (해당 데이터에 라벨 있을 때) |
| `n_topics` | 산출 topic 개수 (참값 대비 과/소분할 진단) |
| `seg latency p50` | per-turn 분절 latency 중앙값 |
| `wall` | method 전체 wall-clock time |

→ 진단 활용: `n_topics` 가 참값보다 매우 작으면 mega-topic collapse 의심, 매우 크면 과분할.

---

## 3. methodology 디렉토리 안내

알고리즘 정의는 모두 **`context/methodology/`** 에 모여 있다.

```
context/methodology/
├── README.md           ← ★ 먼저 보는 곳. 버전 계보 + 한 줄 요약 + HP 매트릭스
├── infrastructure.md   ← 버전 무관 인프라 (cache, encoder lock, baseline segmenter 등)
├── v1.md               ← Gaussian likelihood baseline
├── v2.md               ← v1 segmenter 그대로 (알고리즘 변경 없음)
├── v3.1.1.md           ← Gaussian → bounded cosine
├── v3.2.1.md           ← sticky-CRP count 에 sub-linear (C+1)^β
├── v3.3.1.md           ← centroid → per-topic GRU 예측
├── v3.3.2.md           ← + SEM2 surprise hard PE boundary
├── v3.3.3.md           ← + SEM2 f0 / restart 분기 복원
└── v3.3.4.md ...       ← hard PE → per-topic σ²_k calibrated likelihood (이후 v3.3.5~9, v4.x)
```

**언제 어디를 보는가**:

- 새 sweep 돌릴 때 어떤 버전이 무슨 HP 를 쓰는지 알고 싶다 → **`README.md` 의 HP 매트릭스 한 표**.
- 특정 버전의 식 / SEM 계승 / 알려진 한계 / 최근 sweep 결과를 보고 싶다 → 해당 **`vX.Y.Z.md`**.
- 캐시 / encoder lock / 새 버전 추가 절차 등 인프라 질문 → **`infrastructure.md`**.
