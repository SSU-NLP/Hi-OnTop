# archive/ — 의도적으로 폐기한 실험들

`archive/` 는 *시간 지난 것* 이 아니라 *폐기 결정한 것* 만 보관한다. 새 항목 추가 시 아래 형식으로 한 줄 append 필수:

```
- <date>/<dir>: <한 줄 폐기 사유>
```

## 폐기 항목 이력

- **2026-04-26-baseline/** (2026-04-26): Phase 4 reference baseline. Hi-OnTop persistence + nothink + full-500 LongMemEval = 0.562 overall — 4 method 중 꼴찌 (sliding/full/RAG 모두 우위). multi-session 0.23 / knowledge 0.51. → **Phase 4 종료, baseline 으로 동결, 재실행 금지**. 자세히는 `2026-04-26-baseline/README.md`.
- **2026-04-28/** (2026-04-28, 24 runs): Phase 4 follow-up `freq_shift` / `persistence` / `sliding` / `hi-ontop-full-v1` HP 변형. baseline 결과를 못 뒤집음 → Phase 4 같이 폐기.
- **2026-04-29/** (2026-04-29, 14 runs): 같은 사유로 폐기.
- **2026-04-30/** (2026-04-30, 1 run): Phase 4 마지막 시도, 폐기.
- **legacy_sem_ablation/** (2026-05-22): v4.1.3 의 full SEM2 form (`sem_core_v413.py`). 2026-05-22 실증 audit 으로 v4.1.3 default/canonical setting 의 SEM2 machinery (per-topic EventRNN / f0·restart·re-entry / scaled-inv-χ² variance / sticky-CRP α 및 default λ) 가 출력에 영향 없음이 확인됨 — `_fresh_baseline_for_prev` 의 prior-cancel 설계로 repeat-vs-fresh 결정이 `δ_eff < δ*` 로 환원. 단 낮은 `lmda` 값은 archived full form 에서 일부 출력 차이를 낼 수 있어 전역 dead 로 보지 않는다. main 모델은 `src/hi_ontop/hi_ontop.py` (`HiOnTop`, reduced form, matched-HP byte-identical 검증) 로 대체. 이 파일은 **삭제 아님 — audit 재현용 ablation 증거물** 로 보존 (paper 의 "we implemented and audited SEM2-style ... found degenerate" disclosure 근거). decision-log 2026-05-22 참조.

## archive 에 들어오지 않는 것

- 옛 sweep / ablation 의 raw experiment data → `outputs/runs/<date>/`
- 단순히 오래된 standalone run → `outputs/runs/<date>/`
- 진행 중 실험의 중간 산출물 → `outputs/experiments/<name>/`

archive 는 *"이 디렉토리 건드릴 일 없음"* 의 표식. 부활시켜 비교용으로 쓰려면 그대로 두고 새 experiment 에서 baseline 으로 referencing 하는 게 정석.
