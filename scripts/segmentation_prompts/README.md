# segmentation_prompts/ — 분절 LLM 프롬프트 단일 출처

모든 분절 관련 LLM 프롬프트의 **단일 출처**. 스크립트는 인라인 하드코딩 대신 여기서 `open().read()` 한다.
(2026-06-22 `secom_prompts/` → `segmentation_prompts/` 개명 + 인라인 프롬프트 일원화.)

## 프롬프트 목록

| 파일 | 무엇 | 쓰는 스크립트 | 결과/비고 |
|---|---|---|---|
| `judge_binary_neutral_v1.md` | **우리 메인 judge** — 도메인중립 "현재 발화가 NEW topic인가 SAME인가" binary | `llm_judge_universal.py` (`SYS_PROMPT`) | **헤드라인**: AMI Score 0.503(gpt-4o)/0.451(mini), DTS dialseg 0.796 등 (HANDOFF_01 §H) |
| `judge_binary_meeting_v0.md` | [SUPERSEDED] meeting 전용 초기 judge (agenda/facilitator 언급) | `llm_judge_spikes.py` (`SYS_PROMPT`) | 8/20미팅 프롬프트 반복(§E). neutral_v1 로 대체됨 |
| `baseline_segment_v1.md` | SeCom 원본 — 입력 전체를 segment(partition). B-mode 누적 | `measure_streaming_table.py`(v2), `secom_llm_eval.py` | v1 baseline Score 0.50 (HANDOFF_03). ⚠ segment-all = 20분 붕괴(§K) |
| `baseline_segment_v2.md` | SeCom 요약-carryover streaming | `measure_streaming_table.py` | Score 0.43 (HANDOFF_03 §2) |
| `baseline_segment_v3.md` | raw 슬라이딩 ≤4분 윈도우 | `measure_v3.py` | Score 0.42 (HANDOFF_03 §2) |
| `baseline_segment_binary_v1.md` | **SeCom 최소변경 binary** — Context/criterion 유지 + Question만 "현재 turn이 경계?" | `recompare_binary_judge.py`, `binary_breakpoint_fullpast.py --prompt secom` | §K 붕괴 실험. every-turn Score 0.358(과분절), peak 0.408 |
| `baseline_incremental_v1.md`, `segment_exchange.md`, `segment_incremental.md` | SeCom 기타 변종 | `secom_llm_eval.py` | 참고용 |

## 규칙
- **인라인 프롬프트 금지** — 새 프롬프트는 여기 `.md` 1개로 추가하고 스크립트는 read.
- judge 프롬프트는 cache 키(=user message)와 무관(system msg)이지만 **텍스트 변경 시 동작이 바뀌므로** 버전(`_vN`)을 올린다.
- 어느 프롬프트가 어느 수치를 냈는지 위 표에 1줄 기록 (재현/논문 추적).

## 아직 일원화 안 된 것 (TODO, 저위험)
- 버퍼 윈도우 프롬프트: `WIN_PROMPT`(ami_llm_buffer_eval.py·llm_latency_full.py 중복), `BUF_PROMPT`(compare_buffer_vs_twostage.py),
  `SYS`(buffer_fullpast.py) — 셋 다 인라인. baseline 성격이라 후순위. 통합 시 `buffer_window_v1.md` 로.
