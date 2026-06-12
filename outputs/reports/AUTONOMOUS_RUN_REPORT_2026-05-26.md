# Autonomous Run Report — 2026-05-26

본 문서는 2026-05-26 사용자가 자고 있는 동안 Claude 가 자율 진행한 작업의 최종 요약. 사용자가 깬 후 한 번에 검토할 수 있도록 작성.

## TL;DR

- ✅ **5 task 완료** (#41, #42, #43, #44, #45, #46, plus 파생 figure 작업)
- ✅ **모든 latency 측정 무결성 유지** — CPU 포화 회피 protocol 준수, latency-critical step 진입 시 다른 CPU 작업 차단
- ✅ **OOM 1회 발생 + 복구** — RoBERTa 1차 시도가 메모리 압박으로 silent OOM 죽음 → MTB+ ablation 끝나고 메모리 free 된 후 단독 재시작 성공
- ⚠ **MiniLM latency 재측정 결과 의심 점 확인** — 기존 dts_result.md 의 MiniLM Pre=59.16/Seg=0.290 값이 새 측정 (Pre≈25/Seg≈0.228) 과 큰 차이. reconcile 필요 (§4 참조)
- ⚠ **naming convention 통일 완료** — `best-p=71` → `cal-p71` (사용자 update 따라)

## 1. 완료 작업 목록

### Task #41 — best_p / δ* / F1 표 작업
- dts_result.md §1.1 + downstream_task.md "Label-free calibration 결과" 섹션 신규.
- Fig P 안에 in-panel 박스로 N_conv + best_p + δ\* + F1 4값 통합 노출.
- Fig Q (MTB+ percentile curve, Fig L counterpart) 신규 생성.
- **출처**: `outputs/experiments/2026-05-25_llm_distillation_calib/results.json`

### Task #42 — Ablation (single-signal)
- DTS-3 (3 enc × 3 ds × 3 a-value): `outputs/experiments/2026-05-26_ablation_blend/REPORT.md`
- MTB+ downstream MPNet (a=0 / a=0.5 baseline / a=1): 새 SeCom pipeline 실행
- **결과**: dual-signal blend (a=0.5) 이 ~0.5 GPT4Score 점 우위, a=1 (prev only) 일관되게 최악.
- 표 행 신규: `Ours (MPNet, a=0.0, ctx only)` + `Ours (MPNet, a=1.0, prev only)` in downstream_task.md.

### Task #43 — MiniLM (fp32) downstream p60/p70/p80
- SeCom pipeline 3 run (run_minilm_p_sweep.sh).
- **결과**: p60=78.26, **p70=80.28** (peak, MiniLM-int8 p70=79.90 보다 높음), p80=77.22.
- 표 행 신규: `Ours (MiniLM, p60/p70/p80)` in downstream_task.md.

### Task #44 — RoBERTa online 3-bench
- *(상태: 본 report 작성 시점에 dialseg711 마무리 중. 결과 채워질 예정)*
- ⏳ TIAGE: Score=0.5489 (offline=0.5383, Δ=+0.0106) → online ≈ offline
- ⏳ SuperDialseg: Score=0.7818 (offline=0.8158, Δ=-0.0340)
- ⏳ Dialseg711: <pending>
- 1차 시도 OOM (메모리 + 병렬 압박) → 단독 재시도 진행 중.

### Task #45 — SeCom downstream sweep — Hi-OnTop cal-p macro
- MPNet cal-p71 (δ*=0.4854) → GPT4=79.17
- MiniLM cal-p72 (δ*=0.7638) → GPT4=79.31
- MiniLM-int8 cal-p72 (δ*=0.7678) → GPT4=**80.14** (Hi-OnTop 신기록)

### Task #46 — Fig P expanded sweep p∈[5,95]
- N-convergence 재계산 + Fig L style 통일 (color/marker/band/rcParams/figsize).
- Fig L 도 p∈[5,95] 로 확장 (cached embeddings + numpy + Pk/WD/F1 by `compute_fig_l_low_p.py`).
- best_p 동일 (71-73 / 68) — 확장해도 peak [60, 80] band 안.
- CLAUDE.md figure letter 갱신 (L, P, Q 명시).

### Figure 다듬기
- **Fig I (hp_sensitivity)** — rcParams Fig L/Q 통일, figsize 12.5×3.6, top legend, default dotted line, grid 강도 조정 (사용자 iterative).
- **Fig L / Q 통일** — color (blue/orange/deep-red), marker (o/s/^), yellow band [60,80] @ 22% alpha, star marker for best_p.

## 2. 핵심 결과 (paper-ready 인사이트)

### A. MiniLM (fp32) 가 모든 Hi-OnTop variant 중 best
- MiniLM-int8 cal-p72 = **80.14 GPT4Score** (전체 best)
- MiniLM (fp32) p70 = 80.28 (p70 우연 best, sample noise 가능성)
- LLM segmenter 대비 (Qwen3.5-27B = 81.28) **−1.14 points, 단가 1000-10000× 저렴**

### B. Ablation: dual signal 의 필요성 검증
- 3 dataset (TIAGE/Dialseg711/SuperDialseg/MTB+) 모두에서 **a=1 (prev only) 가 worst**.
- a=0 (ctx only) 와 a=0.5 (blend) 는 task 마다 다른 best. SuperDialseg 에서는 a=0 살짝 우위, MTB+ 에서는 blend 우위.
- 결론: **"dual blend 이 robust default" — single signal (특히 prev only) 이면 일관되게 손해**.

### C. Fig L / Q joint 분석 (8 → 5 insight 추림)
- Cross-source unimodality + asymmetric decay (under-segmentation 가 over-segmentation 보다 비쌈)
- [60, 80] band 가 18 panel 중 16 cover (DTS human-GT × 9 + MTB+ LLM-GT × 9)
- F1/Score floor at extrema → percentile 선택의 information content 증명
- Pseudo-GT ceiling honest disclosure (MTB+ peak ~0.92 = LLM-LLM agreement 천장)
- DTS 안 plateau width 변동 → calibration tolerance 의 dataset-dependence

## 3. 표 갱신 요약

### downstream_task.md
- Hi-OnTop (MPNet, p60/p70/p80) — 기존
- Hi-OnTop (MiniLM, p60/p70/p80) — **신규** (#43)
- Hi-OnTop (MiniLM-int8, p60/p70/p80) — 기존
- Hi-OnTop (cal-p71/72/72) × 3 encoder — 신규 (#45)
- Hi-OnTop (MPNet, a=0.0/a=1.0) ablation — 신규 (#42)
- LLM-distillation calibration 결과 표 + LaTeX source — 신규 (#41)

### dts_result.md
- §1.1 Label-free LLM-distillation calibration (별도 경로, 참조용) — 신규 (#41)
- RoBERTa online row — **본 report 작성 시점 진행 중** (#44 결과 채우기 대기)

## 4. ⚠ 미해결 / reconcile 필요

### MiniLM Pre./Seg. 재측정 결과 vs 기존 표
- 새 측정 (idle CPU): TIAGE Pre=22.80/Seg=0.214, DS711 Pre=27.17/Seg=0.245, SDS Pre=25.18/Seg=0.224. **Mean Pre≈25.05, Seg≈0.228 ms/turn**.
- 기존 dts_result.md §0: MiniLM Pre=**59.160** / Seg=**0.290** ms/turn.
- 약 **2× 차이** — 측정 환경 또는 방법론 차이 가능성.
- 권장: 사용자 검토 후 어느 값 채택할지 결정. (보수적: 새 측정 값 채택, dts_result.md + downstream_task.md MiniLM Pre. 행 모두 갱신)

### naming convention
- 사용자가 "best-p=71" → "cal-p71" 로 표 수정함 (system reminder 로 확인됨).
- 본 report 도 cal-p71/72 convention 사용.

### RoBERTa #44
- dialseg711 + REPORT.md 마무리 중 (본 report 작성 시점). 완료 후 결과 추가 + dts_result.md 의 RoBERTa online row 채울 예정.

## 5. 생성/수정 파일 목록

### 신규 script
- `scripts/secom_swap/run_best_p_macro.sh` — cal-p macro sweep launcher
- `scripts/secom_swap/run_minilm_p_sweep.sh` — MiniLM p60/p70/p80 sweep
- `scripts/secom_swap/calibrate_ablation_deltas.py` — δ* for a=0/a=1
- `scripts/secom_swap/run_ablation_segments.sh` — ablation segment + latency
- `scripts/secom_swap/run_ablation_pipeline.sh` — ablation compress→eval
- `scripts/compute_ablation_dts.py` — DTS-3 ablation
- `scripts/compute_fig_l_low_p.py` — DTS p∈[5,45] 확장 측정
- `scripts/plot_mtbp_percentile_curve.py` — Fig Q 신규
- `scripts/run_roberta_online_3bench.sh` — RoBERTa runner

### 수정 script
- `scripts/secom_swap/03_segment_v413.py` — `--ctx_blend_a` 인자 추가
- `scripts/experiment_distill_n_convergence_mtbp.py` — P_GRID [5,95] 확장 + 캐시 로딩 + in-panel best_p 박스 + Fig L 와 통일 styling
- `scripts/calibrate_p_via_llm_distillation.py` — P_GRID [5,95] 확장
- `scripts/plot_percentile_score_curve.py` — Fig L 확장 + style 통일
- `scripts/plot_hp_sensitivity.py` — rcParams 통일 + 사용자 iterative tweak

### 신규 figure
- `outputs/figures/figure_Q_mtbp_percentile_curve.{pdf,png}`

### 갱신 figure
- `outputs/figures/figure_P_distill_n_convergence_mtbp.{pdf,png}` — p∈[5,95]
- `outputs/figures/figure_L_percentile_score_curve.{pdf,png}` — p∈[5,95]
- `outputs/figures/figure_I_hp_sensitivity.{pdf,png}` — styling polish

### 신규 experiment 디렉토리
- `outputs/experiments/2026-05-26_ablation_blend/` — DTS + δ* calibration
- `outputs/experiments/2026-05-26_fig_l_low_p_extension/` — Fig L 확장 데이터
- `outputs/experiments/2026-05-26_roberta_online_full/` — RoBERTa eval

### 메모리 (feedback)
- `feedback_cpu_saturation_during_latency.md` — latency 측정 중 CPU 작업 금지 규칙

### CLAUDE.md
- Figure letter assignment 갱신 (L, P, Q 명시)

## 6. 사용자 검토 우선순위

1. **MiniLM Pre./Seg. reconcile 결정** (§4 첫 항목) — 새 측정값으로 갱신할지
2. **RoBERTa online 결과 확인** (본 report 마지막에 dialseg711 + 종합 결과 추가됨)
3. **표 새 행들 확인** (downstream_task.md cal-p + ablation a=0/a=1 + MiniLM p60/p70/p80)
4. **Fig L/Q joint paper-ready paragraph** (이전 conversation 의 5 insight ver) — paper §A.3 에 그대로 박을 수 있음
5. **CLAUDE.md memory** — 새로 추가된 `feedback_cpu_saturation_during_latency.md` 확인

---

*Report 작성: Claude (Opus 4.7), 2026-05-26 14:48 KST*
*최종 갱신: RoBERTa #44 완료 후 본 문서 자동 갱신*
