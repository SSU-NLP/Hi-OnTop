# Segment Counts & Boundary Density — Long-MT-Bench+ Baselines

`benchmarks/SeCom/experiment/result/mtbp/*/segments.jsonl` · 2026-05-24

Long-MT-Bench+ 11 conversations · 720 turns 위에서 `downstream_task.md` 의 모든 baseline
(LLM-based 의 GPT-5 / Qwen-122B 포함) 의 분절 통계.

**정의**
- **`분절 밀도` (bnd/turn)** = total boundaries / total turns = (`# Seg total` − 11) / 720.
  *분절 밀도의 표준 metric*. 0 에 가까울수록 sparse, 1 에 가까울수록 over-segmentation.
- `# Seg total` = 11 conv 전체의 segment 합.
- `mean / med / max seg len` = segment 안의 turn 수 통계 (보조 지표).
- *Zero / Full History* 는 분절 안 함 → 해당 컬럼 `--`.

## 값 표 — **분절 밀도 (bnd/turn) 기준**

| Category | Method | GPT4Score | **분절 밀도** ↓ | # Seg | mean len | med | max |
|---|---|---:|---:|---:|---:|---:|---:|
| history | Zero History | 42.12 | -- | -- | -- | -- | -- |
| history | Full History | 77.92 | -- | -- | -- | -- | -- |
| unsup | TextTiling-Style-Seg | 73.47 | 0.297 | 225 | 3.20 | 4 | 7 |
| unsup | GraphSeg-Style-Seg | 62.53 | **0.106** | 87 | **8.28** | 7 | 16 |
| unsup | GreedySeg-Style-Seg | 68.58 | 0.197 | 153 | 4.71 | 5 | 9 |
| unsup | CSM-Style-Seg | 64.24 | **0.100** | 83 | **8.67** | 9 | 16 |
| **unsup (ours)** | **Ours (p60)** | 78.75 | 0.426 | 318 | 2.26 | 2 | 4 |
| **unsup (ours)** | **Ours (p70)** | **79.90** | **0.336** | 253 | 2.85 | 3 | 8 |
| **unsup (ours)** | **Ours (p80)** | 75.87 | 0.242 | 185 | 3.89 | 3 | 11 |
| sup | RoBERTa-Style-Seg | 74.44 | 0.372 | 279 | 2.58 | 2 | 13 |
| LLM | GPT-4o-mini-Seg | 78.13 | 0.426 | 318 | 2.26 | 2 | 7 |
| LLM | **GPT-5-Seg** | **80.62** | **0.310** | 234 | 3.08 | 3 | 7 |
| LLM | **Qwen3.5-122B-Seg** | **80.83** | **0.322** | 243 | 2.96 | 3 | 6 |
| LLM | Qwen3.5-27B-Seg | **81.28** | 0.328 | 247 | 2.91 | 3 | 5 |
| LLM | Qwen3.5-4B-Seg | 76.77 | 0.302 | 228 | 3.15 | 3 | 9 |
| LLM | Qwen3.5-2B-Seg | 72.81 | **0.592** | 435 | **1.65** | 1 | 12 |
| LLM | Llama3.2-3B-Seg | 71.60 | 0.340 | 253 | 2.81 | 3 | 13 |
| LLM | Mistral3-3B-Seg | 76.91 | 0.372 | 276 | 2.58 | 3 | 8 |

**Bold = 극단치** (가장 sparse 0.10 / 가장 dense 0.59) + GPT-5 / Qwen-122B 강조.

## 빠르게 보는 클러스터 — **분절 밀도 (bnd/turn) 기준**

| 분절 밀도 band | Methods | 특징 |
|---|---|---|
| **극단 sparse** (≤ 0.15) | GraphSeg (0.106), CSM-Style (0.100) | 큰 chunks (mean 8+ turn). 분절 *회피* 경향. |
| **sparse** (0.15~0.30) | GreedySeg (0.197), Ours-p80 (0.242), TextTiling (0.297) | 3~5 turn segments. coherent passage 보존. |
| **medium** ⭐ | Qwen-4B (0.302), **GPT-5 (0.310)**, **Qwen-122B (0.322)**, **Qwen-27B (0.328)**, **Ours-p70 (0.336)**, Llama-3B (0.340), RoBERTa (0.372), Mistral-3B (0.372) | 2.6~3.2 turn segments. **강한 LLM (top1~3) cluster + Hi-OnTop default 모두 여기**. |
| **dense** (0.40~0.50) | Ours-p60 (0.426), GPT-4o-mini (0.426) | 2.3 turn segments. boundary 적극. |
| **극단 dense** (≥ 0.50) | Qwen-2B (0.592) | 1.65 turn — 거의 turn 단위. 작은 LLM 의 over-segmentation. |

**핵심 발견**:
- **GPT4Score top3 LLM (Qwen-27B 0.328 / Qwen-122B 0.322 / GPT-5 0.310) 모두 medium
  band (0.30~0.34)** — 큰 LLM 들이 *수렴하는 분절 밀도 영역* 존재.
- **Ours-p70 (0.336) 이 top3 LLM 평균 (0.320) 과 거의 일치** — Hi-OnTop 의 segmentation
  F1 기반 calibration default 가 자동으로 강한 LLM 의 분절 밀도와 같은 위치에 수렴.
- GPT-4o-mini (0.426) 만 LLM 중 *과분절* outlier. small LLM (Qwen-2B 0.592) 은 더 심함.

## ours percentile 별 분절 밀도 (sensitivity)

| Percentile | **분절 밀도** | mean len | GPT4Score | 비교 |
|---|---:|---:|---:|---|
| p60 | 0.426 | 2.26 | 78.75 | = GPT-4o-mini (0.426) |
| **p70 (default)** | **0.336** | **2.85** | **79.90** | ≈ Qwen-27B (0.328) / GPT-5 (0.310) |
| p80 | 0.242 | 3.89 | 75.87 | < TextTiling (0.297), sparser |

## LaTeX (paper §4 또는 supplementary 용)

```latex
%% 모든 baseline 의 분절 밀도 (boundary density = bnd/turn) %%
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{l|c|c|cccc}
\toprule
\textbf{Methods}
& \textbf{GPT4Score}
& \textbf{Boundary Density}
& \textbf{\# Seg}
& \textbf{mean len}
& \textbf{med}
& \textbf{max} \\
\midrule

Zero History & 42.12 & -- & -- & -- & -- & -- \\
Full History & 77.92 & -- & -- & -- & -- & -- \\

\midrule
\multicolumn{7}{c}{\textbf{Unsupervised}} \\
\midrule

TextTiling-Style-Seg  & 73.47 & 0.297 & 225 & 3.20 & 4 & 7  \\
GraphSeg-Style-Seg    & 62.53 & 0.106 &  87 & 8.28 & 7 & 16 \\
\midrule
GreedySeg-Style-Seg   & 68.58 & 0.197 & 153 & 4.71 & 5 & 9  \\
CSM-Style-Seg         & 64.24 & 0.100 &  83 & 8.67 & 9 & 16 \\
Ours (p60)            & 78.75 & 0.426 & 318 & 2.26 & 2 & 4  \\
Ours (p70)            & 79.90 & 0.336 & 253 & 2.85 & 3 & 8  \\
Ours (p80)            & 75.87 & 0.242 & 185 & 3.89 & 3 & 11 \\

\midrule
\multicolumn{7}{c}{\textbf{Supervised}} \\
\midrule

RoBERTa-Style-Seg     & 74.44 & 0.372 & 279 & 2.58 & 2 & 13 \\

\midrule
\multicolumn{7}{c}{\textbf{LLM-based (SeCom)}} \\
\midrule

GPT-4o-mini-Seg       & 78.13 & 0.426 & 318 & 2.26 & 2 & 7  \\
GPT-5-Seg             & 80.62 & 0.310 & 234 & 3.08 & 3 & 7  \\
Qwen3.5-122B-Seg      & 80.83 & 0.322 & 243 & 2.96 & 3 & 6  \\
Qwen3.5-27B-Seg       & 81.28 & 0.328 & 247 & 2.91 & 3 & 5  \\
Qwen3.5-4B-Seg        & 76.77 & 0.302 & 228 & 3.15 & 3 & 9  \\
Qwen3.5-2B-Seg        & 72.81 & 0.592 & 435 & 1.65 & 1 & 12 \\
Llama3.2-3B-Seg       & 71.60 & 0.340 & 253 & 2.81 & 3 & 13 \\
Mistral3-3B-Seg       & 76.91 & 0.372 & 276 & 2.58 & 3 & 8  \\

\bottomrule
\end{tabular}
\caption{Boundary density across all baselines on Long-MT-Bench+ (11 conversations, 720 total turns). \textbf{Boundary Density} = total boundaries / total turns (the standard segmentation density metric). \textbf{\# Seg} = total segments. Zero/Full History do not segment. Hi-OnTop's default (\textit{Ours (p70)}, density 0.336) closely matches the top-3 LLM cluster (Qwen-27B 0.328, Qwen-122B 0.322, GPT-5 0.310), well below the over-segmentation of GPT-4o-mini (0.426) and Qwen-2B (0.592).}
\label{tab:boundary_density_baselines}
\end{table*}
```

## 검증 미해결

- 본 표는 11 conversation 의 segment **합** 을 보고. per-conv 분산은 따로 측정 안 했음.
- GraphSeg / CSM 의 *극단적 under-segmentation* (87 / 83 segs) 은 MTB+ 에 적용된 online
  변형의 특성 — 원 SuperDialseg 결과 (`dts_result.md`) 의 분절 양상과 일치.
- *RoBERTa* (sup) 의 segment count 279 는 다른 sup 비교군이 없어 단독 판단.
- Ours (p60) 와 GPT-4o-mini-Seg 의 segment 통계가 일치 (318/28.91/2.26) — *우연*. boundary
  위치는 다름 (cf. `ours_vs_llm_segmentation_pattern.md` retrieval overlap 58.3%).
- GPT-5 / Qwen-122B 는 추가 분석 (`ours_vs_llm_segmentation_pattern.md`) 의 LLM top3 cluster
  에 포함됨 — Ours-p70 와 boundary F1 ≥ 0.75 예상 (재측정 권장).
