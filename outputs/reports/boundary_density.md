# Boundary Density — Segmentation Benchmarks (Def-DTS Bundle)

`benchmarks/Def-DTS/data/DTS_session_datasets/*_test.jsonl` · 2026-05-24

세 segmentation 벤치마크의 경계 밀도 (boundary density) 통계. δ\* 가 도메인별로
달라지는 원인 (cosine geometry + boundary density) 중 *boundary density* 측을
quantitative 로 뒷받침.

**정의**
- `density_per_turn` = total boundaries / total turns (전체 turn 당 경계 비율,
  가장 핵심 지표).
- `seg_len_*` = segment 길이 (인접 경계 사이 turn 수, *gold* 기준).
- `bnds_per_dial_*` = dialog 당 경계 수.

## 값 표

| Dataset | # Dial | # Turn (mean / med) | # Boundary / Dial (mean / med) | **Density / turn** | # Seg / Dial | Seg Len (mean / med) |
|---|---:|---:|---:|---:|---:|---:|
| TIAGE        | 100  | 15.64 / 16 | 3.15 / 3 | **0.201** | 4.15 | 3.77 / 3 |
| Dialseg711   | 711  | 26.22 / 25 | 3.86 / 4 | **0.147** | 4.86 | 5.40 / 5 |
| SuperDialseg | 1322 | 12.11 / 12 | 3.04 / 3 | **0.251** | 4.04 | 3.00 / 2 |

**밀도 순위 (sparse → dense)**: Dialseg711 (.147) < TIAGE (.201) < SuperDialseg (.251).

## 해석 — boundary density ↔ best $p_x$

`delta_star_calibration.md` 의 best $p_x$ (percentile family 안 최적 $p$):

| Dataset | density | MPNet best p | MiniLM best p | MiniLM-int8 best p |
|---|---:|---|---|---|
| SuperDialseg | **0.251** (densest) | p60 | p50 | p50 |
| TIAGE        | 0.201 | p65 | p70 | p80 |
| Dialseg711   | **0.147** (sparsest) | p85 | p80 | p85 |

**Trend**: density ↑ → best $p_x$ ↓.

- SuperDialseg (densest, 0.251) → best $p_x \in \{p50, p60\}$ (낮은 percentile).
- Dialseg711 (sparsest, 0.147) → best $p_x \in \{p80, p85\}$ (높은 percentile).
- TIAGE (중간) → best $p_x \in \{p65, p80\}$ (중간 percentile).

**직관**: dense boundary 데이터셋은 boundary 가 자주 발생 → 낮은 $\delta_\text{eff}$
threshold 로 잡아야 함 → 낮은 percentile. sparse 데이터셋은 강한 surprise 만 boundary
로 인정해야 false positive 안 남 → 높은 percentile.

## LaTeX (paper §3 또는 §4 ablation 용)

```latex
%% Boundary density 표 %%
\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
\textbf{Dataset}
& \# Dial
& \# Turn
& \# Bnd / Dial
& \textbf{Density}
& Seg Len \\
\midrule
TIAGE         & 100  & 15.6 & 3.15 & 0.201 & 3.77 \\
Dialseg711    & 711  & 26.2 & 3.86 & \textbf{0.147} & 5.40 \\
SuperDialseg  & 1322 & 12.1 & 3.04 & \textbf{0.251} & 3.00 \\
\bottomrule
\end{tabular*}
\caption{Boundary density statistics of the three segmentation benchmarks (Def-DTS test splits). \textbf{Density} = total boundaries / total turns. SuperDialseg has the densest boundaries (one boundary every $\approx 4$ turns); Dialseg711 the sparsest ($\approx 7$ turns). Mean turns per dialogue and mean segment length are also reported. Boundary density correlates with the optimal calibration percentile $p_x$ in Table~\ref{tab:encoder_segmentation} (denser $\Rightarrow$ lower $p_x$).}
\label{tab:boundary_density}
\end{table}
```

## 검증 미해결

- 본 density 는 *test split* 기준. δ\* 보정은 일부 인코더/벤치는 train split
  서브샘플 → train 의 density 도 동일한지는 별도 검증 미수행 (분포가 시기/소스
  같으면 동일 추정).
- "density ↔ best $p_x$" 는 9 셀 (3 encoder × 3 bench) 관찰. 더 많은 도메인에서
  같은 trend 확인하면 strong claim 가능. 현재는 *qualitative trend*.
- Density 외 다른 가능 원인: 평균 segment 길이, turn 길이, 화제 유형 (open
  domain vs doc-grounded). 본 표는 density 만 일차 지표로 보고.
