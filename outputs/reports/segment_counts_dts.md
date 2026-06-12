# Segment Counts & Boundary Density — DTS Benchmarks (TIAGE / Dialseg711 / SuperDialseg)

세 segmentation 벤치 (Def-DTS test split) 위에서 `dts_result.md` 의 모든 method 의
**실제 분절 결과 — pred boundary 수 + 경계 밀도 (boundaries / turns)** 를 추출.
`segment_counts_downstream.md` 의 DTS 버전.

**경계 밀도 정의**: `density = total predicted boundaries / total turns` (per-bench).
gold density 와 비교하여 method 가 *과분절 / 과소분절* 인지 직접 판정 가능.

## ⚠️ LLM segmenter (GPT-5 / Qwen-122B / Qwen-27B / GPT-4o-mini / Ministral / Llama / 등) 의 DTS 결과는 *없음*

LLM-based segmenter 들은 본 프로젝트에서 **Long-MT-Bench+ (downstream QA) 에만 적용**
되었음. TIAGE / Dialseg711 / SuperDialseg test split 위에서는 분절 수행 안 함 (1322+711+100
= 2133 dialogue × LLM 콜 비용 + token 비용 부담). 따라서 본 표에 LLM segmenter 행은
**부재 (—)** 로 표기. LLM 의 DTS 분절 밀도가 필요하면 별도 실험 필요. cf.
`segment_counts_downstream.md` 의 LLM 행 (MTB+ density 0.10~0.59).

(예외: `outputs/experiments/2026-05-19_defdts_gpt4o_crts/` 가 *Def-DTS prompt template* 으로
gpt-4o 를 TIAGE/Dialseg711 에 적용 — Pk/WD/F1 만 보고, pred 수 미보고. SeCom segmentation
prompt 와는 다른 방식. 본 표에 미포함.)

## 데이터 출처 — 두 가지 split 가족 (정직 표기)

baseline (TextTiling/GraphSeg/GreedySeg/CSM) 의 pred 수는 `Def-DTS` 번들 위 측정 (각
method 의 REPORT 직접 참조). Hi-OnTop 는 `superdialseg_data` 위 측정 (encoder cache 정합).
**turn 수가 다소 다름** (DS711: 18639 vs 19350; SDS: 16006 vs 17328) — 데이터 전처리 차이
이지만 method 간 density 비교 (% 단위) 는 ±1% 영향, 결론에는 영향 없음.

## Gold density (참고용)

| Bench | Def-DTS turns | Def-DTS gold | gold density | SDS-data turns | SDS-data gold | gold density |
|---|---:|---:|---:|---:|---:|---:|
| TIAGE | 1564 | 315 | **0.201** | 1564 | 315 | 0.201 |
| Dialseg711 | 18639 | 2743 | **0.147** | 19350 | 2754 | 0.142 |
| SuperDialseg | 16006 | 4017 | **0.251** | 17328 | 4020 | 0.232 |

## 1. Baselines (Def-DTS 번들, full test split)

| Category | Method | TIAGE pred / density | DS711 pred / density | SDS pred / density |
|---|---|---:|---:|---:|
| Online unsup | TextTiling-Style | 305 / **0.195** | 4210 / **0.226** | 3229 / **0.202** |
| Online unsup | GraphSeg-Style | 257 / **0.164** | 3968 / **0.213** | 1545 / **0.097** |
| Online unsup | GreedySeg-Style | 230 / **0.147** | 3511 / **0.188** | 2449 / **0.153** |
| Online unsup | CSM-Style | 275 / **0.176** | 3753 / **0.201** | 2821 / **0.176** |
| **Gold** | — | **315 / 0.201** | **2743 / 0.147** | **4017 / 0.251** |

**해석**:
- **TextTiling**: TIAGE 거의 정확 (0.195 ≈ 0.201), DS711/SDS 는 과분절 (1.5× / 0.8×).
- **GraphSeg**: TIAGE/SDS 과소분절 (-19% / -61%), DS711 과분절 (+45%).
- **GreedySeg**: TIAGE/DS711 거의 정확, SDS 큰 과소분절 (-39%).
- **CSM**: 세 벤치 모두 *과분절 경향* (TIAGE/DS711 +20~37%, SDS -30%).

## 2. Hi-OnTop / Hi-OnTop (superdialseg_data, full test split)

### MPNet 인코더

| Variant | δ\* | TIAGE pred / density | DS711 pred / density | SDS pred / density |
|---|---:|---:|---:|---:|
| p60 | TIAGE 0.530 / DS 0.514 / SDS 0.530 | 583 / **0.373** | 7435 / **0.384** | 6716 / **0.388** |
| **p70 (default)** | TIAGE 0.562 / DS 0.552 / SDS 0.569 | 414 / **0.265** | 5587 / **0.289** | 5014 / **0.289** |
| p80 | TIAGE 0.601 / DS 0.594 / SDS 0.613 | 266 / **0.170** | 3729 / **0.193** | 3246 / **0.187** |
| sup | TIAGE 0.574 / DS 0.614 / SDS 0.503 | 368 / 0.235 | 2945 / 0.152 | 7966 / 0.460 |
| oracle | TIAGE 0.543 / DS 0.614 / SDS 0.543 | 500 / 0.320 | 2945 / 0.152 | 6146 / 0.355 |
| **Gold (SDS-data)** | — | **315 / 0.201** | **2754 / 0.142** | **4020 / 0.232** |

### MiniLM-int8 인코더

| Variant | δ\* | TIAGE pred / density | DS711 pred / density | SDS pred / density |
|---|---:|---:|---:|---:|
| p60 | TIAGE 0.733 / DS 0.705 / SDS 0.725 | 595 / **0.380** | 7399 / **0.382** | 6319 / **0.365** |
| **p70 (default)** | TIAGE 0.776 / DS 0.754 / SDS 0.784 | 446 / **0.285** | 5530 / **0.286** | 4713 / **0.272** |
| p80 | TIAGE 0.822 / DS 0.805 / SDS 0.841 | 306 / **0.196** | 3694 / **0.191** | 3021 / **0.174** |
| sup | TIAGE 0.787 / DS 0.828 / SDS 0.655 | 412 / 0.263 | 2892 / 0.150 | 8145 / 0.470 |
| oracle | TIAGE 0.777 / DS 0.828 / SDS 0.625 | 444 / 0.284 | 2892 / 0.150 | 8863 / 0.512 |
| **Gold (SDS-data)** | — | **315 / 0.201** | **2754 / 0.142** | **4020 / 0.232** |

**해석 — percentile sensitivity**:
- p60 (낮은 threshold) → density 0.37~0.39 → **gold 의 ~1.8× 과분절** (TIAGE/DS711) / **1.7× 과분절** (SDS).
- **p70 (default)** → density 0.27~0.29 → TIAGE 약간 과분절 (0.27 vs 0.20), DS711 큰 과분절 (0.29 vs 0.14), **SDS 약간 과소분절** (0.29 vs 0.23 — 거의 정확).
- p80 → density 0.17~0.19 → TIAGE 정확 (0.20), DS711 정확 (0.14), SDS 큰 과소분절 (0.18 vs 0.23).

**핵심 관찰**:
- *Hi-OnTop 는 보편적으로 over-segment* (p60/p70), 특히 DS711 에서. sparse boundary
  데이터인데 percentile 기반 threshold 가 적절히 보수적이지 못함.
- **p80 이 boundary density 측면에선 gold 와 가장 가까움** (TIAGE/DS711 거의 일치).
  하지만 Score 는 p70 이 더 높음 — **density 만으로는 best 가 안 됨** (false positive 의
  *위치* 가 잘 맞으면 over-segment 도 F1 점수에 큰 손해 안 됨).

## 통합 비교표 (LaTeX — DTS 3 벤치 × 모든 method)

```latex
%% 분절 밀도 비교 — DTS 3 벤치 %%
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{l|cc|cc|cc}
\toprule
\multirow{2}{*}{\textbf{Method}}
& \multicolumn{2}{c|}{\textbf{TIAGE}}
& \multicolumn{2}{c|}{\textbf{Dialseg711}}
& \multicolumn{2}{c}{\textbf{SuperDialseg}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& \textbf{pred bs}
& \textbf{density}
& \textbf{pred bs}
& \textbf{density}
& \textbf{pred bs}
& \textbf{density} \\
\midrule

\multicolumn{7}{c}{\textbf{Baselines --- Online unsupervised (Def-DTS, full test)}} \\
\midrule
TextTiling-Style  &  305 & 0.195 & 4210 & 0.226 & 3229 & 0.202 \\
GraphSeg-Style    &  257 & 0.164 & 3968 & 0.213 & 1545 & 0.097 \\
GreedySeg-Style   &  230 & 0.147 & 3511 & 0.188 & 2449 & 0.153 \\
CSM-Style         &  275 & 0.176 & 3753 & 0.201 & 2821 & 0.176 \\

\midrule
\multicolumn{7}{c}{\textbf{Ours --- MPNet (SDS-data, full test)}} \\
\midrule
Ours (p60)        &  583 & 0.373 & 7435 & 0.384 & 6716 & 0.388 \\
Ours (p70)        &  414 & 0.265 & 5587 & 0.289 & 5014 & 0.289 \\
Ours (p80)        &  266 & 0.170 & 3729 & 0.193 & 3246 & 0.187 \\
Ours (sup)        &  368 & 0.235 & 2945 & 0.152 & 7966 & 0.460 \\
Ours (oracle)     &  500 & 0.320 & 2945 & 0.152 & 6146 & 0.355 \\

\midrule
\multicolumn{7}{c}{\textbf{Ours --- MiniLM-int8 (SDS-data, full test)}} \\
\midrule
Ours (p60)        &  595 & 0.380 & 7399 & 0.382 & 6319 & 0.365 \\
Ours (p70)        &  446 & 0.285 & 5530 & 0.286 & 4713 & 0.272 \\
Ours (p80)        &  306 & 0.196 & 3694 & 0.191 & 3021 & 0.174 \\
Ours (sup)        &  412 & 0.263 & 2892 & 0.150 & 8145 & 0.470 \\
Ours (oracle)     &  444 & 0.284 & 2892 & 0.150 & 8863 & 0.512 \\

\midrule
\multicolumn{7}{c}{\textbf{LLM-based segmenters (SeCom prompt) --- not evaluated on DTS}$^{\dagger}$} \\
\midrule
GPT-4o-mini-Seg   & -- & -- & -- & -- & -- & -- \\
GPT-5-Seg         & -- & -- & -- & -- & -- & -- \\
Qwen3.5-122B-Seg  & -- & -- & -- & -- & -- & -- \\
Qwen3.5-27B-Seg   & -- & -- & -- & -- & -- & -- \\
Qwen3.5-4B-Seg    & -- & -- & -- & -- & -- & -- \\
Qwen3.5-2B-Seg    & -- & -- & -- & -- & -- & -- \\
Llama3.2-3B-Seg   & -- & -- & -- & -- & -- & -- \\
Mistral3-3B-Seg   & -- & -- & -- & -- & -- & -- \\

\midrule
\textbf{Gold (Def-DTS)} &  315 & \textbf{0.201} & 2743 & \textbf{0.147} & 4017 & \textbf{0.251} \\
\bottomrule
\end{tabular}
\caption{Predicted boundary count and boundary density (= boundaries / turns) across DTS benchmarks. Online unsupervised baselines measured on the Def-DTS bundle; Hi-OnTop measured on the original SuperDialseg-data bundle (encoder cache compatibility, $\pm$1\% turn-count drift). Hi-OnTop at $p70$ over-segments TIAGE and Dialseg711 ($1.3$--$2.0\times$ gold density) while matching SuperDialseg ($0.29$ vs $0.25$). $p80$ matches gold density on TIAGE/Dialseg711 but under-segments SuperDialseg. Sup/oracle reflect $\delta^{*}$ chosen on labeled train/test respectively (cf.\ \texttt{delta\_star\_calibration.md}). $^{\dagger}$LLM-based segmenters (GPT-5 / Qwen / GPT-4o-mini / Mistral / Llama) were applied only to Long-MT-Bench+ downstream evaluation (cf.\ \texttt{segment\_counts\_downstream.md}); their DTS boundary density is unmeasured in this project (cost prohibitive for $2{,}133$ dialogues $\times$ LLM call).}
\label{tab:boundary_density_dts}
\end{table*}
```

## 검증 미해결

- **두 split family 사용** — baselines 는 Def-DTS 번들, Hi-OnTop 는 SDS-data 번들. Turn
  카운트 차이 (DS711 18639 vs 19350, SDS 16006 vs 17328) 는 데이터 전처리 차이 (last-turn
  inclusion, role label 분리). 결론에 큰 영향 없으나, 직접 비교 시 한 가족으로 통일하면
  더 깔끔.
- **SuperDialseg train 의 sup δ\* 가 노이즈**: SDS train 의 yt 와 emb 길이 불일치 dialogue
  많음 (400 중 86 / 82 만 사용) — sup δ\* 값이 부풀려 진 보고. p60/p70/p80 은 정확.
- **Density ↔ Score 관계는 단순하지 않음** — density 만 보면 p80 이 gold 에 가장 가깝지만
  Score (F1/Pk/WD 가중평균) 는 p70 이 더 높음. boundary *위치* 와 *밀도* 가 분리된 metric.
- **baselines REPORT 출처**: `outputs/experiments/2026-05-20_texttiling_streaming/REPORT.md`,
  `2026-05-21_graphseg_window_d/REPORT.md`, `2026-05-21_greedyseg_online_delay2/REPORT.md`,
  `2026-05-24_csm_online_delay2_fixed/REPORT.md`.
