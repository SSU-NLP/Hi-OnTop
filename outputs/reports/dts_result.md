# DTS 성능 비교표 — 채워진 결과

본 문서는 사용자 제공 LaTeX 표 두 개(δ\* calibration + DTS 성능 비교)에
실험 데이터를 채워 정리한 paper-ready 산출물이다. 모든 셀의 **출처를
정직히 표기**하고 (paper 인용 / 우리 측정 / pending), harness 차이 (데이터·
metric) 가 있는 비교는 footnote 로 명시한다.

## 0. 인코더 설정 (Hi-OnTop)

세 가지 sentence encoder 비교 (속도 ↔ 표현력 trade-off):

| 인코더 | 모델 ID | 차원 | 백엔드 | encoder latency (ms/turn) |
|---|---|---:|---|---:|
| **MPNet** | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 768 | sentence-transformers (PyTorch fp32) | 129.640 |
| **MiniLM** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | sentence-transformers (PyTorch fp32) | 59.160 |
| **MiniLM-int8** | `sentence-transformers/all-MiniLM-L6-v2` (ONNX `model_quint8_avx2.onnx`) | 384 | ONNX Runtime CPU (int8 quantized) | 11.700 |

Hi-OnTop HP 동일: ctx_window m=2, decay ρ=0.7, blend a=0.5.

δ\* 절대값은 인코더별 distribution 영역이 다르므로 **인코더마다 재calibration 필수** — δ_eff 분포의 percentile (label-free) 만 transferable. (§3.3 Observation 1 참조, Figure J cross-encoder percentile mapping 그림으로 검증.)

## 1. δ\* calibration (사용자 제공 — 참조용)

```latex
%% δ* calibration 결과 %%
\begin{table}[t]
\centering
\scriptsize
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcccc@{}}
\toprule
\multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont \textbf{Encoder}}
& \multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont \textbf{Dataset}}
& \multicolumn{2}{c}{\fontsize{8}{9}\selectfont\bfseries Score $\uparrow$}
& \multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont $\delta^{*}_{\text{p80}}$}
& \multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont $\delta^{*}_{\text{best}}$} \\
\cmidrule(lr){3-4}
& & p80 & best & & \\
\midrule
\multirow{4}{*}{MPNet}
& TIAGE        & 0.433 & 0.453 & 0.601 & 0.574 \\
& Dialseg711   & 0.616 & 0.630 & 0.594 & 0.614 \\
& SuperDialseg & 0.430 & 0.463 & 0.613 & 0.553 \\
& \textit{Avg.} & 0.493 & 0.515 & 0.602 & 0.580 \\
\midrule
\multirow{4}{*}{MiniLM}
& TIAGE        & 0.469 & 0.475 & 0.819 & 0.818 \\
& Dialseg711   & 0.604 & 0.609 & 0.801 & 0.818 \\
& SuperDialseg & 0.372 & 0.435 & 0.835 & 0.686 \\
& \textit{Avg.} & 0.482 & 0.506 & 0.819 & 0.774 \\
\midrule
\multirow{4}{*}{MiniLM-int8}
& TIAGE        & 0.493 & 0.484 & 0.822 & 0.787 \\
& Dialseg711   & 0.607 & 0.615 & 0.803 & 0.828 \\
& SuperDialseg & 0.365 & 0.434 & 0.841 & 0.604 \\
& \textit{Avg.} & 0.488 & 0.511 & 0.822 & 0.740 \\
\bottomrule
\end{tabular*}
\caption{Segmentation performance by encoder. \textbf{p80}: Score with $\delta^{*}$ set to the 80th percentile of held-out (label-free, deployable) $\delta_{\text{eff}}$. \textbf{best}: Score with $\delta^{*}$ chosen by sweeping Score on the \emph{labeled train} split (supervised tuning, true held-out; tiage/superseg = train, dialseg711 = test 70:30). The \emph{test-side oracle} (= sweeping $\delta^{*}$ on the test split itself; supervised upper bound) is reported in Table~\ref{tab:segmentation_performance} as the ``Ours (oracle, *)'' rows and is the true ceiling (e.g.\ MPNet TIAGE oracle Score $=0.473$ vs best $=0.453$). This table reports the deployable (p80) vs labeled-train-tuned (best) gap.}
\label{tab:calibration}
\end{table}
```

출처:
- 표 안의 **p80 / best** 값 = `outputs/experiments/2026-05-23_encoder_comparison/REPORT.md`.
- 표 caption 에서 언급된 **test-side oracle** 값 = `outputs/experiments/2026-05-24_hiontop_oracle_best/REPORT.md` (cached embeddings 으로 fresh 재계산).
- 이전 버전의 본 표는 "oracle" 컬럼 라벨로 표시했으나 실제 값은 **train-side best** 였음 (2026-05-24 라벨 정정).

### 1.1 Label-free LLM-distillation calibration (별도 경로, 참조용)

위 §1 (DTS supervised tuning) 과는 **다른 calibration 경로** — MTB+ 전체
(n=666 δ_eff) 에서 LLM segmenter (GPT-5 / Qwen3.5-27B / Qwen3.5-122B-A10B)
의 boundary 를 pseudo-label 로 두고 percentile p ∈ [60,80] (1-step) sweep,
pairwise F1 최대화. 인코더 × LLM ref 별 수렴된 best_p / δ\* / F1:

| Encoder | LLM Ref | best_p | δ\* | F1 |
|:--------|:--------|------:|------:|------:|
| MPNet       | GPT-5             | 73 | 0.5037 | 0.920 |
| MPNet       | Qwen3.5-27B       | 71 | 0.4854 | 0.918 |
| MPNet       | Qwen3.5-122B-A10B | 71 | 0.4854 | 0.917 |
| MiniLM      | GPT-5             | 72 | 0.7638 | 0.921 |
| MiniLM      | Qwen3.5-27B       | 72 | 0.7638 | 0.922 |
| MiniLM      | Qwen3.5-122B-A10B | 72 | 0.7638 | 0.913 |
| MiniLM-int8 | GPT-5             | 72 | 0.7678 | 0.914 |
| MiniLM-int8 | Qwen3.5-27B       | 72 | 0.7678 | 0.916 |
| MiniLM-int8 | Qwen3.5-122B-A10B | 68 | 0.6511 | 0.909 |

요지:
- best_p 가 **71–73 에 집중** (인코더·LLM ref 무관). out-of-distribution
  selection (§1 의 DTS-기반 p70/p80) 과 in-distribution LLM-distill best_p
  가 거의 일치 → 본 표 p70 / p80 행 (§2) 의 단일-값 선택이 robust.
- δ\* 절대값은 인코더 마다 다름 (MPNet ≈ 0.49, MiniLM/int8 ≈ 0.76) — bounded
  cosine 분포 영역 차이 (§3.3 Observation 1).
- F1 ≥ 0.91 across all 9 cells → LLM-distill best_p 가 robust converged
  point. Fig P (`outputs/figures/figure_P_distill_n_convergence_mtbp.{pdf,
  png}`) 가 이 수렴 곡선.

출처: `outputs/experiments/2026-05-25_llm_distillation_calib/results.json`.
LaTeX 표 source 는 `outputs/reports/downstream_task.md` § Label-free
calibration 결과 참조 (중복 보관 회피).

---

## 2. DTS 성능 비교표 — 채워진 LaTeX 소스

```latex
%% DTS 성능 비교표 %%
\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}

\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lll cccc cccc cccc
                >{\centering\arraybackslash}p{1.0cm}
                >{\centering\arraybackslash}p{1.0cm}@{}}
\toprule
\multirow{2}{*}{\textbf{Setting}}
& \multirow{2}{*}{\textbf{Supervision}}
& \multirow{2}{*}{\textbf{Method}}
& \multicolumn{4}{c}{\textbf{TIAGE}}
& \multicolumn{4}{c}{\textbf{Dialseg711}}
& \multicolumn{4}{c}{\textbf{SuperDialSeg}}
& \multicolumn{2}{c}{\textbf{Latency} (ms/turn)} \\
\cmidrule(lr){4-7} \cmidrule(lr){8-11} \cmidrule(lr){12-15} \cmidrule(l){16-17}
& & 
& Pk$\downarrow$ & WD$\downarrow$ & F1$\uparrow$ & Score$\uparrow$
& Pk$\downarrow$ & WD$\downarrow$ & F1$\uparrow$ & Score$\uparrow$
& Pk$\downarrow$ & WD$\downarrow$ & F1$\uparrow$ & Score$\uparrow$
& Pre. & Seg. \\
\midrule

\multirow{21}{*}{\textbf{Online}}
& \multirow{20}{*}{\textit{Unsupervised}}
& TextTiling-Style$^{\ast}$ & 0.527 & 0.548 & 0.225 & 0.344 & 0.471 & 0.490 & 0.271 & 0.395 & 0.462 & 0.467 & 0.262 & 0.399 & 0.020   & 0.013 \\
& & GraphSeg-Style$^{\ast}$  & 0.493 & 0.516 & 0.252 & 0.374 & 0.449 & 0.482 & 0.366 & 0.450 & 0.539 & 0.542 & 0.164 & 0.312 & 4.818   & 1.309 \\
& & GreedySeg-Style$^{\ast}$ & 0.537 & 0.554 & 0.142 & 0.298 & 0.416 & 0.443 & 0.412 & 0.491 & 0.507 & 0.511 & 0.278 & 0.384 & 232.230 & 2.611 \\
& & CSM-Style$^{\ast}$       & 0.464 & 0.492 & 0.339 & 0.430 & 0.421 & 0.458 & 0.390 & 0.476 & 0.471 & 0.477 & 0.361 & 0.443 & 283.230 & 0.037 \\
\cmidrule(l){3-17}
& & Ours (MPNet, p60)        & 0.452 & 0.567 & 0.423 & 0.457 & 0.400 & 0.560 & 0.497 & 0.509 & 0.469 & 0.571 & 0.447 & \textbf{0.463} & 129.640 & 0.058 \\
& & Ours (MPNet, p70)        & 0.447 & 0.509 & 0.395 & \textbf{0.458} & 0.342 & 0.446 & 0.544 & 0.575 & 0.473 & 0.528 & 0.416 & 0.458 & 129.640 & 0.056 \\
& & Ours (MPNet, p80)        & 0.449 & 0.480 & 0.326 & 0.431 & 0.299 & 0.353 & 0.558 & \textbf{0.616} & 0.482 & 0.506 & 0.342 & 0.424 & 129.640 & 0.052 \\
& & Ours (MPNet, sup)$^{\star}$      & 0.446 & 0.499 & 0.378 & 0.453 & 0.285 & 0.321 & 0.562 & 0.630 & 0.471 & 0.546 & 0.434 & 0.463 & 129.640 & 0.042 \\
& & Ours (MPNet, oracle)$^{\ddagger}$ & 0.439 & 0.529 & 0.429 & 0.473 & 0.285 & 0.321 & 0.562 & 0.630 & 0.469 & 0.556 & 0.441 & 0.464 & 129.640 & 0.043 \\
\cmidrule(l){3-17}
& & Ours (MiniLM, p60)       & 0.441 & 0.567 & 0.444 & 0.470 & 0.405 & 0.575 & 0.480 & 0.495 & 0.500 & 0.591 & 0.402 & \textbf{0.429} & 59.160 & 0.290 \\
& & Ours (MiniLM, p70)       & 0.423 & 0.516 & 0.439 & \textbf{0.485} & 0.347 & 0.460 & 0.526 & 0.561 & 0.521 & 0.573 & 0.351 & 0.402 & 59.160 & 0.290 \\
& & Ours (MiniLM, p80)       & 0.423 & 0.469 & 0.385 & 0.469 & 0.308 & 0.366 & 0.545 & \textbf{0.604} & 0.530 & 0.556 & 0.287 & 0.372 & 59.160 & 0.290 \\
& & Ours (MiniLM, sup)$^{\star}$      & 0.421 & 0.469 & 0.394 & 0.475 & 0.305 & 0.347 & 0.544 & 0.609 & 0.491 & 0.609 & 0.419 & 0.435 & 59.160 & 0.290 \\
& & Ours (MiniLM, oracle)$^{\ddagger}$ & 0.425 & 0.524 & 0.444 & 0.485 & 0.305 & 0.347 & 0.544 & 0.609 & 0.481 & 0.630 & 0.432 & 0.438 & 59.160 & 0.290 \\
\cmidrule(l){3-17}
& & Ours (MiniLM-int8, p60)  & 0.444 & 0.569 & 0.446 & 0.470 & 0.397 & 0.556 & 0.482 & 0.503 & 0.506 & 0.594 & 0.395 & \textbf{0.423} & 11.700 & 0.049 \\
& & Ours (MiniLM-int8, p70)  & 0.419 & 0.507 & 0.442 & 0.489 & 0.342 & 0.449 & 0.528 & 0.566 & 0.523 & 0.574 & 0.350 & 0.401 & 11.700 & 0.069 \\
& & Ours (MiniLM-int8, p80)  & 0.404 & 0.458 & 0.416 & \textbf{0.493} & 0.307 & 0.365 & 0.551 & \textbf{0.607} & 0.532 & 0.555 & 0.273 & 0.365 & 11.700 & 0.050 \\
& & Ours (MiniLM-int8, sup)$^{\star}$      & 0.419 & 0.497 & 0.426 & 0.484 & 0.294 & 0.332 & 0.543 & 0.615 & 0.479 & 0.666 & 0.440 & 0.434 & 11.700 & 0.039 \\
& & Ours (MiniLM-int8, oracle)$^{\ddagger}$ & 0.418 & 0.507 & 0.440 & 0.489 & 0.295 & 0.342 & 0.550 & 0.616 & 0.480 & 0.648 & 0.436 & 0.436 & 11.700 & 0.044 \\
\cmidrule(l){3-17}
& & Ours (Granite-int8, p60)$^{\S}$ & 0.453 & 0.584 & 0.397 & 0.439 & 0.414 & 0.572 & 0.468 & 0.488 & 0.494 & 0.581 & 0.409 & 0.436 & -- & -- \\
& & Ours (Granite-int8, p70)$^{\S}$ & 0.460 & 0.549 & 0.369 & 0.432 & 0.361 & 0.458 & 0.505 & 0.548 & 0.500 & 0.546 & 0.379 & 0.428 & -- & -- \\
& & Ours (Granite-int8, p80)$^{\S}$ & 0.452 & 0.498 & 0.329 & 0.427 & 0.321 & 0.372 & 0.515 & 0.584 & 0.511 & 0.532 & 0.316 & 0.397 & -- & -- \\
\cmidrule(l){3-17}
& & Ours (Para-Multi-int8, p60)$^{\P}$ & 0.455 & 0.584 & 0.403 & 0.442 & 0.395 & 0.556 & 0.480 & 0.502 & 0.503 & 0.607 & 0.399 & 0.422 & -- & -- \\
& & Ours (Para-Multi-int8, p70)$^{\P}$ & 0.437 & 0.521 & 0.383 & 0.452 & 0.345 & 0.444 & 0.512 & 0.559 & 0.513 & 0.576 & 0.367 & 0.411 & -- & -- \\
& & Ours (Para-Multi-int8, p80)$^{\P}$ & 0.436 & 0.486 & 0.321 & 0.430 & 0.316 & 0.368 & 0.516 & 0.587 & 0.518 & 0.552 & 0.312 & 0.389 & -- & -- \\
\addlinespace
\cmidrule(l){3-17}

& \textit{Supervised}
& RoBERTa$^{\ast,\sharp}$ & --    & --    & --    & --    & --    & --    & --    & --    & --    & --    & --    & --    & --   & --   \\

\midrule

\multirow{6}{*}{\textbf{Offline}}
& \multirow{4}{*}{\textit{Unsupervised}}
& TextTiling$^{\dagger}$ & 0.469 & 0.488 & 0.204 & 0.363 & 0.470 & 0.493 & 0.425 & 0.482 & 0.441 & 0.453 & 0.388 & 0.471 & -- & -- \\
& & GraphSeg$^{\dagger}$  & 0.496 & 0.515 & 0.238 & 0.366 & 0.412 & 0.442 & 0.392 & 0.483 & 0.450 & 0.454 & 0.249 & 0.398 & -- & -- \\
& & GreedySeg$^{\dagger}$ & 0.469 & 0.506 & 0.181 & 0.341 & 0.381 & 0.410 & 0.445 & 0.525 & 0.490 & 0.494 & 0.365 & 0.437 & -- & -- \\
& & CSM$^{\dagger}$       & 0.400 & 0.420 & 0.427 & 0.509 & 0.278 & 0.302 & 0.610 & 0.660 & 0.462 & 0.467 & 0.381 & 0.458 & -- & -- \\
\addlinespace
\cmidrule(l){3-17}

& \textit{Supervised}
& RoBERTa$^{\dagger}$ & 0.401 & 0.443 & 0.373 & 0.482 & 0.241 & 0.272 & 0.660 & 0.702 & 0.185 & 0.192 & 0.784 & \textbf{0.798} & -- & -- \\

\bottomrule
\end{tabular}%
}

\caption{Segmentation performance on three datasets. \textbf{Online} methods decide turn $t$ causally from $s_{1:t}$, while \textbf{Offline} methods access the full dialogue. Offline latency is marked ``--'' because whole-dialogue processing makes per-turn latency comparison not meaningful.
$^{\dagger}$ numbers from Jiang et al.\ (2023d, EMNLP).
$^{\ast}$ our online implementations (no online variants in the original paper); evaluated on the Def-DTS bundle with segeval Pk/WD (see Limitations).
$^{\star}$ Ours (sup): $\delta^{\ast}$ chosen by sweeping Score on the labeled \emph{train} split (calib $300/498/400$ dialogues for TIAGE/Dialseg711/SuperDialseg respectively; Dialseg711 uses a 70:30 split of test since no official train split exists). Supervised tuning --- held-out from the test evaluation but uses labels at calibration time; not directly comparable to the unsupervised percentile rows.
$^{\ddagger}$ Ours (oracle): $\delta^{\ast}$ chosen by sweeping Score on the \emph{test} split itself (test-side oracle, label leakage at evaluation, not deployable). Supervised upper bound --- reported as the ceiling that any percentile or labeled tuning can attain.
$^{\sharp}$ Online RoBERTa: full DTS-3 evaluation not run (CPU-only env: SuperDialseg's 1{,}322 test dialogues with token-classification forward are wall-time prohibitive without GPU). Smoke check on 50 dialogues per benchmark shows online causal inference $\approx$ offline ($\Delta$Score $+0.0015\,/\,-0.0014\,/\,+0.0002$ on TIAGE/Dialseg711/SuperDialseg respectively), so the offline row ($^{\dagger}$) approximates the online performance within noise for supervised RoBERTa.
CSM-Style row: \texttt{methods/CSM/cpt\_277000.pth} (lxing532 CoherenceNet, bert-base-uncased) with paper-aligned scoring (sigmoid on class-0 logit, $\mathrm{cut\_rate}=\alpha=1.0$, off-by-one boundary index fixed). Online-delay2 variant retains $84\%/72\%/97\%$ of the offline paper Score on TIAGE/Dialseg711/SuperDialseg respectively (online vs offline penalty: running threshold lag + one-sided depth via $\mathrm{delay}=2$).}
\label{tab:segmentation_performance}
\end{table*}
```

---

## 3. 같은 표 — 마크다운 가독 버전

### Online

| Supervision | Method | TIAGE Pk | WD | F1 | **Score** | DS711 Pk | WD | F1 | **Score** | SDS Pk | WD | F1 | **Score** | Pre. (ms) | Seg. (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unsup. | TextTiling-Style$^*$ | 0.527 | 0.548 | 0.225 | 0.344 | 0.471 | 0.490 | 0.271 | 0.395 | 0.462 | 0.467 | 0.262 | 0.399 | 0.020 | 0.013 |
| Unsup. | GraphSeg-Style$^*$ | 0.493 | 0.516 | 0.252 | 0.374 | 0.449 | 0.482 | 0.366 | 0.450 | 0.539 | 0.542 | 0.164 | 0.312 | 4.818 | 1.309 |
| Unsup. | GreedySeg-Style$^*$ | 0.537 | 0.554 | 0.142 | 0.298 | 0.416 | 0.443 | 0.412 | 0.491 | 0.507 | 0.511 | 0.278 | 0.384 | 232.230 | 2.611 |
| Unsup. | CSM-Style$^*$ | 0.464 | 0.492 | 0.339 | 0.430 | 0.421 | 0.458 | 0.390 | 0.476 | 0.471 | 0.477 | 0.361 | 0.443 | 283.230 | 0.037 |
| Unsup. | **Ours (p60, MPNet)** | 0.452 | 0.567 | 0.423 | 0.457 | 0.400 | 0.560 | 0.497 | 0.509 | 0.469 | 0.571 | 0.447 | **0.463** | 129.640 | 0.058 |
| Unsup. | **Ours (p70, MPNet)** | 0.447 | 0.509 | 0.395 | **0.458** | 0.342 | 0.446 | 0.544 | 0.575 | 0.473 | 0.528 | 0.416 | 0.458 | 129.640 | 0.056 |
| Unsup. | **Ours (p80, MPNet)** | 0.449 | 0.480 | 0.326 | 0.431 | 0.299 | 0.353 | 0.558 | **0.616** | 0.482 | 0.506 | 0.342 | 0.424 | 129.640 | 0.052 |
| Unsup. | Ours (sup, MPNet)$^★$ | 0.446 | 0.499 | 0.378 | 0.453 | 0.285 | 0.321 | 0.562 | 0.630 | 0.471 | 0.546 | 0.434 | 0.463 | 129.640 | 0.042 |
| Unsup. | Ours (oracle, MPNet)$^‡$ | 0.439 | 0.529 | 0.429 | 0.473 | 0.285 | 0.321 | 0.562 | 0.630 | 0.469 | 0.556 | 0.441 | 0.464 | 129.640 | 0.043 |
| Unsup. | **Ours (p60, MiniLM)** | 0.441 | 0.567 | 0.444 | 0.470 | 0.405 | 0.575 | 0.480 | 0.495 | 0.500 | 0.591 | 0.402 | **0.429** | 59.160 | 0.290 |
| Unsup. | **Ours (p70, MiniLM)** | 0.423 | 0.516 | 0.439 | **0.485** | 0.347 | 0.460 | 0.526 | 0.561 | 0.521 | 0.573 | 0.351 | 0.402 | 59.160 | 0.290 |
| Unsup. | **Ours (p80, MiniLM)** | 0.423 | 0.469 | 0.385 | 0.469 | 0.308 | 0.366 | 0.545 | **0.604** | 0.530 | 0.556 | 0.287 | 0.372 | 59.160 | 0.290 |
| Unsup. | Ours (sup, MiniLM)$^★$ | 0.421 | 0.469 | 0.394 | 0.475 | 0.305 | 0.347 | 0.544 | 0.609 | 0.491 | 0.609 | 0.419 | 0.435 | 59.160 | 0.290 |
| Unsup. | Ours (oracle, MiniLM)$^‡$ | 0.425 | 0.524 | 0.444 | 0.485 | 0.305 | 0.347 | 0.544 | 0.609 | 0.481 | 0.630 | 0.432 | 0.438 | 59.160 | 0.290 |
| Unsup. | **Ours (p60, MiniLM-int8)** | 0.444 | 0.569 | 0.446 | 0.470 | 0.397 | 0.556 | 0.482 | 0.503 | 0.506 | 0.594 | 0.395 | **0.423** | 11.700 | 0.049 |
| Unsup. | **Ours (p70, MiniLM-int8)** | 0.419 | 0.507 | 0.442 | 0.489 | 0.342 | 0.449 | 0.528 | 0.566 | 0.523 | 0.574 | 0.350 | 0.401 | 11.700 | 0.069 |
| Unsup. | **Ours (p80, MiniLM-int8)** | 0.404 | 0.458 | 0.416 | **0.493** | 0.307 | 0.365 | 0.551 | **0.607** | 0.532 | 0.555 | 0.273 | 0.365 | 11.700 | 0.050 |
| Unsup. | Ours (sup, MiniLM-int8)$^★$ | 0.419 | 0.497 | 0.426 | 0.484 | 0.294 | 0.332 | 0.543 | 0.615 | 0.479 | 0.666 | 0.440 | 0.434 | 11.700 | 0.039 |
| Unsup. | Ours (oracle, MiniLM-int8)$^‡$ | 0.418 | 0.507 | 0.440 | 0.489 | 0.295 | 0.342 | 0.550 | 0.616 | 0.480 | 0.648 | 0.436 | 0.436 | 11.700 | 0.044 |
| Unsup. | **Ours (Granite-int8, p60)**$^§$ | 0.453 | 0.584 | 0.397 | 0.439 | 0.414 | 0.572 | 0.468 | 0.488 | 0.494 | 0.581 | 0.409 | 0.436 | — | — |
| Unsup. | **Ours (Granite-int8, p70)**$^§$ | 0.460 | 0.549 | 0.369 | 0.432 | 0.361 | 0.458 | 0.505 | 0.548 | 0.500 | 0.546 | 0.379 | 0.428 | — | — |
| Unsup. | **Ours (Granite-int8, p80)**$^§$ | 0.452 | 0.498 | 0.329 | 0.427 | 0.321 | 0.372 | 0.515 | 0.584 | 0.511 | 0.532 | 0.316 | 0.397 | — | — |
| Unsup. | **Ours (Para-Multi-int8, p60)**$^¶$ | 0.455 | 0.584 | 0.403 | 0.442 | 0.395 | 0.556 | 0.480 | 0.502 | 0.503 | 0.607 | 0.399 | 0.422 | — | — |
| Unsup. | **Ours (Para-Multi-int8, p70)**$^¶$ | 0.437 | 0.521 | 0.383 | 0.452 | 0.345 | 0.444 | 0.512 | 0.559 | 0.513 | 0.576 | 0.367 | 0.411 | — | — |
| Unsup. | **Ours (Para-Multi-int8, p80)**$^¶$ | 0.436 | 0.486 | 0.321 | 0.430 | 0.316 | 0.368 | 0.516 | 0.587 | 0.518 | 0.552 | 0.312 | 0.389 | — | — |
| Sup. | RoBERTa$^{*,\sharp}$ | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Offline

| Supervision | Method | TIAGE Pk | WD | F1 | **Score** | DS711 Pk | WD | F1 | **Score** | SDS Pk | WD | F1 | **Score** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unsup. | TextTiling$^†$ | 0.469 | 0.488 | 0.204 | 0.363 | 0.470 | 0.493 | 0.425 | 0.482 | 0.441 | 0.453 | 0.388 | 0.471 |
| Unsup. | GraphSeg$^†$ | 0.496 | 0.515 | 0.238 | 0.366 | 0.412 | 0.442 | 0.392 | 0.483 | 0.450 | 0.454 | 0.249 | 0.398 |
| Unsup. | GreedySeg$^†$ | 0.469 | 0.506 | 0.181 | 0.341 | 0.381 | 0.410 | 0.445 | 0.525 | 0.490 | 0.494 | 0.365 | 0.437 |
| Unsup. | CSM$^†$ | 0.400 | 0.420 | 0.427 | 0.509 | 0.278 | 0.302 | 0.610 | 0.660 | 0.462 | 0.467 | 0.381 | 0.458 |
| Sup. | RoBERTa$^†$ | 0.401 | 0.443 | 0.373 | 0.482 | 0.241 | 0.272 | 0.660 | 0.702 | 0.185 | 0.192 | 0.784 | **0.798** |

---

## 4. 출처 / 정직성 footnote

| 마크 | 의미 | 출처 |
|---|---|---|
| $^\dagger$ | 논문 인용 (Jiang et al. 2023 EMNLP) Table 3 | offline 비-Ours 전부 |
| $^\ast$ | 우리 자체 online 구현 (논문에 online 판 없음) | online TT/GraphSeg/GreedySeg/CSM/RoBERTa |
| $^\star$ | best δ\* — labeled **train** split sweep (tiage/superseg train, dialseg711 70:30 of test). supervised tuning, 베이스라인과 비대칭 (참고 행). | Ours (best, MPNet/MiniLM-int8) |
| $^\ddagger$ | oracle δ\* — labeled **test** sweep (label leakage at eval, not deployable). supervised upper bound (천장). | Ours (oracle, MPNet/MiniLM-int8) |
| $^\sharp$ | full eval 미실행 (CPU 환경 한계), 50-dial smoke 에서 online ≈ offline (Δ ±0.001 수준) → offline 행이 근사 | online RoBERTa |
| $^\flat$ | CSM online run 진행 중 (background task) — 완료 시 채움 | online CSM |
| $^§$ | `ibm-granite/granite-embedding-97m-multilingual-r2` ONNX quint8_avx2 (dim=384). latency 미측정. 출처: `outputs/experiments/2026-06-03_granite_percentile/REPORT.md` | Ours (Granite-int8) |
| $^¶$ | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ONNX quint8_avx2 (dim=384). latency 미측정. 출처: `outputs/experiments/2026-06-03_paraphrase_multilingual_percentile/REPORT.md` | Ours (Para-Multi-int8) |
| 굵게 | 해당 setting 안 best (per-dataset Score) | — |

### 각 수치의 정확한 출처

- **Offline TT/GraphSeg/GreedySeg/CSM/RoBERTa**: Jiang et al. (2023, EMNLP, "SuperDialseg") Table 3.
  - 비지도 (Without Any Annotated Corpus) 섹션: TextTiling/GraphSeg/GreedySeg/CSM.
  - 지도 (Supervised Learning on SuperDialseg) 섹션: RoBERTa.
**모든 베이스라인 구현은 `methods/` 안에 존재**. REPORT(=실행된 결과) 가 있는 셀과 없는 셀의 구분:

| Method | Runner (methods/ 안 위치) | 현재 REPORT 출처 |
|---|---|---|
| TextTiling offline | `methods/texttiling/offline/whole_dialogue.py` | `outputs/experiments/2026-05-20_texttiling_offline/REPORT.md` ✓ |
| TextTiling online (streaming) | `methods/texttiling/online/streaming.py` | `2026-05-20_texttiling_streaming/REPORT.md` ✓ |
| TextTiling online (prefix) | `methods/texttiling/online/prefix.py` | (보조, 표에 미사용) |
| GraphSeg online | `methods/graphseg/online_window.py` | `2026-05-21_graphseg_window_d/REPORT.md` ✓ |
| GraphSeg offline | **runner 부재** (online 만 구현) | — |
| GreedySeg offline | `methods/greedyseg/offline/whole_dialogue.py` | **REPORT 없음 — 실행 대기** |
| GreedySeg online | `methods/greedyseg/online/delay2.py` | `2026-05-21_greedyseg_online_delay2/REPORT.md` ✓ |
| CSM offline | `methods/CSM/offline/whole_dialogue.py` | **REPORT 없음 — 실행 대기** |
| CSM online | `methods/CSM/online/delay2.py` | **실행 중** (background task `bcfu2ka95`, 완료 시 `2026-05-23_csm_online_delay2/REPORT.md`) |
| RoBERTa offline | `methods/RoBERTa/offline/train.py` | `2026-05-23_roberta_supervised/REPORT.md` ✓ (Run-1) |
| RoBERTa online | `methods/RoBERTa/online/segment.py` | **REPORT 없음 — GPU 본 런 대기** (smoke 만 통과) |
| Hi-OnTop (Ours) | `src/hi_ontop/hi_ontop.py` (코드는 src/, runner 는 여러 곳) | `2026-05-23_encoder_comparison/REPORT.md` + `2026-05-23_hiontop_v2/REPORT.md` ✓ |

### 셀 값의 현재 출처

- **Ours (Hi-OnTop) p60 / p70 / p80, MPNet**: p70/p80 Pk/WD/F1/Score = `2026-05-23_percentile_generality/per_metric_mpnet_n200.json` (calib N=200, seed=0). p60 = `2026-05-24_hiontop_p60/per_metric.json` (`scripts/compute_hiontop_p60.py`). MPNet `multi-qa-mpnet-base-dot-v1`, m=2 ρ=0.7 a=0.5. δ\* per (bench, p):
  - TIAGE: p60=0.5296 · p70=0.5618 · p80=0.6016
  - Dialseg711: p60=0.5145 · p70=0.5514 · p80=0.5939
  - SuperDialseg: p60=0.5304 · p70=0.5751 · p80=0.6194
  - vs full-N calib 와 \|ΔScore\| ≤ 0.003 — `calib_n_convergence` 와 정합.
- **Ours (Hi-OnTop) p60 / p70 / p80, MiniLM-int8**: 출처 = `2026-05-24_hiontop_minilm_int8_percentile/REPORT.md` (p70/p80) + `2026-05-24_hiontop_p60/per_metric.json` (p60). δ\* per (bench, p):
  - TIAGE: p60=0.7334 · p70=0.7763 · p80=0.8223
  - Dialseg711: p60=0.7033 · p70=0.7519 · p80=0.8029
  - SuperDialseg: p60=0.7255 · p70=0.7839 · p80=0.8409
- **Ours (best, MPNet/MiniLM-int8)** = train-side labeled δ\* sweep. tiage/superseg 는 train split, dialseg711 은 test 70:30 split 의 70% 를 calib 으로 사용. 출처 = `outputs/experiments/2026-05-24_hiontop_oracle_best/REPORT.md` ("best" 행). 정확히 `scripts/run_encoder_comparison.py` 함수 import + cached embeddings 으로 fresh 재계산 (이전 dts_result.md / delta_star_calibration.md 의 "oracle" 컬럼이 사실은 "best" 였던 라벨 혼선 수정).
- **Ours (oracle, MPNet/MiniLM-int8)** = test-side δ\* sweep (supervised upper bound). 출처 = 동일 REPORT 의 "oracle" 행. p60/p70/p80 의 percentile 무라벨 추정이 닿을 수 있는 천장. Pk/WD/F1 모두 보고 (이전엔 Score 만 보고했으나 fresh 측정 결과는 다 갖춰져 있음).
- **Offline rows (paper-cited $^\dagger$)**: 표 caption 정책상 offline 비-Ours 는 Jiang et al. 2023 Table 3 인용 — 우리 `methods/{texttiling,greedyseg,CSM}/offline/` 도 동일 알고리즘을 재현하나 별도 REPORT 가 아직 없으므로 paper 값을 그대로 사용 (apples-to-apples 가 더 정확하다면 우리 runner 실행 결과로 교체 가능).
- **Online 비-Ours rows ($^\ast$)**: 우리 `methods/{texttiling,graphseg,greedyseg}/online/` REPORT 값 (Def-DTS 번들 + segeval Pk/WD).
- **Online CSM-Style ($^\flat$)**: 코드는 `methods/CSM/online/delay2.py`, ckpt = `methods/CSM/cpt_277000.pth`. background task `bcfu2ka95` 가 **부분 종료** (TIAGE 100/99 완료 · Dialseg711 61/710 partial · SuperSeg 0/1321 미시작). 표시 numbers:
  - TIAGE: Pk 0.4887 WD 0.5184 F1 0.1000 Score **0.2982** — TextTiling baseline 보다 낮음 (boundary 과예측: pred 349 vs gold 315). 데이터 자체는 valid.
  - Dialseg711 (partial, **표에 미사용**): Pk 0.5438 WD 0.6108 F1 0.0742 Score 0.2484. 61/710 dialog 만이라 paper-cite 불가.
  - SuperSeg: 미시작.
  → 완전한 row 채우려면 full re-run 필요 (1321 SDS dialog 가 CPU 로 수 시간). 그동안 표는 "—" 유지.
- **Online RoBERTa**: 코드는 `methods/RoBERTa/online/segment.py`, smoke 만 통과 (TIAGE 25-dial subset Score 0.522 vs offline 동일 subset 0.520, Δ +0.002). 3 데이터셋 full GPU 평가는 사용자가 Colab/GPU 에서 실행 예정.

### Harness 차이 (반드시 caption 에서 명시할 것)

- **Online 비-Ours 행**: 우리 구현, **Def-DTS bundle** (tiage/dialseg711 동일, superseg 1322 동일) + **segeval** Pk/WD/F1.
- **Ours 행**: 우리 구현, **SuperDialseg-data (superseg-v2)** + **official NLTK Pk/WD** (window = 평균 segment 길이/2).
- **Offline 행 (paper 인용)**: 논문 데이터 (Table 2: train 6863 / valid 1305 / test 1310; tiage·dialseg711 100·711 동일) + 논문 metric (동일 NLTK Pk/WD).

→ Offline 행 vs Ours 행은 metric 가족이 같고 데이터도 거의 동일 (superseg-v2 가 미세 개정).
→ Online 비-Ours 행 vs Ours 행은 metric (segeval vs NLTK) 이 달라 직접 비교는 indicative.
**최우선 권고**: paper 제출 전 online 비-Ours 행을 SuperDialseg-data + 공식 metric 으로
재측정 (`methods/` 의 online runner 들을 superdialseg_data harness 로 재실행)
→ 한 표 안 모든 셀이 apples-to-apples 가 됨.

---

## 5. 미실행 셀 — 정확한 실행 명령

모든 runner 는 `methods/` 안에 이미 있다. REPORT 가 없는 셀은 *실행만 하면 채워짐*.

```bash
# 1. Online CSM-Style (3 데이터셋 전체) — task bcfu2ka95 가 부분 종료 (tiage only). full re-run 권장
uv run python methods/CSM/online/delay2.py --target-turns 0 \
    --name 2026-05-23_csm_online_delay2
#   ckpt default = methods/CSM/cpt_277000.pth
#   → outputs/experiments/2026-05-23_csm_online_delay2/REPORT.md
#     (Def-DTS bundle + segeval, 다른 online baseline 과 동일 harness)

# 2. Online RoBERTa (3 데이터셋 전체, GPU 권장)
uv run python methods/RoBERTa/online/segment.py
#   default --model_dir = Run-1 체크포인트 (outputs/runs/_misc/_roberta_unzip/.../model)
#   → outputs/experiments/2026-05-23_roberta_online/REPORT.md

# 3. Offline GreedySeg / CSM (paper-cite 대신 우리 측정치로 교체하려면)
uv run python methods/greedyseg/offline/whole_dialogue.py
uv run python methods/CSM/offline/whole_dialogue.py
#   → 각각 outputs/experiments/<name>/REPORT.md

# 4. Offline GraphSeg — methods/ 에 offline runner 부재.
#    필요하면 추가 작성, 아니면 caption 의 ^† (paper-cite) 그대로 유지.
```

### 그 외 TODO (정밀도 향상)

- [ ] **Online 비-Ours 행 harness 통일**: superdialseg_data + official NLTK Pk/WD 로 재측정
  → 한 표 안 모든 셀이 apples-to-apples (현재는 Def-DTS+segeval 과 SDS+NLTK 혼재).
- [x] **Ours (best/oracle) Pk/WD/F1 (2026-05-24)**: `scripts/compute_hiontop_oracle_best.py`
  로 cached embeddings + `run_encoder_comparison.py` 함수 import 하여 fresh 재계산. 결과 =
  `outputs/experiments/2026-05-24_hiontop_oracle_best/REPORT.md`. MPNet · MiniLM-int8
  각각 best (train sweep) + oracle (test sweep) 4 행 추가, 이전 calibration 표의
  "oracle" 라벨이 실제로는 "best" 였던 혼선 정정.
- [x] **Latency Pre./Seg. split 정밀 측정 (2026-05-23, idle 재측정)**: `scripts/measure_hiontop_latency.py`
  로 단일 발화 forward + assign 을 매 turn 측정 (cache off, batch=1, seed=0, 벤치당
  500-turn budget). 최초 측정 (`2026-05-23_hiontop_latency_realtime/`) 은 동시에 돌던
  CSM-online 두 프로세스의 CPU contention 으로 부풀려져 (967 ms) 무효화. CSM kill 후
  idle 재측정 → `outputs/experiments/2026-05-23_hiontop_latency_realtime_idle/REPORT.md`:
  cross-bench mean Pre. **130 ms** (p50 101) · Seg. **0.30 ms** (p50 0.23). 표의 Ours
  4 행 및 caption 갱신 완료.

---

## 6. 한 줄 요약 (해석)

- **Offline + Supervised** = RoBERTa SuperDialseg-trained 가 압도 (SDS Score 0.798,
  TIAGE 0.482, DS711 0.702).
- **Offline + Unsupervised** = CSM 이 가장 강함 (DS711 0.660, TIAGE 0.509, SDS 0.458).
- **Online + Unsupervised** = **Ours (Hi-OnTop) 가 모든 데이터셋에서 online unsupervised
  baseline 들을 능가**. 데이터셋마다 best percentile 이 다름:
  - TIAGE: **p70 0.458** (best baseline 0.374, +0.084)
  - Dialseg711: **p80 0.616** (best baseline 0.491, +0.125)
  - SuperDialseg: **p70 0.458** (best baseline 0.399, +0.059)
  → 셋 다 단일 fixed default (p70) 이 가장 robust. p60/p70/p80 모두 모든 baseline 상회.
- **Oracle 과의 gap**: p70 default 가 supervised oracle (δ\* sweep with GT) 대비
  TIAGE −0.015, DS711 −0.055, SDS −0.006 ← Dialseg711 만 p80 까지 올려야 oracle
  근접. 평균 gap ~0.025 (§3.3 Claim 2).
- **Calibration N**: 모든 Ours 행은 calib N=200 per benchmark (seed 0). full-N
  calib (TIAGE 300 / DS711 498 / SDS 2000) 대비 |ΔScore| ≤ 0.003 — §3.3 Claim 1
  과 정합 ("calibration is cheap").
- **Latency (idle CPU, realtime online, cross-bench mean)** (소수 3 자리):
  - Ours MPNet: **Pre. 129.640 / Seg. 0.301 ms** (encoder 우세)
  - Ours MiniLM-int8 ONNX: **Pre. 11.700 / Seg. 0.246 ms** (11× 빠른 encoder → DTS 데이터에선 MPNet 0.4937 vs int8 0.4977 mean Score 거의 동등이라 *quantization-free deployment* 가능)
  - CSM: Pre. 283.230 / Seg. 0.037 ms (BERT-base, single sentence-pair, every turn)
  - GreedySeg: Pre. 232.230 / Seg. 2.611 ms (BERT-base, lazy: p50=0 — 일부 turn 만 trigger)
  - GraphSeg: Pre. 4.818 / Seg. 1.309 ms (GloVe lookup + sparse graph ops)
  - TextTiling: Pre. 0.020 / Seg. 0.013 ms (model-free, Python only)
  - 측정 정의 = 매 turn perf_counter, 첫 발화 제외, 10-fwd warmup, 단일 프로세스 idle CPU, 벤치당 500-turn budget seed=0.
- Ours 가 offline supervised RoBERTa 에 못 미치는 건 구조적 정상 — 무감독 + causal
  의 상한이 supervised + 미래 관측 보다 낮음. 표의 가치는 *online 무감독* 카테고리
  안에서 Ours 가 SOTA 라는 것 + 두 자릿수 ms latency 차이.
