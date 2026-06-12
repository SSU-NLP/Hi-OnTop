# δ\* Calibration — Encoder Comparison

`outputs/experiments/2026-05-23_encoder_comparison/` · 2026-05-23

Hi-OnTop 의 boundary threshold δ\* 를 인코더 3종(MPNet / MiniLM / MiniLM-int8)
에 대해 보정하고 segmentation Score 를 비교. 데이터 = superdialseg_data,
metric = 공식 SuperDialseg `Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)`.
Hi-OnTop HP: m=2, ρ=0.7, a=0.5.

**δ\* 보정** — 인코더별, 벤치별 train split 에서:
TIAGE→TIAGE-train · SuperDialseg→SuperDialseg-train(≤400 서브샘플) ·
Dialseg711→train split 부재라 test 70:30 분할(70% 보정 / 30% test).

**세 기준점 (반드시 분리 — 본 파일에서 자주 혼동됨):**

- **p_x** — δ\* = held-out δ_eff 의 *x*-percentile (unsupervised, **배포 가능**).
  p ∈ {50, 55, …, 95}.
- **best p_x** — percentile family 안에서 Score 최대화하는 p_x (10개 옵션 중 label 로 고름; semi-supervised 선택, *deployable 아님*).
- **train-side best** — train split 에서 δ\* 를 **continuous sweep** 으로 Score 최대화 → test 평가. labeled train tuning, 진짜 held-out (별도 표, §2-2 참조).
- **test-side oracle** — δ\* 를 **test 자체** 에서 사후 continuous sweep 으로 Score 최대화 (supervised 상한, GT 필요, *not deployable*). **진짜 천장.**
- **δ\*_p** — best p_x 에 해당하는 실제 임계값.

본 표 (§2) 는 p_x + best p_x + test-side oracle 만 보고. train-side best 는 §2-2 별표.

## 표

**Score cols (12)**: p50..p95 (10 percentile) + **best** (train continuous sweep) + oracle (test continuous sweep). **δ\* cols (3)**: **δ\*_top p** (= δ\* at the bolded best percentile within p_x family) + **δ\*_best** (continuous train sweep δ\*) + **δ\*_oracle** (continuous test sweep δ\*). best p column 은 percentile family 안 최고 p_x indicator.

| Encoder | Dataset | p50 | p55 | p60 | p65 | p70 | p75 | p80 | p85 | p90 | p95 | best | oracle | best p | δ\*_top p | δ\*_best | δ\*_oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| MPNet | TIAGE | 0.422 | 0.447 | 0.457 | **0.476** | 0.459 | 0.447 | 0.433 | 0.427 | 0.402 | 0.371 | 0.453 | 0.473 | p65 | 0.546 | 0.574 | 0.543 |
| MPNet | Dialseg711 | 0.433 | 0.471 | 0.509 | 0.540 | 0.575 | 0.601 | 0.616 | **0.629** | 0.609 | 0.521 | 0.630 | 0.630 | p85 | 0.619 | 0.614 | 0.614 |
| MPNet | SuperDialseg | 0.455 | 0.459 | **0.465** | 0.462 | 0.459 | 0.445 | 0.427 | 0.402 | 0.366 | 0.328 | 0.463 | 0.464 | p60 | 0.536 | 0.553 | 0.543 |
| MPNet | *Avg.* | 0.437 | 0.459 | 0.477 | 0.493 | **0.498** | 0.497 | 0.492 | 0.486 | 0.459 | 0.406 | *0.515* | *0.522* | **p70** | **0.562** | 0.580 | 0.567 |
| MiniLM | TIAGE | 0.439 | 0.455 | 0.470 | 0.472 | **0.485** | 0.481 | 0.469 | 0.466 | 0.416 | 0.371 | 0.475 | 0.485 | p70 | 0.773 | 0.818 | 0.767 |
| MiniLM | Dialseg711 | 0.437 | 0.468 | 0.506 | 0.535 | 0.565 | 0.593 | **0.612** | 0.599 | 0.580 | 0.493 | 0.609 | 0.609 | p80 | 0.803 | 0.818 | 0.818 |
| MiniLM | SuperDialseg | **0.438** | 0.435 | 0.430 | 0.420 | 0.404 | 0.391 | 0.373 | 0.355 | 0.338 | 0.308 | 0.435 | 0.438 | p50 | 0.652 | 0.686 | 0.645 |
| MiniLM | *Avg.* | 0.438 | 0.453 | 0.469 | 0.475 | 0.484 | **0.488** | 0.485 | 0.473 | 0.444 | 0.391 | *0.506* | *0.511* | **p75** | **0.793** | 0.774 | 0.743 |
| MiniLM-int8 | TIAGE | 0.439 | 0.453 | 0.470 | 0.472 | 0.489 | 0.482 | **0.493** | 0.448 | 0.423 | 0.378 | 0.484 | 0.489 | p80 | 0.822 | 0.787 | 0.777 |
| MiniLM-int8 | Dialseg711 | 0.434 | 0.469 | 0.502 | 0.535 | 0.559 | 0.586 | 0.596 | **0.600** | 0.575 | 0.499 | 0.615 | 0.616 | p85 | 0.831 | 0.828 | 0.818 |
| MiniLM-int8 | SuperDialseg | **0.436** | 0.434 | 0.426 | 0.415 | 0.402 | 0.387 | 0.367 | 0.355 | 0.335 | 0.305 | 0.434 | 0.436 | p50 | 0.654 | 0.604 | 0.625 |
| MiniLM-int8 | *Avg.* | 0.436 | 0.452 | 0.466 | 0.474 | 0.483 | 0.485 | **0.485** | 0.468 | 0.444 | 0.394 | *0.511* | *0.514* | **p80** | **0.820** | 0.740 | 0.740 |

## 해석

- **δ\* 는 인코더에 강하게 의존** — MPNet ~0.5~0.6, MiniLM 계열 ~0.65~0.83.
  MiniLM 의 δ_eff 분포가 위로 이동 → **인코더를 바꾸면 δ\* 재보정 필수**.
- **best p 는 (인코더, 데이터) 셀별로 다름** — MPNet: p60/p65/p85, MiniLM:
  p50/p70/p80, MiniLM-int8: p50/p80/p85. **단일 universal p_x 는 없다.**
- **그러나 p70~p75 band 가 가장 robust default** — 9 셀 평균 gap (oracle −
  p_x) 이 p70 = +0.025, p75 = +0.024 로 최소. p50 = +0.077, p95 = +0.117
  대비 5× 작음. cf. `2026-05-23_percentile_generality/REPORT.md`.
- **세 벤치마크 기준 best p 모두 [p65, p85] band** — TIAGE/Dialseg711 (자연
  대화·sparse boundary) 는 p65~p85, SuperDialseg (doc-grounded·dense
  boundary) 는 p50~p60. boundary 밀도가 best p_x 를 끌어내림.
- **양자화 손실 없음**: MiniLM-int8 Avg (0.485 @ p80) ≈ MiniLM-fp32 Avg
  (0.488 @ p75). MPNet Avg (0.498 @ p70) 이 최고.
- **deployable 손실**: 라벨 없이 p70 default 쓰면 oracle 대비 평균 −0.024
  (≈ 95% retention). 라벨 약간 있으면 셀별 best p_x sweep 으로 ±0.004 까지
  근접 가능.

## LaTeX

```latex
%% δ* calibration 결과 원본 %%
\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}

\begin{tabular*}{\textwidth}{
@{\extracolsep{\fill}}
ll
@{\hspace{14pt}}
*{12}{c}
@{\hspace{14pt}}
*{3}{c}
@{}}
\toprule
\multirow{2}{*}[-0.5ex]{\textbf{Encoder}}
& \multirow{2}{*}[-0.5ex]{\textbf{Dataset}}
& \multicolumn{12}{c}{\textbf{Score} $\boldsymbol{\uparrow}$}
& \multicolumn{3}{c}{$\boldsymbol{\delta^{*}}$} \\
\cmidrule(r{12pt}){3-14} \cmidrule(l{-3pt}){15-17}
& & p50 & p55 & p60 & p65 & p70 & p75 & p80 & p85 & p90 & p95 & sup & oracle
& best $p$ & sup & oracle \\
\midrule

\multirow{4}{*}{MPNet}
& TIAGE         & 0.422 & 0.447 & 0.457 & \textbf{0.476} & 0.459 & 0.447 & 0.433 & 0.427 & 0.402 & 0.371 & 0.453 & 0.473 & .546 & .574 & .543 \\
& Dialseg711    & 0.433 & 0.471 & 0.509 & 0.540 & 0.575 & 0.601 & 0.616 & \textbf{0.629} & 0.609 & 0.521 & 0.630 & 0.630 & .619 & .614 & .614 \\
& SuperDialseg  & 0.455 & 0.459 & \textbf{0.465} & 0.462 & 0.459 & 0.445 & 0.427 & 0.402 & 0.366 & 0.328 & 0.463 & 0.464 & .536 & .553 & .543 \\
& \textit{Avg.} & 0.437 & 0.459 & 0.477 & 0.493 & \textbf{0.498} & 0.497 & 0.492 & 0.486 & 0.459 & 0.406 & 0.515 & 0.522 & \textbf{.562} & .580 & .567 \\

\midrule
\multirow{4}{*}{MiniLM}
& TIAGE         & 0.439 & 0.455 & 0.470 & 0.472 & \textbf{0.485} & 0.481 & 0.469 & 0.466 & 0.416 & 0.371 & 0.475 & 0.485 & .773 & .818 & .767 \\
& Dialseg711    & 0.437 & 0.468 & 0.506 & 0.535 & 0.565 & 0.593 & \textbf{0.612} & 0.599 & 0.580 & 0.493 & 0.609 & 0.609 & .803 & .818 & .818 \\
& SuperDialseg  & \textbf{0.438} & 0.435 & 0.430 & 0.420 & 0.404 & 0.391 & 0.373 & 0.355 & 0.338 & 0.308 & 0.435 & 0.438 & .652 & .686 & .645 \\
& \textit{Avg.} & 0.438 & 0.453 & 0.469 & 0.475 & 0.484 & \textbf{0.488} & 0.485 & 0.473 & 0.444 & 0.391 & 0.506 & 0.511 & \textbf{.793} & .774 & .743 \\

\midrule
\multirow{4}{*}{MiniLM-int8}
& TIAGE         & 0.439 & 0.453 & 0.470 & 0.472 & 0.489 & 0.482 & \textbf{0.493} & 0.448 & 0.423 & 0.378 & 0.484 & 0.489 & .822 & .787 & .777 \\
& Dialseg711    & 0.434 & 0.469 & 0.502 & 0.535 & 0.559 & 0.586 & 0.596 & \textbf{0.600} & 0.575 & 0.499 & 0.615 & 0.616 & .831 & .828 & .818 \\
& SuperDialseg  & \textbf{0.436} & 0.434 & 0.426 & 0.415 & 0.402 & 0.387 & 0.367 & 0.355 & 0.335 & 0.305 & 0.434 & 0.436 & .654 & .604 & .625 \\
& \textit{Avg.} & 0.436 & 0.452 & 0.466 & 0.474 & 0.483 & 0.485 & \textbf{0.485} & 0.468 & 0.444 & 0.394 & 0.511 & 0.514 & \textbf{.820} & .740 & .740 \\

\bottomrule
\end{tabular*}

\caption{Segmentation performance by encoder. $p_x \in \{50,\dots,95\}$: Score with $\delta^{*}$ set to the $x$-th percentile of held-out $\delta_{\text{eff}}$ (label-free, deployable). \textbf{sup}: $\delta^{*}$ from continuous sweep on the labeled \emph{train} split applied to test (supervised tuning, held-out evaluation; tiage/superseg = train split, dialseg711 = test 70:30 split with 70\% as calib). \textbf{oracle}: $\delta^{*}$ from continuous sweep on the \emph{test} split itself (supervised upper bound, label leakage, not deployable). $\delta^{*}_{\text{best }p}$ reports the threshold value at the per-row sup $p_x$ within the percentile family (the bolded $p_x$ Score column) --- semi-supervised pick among 10 percentile options. $\delta^{*}_{\text{sup}}$ and $\delta^{*}_{\text{oracle}}$ are the continuous sweep thresholds for the corresponding Score columns. Per-encoder \textit{Avg.}\ rows aggregate the three datasets (Avg.\ sup/oracle $\delta^{*}$ = mean of per-row values; Avg.\ $\delta^{*}_{\text{top }p}$ = bolded family-sup aggregate; see existing convention). Score $= 0.5\,F_1 + 0.25\,(1{-}P_k) + 0.25\,(1{-}\mathrm{WD})$. Hi-OnTop HP: $m{=}2$, $\rho{=}0.7$, $a{=}0.5$.}
\label{tab:encoder_segmentation}
\end{table*}
```

## 2-2. Per-metric Pk/WD/F1 (continuous best vs oracle, 보충)

§2 표는 Score 만 보고하지만 본 절은 동일한 best/oracle 데이터의 **Pk/WD/F1 분해** 를 추가. dts\_result.md 표의 ★ ‡ 행 per-metric 과 동일.

**출처**: `outputs/experiments/2026-05-24_hiontop_oracle_best/REPORT.md` (`scripts/compute_hiontop_oracle_best.py` — `run_encoder_comparison.py` 함수 import + cached embeddings).

| Encoder | Dataset | Pk best ↓ | WD best ↓ | F1 best ↑ | **Score best** | δ\*_best | Pk oracle ↓ | WD oracle ↓ | F1 oracle ↑ | **Score oracle** | δ\*_oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MPNet | TIAGE | 0.446 | 0.499 | 0.378 | **0.453** | 0.574 | 0.439 | 0.529 | 0.429 | **0.473** | 0.543 |
| MPNet | Dialseg711 | 0.285 | 0.321 | 0.562 | **0.630** | 0.614 | 0.285 | 0.321 | 0.562 | **0.630** | 0.614 |
| MPNet | SuperDialseg | 0.471 | 0.546 | 0.434 | **0.463** | 0.553 | 0.469 | 0.556 | 0.441 | **0.464** | 0.543 |
| MiniLM-int8 | TIAGE | 0.419 | 0.497 | 0.426 | **0.484** | 0.787 | 0.418 | 0.507 | 0.440 | **0.489** | 0.777 |
| MiniLM-int8 | Dialseg711 | 0.294 | 0.332 | 0.543 | **0.615** | 0.828 | 0.295 | 0.342 | 0.550 | **0.616** | 0.818 |
| MiniLM-int8 | SuperDialseg | 0.479 | 0.666 | 0.440 | **0.434** | 0.604 | 0.480 | 0.648 | 0.436 | **0.436** | 0.625 |

### LaTeX (supplementary)

```latex
\begin{table}[t]
\centering
\scriptsize
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcccc@{}}
\toprule
\multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont \textbf{Encoder}}
& \multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont \textbf{Dataset}}
& \multicolumn{2}{c}{\fontsize{8}{9}\selectfont\bfseries Score $\uparrow$}
& \multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont $\delta^{*}_{\text{best}}$}
& \multirow{2}{*}[-0.6ex]{\fontsize{8pt}{9pt}\selectfont $\delta^{*}_{\text{oracle}}$} \\
\cmidrule(lr){3-4}
& & best & oracle & & \\
\midrule
\multirow{4}{*}{MPNet}
& TIAGE        & 0.453 & 0.473 & 0.574 & 0.543 \\
& Dialseg711   & 0.630 & 0.630 & 0.614 & 0.614 \\
& SuperDialseg & 0.463 & 0.464 & 0.553 & 0.543 \\
& \textit{Avg.} & 0.515 & \textit{0.522} & 0.580 & 0.567 \\
\midrule
\multirow{4}{*}{MiniLM-int8}
& TIAGE        & 0.484 & 0.489 & 0.787 & 0.777 \\
& Dialseg711   & 0.615 & 0.616 & 0.828 & 0.818 \\
& SuperDialseg & 0.434 & 0.436 & 0.604 & 0.625 \\
& \textit{Avg.} & 0.511 & \textit{0.514} & 0.740 & 0.740 \\
\bottomrule
\end{tabular*}
\caption{Continuous-$\delta^{*}$ supervised calibration --- \textbf{best}: $\delta^{*}$ chosen by sweeping Score on the labeled \emph{train} split (tiage/superseg = train, dialseg711 = test 70:30 split, 70\% as calib); supervised tuning with held-out test evaluation. \textbf{oracle}: $\delta^{*}$ chosen by sweeping Score on the \emph{test} split itself (supervised upper bound, label leakage at evaluation, not deployable). Both are continuous sweeps over $\delta^{*} \in [0.35, 0.95]$ ($N=60$). Distinct from Table~\ref{tab:encoder_segmentation}'s \emph{best $p_x$}, which is the best percentile within $\{50,55,\dots,95\}$. Hi-OnTop HP: $m{=}2$, $\rho{=}0.7$, $a{=}0.5$.}
\label{tab:train_best_vs_oracle}
\end{table}
```

## 한계 / 검증 미해결

- δ\* 보정은 인코더별 단일 scalar — SuperDialseg-train 은 ≤400 dialog
  서브샘플, Dialseg711 은 train split 부재로 test 70:30 분할.
- oracle 은 supervised 상한 (test GT 로 사후 sweep, label leakage). 운용 δ\* = label-free p_x.
- §2 의 "best p" 는 percentile family 안 (10 옵션) 선택 — §2-2 의 train-side continuous best 와 다른 양 (값도 약간 다름; 예: MPNet TIAGE best p = p65=0.476 vs train-side continuous best Score = 0.453).
- 인코더별 per-utterance 인코딩 latency: MPNet 600ms · MiniLM-fp32 111ms
  · MiniLM-int8 ONNX 13ms (별도 측정,
  `outputs/experiments/2026-05-23_encoder_latency/`). Score↔latency
  trade-off 판단은 본 표 + latency 함께 볼 것.
- 본 파일의 §2 "oracle" 컬럼은 진짜 test-side oracle 로 라벨 정확. 한때 dts\_result.md 의 calibration 표가 "oracle" 라벨로 train-side best 값을 박았던 혼선 (2026-05-24 정정) 과는 별개.
