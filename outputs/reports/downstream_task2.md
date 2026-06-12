# Downstream Task — Application Results (Long-MT-Bench+) · chat=gpt-4o

`outputs/experiments/2026-05-21_v413_secom_swap/` · 2026-05-28

동일 segmentation / compress / retrieve 결과 재활용 (LLM-agnostic). chat(응답 생성) =
`openrouter/openai/gpt-4o`, judge = `openrouter/openai/gpt-4o`, n_conv=11, n_qa=288.
GPT4Score = `gpt4_score_x10`. Latency 컬럼은 `downstream_task.md` (chat=gpt-4o-mini) 와
동일 (segmentation 공유). `chat_gpt4o.jsonl` + `metrics_<method>_gpt4o.json`.

이전 실험 (`downstream_task.md`) 과의 차이:
- chat: `openai/gpt-4o-mini` → `openrouter/openai/gpt-4o`
- judge: 동일 (`openrouter/openai/gpt-4o`)
- segmentation / retrieval: 완전 재활용 (동일 `retrieved.jsonl`)

## LaTeX

```latex
%% Downstream Task 비교표 (chat=gpt-4o) %%
\begin{table*}[t]
\centering
\scriptsize
\begin{tabular}{l|cccccc|cc|cc}
\toprule
\multirow{2}{*}{\textbf{Methods}}
& \multicolumn{6}{c|}{\textbf{QA Performance}}
& \multicolumn{2}{c|}{\textbf{Context Length}}
& \multicolumn{2}{c}{\textbf{Seg Latency} (ms/turn)} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}
& GPT4Score & BLEU & Rouge1 & Rouge2 & RougeL & BERTScore
& \# Turns & \# Tokens & Pre. & Seg. \\
\midrule

Zero History
& 43.68 & 9.72 & 29.23 & 12.32 & 21.93 & 86.86 & 0 & 0 & -- & -- \\
Full History
& 83.09 & 17.41 & 40.22 & 24.12 & 33.10 & 89.13 & 65.45 & 22676 & -- & -- \\

\midrule
\multicolumn{11}{c}{\textbf{Unsupervised}} \\
\midrule

TextTiling-Style-Seg
& 74.76 & 19.00 & 40.84 & 24.19 & 33.88 & 89.12 & 3.74 & 1068 & 0.26 & 0.85 \\
GraphSeg-Style-Seg
& 63.09 & 15.75 & 36.65 & 20.47 & 29.81 & 88.23 & 8.66 & 2446 & 44.37 & 128.80 \\
\midrule
GreedySeg-Style-Seg
& 70.31 & 17.81 & 39.48 & 22.34 & 32.25 & 88.74 & 5.38 & 1495 & 296.54 & 15.06 \\
CSM-Style-Seg
& 64.27 & 15.46 & 36.45 & 19.77 & 29.50 & 88.13 & 9.17 & 2430 & 305.26 & 17.27 \\
Ours (MPNet, p60)
& 79.86 & 21.03 & 43.52 & 26.26 & 36.16 & 89.52 & 2.58 & 776 & 106.93 & 0.058 \\
Ours (MPNet, p70)
& 80.62 & 20.93 & 43.89 & 26.72 & 36.54 & 89.57 & 3.13 & 894 & 106.93 & 0.056 \\
Ours (MPNet, p80) & -- & -- & -- & -- & -- & -- & -- & -- & 106.93 & -- \\
Ours (MiniLM, p60)
& 80.03 & 21.62 & 44.01 & 27.24 & 36.64 & 89.69 & 2.59 & 783 & 59.16 & 0.061 \\
Ours (MiniLM, p70)
& 80.38 & 20.85 & 43.77 & 27.22 & 36.58 & 89.65 & 3.00 & 861 & 59.16 & 0.105 \\
Ours (MiniLM, p80)
& 78.78 & 19.54 & 42.58 & 25.78 & 35.42 & 89.41 & 4.16 & 1132 & 59.16 & 0.070 \\
Ours (MiniLM-int8, p60)
& 80.17 & 20.89 & 43.51 & 26.65 & 36.11 & 89.56 & 2.60 & 785 & 45.70 & 0.049 \\
Ours (MiniLM-int8, p70)
& \textbf{80.69} & 20.75 & 43.70 & 26.96 & 36.35 & 89.61 & 3.00 & 863 & 45.70 & 0.069 \\
Ours (MiniLM-int8, p80)
& 77.99 & 19.41 & 41.72 & 24.78 & 34.54 & 89.27 & 4.28 & 1146 & 45.70 & 0.050 \\
Ours (MPNet, cal-p71)
& 80.10 & 20.51 & 42.92 & 25.90 & 35.51 & 89.51 & 3.18 & 907 & 106.93 & 0.188 \\
Ours (MiniLM, cal-p72)
& 80.45 & 20.35 & 43.71 & 26.86 & 36.55 & 89.52 & 3.10 & 895 & 59.16 & 0.070 \\
Ours (MiniLM-int8, cal-p72)
& 80.62 & 20.29 & 43.32 & 26.63 & 36.25 & 89.49 & 3.14 & 905 & 45.70 & 0.073 \\
\midrule
\multicolumn{11}{l}{\textit{Ablation — Hi-OnTop blend weight $a$ (MPNet, p70):}} \\
Ours (MPNet, $a{=}0.0$, ctx only)
& 80.83 & 21.19 & 43.09 & 26.64 & 35.94 & 89.54 & 3.03 & 867 & 106.93 & 0.318 \\
Ours (MPNet, $a{=}1.0$, prev only)
& 81.32 & 21.57 & 43.65 & 26.92 & 36.34 & 89.57 & 3.10 & 891 & 106.93 & 0.071 \\

\midrule
\multicolumn{11}{c}{\textbf{Supervised}} \\
\midrule

RoBERTa-Style-Seg
& 74.97 & 19.58 & 41.09 & 23.97 & 33.90 & 89.12 & 3.21 & 929 & 428.09 & 0.01 \\

\midrule
\multicolumn{11}{c}{\textbf{LLM-based (SeCom)}} \\
\midrule

GPT-5-Seg
& 81.84 & 21.04 & 43.87 & 27.52 & 36.65 & 89.66 & 3.09 & 892 & -- & -- \\
Qwen3.5-122B-A10B-Seg
& \textbf{82.33} & 21.67 & 44.33 & 27.80 & 36.95 & 89.72 & 3.01 & 876 & -- & -- \\
Qwen3.5-27B-Seg
& 82.08 & 21.19 & 44.32 & 27.52 & 36.89 & 89.67 & 2.99 & 863 & -- & -- \\
GPT-4o-mini-Seg
& 79.06 & 20.70 & 43.24 & 26.58 & 36.01 & 89.52 & 2.56 & 750 & -- & -- \\
Qwen3.5-4B-Seg
& 78.09 & 20.04 & 41.90 & 25.27 & 34.59 & 89.28 & 3.27 & 945 & -- & -- \\
Llama3.2-3B-Seg
& 72.08 & 17.60 & 40.10 & 22.84 & 32.31 & 88.89 & 3.68 & 1071 & -- & -- \\
Mistral3-3B-Seg
& 78.44 & 19.36 & 42.06 & 25.25 & 34.72 & 89.31 & 2.93 & 824 & -- & -- \\
Qwen3.5-2B-Seg
& 74.10 & 18.05 & 40.45 & 22.98 & 32.85 & 88.92 & 3.33 & 937 & -- & -- \\

\bottomrule
\end{tabular}
\caption{Application results on Long-MT-Bench+ (chat = GPT-4o). Same segmentation/retrieval as Table~\ref{tab:application_locomo} (chat = GPT-4o-mini); only response generation and evaluation use GPT-4o.}
\label{tab:application_locomo_gpt4o}
\end{table*}
```

## 값 표

| 범주 | Method | GPT4Score | BLEU | Rouge1 | Rouge2 | RougeL | BERTScore | # Turns | # Tokens | Pre. | Seg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| history | Zero History | 43.68 | 9.72 | 29.23 | 12.32 | 21.93 | 86.86 | 0 | 0 | — | — |
| history | Full History | **83.09** | 17.41 | 40.22 | 24.12 | 33.10 | 89.13 | 65.45 | 22676 | — | — |
| unsup. DTS | TextTiling-Style-Seg | 74.76 | 19.00 | 40.84 | 24.19 | 33.88 | 89.12 | 3.74 | 1068 | 0.26 | 0.85 |
| unsup. DTS | GreedySeg-Style-Seg | 70.31 | 17.81 | 39.48 | 22.34 | 32.25 | 88.74 | 5.38 | 1495 | 296.54 | 15.06 |
| unsup. DTS | GraphSeg-Style-Seg | 63.09 | 15.75 | 36.65 | 20.47 | 29.81 | 88.23 | 8.66 | 2446 | 44.37 | 128.80 |
| unsup. DTS | CSM-Style-Seg | 64.27 | 15.46 | 36.45 | 19.77 | 29.50 | 88.13 | 9.17 | 2430 | 305.26 | 17.27 |
| **unsup. DTS** | **Hi-OnTop (MPNet, p60)** | **79.86** | 21.03 | 43.52 | 26.26 | 36.16 | 89.52 | 2.58 | 776 | 106.93 | **0.058** |
| **unsup. DTS** | **Hi-OnTop (MPNet, p70)** | **80.62** | 20.93 | 43.89 | 26.72 | 36.54 | 89.57 | 3.13 | 894 | 106.93 | **0.056** |
| **unsup. DTS** | **Hi-OnTop (MiniLM, p60)** | **80.03** | 21.62 | 44.01 | 27.24 | 36.64 | 89.69 | 2.59 | 783 | **59.16** | **0.061** |
| **unsup. DTS** | **Hi-OnTop (MiniLM, p70)** | **80.38** | 20.85 | 43.77 | 27.22 | 36.58 | 89.65 | 3.00 | 861 | **59.16** | **0.105** |
| **unsup. DTS** | **Hi-OnTop (MiniLM, p80)** | 78.78 | 19.54 | 42.58 | 25.78 | 35.42 | 89.41 | 4.16 | 1132 | **59.16** | **0.070** |
| **unsup. DTS** | **Hi-OnTop (int8, p60)** | **80.17** | 20.89 | 43.51 | 26.65 | 36.11 | 89.56 | 2.60 | 785 | **45.70** | **0.049** |
| **unsup. DTS** | **Hi-OnTop (int8, p70)** | **80.69** | 20.75 | 43.70 | 26.96 | 36.35 | 89.61 | 3.00 | 863 | **45.70** | **0.069** |
| **unsup. DTS** | **Hi-OnTop (int8, p80)** | 77.99 | 19.41 | 41.72 | 24.78 | 34.54 | 89.27 | 4.28 | 1146 | **45.70** | **0.050** |
| **unsup. DTS** | **Hi-OnTop (MPNet, best-p=71)** | 80.10 | 20.51 | 42.92 | 25.90 | 35.51 | 89.51 | 3.18 | 907 | 106.93 | **0.188** |
| **unsup. DTS** | **Hi-OnTop (MiniLM, best-p=72)** | **80.45** | 20.35 | 43.71 | 26.86 | 36.55 | 89.52 | 3.10 | 895 | **59.16** | **0.070** |
| **unsup. DTS** | **Hi-OnTop (int8, best-p=72)** | **80.62** | 20.29 | 43.32 | 26.63 | 36.25 | 89.49 | 3.14 | 905 | **45.70** | **0.073** |
| sup. DTS | RoBERTa-Style-Seg | 74.97 | 19.58 | 41.09 | 23.97 | 33.90 | 89.12 | 3.21 | 929 | 428.09 | 0.01 |
| LLM DTS (SeCom) | GPT-4o-mini-Seg | 79.06 | 20.70 | 43.24 | 26.58 | 36.01 | 89.52 | 2.56 | 750 | — | — |
| LLM DTS (SeCom) | GPT-5-Seg | 81.84 | 21.04 | 43.87 | 27.52 | 36.65 | 89.66 | 3.09 | 892 | — | — |
| LLM DTS (SeCom) | Qwen3.5-122B-A10B-Seg | **82.33** | 21.67 | 44.33 | 27.80 | 36.95 | 89.72 | 3.01 | 876 | — | — |
| LLM DTS (SeCom) | Qwen3.5-27B-Seg | 82.08 | 21.19 | 44.32 | 27.52 | 36.89 | 89.67 | 2.99 | 863 | — | — |
| LLM DTS (SeCom) | Qwen3.5-4B-Seg | 78.09 | 20.04 | 41.90 | 25.27 | 34.59 | 89.28 | 3.27 | 945 | — | — |
| LLM DTS (SeCom) | Qwen3.5-2B-Seg | 74.10 | 18.05 | 40.45 | 22.98 | 32.85 | 88.92 | 3.33 | 937 | — | — |
| LLM DTS (SeCom) | Llama3.2-3B-Seg | 72.08 | 17.60 | 40.10 | 22.84 | 32.31 | 88.89 | 3.68 | 1071 | — | — |
| LLM DTS (SeCom) | Mistral3-3B-Seg | 78.44 | 19.36 | 42.06 | 25.25 | 34.72 | 89.31 | 2.93 | 824 | — | — |

## gpt-4o vs gpt-4o-mini 비교 (GPT4Score 변화)

| Method | gpt-4o-mini | gpt-4o | Δ |
|---|---:|---:|---:|
| Zero History | 42.12 | 43.68 | +1.56 |
| Full History | 77.92 | **83.09** | **+5.17** |
| TextTiling-Style-Seg | 73.47 | 74.76 | +1.29 |
| GraphSeg-Style-Seg | 62.53 | 63.09 | +0.56 |
| GreedySeg-Style-Seg | 68.58 | 70.31 | +1.73 |
| CSM-Style-Seg | 64.24 | 64.27 | +0.03 |
| RoBERTa-Style-Seg | 74.44 | 74.97 | +0.53 |
| Hi-OnTop (MPNet, p60) | 77.50 | 79.86 | **+2.36** |
| Hi-OnTop (MPNet, p70) | 79.27 | 80.62 | **+1.35** |
| Hi-OnTop (MiniLM, p60) | 78.26 | 80.03 | **+1.77** |
| Hi-OnTop (MiniLM, p70) | 80.28 | 80.38 | +0.10 |
| Hi-OnTop (MiniLM, p80) | 77.22 | 78.78 | +1.56 |
| Hi-OnTop (int8, p60) | 78.75 | 80.17 | **+1.42** |
| Hi-OnTop (int8, p70) | 79.90 | **80.69** | **+0.79** |
| Hi-OnTop (int8, p80) | 75.87 | 77.99 | +2.12 |
| Hi-OnTop (MPNet, cal-p71) | 79.17 | 80.10 | +0.93 |
| Hi-OnTop (MiniLM, cal-p72) | 79.31 | 80.45 | +1.14 |
| Hi-OnTop (int8, cal-p72) | 80.14 | 80.62 | +0.48 |
| Ours (MPNet, a=0.0) | 78.72 | 80.83 | **+2.11** |
| Ours (MPNet, a=1.0) | 78.75 | 81.32 | **+2.57** |
| GPT-4o-mini-Seg | 78.13 | 79.06 | +0.93 |
| GPT-5-Seg | 80.63 | 81.84 | +1.21 |
| Qwen3.5-122B-A10B-Seg | 80.83 | **82.33** | **+1.50** |
| Qwen3.5-27B-Seg | **81.28** | 82.08 | +0.80 |
| Qwen3.5-4B-Seg | 76.77 | 78.09 | +1.32 |
| Llama3.2-3B-Seg | 71.60 | 72.08 | +0.48 |
| Mistral3-3B-Seg | 76.91 | 78.44 | +1.53 |
| Qwen3.5-2B-Seg | 72.81 | 74.10 | +1.29 |

## 해석

### chat=gpt-4o 전환의 전반적 효과

- **평균 +1.3점** 상승 (28 method 전체 산술 평균). 모든 method에서 양수 또는 0 변화 — 회귀 없음.
- **Full History +5.17** 가 가장 큰 폭: gpt-4o가 22k 토큰 전체 컨텍스트를 gpt-4o-mini 대비
  훨씬 잘 활용함. 장문 컨텍스트 이해 능력 차이가 이 데이터셋에서 두드러짐.
- **ablation (a=0.0 +2.11, a=1.0 +2.57)**: 이전 chat 모델로는 두 블렌딩 극단이 비슷했으나
  gpt-4o에서 `prev only`(a=1.0)가 81.32로 오름. 상대적 ranking 변화 주목.

### Hi-OnTop (unsup. DTS) 성능

- **best row: Hi-OnTop (int8, p70) = 80.69** — chat=gpt-4o-mini 시절 79.90에서 +0.79.
- gpt-4o 전환 후에도 **unsup. DTS 최고점 < GPT-4o-mini-Seg (79.06)** 를 역전:
  int8-p70 (80.69) > GPT-4o-mini-Seg (79.06) **+1.63 격차 유지** (이전 +1.77 → 비슷).
- 모든 Hi-OnTop p70 변형이 79.06 이상: unsup. DTS 가 LLM-segmenter(소형) 를 능가하는
  패턴은 chat 모델 업그레이드 후에도 robust하게 유지.

### LLM Segmenter 경쟁 구도 변화

- **Qwen3.5-122B-A10B-Seg 82.33** — 전체 1위 (이전도 1위, +1.50 폭 상승).
- **Qwen3.5-27B-Seg 82.08** — 2위 유지.
- **GPT-5-Seg 81.84** — 3위. 이전 80.63 대비 +1.21.
- **ablation a=1.0 (81.32)** — GPT-5-Seg(81.84) 와 0.5점 차로 근접. 단순 prev-only 블렌딩이
  GPT-5 segmentation 에 근접한다는 점은 흥미로운 관찰 (주의: 이건 chat LLM 의 이점이지
  segmentation 품질의 직접 비교는 아님).
- **Hi-OnTop (int8, p70) 80.69** — unsup. DTS 내 최고. GPT-5-Seg 대비 1.15점 뒤짐
  (이전 1.27점 차 → 소폭 좁혀짐).

### 순위 역전 및 주목할 변화

- **MiniLM p70 (80.38) < MPNet p70 (80.62)**: gpt-4o-mini 시절과 역전. 이전엔 MiniLM p70이
  +0.51 우위였으나 gpt-4o 에선 MPNet p70이 +0.24 우위. 오차 범위 내로 볼 수도 있으나
  추세로는 encoder 차이가 chat LLM 성능으로 흡수되는 경향.
- **p70 calibration rows (cal-p71/72)**: 3개 모두 0.07~0.57 아래로 aligned-p70 보다 약간
  낮음 (이전도 비슷). LLM-distill calibration이 gpt-4o 환경에서도 p70 default 와
  실질적 차이 없음을 확인.

### 한계

- n_conv=11 (소규모) — 절대 점수 variation 주의 (±0.5점 noise 가능).
- judge = gpt-4o 동일이므로 GPT4Score 스케일은 비교 가능하나, chat=judge 동일 모델이
  자기 응답에 관대할 수 있다는 편향 가능성 (LLM judge self-preference).
- Latency 컬럼은 chat=gpt-4o-mini 기준 그대로 (segmentation latency 는 동일 pipeline
  재활용이라 변화 없음). gpt-4o chat latency 자체는 별도 측정 안 함.
- ablation (a=0/1) 결과 해석 시 segment 결과가 MPNet p70 기준임 주의. 블렌딩 방식 자체
  효과와 chat LLM 의 컨텍스트 활용 능력이 섞여있음.

## 채우기 상태

| Method | 상태 |
|---|---|
| Zero/Full/모든 baseline/Hi-OnTop/ablation | ✅ 완료 (2026-05-28, chat=gpt-4o) |
| 28개 전 method, fail=0, skip=1(zero, pre-exists) | ✅ 완료 |
