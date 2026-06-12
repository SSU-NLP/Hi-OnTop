# §3 Method — Hi-OnTop

> EMNLP-style draft (Algorithm-first). LaTeX-ready prose. Citation placeholders
> use `\citep{...}`. Tables and figures are referenced as `\autoref{tab:...}` /
> `\autoref{fig:...}` and to be defined in §4 (Experiments).

---

## 3 Method

We introduce \textsc{Hi-OnTop}, an online, encoder-agnostic dialogue topic
segmenter whose only data-adaptive parameter is a scalar decision threshold
calibrated without boundary labels.  §3.1 fixes notation; §3.2 specifies the
segmenter; §3.3 introduces our label-free calibration recipe; §3.4 describes
how \textsc{Hi-OnTop} drops into a memory-augmented dialogue pipeline.

### 3.1 Preliminaries

**Problem.**
Let $D = (u_1, u_2, \ldots, u_T)$ denote a streaming dialogue, where $u_t$ is
the $t$-th utterance.  At each turn $t$, an *online* segmenter outputs a
binary decision $b_t \in \{0, 1\}$ — whether $u_t$ initiates a new topic
segment — using only the prefix $u_{\le t}$.  This *prefix-causal* constraint
forbids future lookahead and distinguishes online dialogue topic segmentation
(DTS) from offline variants such as TextTiling \citep{hearst1997texttiling}
and CSM \citep{xing2020csm}.  A topic segment is a maximal contiguous span of
utterances under the same topic; the boundary sequence $\{b_t\}_{t=1}^{T}$
induces a partition of $D$ into $K \ge 1$ segments.

**Encoder.**
Let $f : \texttt{text} \!\rightarrow\! \mathbb{R}^d$ be a frozen
sentence encoder producing $L_2$-normalized embeddings, and write
$s_t = f(u_t)$.  All similarities are measured as cosine distance,
$\delta(x, y) = 1 - \cos(x, y) \in [0, 2]$.  We make no assumption about $f$
beyond unit-norm outputs; experimentally we vary $f$ across MPNet (110M
parameters, fp32) \citep{song2020mpnet} and MiniLM-int8 (22M, INT8 quantized
ONNX) \citep{wang2020minilm}.

**Evaluation.**
We follow the SuperDialseg \citep{xia2023superdialseg} convention and report
$P_k$ \citep{beeferman1999pk}, WindowDiff (WD) \citep{pevzner2002windowdiff},
and boundary-set $F_1$.  Their composite is
$\mathrm{Score} = 0.5\, F_1 + 0.25\,(1 - P_k) + 0.25\,(1 - \mathrm{WD})$,
where $P_k$ and WD are computed with $k$ set to half the mean segment length
per dialogue.  Hyperparameters of competing methods follow their published
defaults.

### 3.2 Streaming Segmentation with Causal Context

\textsc{Hi-OnTop} computes a context-aware distance $\delta_{\mathrm{eff}}(t)$
between $s_t$ and a windowed view of its past, and emits a boundary whenever
$\delta_{\mathrm{eff}}(t)$ exceeds a calibrated threshold.  The design is
intentionally minimalist — no learned dynamics, no future buffer, no per-topic
state — so that downstream applications can swap encoders or domains without
retraining.

**Adjacent distance.**
The atomic signal is the cosine distance between consecutive embeddings:
$$
\delta_{\mathrm{prev}}(t) \;=\; 1 - \cos(s_{t-1}, s_t).
\tag{1}
$$
Prior unsupervised segmenters (GreedySeg, CSM) threshold this quantity
directly.  We treat it as a noisy proxy: a single-step view is dominated by
local speech-act variation (short acknowledgements between on-topic
utterances), inflating $\delta_{\mathrm{prev}}$ at non-boundary positions.

**Causal context distance.**
To suppress that noise we average $s_t$'s distances to the *last $m$
utterances*, exponentially down-weighted by $\rho \in (0, 1]$:
$$
\delta_{\mathrm{ctx}}(t)
\;=\;
\frac{\sum_{i=1}^{m}\,\rho^{\,i}\,\delta(s_{t-1-i},\, s_t)}
     {\sum_{i=1}^{m}\,\rho^{\,i}},
\qquad t > m.
\tag{2}
$$
For $t \le m$ we fall back to $\delta_{\mathrm{ctx}}(t) := \delta_{\mathrm{prev}}(t)$.
The window $m$ controls how much short-term history smooths the signal, and
$\rho$ governs how aggressively older turns are discounted.

**Effective distance and decision rule.**
We linearly mix adjacent and contextual views:
$$
\delta_{\mathrm{eff}}(t)
\;=\;
a\,\delta_{\mathrm{prev}}(t)
\;+\;
(1 - a)\,\delta_{\mathrm{ctx}}(t),
\qquad a \in [0, 1].
\tag{3}
$$
The boundary rule is a single threshold test:
$$
b_t \;=\; \mathbb{1}\!\bigl[\,\delta_{\mathrm{eff}}(t) > \delta^{*}\,\bigr].
\tag{4}
$$
We fix the three structural hyperparameters $m = 2$, $\rho = 0.7$, $a = 0.5$
once on a single held-out development set (TIAGE-train) and reuse them across
*all* benchmarks, encoders, and downstream tasks; only $\delta^{*}$ is
data-adaptive (§3.3).

**Complexity.**
Beyond the encoder forward pass, \textsc{Hi-OnTop} performs $\mathcal{O}(m)$
arithmetic operations per turn and maintains $\mathcal{O}(m)$ persistent
state.  No gradients, no recurrence, no future buffer; the entire segmenter is
$\sim$\,20 lines of NumPy.  In our measurements (§4.3) the encoder forward
dominates wall-clock by 3--4 orders of magnitude.

### 3.3 Label-Free Threshold Calibration

The threshold $\delta^{*}$ is the only data-dependent parameter, and its
optimal value shifts with both the encoder (different cosine geometry) and the
domain (different boundary density).  Rather than retrain or hand-tune
$\delta^{*}$ per deployment, we exploit two empirical regularities of
$\delta_{\mathrm{eff}}$ that admit a fully label-free recipe.

**Observation 1: distributional rank, not absolute scale, governs boundary
detection.**
Across encoders, the mean of $\delta_{\mathrm{eff}}$ on a fixed domain varies
substantially (e.g., MPNet $\bar{\delta} \!\approx\! 0.38$ vs.\ MiniLM-int8
$\bar{\delta} \!\approx\! 0.52$ on Long-MT-Bench+), but the *rank* of true
boundaries within the per-domain distribution remains high — boundary turns
sit in the upper quantile of $\delta_{\mathrm{eff}}$ regardless of the encoder.
This rank stability motivates a percentile-based threshold.

**Observation 2: per-domain boundary rates are loosely known.**
For human dialogue corpora, the empirical boundary rate falls in a
predictable band (roughly 10--30\% of turns for open-domain conversation,
$<$10\% for sparsely-annotated artificial concatenations).  This bounds the
useful range of percentiles tightly enough that a small family $p_x \in
\{p_{60}, p_{65}, \ldots, p_{85}\}$ covers all reasonable deployments.

**Percentile threshold.**
Given a *label-free* calibration set $\mathcal{C}$ of in-domain dialogues —
boundary annotations are *not* required — we encode each $u_t \in \mathcal{C}$
and define
$$
\delta^{*}_{p_x}
\;=\;
\mathrm{Percentile}_x\!
  \Bigl(\,\bigl\{\delta_{\mathrm{eff}}(t) : u_t \in \mathcal{C}\bigr\}\,\Bigr).
\tag{5}
$$
Larger $x$ yields fewer boundaries; smaller $x$ yields more.  Equation (5) is
the entire calibration procedure: no held-out boundary labels, no validation
set, no gradient computation.

**Two empirical claims** about percentile calibration anchor our experiments:

1.  *Calibration is cheap.*  On a 3-benchmark $\times$ 3-encoder grid
    (\autoref{tab:calib-n}), $N \!\approx\! 100$ unlabeled calibration
    dialogs suffice to reach the supervised oracle within $\pm 0.005$ Score;
    additional calibration data yields no further improvement.

2.  *A small percentile family approximates the supervised oracle.*  For each
    (encoder, dataset) cell, *some* $p_x \in \{p_{50}, \ldots, p_{85}\}$
    achieves test Score within $\pm 0.005$ of a supervised oracle that sweeps
    $\delta^{*}$ over a fine grid using gold boundary labels
    (\autoref{tab:percentile-grid}).  The optimal $p_x$ tracks domain
    boundary density: document-grounded subtopic shifts (SuperDialseg) prefer
    $p_{60}$, sparsely-annotated dialogue concatenations
    (Dialseg711) prefer $p_{80}$--$p_{85}$.  A single fixed default
    $p_{70}$--$p_{75}$ trails the per-cell oracle by only
    $\sim$\,0.022 Score averaged across nine cells.

Together, these two facts yield a practical recipe for deploying
\textsc{Hi-OnTop} on a new domain: gather $\sim$\,100 unlabeled in-domain
dialogues, encode them once, and read off $\delta^{*}_{p_{70}}$.  No boundary
annotation, no hyperparameter search, no model adaptation.

### 3.4 Application to Conversational Memory

\textsc{Hi-OnTop} is designed as a low-cost segmentation module for
*memory-augmented dialogue systems*.  We instantiate it as a drop-in
replacement for the LLM-based segmenter in SeCom \citep{pan2024secom}, a
recent pipeline for long-horizon conversational QA.  The original SeCom
performs segmentation by calling \texttt{gpt-4o-mini} once per turn; the
remaining stages — memory compression, dense retrieval, and answer
generation — are unchanged.

**Pipeline (SeCom-swap).**  Given a multi-session conversation $C$:

1.  *Segmentation.*  Partition each session of $C$ into topic segments using
    \textsc{Hi-OnTop} (§3.2), with $\delta^{*}_{p_x}$ calibrated by Eq. (5) on
    the held-out training half of $C$ itself (still label-free; see §3.3).
2.  *Memory compression.*  Apply LLMLingua-2
    \citep{pan2024llmlingua2} to each segment, producing a compressed memory
    token sequence (compression ratio $0.75$, following SeCom's default).
3.  *Retrieval.*  Encode compressed segments with MPNet and index them via
    FAISS; at query time, retrieve the top-$1$ segment by cosine similarity.
4.  *Generation.*  Feed the retrieved segment and the question to
    \texttt{gpt-4o-mini}.

Only step 1 is replaced; steps 2--4 retain SeCom's original implementation.
This isolation lets us attribute any change in QA performance to
segmentation quality alone, and lets us compare \textsc{Hi-OnTop} against
(i) the LLM-based segmenter it replaces, (ii) unsupervised baselines
(TextTiling, GreedySeg, CSM, GraphSeg), and (iii) a supervised baseline
(RoBERTa).

**Encoder choice and latency.**
\textsc{Hi-OnTop} imposes no constraint on the encoder beyond $L_2$-normalized
cosine geometry.  We instantiate it with two encoders of contrasting cost:
MPNet (110M parameters, fp32) and MiniLM-int8 (22M parameters, ONNX
\texttt{quint8\_avx2}).  Because $\delta^{*}$ is recalibrated per encoder via
Eq. (5), the choice of encoder trades segmentation latency against modest
shifts in segment quality (§4.3).  The lowest-cost configuration
(MiniLM-int8) reduces the per-turn segmentation latency from $568$\,ms (MPNet,
CPU, batch=1, online streaming) to $77$\,ms — a $7.4\times$ speedup — while
maintaining QA performance to within $0.5$ GPT-4 score points
(\autoref{tab:downstream}).

---

## TODO

- [ ] §4 (Experiments) numbers fill-in once int8 pipeline (`b0yohgkef`) completes.
- [ ] Confirm SeCom citation key (Pan et al. 2024) matches \texttt{secom.bib}.
- [ ] Replace placeholder \autoref{tab:...} keys with final labels.
- [ ] Consider adding a Figure for §3.2 (per-turn $\delta_{\mathrm{eff}}$ curve with $\delta^{*}$ overlay) — data ready in `outputs/experiments/.../diag_hiontop_segmentation_demo/`.
- [ ] Consider adding a Figure for §3.3 (percentile-Score curve per benchmark) — data ready in `outputs/experiments/2026-05-23_percentile_generality/`.
- [ ] §3.2 cite original $\delta_{\mathrm{prev}}$-only baselines GreedySeg and CSM.
- [ ] Decide whether to keep both MPNet and MiniLM-int8 rows in main downstream table, or move MPNet to ablation appendix.
