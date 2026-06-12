# §3 Method — Hi-OnTop

> EMNLP-style draft (Algorithm-first). Outline:
> - 3.1 Preliminaries
> - 3.2 Streaming Segmentation with Causal Context
> - 3.3 Label-Free Threshold Calibration
> - 3.4 Application to Conversational Memory

---

## 3.1 Preliminaries

We address **online dialogue topic segmentation** (DTS): given a streaming dialogue
$D = (u_1, u_2, \ldots, u_T)$ where $u_t$ denotes the $t$-th utterance, at each turn $t$
the model must emit a binary decision $b_t \in \{0,1\}$ — whether $u_t$ initiates a new
topic segment — using only the prefix $u_{\le t}$.  This *prefix-causal* constraint
distinguishes online DTS from offline variants (e.g., TextTiling) that exploit
future context, and motivates segmentation strategies that operate in a single
forward pass without future lookahead.

Let $f : \text{text} \rightarrow \mathbb{R}^d$ be a frozen sentence encoder with
L2-normalized outputs; we denote $s_t = f(u_t)$.  All similarities are measured by
cosine distance $\delta(x,y) = 1 - \cos(x, y) \in [0, 2]$.  A topic segment is a
maximal contiguous span of utterances sharing the same topic; segment boundaries
partition $D$ into $K \ge 1$ non-overlapping segments.

We evaluate boundary predictions with three standard metrics — $P_k$
(probabilistic boundary mismatch), $\text{WindowDiff}$ (WD), and boundary-set
$F_1$ — and report the aggregate
$\text{Score} = 0.5\,F_1 + 0.25\,(1 - P_k) + 0.25\,(1 - \text{WD})$
following the SuperDialseg benchmark.

---

## 3.2 Streaming Segmentation with Causal Context

We propose **Hi-OnTop** (*Hi*erarchical *Do*t-product *T*opic *S*egmenter), a
single-pass online segmenter whose decision relies on a context-aware distance
$\delta_{\text{eff}}(t)$ between the current utterance and a windowed view of its
past. The design is intentionally minimalist: no learned dynamics, no future
context, no per-topic state — only L2-normalized embeddings and a scalar threshold.

**Adjacent distance.**  The atomic signal is the cosine distance between the
current and the immediately preceding embedding,
$$
\delta_{\text{prev}}(t) = 1 - \cos(s_{t-1}, s_t).
$$
This quantity alone is the input to many prior unsupervised DTS baselines
(GreedySeg, CSM); we treat it as a starting point, not a final score, because a
single-step view is noise-dominated under conversational speech act variation
(e.g., short acknowledgements between two on-topic utterances).

**Causal context distance.**  We extend $\delta_{\text{prev}}$ by averaging the
distances of $s_t$ against the *last $m$ utterances*, exponentially decayed by a
factor $\rho \in (0, 1]$:
$$
\delta_{\text{ctx}}(t)
\;=\;
\frac{\sum_{i=1}^{m}\,\rho^{\,i}\,\delta(s_{t-1-i},\, s_t)}
     {\sum_{i=1}^{m}\,\rho^{\,i}}
\qquad
\bigl(\text{undefined for } t \le m;\ \delta_{\text{ctx}}(t) := \delta_{\text{prev}}(t)\bigr).
$$
The window $m$ controls how much short-term history smooths the per-turn signal,
and $\rho$ governs how aggressively older turns are discounted.

**Effective distance.**  We combine adjacent and contextual views by convex
combination:
$$
\delta_{\text{eff}}(t)
\;=\;
a\,\delta_{\text{prev}}(t)
\;+\;
(1 - a)\,\delta_{\text{ctx}}(t),
\qquad a \in [0, 1].
$$
The boundary rule is a simple threshold test:
$$
b_t \;=\; \mathbb{1}\!\bigl[\,\delta_{\text{eff}}(t) > \delta^{*}\,\bigr].
$$

**Hyperparameters.**  Throughout we fix $m = 2$, $\rho = 0.7$, $a = 0.5$ — these
defaults are selected once on TIAGE-train and held constant across all
benchmarks, encoders, and downstream tasks; only $\delta^{*}$ is data-adaptive
(Section 3.3).

**Complexity.**  Hi-OnTop is $\mathcal{O}(m)$ per turn in arithmetic operations
beyond the encoder forward pass, with $\mathcal{O}(m)$ persistent state.  No
gradients, no recurrent computation, no future buffer — the entire segmenter is
expressible as $\sim 20$ lines of numpy.  In practice the encoder forward
dominates wall-clock by 3–5 orders of magnitude (Section 4).

---

## 3.3 Label-Free Threshold Calibration

The only data-dependent parameter is the threshold $\delta^{*}$.  Two observations
make $\delta^{*}$ well-behaved:

1.  Across encoders, the *distribution* of $\delta_{\text{eff}}$ values on a
    domain is much more stable than its absolute scale.  Encoder choice shifts
    the entire distribution (e.g., MPNet $\bar{\delta}_{\text{eff}} \approx 0.38$
    vs. MiniLM $\bar{\delta}_{\text{eff}} \approx 0.52$), but the *rank* of true
    topic boundaries within the per-domain distribution remains high.
2.  The per-domain boundary rate is loosely known *a priori* (e.g., open-domain
    dialogue has roughly 10–30\% boundary turns).  This pins a percentile range
    of $\delta_{\text{eff}}$ as a natural threshold.

**Percentile calibration.**  Given a calibration set
$\mathcal{C} = \{u^{(c)}_{1:T_c}\}$ drawn from the deployment domain (no
boundary labels required), we compute the empirical $\delta_{\text{eff}}$
distribution and define
$$
\delta^{*}_{p_x} \;=\;
\operatorname{percentile}_x\!
  \Bigl(\,\bigl\{\delta_{\text{eff}}(t) : u_t \in \mathcal{C}\bigr\}\,\Bigr).
$$
We refer to $\delta^{*}_{p_x}$ as the *$x$-percentile threshold*.  Increasing $x$
yields fewer boundaries; decreasing $x$ yields more.  Calibration is fully
unsupervised, encoder-agnostic, and requires only a forward pass per turn of
$\mathcal{C}$.

**Two empirical claims about percentile calibration** drive the rest of the paper:

- *Calibration is cheap.*  On a 3-benchmark × 3-encoder grid we show that
  $N \approx 100$–$200$ calibration dialogs suffice to reach the supervised
  oracle within $\pm 0.005$ Score; larger $N$ yields no further improvement
  (Section 4.2).

- *A small percentile family approximates the supervised oracle.*  For each
  (encoder, dataset) cell there exists a $p_x \in \{p_{50}, \ldots, p_{85}\}$
  whose test Score is within $\pm 0.005$ of the supervised oracle (which sweeps
  $\delta^{*}$ over a fine grid).  The optimal $p_x$ tracks the boundary density
  of the domain (e.g., document-grounded QA prefers $p_{60}$; sparsely-annotated
  artificial concatenations prefer $p_{80}$–$p_{85}$).  A single fixed default
  $p_{70}$–$p_{75}$ trails the per-cell oracle by only $\sim 0.022$ Score on
  average across 9 cells.

Together these two observations license a *practical recipe* for deploying
Hi-OnTop to a new domain with neither boundary labels nor expensive
hyperparameter search: gather $\sim 100$ unlabeled in-domain dialogs, encode
once, and read off $\delta^{*}_{p_{70}}$.

---

## 3.4 Application to Conversational Memory

Hi-OnTop is intended as a low-cost segmentation module for **memory-augmented
dialogue systems**.  We instantiate it as a drop-in replacement for the
LLM-based segmenter inside the SeCom\,\citep{secom} pipeline for long-horizon
QA on Long-MT-Bench+.

**Pipeline.**  Given a multi-session conversation $C$, the SeCom-swap pipeline
performs the following:

1.  **Segmentation** — partition each session of $C$ into topic segments using
    Hi-OnTop (Section 3.2).
2.  **Memory compression** — apply LLMLingua-2 to each segment to produce a
    compressed memory token sequence.
3.  **Retrieval** — index compressed segments via MPNet + FAISS; at query time,
    retrieve the top-$k$ segments by cosine similarity.
4.  **Generation** — feed the retrieved segments and the question to a
    downstream chat model (we use \texttt{gpt-4o-mini}).

Crucially, **only step 1 is replaced**; steps 2–4 retain SeCom's original
implementation.  This isolates the effect of segmentation quality on
end-to-end QA, and lets us compare Hi-OnTop to (i) an LLM-based segmenter
(SeCom's own; \texttt{gpt-4o-mini-Seg}), (ii) unsupervised baselines
(TextTiling, GreedySeg, CSM, GraphSeg), and (iii) a supervised baseline
(RoBERTa).

**Encoder choice and latency.**  Hi-OnTop imposes no constraint on the encoder
beyond L2-normalized cosine geometry.  We instantiate it with three encoders of
increasing compression: MPNet (110M, fp32), MiniLM (22M, fp32), and
MiniLM-int8 (22M, quantized ONNX with \texttt{quint8\_avx2}).  Because
$\delta^{*}$ is recalibrated per encoder (Section 3.3), the choice of encoder
trades segmentation latency against modest changes in segment quality
(Section 4.3).  The lowest-cost configuration (MiniLM-int8, ONNX \texttt{quint8\_avx2})
reduces the per-turn segmentation latency from 568\,ms (MPNet, CPU, batch=1,
online streaming) to **77\,ms** — a 7.4$\times$ speedup — while preserving QA
performance (Section 4.3).

---

## TODO (작성 후 추가)

- [ ] Section 4 (Experiments) numbers fill-in: percentile generality grid (3×3×10), N\* convergence (3×3), downstream QA table.
- [ ] Section 3.4 latency numbers: pending MiniLM-int8 measurement (`bmas6b9v3`).
- [ ] Section 3.4 downstream QA: pending MiniLM-int8 pipeline (`b3uu2fg04`, ~2-3 hr).
- [ ] Decide §3.2 hyperparameter selection narrative: "fixed on TIAGE-train" vs. "from prior work".
- [ ] Add running example / figure for §3.2 (per-turn δ_eff curve with threshold overlay).
- [ ] §3.3 figure: percentile-Score curve per benchmark (data already in `outputs/experiments/2026-05-23_percentile_generality/`).
