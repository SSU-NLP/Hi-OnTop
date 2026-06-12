```latex
\section{Method}

We introduce \textsc{Hi-OnTop}, an online dialogue topic segmenter that operates under the response-unknown, future-blind regime motivated in §1. The segmenter realizes two cognitively-grounded principles operationally: (i) it accumulates prediction error at \emph{two time scales}—an adjacent-turn distance and a decayed causal-context distance—and (ii) it preserves the \emph{strength} of each boundary alongside the discrete decision, so that downstream consumers can act on graded confidence rather than a hard $0/1$ stream. §3.1 fixes the problem setup and notation; §3.2 specifies the segmenter and its graded readout; §3.3 introduces a label-free calibration recipe that requires no boundary annotations; §3.4 instantiates the segmenter as a drop-in module inside a memory-augmented conversational pipeline.

\subsection{Preliminaries}
\label{sec:3.1}

\textbf{Problem.} Let $D = (u_1, u_2, \ldots, u_T)$ be a streaming dialogue with $u_t$ the $t$-th utterance. At every step $t$, an \emph{online} segmenter must emit a binary decision
$$b_t \in \{0, 1\},\qquad b_t = 1 \iff u_t \text{ opens a new topic segment},$$
using only the causal prefix $u_{\le t}$. This \emph{prefix-causal} constraint—the response-unknown setting of \citet{lin2023topic, lin2023multi}—forbids access to $u_{t+1}, \ldots, u_T$ and distinguishes our setup from offline dialogue topic segmentation, where every candidate boundary is scored with full bilateral context~\citep{hearst1997texttiling, glavas2016graphseg, xing2021csm}. A topic segment is a maximal contiguous span of utterances under a single topic; the boundary sequence $\{b_t\}_{t=1}^{T}$ induces a partition of $D$ into $K \ge 1$ segments.

\textbf{Encoder.} Let $f : \texttt{text} \to \mathbb{R}^d$ be a frozen sentence encoder producing $L_2$-normalized embeddings, and write $s_t = f(u_t)$. All similarities are measured as cosine distance
$$\delta(x, y) = 1 - \cos(x, y) \in [0, 2].$$
We make no assumption about $f$ beyond unit-norm outputs; the segmenter is therefore agnostic to the choice of encoder.

\subsection{Hi-OnTop: Dual-Time-Scale Segmenter with Graded Output}
\label{sec:3.2}

\textsc{Hi-OnTop} computes a context-aware effective distance $\delta_{\mathrm{eff}}(t)$ between $s_t$ and a windowed view of its causal past, then emits a boundary when $\delta_{\mathrm{eff}}(t)$ crosses a calibrated threshold $\delta^{*}$. The design is intentionally minimalist—no learned dynamics, no future buffer, no per-topic state—so that the segmenter can be transplanted across encoders and domains without retraining.

\textbf{Adjacent (short-scale) distance.} The atomic signal is the cosine distance between consecutive embeddings,
\begin{equation}
\delta_{\mathrm{prev}}(t) \;=\; 1 - \cos(s_{t-1}, s_t). \label{eq:dprev}
\end{equation}
Prior unsupervised online segmenters threshold this quantity directly~\citep{xing2021csm}. We treat it as a noisy proxy: a one-step view is dominated by local speech-act variation—short acknowledgements between on-topic utterances, fillers, and elliptical follow-ups inflate $\delta_{\mathrm{prev}}$ at non-boundary positions.

\textbf{Causal-context (long-scale) distance.} To suppress short-scale noise we measure $s_t$ against the \emph{last $m$ utterances}, exponentially down-weighted by $\rho \in (0, 1]$:
\begin{equation}
\delta_{\mathrm{ctx}}(t) \;=\;
\frac{\sum_{i=1}^{m}\,\rho^{\,i}\,\delta(s_{t-1-i},\, s_t)}
     {\sum_{i=1}^{m}\,\rho^{\,i}},
\qquad t > m, \label{eq:dctx}
\end{equation}
with the fallback $\delta_{\mathrm{ctx}}(t) := \delta_{\mathrm{prev}}(t)$ for $t \le m$. The window $m$ controls how much short-term history smooths the signal; $\rho$ governs how aggressively older turns are discounted. Equation~\ref{eq:dctx} requires only $\mathcal{O}(m)$ persistent state and $\mathcal{O}(m)$ arithmetic per turn.

\textbf{Effective distance and decision rule.} We linearly mix the adjacent and contextual views,
\begin{equation}
\delta_{\mathrm{eff}}(t) \;=\; a\,\delta_{\mathrm{prev}}(t) + (1 - a)\,\delta_{\mathrm{ctx}}(t), \qquad a \in [0, 1], \label{eq:deff}
\end{equation}
and threshold:
\begin{equation}
b_t \;=\; \mathbb{1}\!\bigl[\,\delta_{\mathrm{eff}}(t) > \delta^{*}\,\bigr]. \label{eq:decision}
\end{equation}
Equation~\ref{eq:deff} operationalizes Principle~1 of §1: the short- and long-scale prediction errors are tracked jointly, in line with the multi-time-scale account of event boundary formation in the brain~\citep{geerligs2022partially}. The mixing weight $a$ governs how much of the signal comes from the immediate transition versus the accumulated context; we keep $a$, together with $m$ and $\rho$, as small fixed structural hyperparameters of the segmenter, distinct from the data-adaptive threshold $\delta^{*}$ that §3.3 calibrates.

\textbf{Graded boundary readout.} Beyond the binary decision in Eq.~\ref{eq:decision}, \textsc{Hi-OnTop} exposes the normalized confidence
\begin{equation}
g_t \;=\; \delta_{\mathrm{eff}}(t) \,/\, \delta^{*}, \label{eq:graded}
\end{equation}
mapped to a four-level strength band: $g_t \!<\! 0.7$ (very weak), $0.7 \!\le\! g_t \!<\! 1.0$ (weak), $1.0 \!\le\! g_t \!<\! 1.3$ (normal), and $g_t \!\ge\! 1.3$ (strong). The bands operationalize Principle~2 of §1: the magnitude of the prediction error is preserved alongside the boundary decision, mirroring the graded hippocampal boundary response of \citet{ben2018hippocampal}. Downstream consumers that benefit from confidence-aware behavior—deferring memory writes on weak boundaries or committing eagerly on strong ones—can act on $g_t$ directly; the binary stream $\{b_t\}$ remains the default interface.

\textbf{Complexity.} Beyond the encoder forward pass, each turn requires $\mathcal{O}(m)$ arithmetic on $d$-dimensional vectors and $\mathcal{O}(m)$ persistent state; no gradient, no recurrence, no future buffer. The segmenter is therefore $\mathcal{O}(1)$ per turn in the dialogue length $T$—a property that distinguishes it from offline segmenters whose per-turn cost grows with the cumulative prefix.

\subsection{Label-Free Threshold Calibration}
\label{sec:3.3}

The threshold $\delta^{*}$ is the only data-dependent parameter of the segmenter, and its optimum shifts with both the encoder (different cosine geometry) and the domain (different boundary density). Rather than retrain or hand-tune $\delta^{*}$ per deployment, we exploit two empirical regularities of the $\delta_{\mathrm{eff}}$ distribution that admit a fully label-free recipe.

\textbf{Observation 1 (rank stability across encoders).} On a fixed domain, the \emph{absolute scale} of $\delta_{\mathrm{eff}}$ varies substantially with the encoder, but the \emph{rank} of true topic boundaries within the per-domain distribution is preserved: boundary turns consistently sit in the upper quantile of $\delta_{\mathrm{eff}}$ regardless of the encoder. This motivates a percentile-based threshold rather than a fixed scalar.

\textbf{Observation 2 (loose prior on boundary density).} Per-domain boundary rates fall in a predictable band—a small fraction of turns initiate new topics in human dialogue, with the exact ratio dictated by genre rather than by any per-deployment quantity. This bounds the useful percentile range tightly enough that a small family of candidate percentiles covers all reasonable deployments.

\textbf{Percentile threshold.} Given a label-free calibration set $\mathcal{C}$ of in-domain dialogues—boundary annotations are \emph{not} required—we encode each utterance, run \textsc{Hi-OnTop} forward over $\mathcal{C}$ to collect the $\delta_{\mathrm{eff}}$ samples, and set
\begin{equation}
\delta^{*}_{p_x} \;=\; \mathrm{Percentile}_{x}\!\Bigl(\,\bigl\{\delta_{\mathrm{eff}}(t) : u_t \in \mathcal{C}\bigr\}\,\Bigr). \label{eq:px}
\end{equation}
Larger $x$ yields fewer, more conservative boundaries; smaller $x$ yields more. Equation~\ref{eq:px} is the entire calibration procedure: no held-out boundary labels, no validation set, no gradient computation.

\textbf{Two operational claims} license the recipe; both are quantified in §4.

\smallskip
\noindent\emph{Claim~1 (calibration is cheap).} A modest number of unlabeled in-domain dialogues suffices to reach a percentile estimate whose downstream segmentation Score is statistically indistinguishable from that obtained with the entire calibration corpus. Calibration noise is dominated by the percentile estimator's intrinsic variance at small $|\mathcal{C}|$ rather than by domain coverage.

\smallskip
\noindent\emph{Claim~2 (oracle-tight percentile family).} For every (encoder, domain) cell of interest, there exists a percentile $p_x^{\star}$ in a small fixed family whose test Score lies essentially at the supervised oracle obtained by sweeping $\delta^{*}$ with gold boundary labels. The optimal $p_x^{\star}$ varies monotonically with boundary density, but a single fixed default within the family trails the per-cell oracle by a small margin, and a cheap label-free sweep on a held-out unlabeled fraction of $\mathcal{C}$ closes the remaining gap.

\smallskip
Together, the two claims yield a deployment recipe that requires no boundary labels, no hyperparameter search, and no encoder retraining: collect a small set of unlabeled in-domain dialogues, encode them once, and read off either a single fixed-percentile threshold or the argmax over a small percentile family on the same unlabeled set.

\subsection{Application to Conversational Memory}
\label{sec:3.4}

\textsc{Hi-OnTop} is designed as a low-cost segmentation module for memory-augmented dialogue systems, which typically interleave segmentation with downstream stages of memory compression, retrieval, and answer generation. We instantiate it as a drop-in replacement for the segmenter inside such a pipeline; all other stages are reused without modification, isolating segmentation quality as the only controlled variable.

\textbf{Pipeline.} Given a multi-session conversation $C$:

\begin{enumerate}
\item \emph{Segmentation.} Partition each session of $C$ into topic segments using \textsc{Hi-OnTop} (§3.2). $\delta^{*}_{p_x}$ is calibrated on the in-domain dialogues themselves via Eq.~\ref{eq:px}; because calibration consumes no boundary labels, sharing dialogues between calibration and downstream evaluation introduces no QA-side leakage.
\item \emph{Memory compression.} Each segment is compressed by the pipeline's original compression module.
\item \emph{Retrieval.} Compressed segments are encoded, indexed, and retrieved by cosine similarity at query time.
\item \emph{Generation.} The retrieved segment(s) and the question are passed to the answer generator.
\end{enumerate}
Only step~1 is replaced; the remaining stages preserve the host pipeline's behavior so that any change in downstream performance can be attributed to the segmentation module alone.

\textbf{Encoder as a latency–quality knob.} Because $\delta^{*}$ is recalibrated per encoder via Eq.~\ref{eq:px}, the encoder used by \textsc{Hi-OnTop} is a pure latency–quality knob, not an algorithmic change: smaller or quantized encoders can be substituted to reduce per-turn segmentation latency without modifying any other component of the segmenter, while larger encoders may be used when quality is the dominant constraint. The empirical trade-off and a comparison against an LLM-based segmenter in the same host pipeline are reported in §4.5.
```
