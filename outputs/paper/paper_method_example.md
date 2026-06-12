```latex
% long-horizon agent에서 발생하는 Escalation prior에 대해 어디서 오는지 추적하고 deployment단계에서 완화시킬수 있는 방안을 제안한다.
\section{Methodology}
We propose three complementary measurements that together (i) quantify the fraction of premature escalations—where the agent could have succeeded independently without the escape hatch—under a same-model action-space ablation, (ii) localize where in a four-stage post-training pipeline the preventable-quitting prior accrues, and (iii) test whether deployment-time prompts can move that prior end-to-end. 

% react style equation 설명
\subsection{Preliminary}
\label{sec:3.1}
We use a ReAct-style agent \citep{yao2022react} augmented with a single additional terminal action `Escalate[reason]`. At step $t$, the agent receives a prompt
$$x_t = (\mathrm{Header},\ q,\ o_1, \ldots, o_{t-1}),$$
where $\mathrm{Header}$ bundles a fixed task instruction and a few-shot in-context demonstration of Thought–Action–Observation cycles in the ReAct format (Detailed prompt templates are provided in Appendix ~\ref{sec:appendix A}), $q$ is the task query, and $o_{1:t-1}$ is the prior observation history. The policy $\pi_\theta$ emits a Thought $\tau_t$ followed by an action token $a_t \in \mathcal{A} \cup \{\textsc{Esc}\}$, where $\mathcal{A}$ is the domain action vocabulary and $\textsc{Esc} = $ `Escalate[reason]`. We write $\pi_\theta(a_t \mid x_t, \tau_t)$ for the next-token distribution at the action slot, after the Thought has been committed but before any action token is sampled, and define
$$P_\theta(\textsc{Esc} \mid x_t, \tau_t) = \sum_{v \in \mathcal{V}_{\textsc{Esc}}} \exp(\ell_v),$$
where $\mathcal{V}_{\textsc{Esc}}$ is the set of token prefixes that begin an escalation action under the agent's tokenization and $\ell_v$ are top-$k$ logprobs.

A trajectory terminates in exactly one of
$$\sigma \in \{\textsc{success},\ \textsc{escalated},\ \textsc{failed}\},$$
where $\textsc{success}$ is defined by a per-domain verifier and $\textsc{escalated}$ is reachable only when \mbox{$\text{Esc} \in \mathcal{A} \cup \{\text{Esc}\}$}.

For Section~\ref{sec:3.3} and \ref{sec:3.4}, "stage" indexes the four checkpoints of the Tülu 3 \citep{lambert2024tulu} post-training pipeline — base, supervised fine-tuning (SFT), Direct Preference Optimization (DPO) \citep{rafailov2023direct}, and Reinforcement Learning with Verifiable Rewards (RLVR) \citep{shao2024deepseekmath, wen2025reinforcement} — applied to a single shared pretraining trunk,
$$s \in \mathcal{S} = \{\text{base},\ \text{SFT},\ \text{DPO},\ \text{RLVR}\},$$
so differences in $\pi_\theta$ across $s$ are attributable to post-training only. We apply a uniform raw-text inference encoding across all stages — the base checkpoint has no built-in chat template, and a uniform encoding is the only way to remove encoding as a stage-level confound.

% 제목 수정 필요
% agent가 escalation 할 수 있는 권한이 주어 졌을때 실제로 조기(premature) escalation 하는것을 평가하는 방법. 
\subsection{Premature Escalation via Same-Model Counterfactual} \label{sec:3.2}
To rigorously quantify the utility of the escalation mechanism, we evaluate a fixed agent $\pi$ across a diverse task suite $\mathcal{T}$ under two comparative configurations:
\begin{itemize}
    \item \textbf{Escalation-Augmented ($\mathcal{M}_{\text{esc}}$):} The action space is expanded to include the explicit escalation option, denoted as $\mathcal{A} \cup \{\textsc{Esc}\}$.
    \item \textbf{Base ($\mathcal{M}_{\text{base}}$):} A controlled baseline where the escalation option is completely ablated from both the action space and the prompt context ($\mathcal{A}$).
\end{itemize}
By isolating the escalation mechanism while keeping all other experimental variables invariant, we contrast the per-task terminal states under the two configurations to establish our core evaluation metrics. Let $\mathrm{SR}(\mathcal{M})$ denote the empirical success rate of $\pi$ over $\mathcal{T}$ under configuration $\mathcal{M}$. We report
$$\Delta_{\text{success}} = \mathrm{SR}(\mathcal{M}_{\text{base}}) - \mathrm{SR}(\mathcal{M}_{\text{esc}}),$$
$$\rho_{\text{recovery}} = \Pr[\sigma_{\text{base}}{=}\mathrm{success} \mid \sigma_{\text{esc}}{=}\mathrm{escalated}].$$

\noindent$\Delta_{\text{success}}$ is the aggregate success rate difference between the two configurations. $\rho_{\text{recovery}}$ is the conditional probability that a task escalated under $\mathcal{M}_{\text{esc}}$ is solved by the same model under $\mathcal{M}_{\text{base}}$. Together, they quantify the fraction of escalations that are premature — cases the agent could have resolved without the escape hatch.

Using a fixed pool of trajectories ending in an escalation at step $N_i$, we strictly fix the context prefix $x_{N_i}$ and replace the original thought $\tau_{N_i}$ with five counterfactual variants (Table~\ref{tab:conditions}, Top). We evaluate these conditions across the four training stages $s \in \mathcal{S}$ to strictly isolate the effects of model parameters $\theta(s)$ and thought interventions. For each (stage, trajectory, condition) tuple, we extract the marginal escalation probability $P_\theta(\textsc{Esc} \mid x_{N_i}, \tau)$ from the action slot's next-token distribution using the mapping $\mathcal{V}_{\textsc{Esc}}$. 

To understand behavioral shifts within the broader action space $\mathcal{A}$, we decompose the residual non-escalation probability mass. This reveals whether suppressed escalation is redirected toward productive actions or merely diffuses. Crucially, Condition B traces the emergence of the premature-quitting prior across the alignment pipeline ($\text{Base} \to \text{SFT} \to \text{DPO} \to \text{RLVR}$). Concurrently, Conditions C--E measure the steerability of this behavior via thought polarity, tracking the model's evolving susceptibility to internal interventions.

\subsection{End-to-End Intervention Scaling}
\label{sec:3.4}

To mitigate premature escalation during deployment, prior work has explored various strategies, including in-context learning and explicit instruction following~\citep{huang2024biastestingmitigation, kaneko2024evaluating}. As outlined in Table~\ref{tab:conditions} (Bottom), we design three intervention conditions, scaling from implicit in-context demonstrations to direct system-level instructions.

For each condition $c$, we evaluate the final deployed agent (RLVR-final checkpoint) end-to-end on a fixed task pool. Using the identical encoding setup as in Section~\ref{sec:3.3}, we compute the empirical termination rate for each outcome:
$$r_{\sigma}(c) = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[\text{traj}_i^{(c)} = \sigma\right]$$

\noindent where $r_\sigma(c)$ denotes the proportion of the $N$ trajectories evaluated under condition $c$ that conclude in state $\sigma$, with $\sigma$ spanning the four terminal states defined in Section~\ref{sec:3.1}.
```
