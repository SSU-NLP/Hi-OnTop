"""Online MAP segmenter — v3.3.7 + SEM2-calibrated fresh baseline (v3.3.8).

Empirically verifying v3.3.7 on idx=374 *falsified* the "variance-gate is
the dominant cause" diagnosis: ``map_variance`` from n≥2 actually pulled
σ² *below* the prior mode for the low-spread 2-sample case (0.04→0.028),
so #14|#15 still split and #12+#13 stayed merged — boundary placement
unchanged from v3.3.6.

codex (2026-05-17, re-diagnosis) confirmed the invariant root cause that
had been deferred twice as "blunt HP": the **fresh / untrained baseline
calibration**. v3.3.x scored a brand-new (and any untrained) topic at
``L0 = -(1-cos_threshold)²/(2σ0²) = -0.125`` — i.e. it *assumes a fresh
topic predicts the incoming turn at cos 0.9*. No real conversational
topic predicts the next (sub-question-shifted) turn at cos ≳0.85, so the
moment a topic is trained and judged by PE it loses to this over-lenient
fresh slot. That is a structural SEM2 deviation, not an HP value.

SEM2 (``SEM/sem/event_models.py``) does **not** assume a good fresh
prediction: an untrained ``log_likelihood_next`` / ``log_likelihood_f0``
returns the *prior predictive* of the observation — its probability under
no learned model. A literal d-dim Gaussian port would clash with Hi-OnTop's
scalar cosine-PE likelihood scale. v3.3.8 instead translates the SEM2
*principle* into the scalar-PE world: an untrained model explains the
turn no better than **chance**, and for unit-norm embeddings a chance
(uninformative) prediction has ``E[cos]≈0`` ⇒ ``PE≈1``. So:

* fresh / untrained baseline:
  ``L0 = var_w · _calibrated_loglik(pe_prior, σ0²)`` with
  ``pe_prior = 1.0`` (chance, cos≈0) — replacing the ``cos_threshold``
  assumption. ``cos_threshold`` no longer feeds ``L0``.
* SEM2 ``k0 != k_prev`` path: a non-previous existing topic is scored by
  its **f0** likelihood ("does this look like the *start* of that
  topic"), not the within-event repeat predictor (SEM2 ``sem.py`` uses
  ``log_likelihood_f0`` for non-current clusters). f0 uses the topic's
  own episode-start centroid (importance-free, retrieval-rule clean).

Everything else (v3.3.7 ``map_variance`` σ², v3.3.6 persistence+replay
dynamics + seed, v3.3.5 ``f_is_trained`` gating, f0/restart) is unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v336 import TopicV336


class HiOnTopSegmenterV411:
    """v3.3.9 + causal-window δ (codex 2026-05-18 (A); identity dynamics).

    v3.3.9 used δ_prev = 1-cos(s_{t-1}, s_t) — the SINGLE previous
    utterance. The TIAGE failure dump showed the dominant error is FP
    over-segmentation (FP 393 vs FN 117): within one topic, chit-chat
    varies turn-to-turn so δ_prev spikes on same-topic turns → spurious
    boundaries. A *causal window* context vector averages out that
    single-turn noise:

        c_{t-1} = normalize( Σ_{i=1..m} ρ^{i-1} · s_{t-i} )   # past-only
        δ_ctx   = 1 - cos(c_{t-1}, s_t)
        δ_adj   = a·δ_prev + (1-a)·δ_ctx          # blend (a=ctx_blend_a)
        δ_eff²  = η·δ_adj² + (1-η)·δ_model²        # η=1 default (identity)

    "identity" = η=1 (RNN/learned-f OFF — η-ablation 2026-05-18 showed
    every η<1 monotonically worse; A3). The RNN code path is retained
    (δ_model) but default-off; permanent removal deferred to a
    model-calibrated fair ablation.

    SEM inheritance: SEM's f is history-conditioned f(x_{1:n}); v3.3.9
    collapsed it to 1-step identity. causal-window is a closer
    approximation to that history-conditioned cold-start dynamics — not
    a new external mechanism (codex 3-step Step 1). δ* MUST be
    re-estimated on TIAGE *train* per (m, ρ, a) — no test leakage.

    causal = past turns only (online segmentation, no lookahead) —
    distinct from bidirectional TextTiling.

    Does NOT fix FN where topic changes but wording stays similar
    (δ≈0.13 dump cases) — structural signal-absence limit; expected
    realistic F1 ceiling ≈ 0.48~0.55 (codex), not 0.65.

    ─── inherited v3.3.9 docstring ───
    v3.3.8 + adjacent-utterance PE restoration (codex 2026-05-17 (A)).

    Root cause of the SEM-vs-prev-cos collapse (codex diagnosis): the
    fresh/untrained baseline ``L0`` and the scalar cosine-PE σ² scale were
    mis-calibrated, so a 1-line ``cos(s_{t-1},s_t) < θ`` baseline (F1≈0.44)
    beat every SEM variant (F1 0.25~0.36) on identical encoder + eval.

    ``prev-cos`` is *not* a non-SEM heuristic — it is exactly the cosine
    PE of SEM2's cold-start identity dynamics (``f(x)=x_prev``). v3.3.9
    restores it coherently inside the SEM likelihood:

    * **δ_eff** for the within-event (``k==prev_k``) repeat predictor::

        δ_prev   = 1 - cos(s_{t-1}, s_t)        # SEM2 identity-dynamics PE
        δ_model  = topic.prediction_error(s)    # learned/persistence f PE
        δ_eff²   = η·δ_prev² + (1-η)·δ_model²

      ``η=eta_prev`` (default 1.0 for TIAGE short dialogues — identity
      cold-start dominant; learned f mixed in only when proven better).
    * **calibrated fresh baseline** ``B0 = -δ*²/(2σδ²) + log r0`` —
      replaces both v3.3.5's ``cos_threshold`` L0 (over-seg) and v3.3.8's
      ``pe_prior=1`` L0 (collapse). ``δ*`` = prev-cos best boundary
      distance, ``r0`` = typical sCRP prior ratio (≈ λ/α).
    * **σδ²** is calibrated online from the observed δ_prev distribution
      (fixed 0.04 forbidden — codex), defaulting until enough samples.

    RNN/learned f is retained (SEM2 inheritance) but only mixed via η
    when it demonstrably beats persistence — not removed.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        tau: float = 50.0,  # legacy CLI compat — unused
        cos_threshold: float = 0.9,
        beta: float = 0.25,
        pe_threshold: float = 1.0,
        k_max: int = 256,
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 3,  # legacy CLI compat — unused
        rnn_max_context: int = 8,  # legacy CLI compat — unused
        rnn_min_history: int = 2,  # legacy CLI compat — superseded
        # variance prior (SEM2 scaled-inv-χ²)
        pe_var_sigma0_sq: float = 0.04,  # prior MODE of the PE variance
        pe_var_df0: float = 1.0,  # ν₀ — SEM2 var_df0 default
        pe_var_min_sq: float = 1e-4,
        pe_var_max_sq: float = 0.25,
        pe_var_window: int = 256,  # cap on retained PE samples / topic
        pe_var_min_samples: int = 2,  # legacy CLI compat — fixed at 2 (SEM2)
        pe_var_decay: float = 0.95,  # legacy CLI compat — unused
        # v3.3.8: SEM2-calibrated fresh/untrained baseline. pe_prior is the
        # chance-level PE of an uninformative prediction (unit embeddings:
        # E[cos]≈0 ⇒ PE≈1). Replaces the cos_threshold assumption in L0.
        pe_prior: float = 1.0,
        # v3.3.9 — adjacent-utterance PE restoration (codex 2026-05-17/18 (A))
        use_rnn: bool = True,  # False → skip RNN compute/train (optional
        # off, NOT removal). Auto-skipped when η==1 (δ_model ×0 anyway).
        eta_prev: float = 1.0,  # η: weight on δ_prev² vs δ_model² in repeat
        delta_star: float = 0.5557,  # δ* from TIAGE *train* prev-cos (no leak;
        # train F1 0.437 @ cos_th 0.4443, n_conv=300 — test never touched)
        sigma_delta_c: float = 0.0625,  # σδ² = c·δ*² (codex: temperature, 1/16)
        sigma_delta_sq: float | None = None,  # explicit override of c·δ*²
        # v4.1.1 — causal-window δ (codex 2026-05-18 (A))
        ctx_window: int = 3,  # m: # past utterances in context vector
        ctx_decay: float = 0.7,  # ρ: geometric decay (recent weighted more)
        ctx_blend_a: float = 0.5,  # a: δ_adj = a·δ_prev + (1-a)·δ_ctx
        # NOTE δ* default below is the v3.3.9 prev-cos value — a PLACEHOLDER.
        # It MUST be re-estimated on TIAGE train for each (m,ρ,a) before
        # any test eval (scripts/calibrate_v411_delta_star.py).
        var_likelihood_weight: float = 1.0,
        hard_pe_fallback: bool = False,
        # v3.3.5 cold-start gating
        min_transitions_for_pe: int = 1,
        # v3.3.3 f0 / restart branch
        restart_pe_threshold: float = 0.5,
        restart_margin: float = 0.0,
        f0_min_starts: int = 2,
        # v3.3.6 SEM2-faithful event dynamics
        rnn_n_epochs: int = 10,
        rnn_ready_min_transitions: int = 3,
        rnn_max_history: int = 64,
        seed: int = 0,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if rnn_lr <= 0:
            raise ValueError(f"rnn_lr must be > 0, got {rnn_lr}")
        if not 0.0 <= pe_threshold <= 1.0:
            raise ValueError(f"pe_threshold must be in [0, 1], got {pe_threshold}")
        if pe_var_df0 <= 0:
            raise ValueError(f"pe_var_df0 must be > 0, got {pe_var_df0}")
        if not pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq:
            raise ValueError(
                "expect pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq, got "
                f"{pe_var_min_sq}, {pe_var_sigma0_sq}, {pe_var_max_sq}"
            )
        if pe_var_window < 2:
            raise ValueError(f"pe_var_window must be >= 2, got {pe_var_window}")
        if var_likelihood_weight <= 0:
            raise ValueError(
                f"var_likelihood_weight must be > 0, got {var_likelihood_weight}"
            )
        if min_transitions_for_pe < 1:
            raise ValueError(
                f"min_transitions_for_pe must be >= 1, got {min_transitions_for_pe}"
            )
        if not 0.0 <= restart_pe_threshold <= 2.0:
            raise ValueError(
                f"restart_pe_threshold must be in [0, 2], got {restart_pe_threshold}"
            )
        if f0_min_starts < 1:
            raise ValueError(f"f0_min_starts must be >= 1, got {f0_min_starts}")
        if rnn_n_epochs < 1:
            raise ValueError(f"rnn_n_epochs must be >= 1, got {rnn_n_epochs}")
        if rnn_ready_min_transitions < 1:
            raise ValueError(
                "rnn_ready_min_transitions must be >= 1, got "
                f"{rnn_ready_min_transitions}"
            )
        if rnn_max_history < 2:
            raise ValueError(f"rnn_max_history must be >= 2, got {rnn_max_history}")

        self.dim = dim
        self.alpha = alpha
        self.lmda = lmda
        self.tau = tau
        self.cos_threshold = cos_threshold
        self.beta = beta
        self.pe_threshold = pe_threshold
        self.k_max = k_max
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_lr = rnn_lr

        # SEM2 scaled-inverse-χ² prior: pick var0 so the prior mode
        # (n=0) equals pe_var_sigma0_sq  →  var0 = mode·(ν0+2)/ν0.
        self.pe_var_sigma0_sq = pe_var_sigma0_sq
        self.pe_var_df0 = pe_var_df0
        self._var0 = pe_var_sigma0_sq * (pe_var_df0 + 2.0) / pe_var_df0
        self.pe_var_min_sq = pe_var_min_sq
        self.pe_var_max_sq = pe_var_max_sq
        self.pe_var_window = pe_var_window
        self.var_likelihood_weight = var_likelihood_weight
        self.hard_pe_fallback = hard_pe_fallback

        self.min_transitions_for_pe = min_transitions_for_pe
        self.restart_pe_threshold = restart_pe_threshold
        self.restart_margin = restart_margin
        self.f0_min_starts = f0_min_starts

        self.rnn_n_epochs = rnn_n_epochs
        self.rnn_ready_min_transitions = rnn_ready_min_transitions
        self.rnn_max_history = rnn_max_history
        self.seed = seed

        if not 0.0 <= pe_prior <= 2.0:
            raise ValueError(f"pe_prior must be in [0, 2], got {pe_prior}")
        self.pe_prior = pe_prior

        # v3.3.9 calibrated fresh baseline + δ_eff state.
        if not 0.0 <= eta_prev <= 1.0:
            raise ValueError(f"eta_prev must be in [0, 1], got {eta_prev}")
        if not 0.0 <= delta_star <= 2.0:
            raise ValueError(f"delta_star must be in [0, 2], got {delta_star}")
        if sigma_delta_c <= 0.0:
            raise ValueError(f"sigma_delta_c must be > 0, got {sigma_delta_c}")
        self.use_rnn = use_rnn
        self.eta_prev = eta_prev
        self.delta_star = delta_star
        # codex 2026-05-18: σδ² is a FIXED softness temperature, NOT the
        # δ_prev population variance (which mixes in boundary jumps and
        # washes out discrimination → v3.3.9-pre over-merge). σδ² = c·δ*².
        self.sigma_delta_c = sigma_delta_c
        self._sigma_delta_sq = (
            sigma_delta_sq if sigma_delta_sq is not None
            else sigma_delta_c * delta_star * delta_star
        )
        if self._sigma_delta_sq <= 0.0:
            raise ValueError("σδ² must be > 0")
        if ctx_window < 1:
            raise ValueError(f"ctx_window must be >= 1, got {ctx_window}")
        if not 0.0 < ctx_decay <= 1.0:
            raise ValueError(f"ctx_decay must be in (0,1], got {ctx_decay}")
        if not 0.0 <= ctx_blend_a <= 1.0:
            raise ValueError(f"ctx_blend_a must be in [0,1], got {ctx_blend_a}")
        self.ctx_window = ctx_window
        self.ctx_decay = ctx_decay
        self.ctx_blend_a = ctx_blend_a
        self._recent: list[np.ndarray] = []  # last turns' s (causal window)
        self._prev_s: np.ndarray | None = None  # s_{t-1} (adjacent utterance)
        # Fixed prior-free baseline (r0=1) for the f0 / non-prev paths;
        # the prev-vs-fresh decision uses _fresh_baseline_for_prev (which
        # is prior-corrected → exactly equivalent to δ_eff < δ*).
        self._L0 = self._delta_loglik(delta_star)

        self.topics: list[TopicV336] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        self._transition_count: list[int] = []
        # SEM2-style: retained scalar PE samples per topic (the
        # ``prediction_errors`` buffer, capped like ``variance_window``).
        self._pe_samples: list[list[float]] = []
        self._f0_centroids: list[np.ndarray] = []
        self._f0_counts: list[int] = []
        self._pe_run_mean: list[float] = []
        self._pe_run_max: list[float] = []
        self._pe_run_count: list[int] = []
        self._n_boundaries: list[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_topic(self) -> TopicV336:
        return TopicV336(
            topic_id=len(self.topics),
            dim=self.dim,
            hidden_dim=self.rnn_hidden_dim,
            lr=self.rnn_lr,
            n_epochs=self.rnn_n_epochs,
            max_history=self.rnn_max_history,
            seed=self.seed,
        )

    def _ensure_topic_slot(self, k: int) -> None:
        while len(self.topics) <= k:
            self.topics.append(self._new_topic())
            self._transition_count.append(0)
            self._pe_samples.append([])
            self._f0_centroids.append(np.zeros(self.dim, dtype=np.float64))
            self._f0_counts.append(0)
            self._pe_run_mean.append(0.0)
            self._pe_run_max.append(0.0)
            self._pe_run_count.append(0)
            self._n_boundaries.append(0)

    def _is_trained(self, k: int) -> bool:
        return self._transition_count[k] >= self.min_transitions_for_pe

    def _rnn_ready(self, k: int) -> bool:
        return self._transition_count[k] >= self.rnn_ready_min_transitions

    def _pe(self, k: int, s: np.ndarray) -> float:
        return self.topics[k].prediction_error(s, use_rnn=self._rnn_ready(k))

    def _sigma_sq_for(self, k: int) -> float:
        """SEM2 ``map_variance`` posterior mode of the PE variance.

        ``σ² = (ν₀·var₀ + n·v) / (ν₀ + n + 2)`` for ``n ≥ 2`` samples
        (``v`` = population variance of the topic's PE samples); the
        scaled-inverse-χ² prior alone (mode == ``pe_var_sigma0_sq``)
        for ``n < 2``. Clipped to ``[min_sq, max_sq]``.
        """
        samples = self._pe_samples[k]
        n = len(samples)
        if n < 2:
            sigma_sq = self.pe_var_sigma0_sq
        else:
            v = float(np.var(samples))  # population variance (SEM2 np.var)
            nu0 = self.pe_var_df0
            sigma_sq = (nu0 * self._var0 + n * v) / (nu0 + n + 2.0)
        return float(np.clip(sigma_sq, self.pe_var_min_sq, self.pe_var_max_sq))

    def _record_pe(self, k: int, pe: float) -> None:
        """Append a PE sample to topic ``k``'s buffer (capped, FIFO)."""
        buf = self._pe_samples[k]
        buf.append(float(pe))
        if len(buf) > self.pe_var_window:
            del buf[0]

    def _calibrated_loglik(self, pe: float, sigma_sq: float) -> float:
        return -(pe * pe) / (2.0 * sigma_sq)

    def _trained_loglik(self, k: int, pe: float) -> float:
        return self.var_likelihood_weight * self._calibrated_loglik(
            pe, self._sigma_sq_for(k)
        )

    # ---- v3.3.9: δ_eff + calibrated fresh baseline -------------------

    def _delta_loglik(self, delta_eff: float) -> float:
        """δ-scale log-likelihood ``var_w·(-δ²/(2σδ²))`` (σδ² fixed)."""
        return self.var_likelihood_weight * (
            -(delta_eff * delta_eff) / (2.0 * self._sigma_delta_sq)
        )

    def _l0(self) -> float:
        """Prior-free baseline ``Lδ(δ*)`` (r0=1) — used only by the f0 /
        non-prev re-entry paths. The prev-vs-fresh decision does NOT use
        this; it uses :meth:`_fresh_baseline_for_prev` (prior-corrected)."""
        return self._delta_loglik(self.delta_star)

    def _fresh_baseline_for_prev(
        self, prior_prev: float, prior_new: float
    ) -> float:
        """codex 2026-05-18: time-dependent, prior-corrected fresh
        baseline. With ``_scores`` adding ``log prior_new`` to the fresh
        slot, ``B0_t = Lδ(δ*) + log(prior_prev/prior_new)`` makes the
        repeat-vs-fresh MAP decision EXACTLY ``δ_eff < δ*`` at every turn
        (prior terms cancel) — fixing the v3.3.9-pre prior mismatch that
        over-merged as ``prior_prev/α`` grew past a fixed ``r0``."""
        r_t = max(prior_prev, 1e-300) / max(prior_new, 1e-300)
        return self._delta_loglik(self.delta_star) + math.log(r_t)

    def _delta_prev(self, s: np.ndarray) -> float | None:
        """δ_prev = 1 - cos(s_{t-1}, s) — SEM2 identity-dynamics PE.
        None before the first turn."""
        if self._prev_s is None:
            return None
        a, b = self._prev_s, s
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return None
        return 1.0 - float(np.dot(a, b) / (na * nb))

    def _delta_ctx(self, s: np.ndarray) -> float | None:
        """v4.1.1: δ_ctx = 1 - cos(c_{t-1}, s), where
        c_{t-1} = normalize(Σ_{i=1..m} ρ^{i-1} s_{t-i}) over the causal
        window (most-recent weighted most). None before any past turn."""
        if not self._recent:
            return None
        win = self._recent[-self.ctx_window:]  # past only, ≤ m
        # win[-1] = s_{t-1} (i=1, weight ρ^0), win[-2] = s_{t-2}, ...
        c = np.zeros_like(win[-1], dtype=np.float64)
        for i, vec in enumerate(reversed(win)):  # i=0 → s_{t-1}
            c += (self.ctx_decay ** i) * vec
        nc = float(np.linalg.norm(c))
        ns = float(np.linalg.norm(s))
        if nc <= 1e-12 or ns <= 1e-12:
            return None
        return 1.0 - float(np.dot(c, s) / (nc * ns))

    def _uses_rnn(self) -> bool:
        """RNN consumed only if explicitly enabled AND it contributes
        (η<1; at η=1 δ_model weight is exactly 0)."""
        return self.use_rnn and self.eta_prev < 1.0

    def _delta_eff(self, k: int, s: np.ndarray) -> float:
        """v4.1.1: δ_adj = a·δ_prev + (1-a)·δ_ctx (causal-window blend),
        then δ_eff² = η·δ_adj² + (1-η)·δ_model². RNN (δ_model) skipped
        when not _uses_rnn (η=1 or use_rnn=False) → δ_eff = δ_adj,
        identical result, no GRU forward. Falls back before s_{t-1} /
        context exist (stream start)."""
        d_prev = self._delta_prev(s)
        if d_prev is None:
            return (float(self._pe(k, s)) if self._uses_rnn() else 0.0)
        d_ctx = self._delta_ctx(s)
        a = self.ctx_blend_a
        d_adj = d_prev if d_ctx is None else a * d_prev + (1.0 - a) * d_ctx
        if not self._uses_rnn():
            return d_adj
        d_model = float(self._pe(k, s))
        eta = self.eta_prev
        return math.sqrt(
            eta * d_adj * d_adj + (1.0 - eta) * d_model * d_model
        )


    def _f0_dir(self, k: int) -> np.ndarray | None:
        if self._f0_counts[k] < self.f0_min_starts:
            return None
        ref = self._f0_centroids[k]
        n = float(np.linalg.norm(ref))
        if n <= 1e-12:
            return None
        return ref / n

    def _f0_loglik(self, k: int, s: np.ndarray) -> float:
        """SEM2 ``log_likelihood_f0``: probability that ``s`` is the
        *start* of (re-entering) topic ``k``. Untrained f0 (centroid not
        yet seeded) → the chance prior constant ``L0`` (SEM2:
        ``f0_is_trained == False`` → prior probability)."""
        f0_dir = self._f0_dir(k)
        if f0_dir is None:
            return self._l0()
        pe_f0 = 1.0 - float(np.dot(f0_dir, s))
        return self._trained_loglik(k, pe_f0)

    def _update_f0(self, k: int, s: np.ndarray) -> None:
        self._f0_counts[k] += 1
        delta = s - self._f0_centroids[k]
        self._f0_centroids[k] = self._f0_centroids[k] + delta / self._f0_counts[k]
        nrm = float(np.linalg.norm(self._f0_centroids[k]))
        if nrm > 1e-12:
            self._f0_centroids[k] = self._f0_centroids[k] / nrm

    def _surprise_forces_new(self, s: np.ndarray) -> bool:
        if (
            not self.hard_pe_fallback
            or not self.topics
            or self.pe_threshold >= 1.0
        ):
            return False
        threshold = 1.0 - self.pe_threshold
        for i, topic in enumerate(self.topics):
            if topic.cosine_score(s, use_rnn=self._rnn_ready(i)) >= threshold:
                return False
        return True

    def _new_cluster_index_in_active(self, active: np.ndarray) -> int | None:
        for i, k in enumerate(active):
            if int(k) >= len(self.topics):
                return i
        return None

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _scores(
        self, s: np.ndarray, prior: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        repeat_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)
        restart_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)

        for i, k in enumerate(active):
            k_int = int(k)
            log_prior = math.log(prior[k_int])

            if k_int >= len(self.topics):  # fresh cluster slot
                # codex 2026-05-18: prior-corrected fresh baseline. With
                # log_prior(=log P_new) added here, total fresh score
                # = log P_prev + Lδ(δ*) → repeat>fresh ⟺ δ_eff<δ*
                # exactly, every turn (prior cancels). Pre-prev fallback
                # (no prev_k) keeps the prior-free r0=1 form.
                if self.prev_k is not None and prior[self.prev_k] > 0.0:
                    b0 = self._fresh_baseline_for_prev(
                        prior[self.prev_k], prior[k_int]
                    )
                else:
                    b0 = self._l0()
                log_scores[i] = log_prior + b0
                continue

            if k_int == self.prev_k:
                # Current event repeat. v3.3.9: δ_eff (η·δ_prev² +
                # (1-η)·δ_model²) for trained AND untrained — η=1 ⇒ pure
                # δ_prev (SEM2 identity dynamics), so untrained is well
                # defined and no longer collapses to the prior constant
                # (which forced ≥2-turn merge bias). codex 2026-05-18.
                repeat_lik = self._delta_loglik(self._delta_eff(k_int, s))
                repeat = log_prior + repeat_lik
                repeat_scores[i] = repeat

                if self._is_trained(k_int) and (
                    self._delta_eff(k_int, s) > self.restart_pe_threshold
                ):
                    # SEM2 same-label restart: log_likelihood_f0, prior
                    # without the +λ stickiness.
                    base = max(
                        (self.counts[k_int] + 1.0) ** self.beta, 1e-300
                    )
                    log_prior_no_sticky = math.log(base)
                    restart = log_prior_no_sticky + self._f0_loglik(k_int, s)
                    restart_scores[i] = restart
                    log_scores[i] = max(repeat, restart)
                else:
                    log_scores[i] = repeat
            else:
                # SEM2 ``k0 != k_prev``: a non-current cluster is scored by
                # log_likelihood_f0 — "does this look like the *start* of
                # (re-entering) topic k", NOT the within-event repeat
                # predictor. f0 uses topic k's own episode-start centroid
                # (importance-free; retrieval-rule clean).
                f0_score = log_prior + self._f0_loglik(k_int, s)
                log_scores[i] = f0_score
                repeat_scores[i] = f0_score

        return log_scores, repeat_scores, restart_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        # v3.3.9 (codex 2026-05-18): σδ² is a FIXED softness temperature
        # (c·δ*²), not online-calibrated — online δ_prev variance mixed
        # boundary jumps and washed out discrimination (→ over-merge).
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, repeat_scores, restart_scores = self._scores(s, prior, active)

        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            chosen_idx = (
                new_idx if new_idx is not None else int(np.argmax(log_scores))
            )
        else:
            chosen_idx = int(np.argmax(log_scores))
        k = int(active[chosen_idx])

        is_restart = False
        if (
            self.prev_k is not None
            and k == self.prev_k
            and not math.isinf(restart_scores[chosen_idx])
            and restart_scores[chosen_idx]
            > repeat_scores[chosen_idx] + self.restart_margin
        ):
            is_restart = True

        self._ensure_topic_slot(k)

        is_label_change = self.prev_k is not None and k != self.prev_k
        is_boundary = is_label_change or is_restart
        is_continuation = (
            self.prev_k is not None and k == self.prev_k and not is_restart
        )

        # σ² sample buffer: feed on continuation AND same-label restart
        # (codex processing C — a restart is still a PE observation of
        # topic k, so σ² may progress even though transition_count, the
        # f_is_trained gate, is frozen by the restart).
        if is_continuation or is_restart:
            pe = float(self._pe(k, s))
            self._record_pe(k, pe)
            if is_continuation:
                pe_clamped = max(0.0, min(2.0, pe))
                n_prev = self._pe_run_count[k]
                n_new = n_prev + 1
                self._pe_run_mean[k] = (
                    self._pe_run_mean[k] * n_prev + pe_clamped
                ) / n_new
                self._pe_run_max[k] = max(self._pe_run_max[k], pe_clamped)
                self._pe_run_count[k] = n_new

        self.topics[k].update(s, train_rnn=self._uses_rnn())
        self.counts[k] += 1

        if is_continuation:
            self._transition_count[k] += 1

        if is_boundary or self.counts[k] == 1:
            self._update_f0(k, s)
            self._n_boundaries[k] += 1

        self.prev_k = k
        s_copy = np.array(s, dtype=np.float64, copy=True)
        self._prev_s = s_copy  # s_{t-1}
        self._recent.append(s_copy)  # causal window (v4.1.1)
        if len(self._recent) > self.ctx_window:
            del self._recent[0]
        return k, is_boundary

    def predict_topic(self, s: np.ndarray) -> int:
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, _, _ = self._scores(s, prior, active)
        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            if new_idx is not None:
                return int(active[new_idx])
        return int(active[int(np.argmax(log_scores))])

    # ------------------------------------------------------------------
    # Importance v2 readouts
    # ------------------------------------------------------------------

    def pe_stats(self) -> dict[int, dict[str, float]]:
        return {
            k: {
                "mean": float(self._pe_run_mean[k]),
                "max": float(self._pe_run_max[k]),
                "count": int(self._pe_run_count[k]),
            }
            for k in range(len(self.topics))
            if self._pe_run_count[k] > 0
        }

    def boundary_counts(self) -> dict[int, int]:
        return {
            k: int(self._n_boundaries[k])
            for k in range(len(self.topics))
            if self.counts[k] > 0
        }
