"""Online MAP segmenter — v3.3.4 variance likelihood + SEM2 ``f_is_trained``
cold-start gating + v3.3.3 f0/restart branch (v3.3.5).

Motivation
----------
v3.3.4 judged *every* existing topic by its centroid/RNN cosine PE, even a
1-turn topic whose predictor has never seen a within-event transition. The
fresh-cluster slot competed at a fixed baseline ``L0``. A young topic was
therefore *penalised by a bad prediction* (centroid of a single turn) while
the fresh slot was not, so it lost the MAP argmax and died at 1 turn — a
chicken-and-egg loop (good prediction needs accumulation; accumulation needs
survival; survival needs good prediction). v3.3.4's per-topic ``σ_k²``
calibration never activated because no topic reached ``pe_var_min_samples``.

SEM2 does not have this loop. In ``sem/event_models.py`` an event model's
``log_likelihood_next`` / ``log_likelihood_f0`` return a *fixed prior
constant* while ``f_is_trained`` / ``f0_is_trained`` is ``False`` — the model
is **not** scored by a half-baked prediction. ``f_is_trained`` flips to
``True`` only after one within-event ``update(x_prev, x_curr)`` (a
transition; *not* an ``update_f0`` episode start). Consequently a brand-new
cluster and an untrained young ``k_prev`` receive the *same* likelihood
constant, so the sCRP prior alone decides between them, and ``k_prev``'s
``+λ`` stickiness keeps the young topic alive long enough to observe its
first transition and become trained.

v3.3.5 restores exactly that gating (the part Hi-OnTop dropped when it
substituted a centroid fallback), keeping v3.3.4's per-topic variance
likelihood for *trained* topics and v3.3.3's same-label f0/restart branch.

Locked spec (codex 2026-05-17)
------------------------------
* ``f_trained_k := transition_count_k >= min_transitions_for_pe`` (default 1).
  ``transition_count_k`` counts within-event continuations only
  (``k == prev_k`` and not a boundary); ``rnn_min_history`` stays a separate
  RNN-activation HP and does *not* gate this.
* ``L0 = var_likelihood_weight · _calibrated_loglik(pe0, σ0²)``,
  ``pe0 = 1 - cos_threshold``, ``σ0² = pe_var_sigma0_sq``.
* fresh slot: ``log prior_new + L0``.
* existing **untrained**: ``log prior_k + L0``  (identical constant, PE
  unused — likelihood ties with the fresh slot; prior decides).
* existing **trained**: ``log prior_k - var_w · PE_k² / (2 σ_k²)``
  (v3.3.4 likelihood; ``σ_k²`` per-topic EMA, cold-start ``σ0²``).
* ``k_prev`` repeat/restart (v3.3.3): restart considered only for a
  *trained* ``k_prev`` whose repeat PE exceeds ``restart_pe_threshold``;
  ``restart_prob`` uses the prior **without** the ``+λ`` stickiness, so an
  untrained restart (== ``L0``) can never spuriously beat repeat.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


class HiOnTopSegmenterV335:
    """v3.3.4 variance likelihood + SEM2 f_is_trained gating + f0/restart."""

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        tau: float = 50.0,  # legacy CLI compat — unused by var-calibrated lik
        cos_threshold: float = 0.9,
        beta: float = 0.25,
        pe_threshold: float = 1.0,
        k_max: int = 256,
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 3,
        rnn_max_context: int = 8,
        rnn_min_history: int = 2,
        # v3.3.4 variance likelihood
        pe_var_decay: float = 0.95,
        pe_var_min_samples: int = 5,
        pe_var_sigma0_sq: float = 0.04,
        pe_var_min_sq: float = 1e-4,
        pe_var_max_sq: float = 0.25,
        var_likelihood_weight: float = 1.0,
        hard_pe_fallback: bool = False,
        # v3.3.5 cold-start gating
        min_transitions_for_pe: int = 1,
        # v3.3.3 f0 / restart branch
        restart_pe_threshold: float = 0.5,
        restart_margin: float = 0.0,
        f0_min_starts: int = 2,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if rnn_lr <= 0:
            raise ValueError(f"rnn_lr must be > 0, got {rnn_lr}")
        if rnn_train_steps < 0:
            raise ValueError(f"rnn_train_steps must be >= 0, got {rnn_train_steps}")
        if not 0.0 <= pe_threshold <= 1.0:
            raise ValueError(f"pe_threshold must be in [0, 1], got {pe_threshold}")
        if not 0.0 < pe_var_decay < 1.0:
            raise ValueError(f"pe_var_decay must be in (0, 1), got {pe_var_decay}")
        if pe_var_min_samples < 1:
            raise ValueError(
                f"pe_var_min_samples must be >= 1, got {pe_var_min_samples}"
            )
        if not pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq:
            raise ValueError(
                "expect pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq, got "
                f"{pe_var_min_sq}, {pe_var_sigma0_sq}, {pe_var_max_sq}"
            )
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
        self.rnn_train_steps = rnn_train_steps
        self.rnn_max_context = rnn_max_context
        self.rnn_min_history = rnn_min_history

        self.pe_var_decay = pe_var_decay
        self.pe_var_rho = 1.0 - pe_var_decay
        self.pe_var_min_samples = pe_var_min_samples
        self.pe_var_sigma0_sq = pe_var_sigma0_sq
        self.pe_var_min_sq = pe_var_min_sq
        self.pe_var_max_sq = pe_var_max_sq
        self.var_likelihood_weight = var_likelihood_weight
        self.hard_pe_fallback = hard_pe_fallback

        self.min_transitions_for_pe = min_transitions_for_pe
        self.restart_pe_threshold = restart_pe_threshold
        self.restart_margin = restart_margin
        self.f0_min_starts = f0_min_starts

        # SEM2-faithful untrained likelihood constant (== fresh-slot baseline).
        pe0 = 1.0 - cos_threshold
        self._L0 = var_likelihood_weight * self._calibrated_loglik(
            pe0, pe_var_sigma0_sq
        )

        self.model = EventRNN(input_dim=dim, hidden_dim=rnn_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rnn_lr)
        self.topics: list[TopicV33RNN] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        # v3.3.5: per-topic within-event transition count (SEM2 f_is_trained).
        self._transition_count: list[int] = []

        # Per-topic PE EMA state (v3.3.4).
        self._pe_mean: list[float] = []
        self._pe_var: list[float] = []
        self._pe_var_count: list[int] = []

        # f0 episode-start centroid state (v3.3.3).
        self._f0_centroids: list[np.ndarray] = []
        self._f0_counts: list[int] = []

        # Importance v2 readouts (PE running stats + boundary counts).
        self._pe_run_mean: list[float] = []
        self._pe_run_max: list[float] = []
        self._pe_run_count: list[int] = []
        self._n_boundaries: list[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_topic(self) -> TopicV33RNN:
        return TopicV33RNN(
            topic_id=len(self.topics),
            dim=self.dim,
            hidden_dim=self.rnn_hidden_dim,
            lr=self.rnn_lr,
            train_steps=self.rnn_train_steps,
            min_rnn_history=self.rnn_min_history,
            model=self.model,
            optimizer=self.optimizer,
        )

    def _ensure_topic_slot(self, k: int) -> None:
        while len(self.topics) <= k:
            self.topics.append(self._new_topic())
            self._transition_count.append(0)
            self._pe_mean.append(0.0)
            self._pe_var.append(self.pe_var_sigma0_sq)
            self._pe_var_count.append(0)
            self._f0_centroids.append(np.zeros(self.dim, dtype=np.float64))
            self._f0_counts.append(0)
            self._pe_run_mean.append(0.0)
            self._pe_run_max.append(0.0)
            self._pe_run_count.append(0)
            self._n_boundaries.append(0)

    def _is_trained(self, k: int) -> bool:
        """SEM2 ``f_is_trained``: ≥1 within-event transition observed."""
        return self._transition_count[k] >= self.min_transitions_for_pe

    def _sigma_sq_for(self, k: int) -> float:
        if self._pe_var_count[k] < self.pe_var_min_samples:
            return self.pe_var_sigma0_sq
        return float(np.clip(self._pe_var[k], self.pe_var_min_sq, self.pe_var_max_sq))

    def _calibrated_loglik(self, pe: float, sigma_sq: float) -> float:
        return -(pe * pe) / (2.0 * sigma_sq)

    def _trained_loglik(self, k: int, pe: float) -> float:
        return self.var_likelihood_weight * self._calibrated_loglik(
            pe, self._sigma_sq_for(k)
        )

    def _f0_score_pe(self, k: int) -> float | None:
        """Cosine PE of ``s`` vs topic ``k``'s f0 centroid; ``None`` if the
        f0 centroid is not yet seeded (untrained f0 → caller uses ``L0``)."""
        if self._f0_counts[k] < self.f0_min_starts:
            return None
        ref = self._f0_centroids[k]
        n = float(np.linalg.norm(ref))
        if n <= 1e-12:
            return None
        return ref / n

    def _update_f0(self, k: int, s: np.ndarray) -> None:
        self._f0_counts[k] += 1
        delta = s - self._f0_centroids[k]
        self._f0_centroids[k] = self._f0_centroids[k] + delta / self._f0_counts[k]
        nrm = float(np.linalg.norm(self._f0_centroids[k]))
        if nrm > 1e-12:
            self._f0_centroids[k] = self._f0_centroids[k] / nrm

    def _surprise_forces_new(self, s: np.ndarray) -> bool:
        """v3.3.2-style hard PE rule. Off by default (``hard_pe_fallback``)."""
        if (
            not self.hard_pe_fallback
            or not self.topics
            or self.pe_threshold >= 1.0
        ):
            return False
        threshold = 1.0 - self.pe_threshold
        for topic in self.topics:
            if topic.cosine_score(s) >= threshold:
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
        """MAP log-scores per active slot, plus repeat/restart components
        for ``k_prev`` (``-inf`` elsewhere).

        * fresh slot / existing-untrained → ``log prior + L0`` (identical
          constant; SEM2 ``f_is_trained == False``).
        * existing-trained → ``log prior - var·PE²/(2σ_k²)`` (v3.3.4).
        * ``k_prev`` trained → ``max(repeat, restart)`` where restart uses
          the f0 centroid and the prior *without* ``+λ`` (v3.3.3).
        """
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        repeat_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)
        restart_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)

        for i, k in enumerate(active):
            k_int = int(k)
            log_prior = math.log(prior[k_int])

            if k_int >= len(self.topics):  # fresh cluster slot
                log_scores[i] = log_prior + self._L0
                continue

            if self._is_trained(k_int):
                pe = self.topics[k_int].prediction_error(s)
                repeat_lik = self._trained_loglik(k_int, pe)
            else:
                # SEM2: untrained model → fixed prior constant, no PE.
                repeat_lik = self._L0

            repeat = log_prior + repeat_lik
            repeat_scores[i] = repeat

            if k_int == self.prev_k and self._is_trained(k_int):
                pe_repeat = self.topics[k_int].prediction_error(s)
                if pe_repeat > self.restart_pe_threshold:
                    # restart: same label, new episode. Prior WITHOUT +λ.
                    base = max(
                        (self.counts[k_int] + 1.0) ** self.beta, 1e-300
                    )
                    log_prior_no_sticky = math.log(base)
                    f0_dir = self._f0_score_pe(k_int)
                    if f0_dir is None:
                        restart_lik = self._L0
                    else:
                        pe_f0 = 1.0 - float(np.dot(f0_dir, s))
                        restart_lik = self._trained_loglik(k_int, pe_f0)
                    restart = log_prior_no_sticky + restart_lik
                    restart_scores[i] = restart
                    log_scores[i] = max(repeat, restart)
                else:
                    log_scores[i] = repeat
            else:
                log_scores[i] = repeat

        return log_scores, repeat_scores, restart_scores

    # ------------------------------------------------------------------
    # PE variance EMA (v3.3.4)
    # ------------------------------------------------------------------

    def _update_pe_stats(self, k: int, pe: float) -> None:
        rho = self.pe_var_rho
        old_mean = self._pe_mean[k]
        new_mean = (1.0 - rho) * old_mean + rho * pe
        new_var = (1.0 - rho) * self._pe_var[k] + rho * (pe - old_mean) * (
            pe - new_mean
        )
        self._pe_mean[k] = new_mean
        self._pe_var[k] = float(
            np.clip(new_var, self.pe_var_min_sq, self.pe_var_max_sq)
        )
        self._pe_var_count[k] += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
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

        # Same-label restart? (k_prev chosen and restart beat repeat.)
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

        # Realised PE for σ_k² EMA + importance readouts — only on a genuine
        # within-event continuation (PE is meaningless for an episode start).
        if is_continuation:
            pe = float(self.topics[k].prediction_error(s))
            self._update_pe_stats(k, pe)
            pe_clamped = max(0.0, min(2.0, pe))
            n_prev = self._pe_run_count[k]
            n_new = n_prev + 1
            self._pe_run_mean[k] = (
                self._pe_run_mean[k] * n_prev + pe_clamped
            ) / n_new
            self._pe_run_max[k] = max(self._pe_run_max[k], pe_clamped)
            self._pe_run_count[k] = n_new

        # Update topic state (RNN + centroid), then counts.
        self.topics[k].update(s)
        self.counts[k] += 1

        if is_continuation:
            # SEM2: f_is_trained flips after the first within-event update().
            self._transition_count[k] += 1

        # f0 episode-start: brand-new topic's first turn, or any boundary.
        if is_boundary or self.counts[k] == 1:
            self._update_f0(k, s)
            self._n_boundaries[k] += 1

        self.prev_k = k
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
    # Importance v2 readouts (optional; read via hasattr in RoundProcessor)
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
