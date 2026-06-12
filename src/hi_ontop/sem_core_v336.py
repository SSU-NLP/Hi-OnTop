"""Online MAP segmenter — v3.3.5 + SEM2-faithful event dynamics (v3.3.6).

v3.3.5 fixed the *untrained* side (a 1-turn topic is no longer punished by
a centroid PE; it ties the fresh slot at ``L0`` and survives via ``+λ``).
But the idx=374 trace showed a residual hard **3-turn ceiling**: the moment
``transition_count`` hit 1 the topic switched from the centroid predictor
to a shared, single-pair-3-step EventRNN whose prediction was near-random
(cos ≈ 0), so every "trained" topic's repeat likelihood went catastrophic
and it died after one f0-restart rescue turn.

codex (2026-05-17) diagnosed the dominant cause as the EventRNN training
itself (under-trained + cross-topic weight interference), not merely the
early switch, and recommended a SEM2-faithful restoration split off as a
new version. v3.3.6 keeps every v3.3.5 term (``f_is_trained`` gating,
per-topic variance likelihood, f0/restart) and only swaps the topic
dynamics for :class:`hi_ontop.topic_v336.TopicV336`:

* per-topic model/optimizer (no shared-weight interference);
* untrained predictor = identity/persistence (last observed embedding),
  *not* a random RNN (SEM2 ``predict_next`` cold-start);
* full-history replay training over ``n_epochs`` (SEM2 ``estimate``);
* ``rnn_ready`` is a **separate** gate from ``f_is_trained``: a trained
  topic uses persistence until it has accumulated
  ``rnn_ready_min_transitions`` within-event transitions, only then the
  GRU prediction.

Likelihood spec is unchanged from v3.3.5; the only change is *what
``prediction_error`` returns* once a topic is trained.
"""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v336 import TopicV336


class HiOnTopSegmenterV336:
    """v3.3.5 likelihood + SEM2-faithful persistence/replay event model."""

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
        rnn_train_steps: int = 3,  # legacy CLI compat — unused (replay uses n_epochs)
        rnn_max_context: int = 8,  # legacy CLI compat — unused
        rnn_min_history: int = 2,  # legacy CLI compat — superseded by rnn_ready_*
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

        self.rnn_n_epochs = rnn_n_epochs
        self.rnn_ready_min_transitions = rnn_ready_min_transitions
        self.rnn_max_history = rnn_max_history
        self.seed = seed

        pe0 = 1.0 - cos_threshold
        self._L0 = var_likelihood_weight * self._calibrated_loglik(
            pe0, pe_var_sigma0_sq
        )

        self.topics: list[TopicV336] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        self._transition_count: list[int] = []
        self._pe_mean: list[float] = []
        self._pe_var: list[float] = []
        self._pe_var_count: list[int] = []
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

    def _rnn_ready(self, k: int) -> bool:
        """Separate from ``_is_trained``: the GRU is used only after enough
        within-event transitions; until then the topic predicts by
        persistence (SEM2 cold-start identity)."""
        return self._transition_count[k] >= self.rnn_ready_min_transitions

    def _pe(self, k: int, s: np.ndarray) -> float:
        return self.topics[k].prediction_error(s, use_rnn=self._rnn_ready(k))

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

    def _f0_dir(self, k: int) -> np.ndarray | None:
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
                log_scores[i] = log_prior + self._L0
                continue

            if self._is_trained(k_int):
                pe = self._pe(k_int, s)
                repeat_lik = self._trained_loglik(k_int, pe)
            else:
                repeat_lik = self._L0  # SEM2: untrained → fixed prior const

            repeat = log_prior + repeat_lik
            repeat_scores[i] = repeat

            if k_int == self.prev_k and self._is_trained(k_int):
                pe_repeat = self._pe(k_int, s)
                if pe_repeat > self.restart_pe_threshold:
                    base = max(
                        (self.counts[k_int] + 1.0) ** self.beta, 1e-300
                    )
                    log_prior_no_sticky = math.log(base)
                    f0_dir = self._f0_dir(k_int)
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

        if is_continuation:
            pe = float(self._pe(k, s))
            self._update_pe_stats(k, pe)
            pe_clamped = max(0.0, min(2.0, pe))
            n_prev = self._pe_run_count[k]
            n_new = n_prev + 1
            self._pe_run_mean[k] = (
                self._pe_run_mean[k] * n_prev + pe_clamped
            ) / n_new
            self._pe_run_max[k] = max(self._pe_run_max[k], pe_clamped)
            self._pe_run_count[k] = n_new

        self.topics[k].update(s)
        self.counts[k] += 1

        if is_continuation:
            self._transition_count[k] += 1

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
