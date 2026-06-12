"""Online MAP segmenter — v3.3.2 + SEM2 f0 restart branch (v3.3.3).

Hi-OnTop-full-v3.3.3 keeps every term of v3.3.2 but restores SEM2's
``log_likelihood_f0`` *restart* path. SEM2 computes two likelihoods
against the previous event ``k_prev``:

    repeat:  log_likelihood_next(x_prev, x_curr)   — direct continuation
    restart: log_likelihood_f0(x_curr)             — same event label,
                                                     new episode start

The MAP score for ``k_prev`` is the maximum of the two. When the
restart branch wins (by an optional margin), the assignment stays on
the same topic but the turn is marked as a boundary — i.e. "same
label, new episode". v1 / v3.1.1 / v3.3.x prior to this version
collapsed both paths into a single repeat score, so a same-label
restart could only become a boundary by spawning a brand-new topic.

f0 centroid: episode-start embedding running mean kept per topic.
Cold-start (`f0_count < f0_min_starts`) falls back to the topic's
ordinary centroid `mu`. ``restart_pe_threshold`` guards against
restart winning when the repeat prediction is already very good — by
default, only consider restart when repeat PE > 0.5.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


class HiOnTopSegmenterV333:
    """v3.3.2 + per-topic f0 centroid / SEM2 restart branch."""

    def __init__(
        self,
        dim: int,
        alpha: float = 100.0,
        lmda: float = 10.0,
        tau: float = 50.0,
        cos_threshold: float = 0.9,
        beta: float = 0.25,
        pe_threshold: float = 0.5,
        k_max: int = 256,
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 3,
        rnn_max_context: int = 8,
        rnn_min_history: int = 2,
        # v3.3.3 only
        restart_pe_threshold: float = 0.5,
        restart_margin: float = 0.0,
        f0_tau: float | None = None,
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

        self.restart_pe_threshold = restart_pe_threshold
        self.restart_margin = restart_margin
        self.f0_tau = tau if f0_tau is None else f0_tau
        self.f0_min_starts = f0_min_starts

        self.model = EventRNN(input_dim=dim, hidden_dim=rnn_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rnn_lr)
        self.topics: list[TopicV33RNN] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        # f0 state (parallel to self.topics).
        self._f0_centroids: list[np.ndarray] = []
        self._f0_counts: list[int] = []

        # PE / boundary stats for importance v2 (2026-05-11). Optional usage —
        # RoundProcessor reads via pe_stats() / boundary_counts() accessors.
        self._pe_mean: list[float] = []
        self._pe_max: list[float] = []
        self._pe_count: list[int] = []
        self._n_boundaries: list[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_cluster_score(self, s: np.ndarray) -> float:
        return self.tau * self.cos_threshold

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
            self._f0_centroids.append(np.zeros(self.dim, dtype=np.float64))
            self._f0_counts.append(0)
            self._pe_mean.append(0.0)
            self._pe_max.append(0.0)
            self._pe_count.append(0)
            self._n_boundaries.append(0)

    def _surprise_forces_new(self, s: np.ndarray) -> bool:
        if not self.topics or self.pe_threshold >= 1.0:
            return False
        cos_max = -float("inf")
        threshold = 1.0 - self.pe_threshold
        for topic in self.topics:
            c = topic.cosine_score(s)
            if c > cos_max:
                cos_max = c
            if cos_max >= threshold:
                return False
        return True

    def _new_cluster_index_in_active(self, active: np.ndarray) -> int | None:
        for i, k in enumerate(active):
            if int(k) >= len(self.topics):
                return i
        return None

    def _f0_score(self, k: int, s: np.ndarray) -> float:
        """Cosine score against topic ``k``'s f0 centroid.

        Cold-start (`f0_count < f0_min_starts`) falls back to the
        ordinary centroid ``mu``.
        """
        if self._f0_counts[k] < self.f0_min_starts:
            ref = self.topics[k].mu
        else:
            ref = self._f0_centroids[k]
        n = float(np.linalg.norm(ref))
        if n <= 1e-12:
            return 0.0
        return float(np.dot(ref / n, s))

    def _update_f0(self, k: int, s: np.ndarray) -> None:
        """Update topic ``k``'s f0 centroid with a fresh episode-start sample."""
        self._f0_counts[k] += 1
        delta = s - self._f0_centroids[k]
        self._f0_centroids[k] = self._f0_centroids[k] + delta / self._f0_counts[k]
        nrm = float(np.linalg.norm(self._f0_centroids[k]))
        if nrm > 1e-12:
            self._f0_centroids[k] = self._f0_centroids[k] / nrm

    # ------------------------------------------------------------------
    # Score / branch selection
    # ------------------------------------------------------------------

    def _scores_with_repeat_restart(
        self, s: np.ndarray, prior: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MAP scores. For ``k == prev_k`` track repeat / restart."""
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        repeat_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)
        restart_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)
        for i, k in enumerate(active):
            k_int = int(k)
            log_prior = math.log(prior[k_int])
            if k_int >= len(self.topics):
                log_scores[i] = log_prior + self._new_cluster_score(s)
                continue
            repeat_lik = self.tau * self.topics[k_int].log_likelihood(s)
            repeat = log_prior + repeat_lik
            repeat_scores[i] = repeat
            if k_int == self.prev_k:
                # SEM2 restart branch: subtract sticky lambda from prior,
                # use f0 likelihood instead of repeat.
                base = max((self.counts[k_int] + 1) ** self.beta, 1e-300)
                log_prior_no_sticky = math.log(base)
                f0_lik = self.f0_tau * self._f0_score(k_int, s)
                restart = log_prior_no_sticky + f0_lik
                restart_scores[i] = restart
                # gate restart by repeat-PE: only consider restart when
                # repeat prediction is *not* very good
                pe_repeat = self.topics[k_int].prediction_error(s)
                if pe_repeat > self.restart_pe_threshold:
                    log_scores[i] = max(repeat, restart)
                else:
                    log_scores[i] = repeat
            else:
                log_scores[i] = repeat
        return log_scores, repeat_scores, restart_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, repeat_scores, restart_scores = self._scores_with_repeat_restart(
            s, prior, active
        )

        forced = self._surprise_forces_new(s)
        if forced:
            new_idx = self._new_cluster_index_in_active(active)
            chosen_idx = new_idx if new_idx is not None else int(np.argmax(log_scores))
        else:
            chosen_idx = int(np.argmax(log_scores))
        k = int(active[chosen_idx])

        # Boundary determination.
        is_restart = False
        if (
            self.prev_k is not None
            and k == self.prev_k
            and not math.isinf(restart_scores[chosen_idx])
        ):
            if (
                restart_scores[chosen_idx]
                > repeat_scores[chosen_idx] + self.restart_margin
            ):
                is_restart = True

        self._ensure_topic_slot(k)
        # PE running stats (for importance v2 salience export). Before topic.update.
        if self.counts[k] >= 1:
            pe = float(self.topics[k].prediction_error(s))
            pe = max(0.0, min(2.0, pe))
            n_prev = self._pe_count[k]
            n_new = n_prev + 1
            self._pe_mean[k] = (self._pe_mean[k] * n_prev + pe) / n_new
            self._pe_max[k] = max(self._pe_max[k], pe)
            self._pe_count[k] = n_new
        # update topic state (RNN + centroid)
        self.topics[k].update(s)
        self.counts[k] += 1

        is_label_change = self.prev_k is not None and k != self.prev_k
        is_boundary = is_label_change or is_restart
        # f0 update on episode-start: every boundary (incl. new topic) and
        # the very first turn of a brand-new topic.
        if is_boundary or self.counts[k] == 1:
            self._update_f0(k, s)
            self._n_boundaries[k] += 1

        self.prev_k = k
        return k, is_boundary

    def pe_stats(self) -> dict[int, dict[str, float]]:
        return {
            k: {"mean": float(self._pe_mean[k]), "max": float(self._pe_max[k]),
                "count": int(self._pe_count[k])}
            for k in range(len(self.topics)) if self._pe_count[k] > 0
        }

    def boundary_counts(self) -> dict[int, int]:
        return {
            k: int(self._n_boundaries[k])
            for k in range(len(self.topics)) if self.counts[k] > 0
        }

    def predict_topic(self, s: np.ndarray) -> int:
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, _, _ = self._scores_with_repeat_restart(s, prior, active)

        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            if new_idx is not None:
                return int(active[new_idx])
        chosen_idx = int(np.argmax(log_scores))
        return int(active[chosen_idx])
