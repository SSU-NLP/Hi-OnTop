"""Online MAP segmenter — v3.3.1 + hard PE boundary (v3.3.2).

Hi-OnTop-full-v3.3.2 keeps every term of v3.3.1's score function

    log_score_k = log[(C_k + 1)^beta + lambda * 1[prev=k]]
                  + tau * cos(s, predicted_next_k)

but adds a *Bayesian-surprise hard reset*: when no existing topic
predicts the incoming embedding well enough — concretely when

    max_k cos(s, predicted_next_k) < (1 - pe_threshold)

— the segmenter forces a new topic regardless of MAP. This restores
SEM2's "prediction-error spike triggers an event boundary" behaviour
that v3.3.1 only encodes softly through the MAP score, allowing the
mega-topic prior + bulk-direction RNN prediction to win even when the
content has clearly shifted.

``pe_threshold`` defaults to ``1.0`` so that the rule is *disabled* by
default — v3.3.2 is then numerically identical to v3.3.1. Use
``pe_threshold = 0.5`` (cosine < 0.5 forces new) as a moderate setting.

The shared GRU + per-topic hidden state architecture from v3.3.1 (A+B)
is reused verbatim via :class:`hi_ontop.topic_v331_rnn.TopicV33RNN`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


class HiOnTopSegmenterV332:
    """Segmenter with shared-GRU PE *and* explicit surprise-driven boundary."""

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        tau: float = 50.0,
        cos_threshold: float = 0.7,
        beta: float = 0.5,
        pe_threshold: float = 1.0,
        k_max: int = 256,
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 1,
        rnn_max_context: int = 8,  # legacy CLI compat — unused after A+B
        rnn_min_history: int = 2,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if rnn_lr <= 0:
            raise ValueError(f"rnn_lr must be > 0, got {rnn_lr}")
        if rnn_train_steps < 0:
            raise ValueError(f"rnn_train_steps must be >= 0, got {rnn_train_steps}")
        if not 0.0 <= pe_threshold <= 1.0:
            raise ValueError(f"pe_threshold must be in [0, 1], got {pe_threshold}")
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

        self.model = EventRNN(input_dim=dim, hidden_dim=rnn_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rnn_lr)
        self.topics: list[TopicV33RNN] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

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

    def _surprise_forces_new(self, s: np.ndarray) -> bool:
        """Return True iff every existing topic's cosine prediction falls
        below ``1 - pe_threshold`` — i.e., the embedding is "surprising"
        for *all* known events. Disabled (always False) when
        ``pe_threshold >= 1.0`` or when no topics exist yet.
        """
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
        """Index in ``active`` of the K+1 (fresh) slot, if any."""
        for i, k in enumerate(active):
            if int(k) >= len(self.topics):
                return i
        return None

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        """Assign scene ``s`` and return ``(topic_id, is_boundary)``."""
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)

        log_scores = np.empty(active.shape[0], dtype=np.float64)
        for i, k in enumerate(active):
            k_int = int(k)
            if k_int < len(self.topics):
                log_lik = self.tau * self.topics[k_int].log_likelihood(s)
            else:
                log_lik = self._new_cluster_score(s)
            log_scores[i] = math.log(prior[k_int]) + log_lik

        forced = self._surprise_forces_new(s)
        if forced:
            new_idx = self._new_cluster_index_in_active(active)
            chosen_idx = new_idx if new_idx is not None else int(np.argmax(log_scores))
        else:
            chosen_idx = int(np.argmax(log_scores))
        k = int(active[chosen_idx])

        while len(self.topics) <= k:
            self.topics.append(self._new_topic())
        self.topics[k].update(s)
        self.counts[k] += 1

        is_boundary = self.prev_k is not None and k != self.prev_k
        self.prev_k = k
        return k, is_boundary

    def predict_topic(self, s: np.ndarray) -> int:
        """Read-only MAP assignment. Surprise rule applies here too — at
        eval time a query that no topic predicts well falls into the
        K+1 (fresh) slot, which downstream retrieval treats as
        "no matching topic". Does not mutate state.
        """
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        for i, k in enumerate(active):
            k_int = int(k)
            if k_int < len(self.topics):
                log_lik = self.tau * self.topics[k_int].log_likelihood(s)
            else:
                log_lik = self._new_cluster_score(s)
            log_scores[i] = math.log(prior[k_int]) + log_lik

        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            if new_idx is not None:
                return int(active[new_idx])
        chosen_idx = int(np.argmax(log_scores))
        return int(active[chosen_idx])
