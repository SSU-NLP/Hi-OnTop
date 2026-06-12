"""Online MAP segmenter — sub-linear sCRP + shared-GRU PE (v3.3.1, A+B).

Hi-OnTop-full-v3.3.1 revives SEM's event-model idea in the v3 line. After the
2026-05-07 A+B refactor, the segmenter holds **one** ``EventRNN`` and its
optimizer; topics carry only a per-topic hidden state ``h_k`` plus a cached
prediction. This (a) removes the redundant per-turn RNN forward across all
active topics, since cached predictions invalidate only when a topic gets a
new assignment, and (b) replaces per-topic ``nn.Module`` instances with a
single shared one, mirroring SEM2's actual architecture. Score:

    log_score_k = log[(C_k + 1)^beta + lambda * 1[prev=k]]
                  + tau * cos(s, predicted_next_k)

The fresh-topic score keeps v3.1/v3.2's ``tau * cos_threshold`` baseline.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


class HiOnTopSegmenterV331:
    """Online segmenter with shared-GRU prediction-error likelihood."""

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        tau: float = 50.0,
        cos_threshold: float = 0.7,
        beta: float = 0.5,
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
        self.dim = dim
        self.alpha = alpha
        self.lmda = lmda
        self.tau = tau
        self.cos_threshold = cos_threshold
        self.beta = beta
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
        """Read-only MAP assignment. Does not mutate topics/counts/``prev_k``."""
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
        chosen_idx = int(np.argmax(log_scores))
        return int(active[chosen_idx])
