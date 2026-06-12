"""Online MAP segmenter — sticky-CRP prior + option A Gaussian likelihood.

Implements Hi-OnTop's topic segmentation loop
(``context/01-hi-ontop-design.md`` §5 and ``02-math-model.md`` §Topic 배정)::

    e_hat_n = argmax_k [ log prior(k | prev) + log P(s_n | e=k) ]

Under option A, ``P(s | e=k) = N(mu_k, diag(sigma_k^2))`` has no
dependence on ``s_{n-1}``. Therefore SEM2's restart-vs-repeat branch
(``log_likelihood_next`` vs ``log_likelihood_f0``) and its
``prior[k_prev] -= lmda/2`` halving are **not ported** — both are
relevant only when the event dynamics use scene history. See
``context/00-sem-paper.md`` §7 (검증 미해결 1, 2).
"""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic import Topic


class HiOnTopSegmenter:
    """Online topic segmenter over L2-normalized scene vectors.

    Each call to :meth:`assign` does:

    1. Compute unnormalized sticky-CRP prior over existing topics +
       one new-cluster slot.
    2. Score each candidate with ``log prior + log likelihood``.
       New-cluster likelihood uses a cold-start Gaussian centered
       at the origin with variance ``sigma0_sq``.
    3. Pick MAP topic, update its Welford state, update counts.

    Args:
        dim: Feature dimension (768 for ``bge-base-en-v1.5``).
        alpha: sticky-CRP concentration. Default 1.0 (Hi-OnTop).
        lmda: sticky-CRP stickiness. Default 10.0 (Hi-OnTop).
        sigma0_sq: Topic cold-start variance prior. Default 0.01.
        sigma_min_sq: Variance floor. Default 1e-6.
        k_max: Maximum allocated topic capacity. Default 256.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        sigma0_sq: float = 0.01,
        sigma_min_sq: float = 1e-6,
        k_max: int = 256,
    ) -> None:
        self.dim = dim
        self.alpha = alpha
        self.lmda = lmda
        self.sigma0_sq = sigma0_sq
        self.sigma_min_sq = sigma_min_sq
        self.k_max = k_max

        self.topics: list[Topic] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

    def _cold_start_log_lik(self, s: np.ndarray) -> float:
        """``log N(s; 0, sigma0_sq * I)`` — likelihood for a fresh cluster.

        Using the origin as the mean of a never-seen cluster is a
        convention (no prior information about its centroid). For
        L2-normalized ``s`` this keeps the score O(1) per dim.
        """
        sigma2 = self.sigma0_sq
        return float(
            -0.5
            * (np.sum(s * s) / sigma2 + self.dim * math.log(2.0 * math.pi * sigma2))
        )

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        """Assign scene ``s`` to a topic and return ``(topic_id, is_boundary)``.

        Args:
            s: Scene embedding (``dim,``). Callers normalize in the
               encoder; this method does **not** renormalize.

        Returns:
            ``topic_id``: The chosen topic's id (0-based).
            ``is_boundary``: True iff topic changed from previous turn
            (False on the very first turn).
        """
        prior = sticky_crp_unnormed(self.counts, self.prev_k, self.alpha, self.lmda)
        active = np.flatnonzero(prior)

        log_scores = np.empty(active.shape[0], dtype=np.float64)
        for i, k in enumerate(active):
            k_int = int(k)
            if k_int < len(self.topics):
                log_lik = self.topics[k_int].log_likelihood(s)
            else:
                log_lik = self._cold_start_log_lik(s)
            log_scores[i] = math.log(prior[k_int]) + log_lik

        chosen_idx = int(np.argmax(log_scores))
        k = int(active[chosen_idx])

        while len(self.topics) <= k:
            self.topics.append(
                Topic(
                    topic_id=len(self.topics),
                    dim=self.dim,
                    sigma0_sq=self.sigma0_sq,
                    sigma_min_sq=self.sigma_min_sq,
                )
            )
        self.topics[k].update(s)
        self.counts[k] += 1

        is_boundary = self.prev_k is not None and k != self.prev_k
        self.prev_k = k
        return k, is_boundary

    def predict_topic(self, s: np.ndarray) -> int:
        """Read-only MAP assignment. Does NOT mutate counts, topics, or
        ``prev_k``. Used during evaluation queries where the question
        should select retrieval target without contributing to topic
        statistics (the conversation that built those statistics is
        already complete).

        Returns the most-likely topic_id given current state, or ``0`` if
        no topics exist yet.
        """
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(self.counts, self.prev_k, self.alpha, self.lmda)
        active = np.flatnonzero(prior)
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        for i, k in enumerate(active):
            k_int = int(k)
            if k_int < len(self.topics):
                log_lik = self.topics[k_int].log_likelihood(s)
            else:
                log_lik = self._cold_start_log_lik(s)
            log_scores[i] = math.log(prior[k_int]) + log_lik
        chosen_idx = int(np.argmax(log_scores))
        return int(active[chosen_idx])
