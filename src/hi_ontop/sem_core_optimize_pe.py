"""Online MAP segmenter — sticky-CRP prior + Bounded Cosine + sub-linear count (v3.2).

Hi-OnTop-full-v3.2 keeps v3.1's Bounded Cosine MAP likelihood verbatim and
adds the **sub-linear count** P0 modification to the sticky-CRP prior:

    log_score_k = log[(C_k + 1)^beta + lambda * 1[prev=k]] + tau * cos(s, mu_k)

The cosine likelihood term and the new-cluster baseline (
``tau * cos_threshold``) are identical to v3.1; only the prior changes.
``beta = 1.0`` recovers v3.1 exactly. ``beta < 1.0`` squashes the
``log C_k`` accumulation that drives mega-topic collapse on long
conversations (LoCoMo 600+ turn).

Diagnostic motivation: with ``C_k ~ 600, alpha = 10, tau = 50,
cos_threshold = 0.3``, raw v3.1 lets an existing topic win whenever
``cos(s, mu_k) > 0.218`` — almost always within a single conv. With
``beta = 0.5`` the threshold rises to ~``0.275``; with ``beta = 0.25``,
to ~``0.292`` — letting genuine semantic shifts open new topics.

The previous v3.2 (cosine prediction-error with AR(1) ``rho`` blending)
is removed; the AR(1) variant did not address the dominant mega-topic
cause and added a hyperparameter without empirical payoff. See
``context/06-decision-log.md`` for the migration note.
"""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_optimize import TopicV3


class HiOnTopSegmenterV32:
    """Online topic segmenter — Bounded Cosine MAP + sub-linear sCRP count.

    Args:
        dim: Feature dimension.
        alpha: sticky-CRP concentration. Default 1.0.
        lmda: sticky-CRP stickiness. Default 10.0.
        tau: Cosine temperature. Default 50.0 (v3.1-tuned).
        cos_threshold: New-cluster cosine baseline. Default 0.7.
        beta: Sub-linear count exponent in (0, 1]. Default 0.5
            (sqrt). ``1.0`` reduces to v3.1 exactly.
        k_max: Maximum allocated topic capacity. Default 256.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        tau: float = 50.0,
        cos_threshold: float = 0.7,
        beta: float = 0.5,
        k_max: int = 256,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        self.dim = dim
        self.alpha = alpha
        self.lmda = lmda
        self.tau = tau
        self.cos_threshold = cos_threshold
        self.beta = beta
        self.k_max = k_max

        self.topics: list[TopicV3] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

    def _new_cluster_score(self, s: np.ndarray) -> float:
        """Score for the K+1 (fresh) cluster: ``tau * cos_threshold``.

        Identical to v3.1 — only the prior side differs.
        """
        return self.tau * self.cos_threshold

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
            self.topics.append(TopicV3(topic_id=len(self.topics), dim=self.dim))
        self.topics[k].update(s)
        self.counts[k] += 1

        is_boundary = self.prev_k is not None and k != self.prev_k
        self.prev_k = k
        return k, is_boundary

    def predict_topic(self, s: np.ndarray) -> int:
        """Read-only MAP assignment. Mirrors v2/v3.1 ``predict_topic``."""
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
