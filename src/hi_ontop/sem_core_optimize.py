"""Online MAP segmenter — sticky-CRP prior + Bounded Cosine score (v3.1).

Hi-OnTop-full-v3.1 replaces the v2 Gaussian likelihood with a cosine score::

    e_hat_n = argmax_k [ log prior(k | prev) + tau * cos(s_n, mu_k) ]

This is the centroid-cosine variant. v3.2 (in
:mod:`hi_ontop.sem_core_optimize_pe`) replaces the centroid with an
AR(1) on-sphere prediction ``mu_k`` ⇒ ``predicted_next_k`` so the
score becomes ``tau * cos(s_n, predicted_next_k)`` — a cosine
prediction-error formulation.

Only the likelihood changed; the sticky-CRP prior is reused verbatim
from :mod:`hi_ontop.scrp`. Subsequent v3 phases (prior decay /
sqrt count / boundary-start f0) plug in on top of either v3.1 or v3.2.
"""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_optimize import TopicV3


class HiOnTopSegmenterV3:
    """Online topic segmenter with cosine likelihood.

    Args:
        dim: Feature dimension.
        alpha: sticky-CRP concentration. Default 1.0.
        lmda: sticky-CRP stickiness. Default 10.0.
        tau: Cosine temperature — scales the cosine term against the
            ``log prior`` term. Default 10.0 (decision-log 2026-05-03).
        cos_threshold: New-cluster cosine baseline. The K+1 score is
            ``log alpha + tau * cos_threshold`` — a fresh topic competes
            as if it had cosine = ``cos_threshold`` with the query.
            Existing topic ``k`` only beats the new-cluster slot when
            ``cos(s, mu_k) > cos_threshold + (log alpha - log prior_k)/tau``.
            Default 0.5 (treats moderate-strength matches as "good
            enough to keep" vs. starting a new cluster). With the v2
            cold-start of 0.0, mega-topic collapse is unavoidable in
            long conversations because every existing topic with
            positive cos beats the new-cluster slot.
        k_max: Maximum allocated topic capacity. Default 256.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        tau: float = 50.0,
        cos_threshold: float = 0.7,
        k_max: int = 256,
    ) -> None:
        self.dim = dim
        self.alpha = alpha
        self.lmda = lmda
        self.tau = tau
        self.cos_threshold = cos_threshold
        self.k_max = k_max

        self.topics: list[TopicV3] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

    def _new_cluster_score(self, s: np.ndarray) -> float:
        """Score for the K+1 (fresh) cluster: ``tau * cos_threshold``.

        Existing topic ``k`` only beats the new-cluster slot on the
        likelihood term when ``cos(s, mu_k) > cos_threshold``. The
        prior term still favors existing topics (``log(C_k + lmda)``
        vs ``log(alpha)``); ``cos_threshold`` is the cosine margin a
        match must clear to be worth keeping over a fresh start.
        """
        return self.tau * self.cos_threshold

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        """Assign scene ``s`` to a topic and return ``(topic_id, is_boundary)``.

        Args:
            s: Scene embedding (``dim,``). Caller normalizes in the
               encoder; this method does **not** renormalize.
        """
        prior = sticky_crp_unnormed(self.counts, self.prev_k, self.alpha, self.lmda)
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
        """Read-only MAP assignment. Mirrors v2 :meth:`predict_topic`."""
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(self.counts, self.prev_k, self.alpha, self.lmda)
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
