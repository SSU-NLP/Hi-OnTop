"""Topic (v3.1 / optimize) — centroid only, cosine score.

Hi-OnTop-full-v3.1 (model name: ``hi-ontop-full-v3.1.1``) replaces the diagonal
Gaussian likelihood of the v2 :class:`hi_ontop.topic.Topic` with a
**Bounded Cosine score** suited to L2-normalized embeddings::

    score(s; k) = mu_k^T s   (with mu_k re-normalized on every update)

Rationale (decision-log 2026-05-03):

* L2-normalized 768-D embeddings live on the unit sphere; the
  v2 Gaussian normalization term ``sum_j log(2*pi*sigma_j^2)``
  swamps the actual distance term, flattening the likelihood and
  letting the sCRP prior count ``C_k`` dominate the assignment.
* Cosine = inner product on the sphere is the native similarity for
  this representation (encoder contract, see ``embedding.py:10-12``).

This file is **only** used when ``HiOnTop(version="v3.1.1")``. v2 paths
import ``hi_ontop.topic.Topic`` unchanged. v3.2 (cosine prediction
error) lives in ``hi_ontop.topic_optimize_pe``.
"""

from __future__ import annotations

import numpy as np


class TopicV3:
    """Topic with running mean direction; cosine likelihood.

    State is just the direction estimate ``mu`` (unit-norm) plus a sample
    count ``n``. Welford-style variance is intentionally dropped — cosine
    score has no per-dimension variance term.

    Args:
        topic_id: Integer identifier.
        dim: Feature dimension (e.g., 768 for ``bge-base-en-v1.5``).
    """

    def __init__(self, topic_id: int, dim: int) -> None:
        self.topic_id = topic_id
        self.dim = dim
        self.mu: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.n: int = 0

    def update(self, s: np.ndarray) -> None:
        """Online running-mean update, then re-normalize ``mu`` to unit norm.

        Re-normalization keeps the score ``mu^T s`` in ``[-1, 1]``. If the
        running sum collapses to ~0 (extremely rare with semantic
        embeddings), ``mu`` is left unchanged — the next update will
        reseed it.
        """
        self.n += 1
        delta = s - self.mu
        self.mu = self.mu + delta / self.n
        norm = float(np.linalg.norm(self.mu))
        if norm > 1e-12:
            self.mu = self.mu / norm

    def log_likelihood(self, s: np.ndarray) -> float:
        """Cosine similarity score, used as a log-likelihood surrogate.

        Range: ``[-1, 1]``. The MAP score is ``log prior_k + tau * cos``
        — the temperature ``tau`` is applied by the segmenter, not here.
        Returning the raw cosine keeps this primitive composable.
        """
        return float(np.dot(self.mu, s))
