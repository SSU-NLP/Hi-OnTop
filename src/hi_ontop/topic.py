"""Topic — centroid + diag variance with Welford online update.

Implements option A of Hi-OnTop's event model
(``context/01-hi-ontop-design.md`` §4)::

    P(s | e=k) = N(s; mu_k, diag(sigma_k^2))

Cold start: while ``n < 3``, variance is a constant prior ``sigma0_sq``.
After ``n >= 3``, variance is running ``M2 / n`` with a ``sigma_min_sq``
floor (prevents div-by-zero on identical samples).
"""

from __future__ import annotations

import math

import numpy as np

_COLD_START_N = 3


class Topic:
    """A topic cluster with online centroid + diagonal variance.

    Args:
        topic_id: Integer identifier.
        dim: Feature dimension (e.g., 768 for ``bge-base-en-v1.5``).
        sigma0_sq: Cold-start variance prior (used while ``n < 3``).
        sigma_min_sq: Variance floor after ``n >= 3``.
    """

    def __init__(
        self, topic_id: int, dim: int, sigma0_sq: float, sigma_min_sq: float
    ) -> None:
        self.topic_id = topic_id
        self.dim = dim
        self.sigma0_sq = sigma0_sq
        self.sigma_min_sq = sigma_min_sq
        self.mu: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.M2: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.n: int = 0

    def variance(self) -> np.ndarray:
        """Current variance vector (``dim,``).

        Returns the cold-start prior ``sigma0_sq * 1`` while ``n < 3``,
        otherwise the running variance ``M2 / n`` floored at
        ``sigma_min_sq``.
        """
        if self.n < _COLD_START_N:
            return np.full(self.dim, self.sigma0_sq, dtype=np.float64)
        return np.maximum(self.M2 / self.n, self.sigma_min_sq)

    def update(self, s: np.ndarray) -> None:
        """Welford online update with a new sample ``s`` (``dim,``)."""
        self.n += 1
        delta = s - self.mu
        self.mu = self.mu + delta / self.n
        delta2 = s - self.mu
        self.M2 = self.M2 + delta * delta2

    def log_likelihood(self, s: np.ndarray) -> float:
        """Gaussian log-likelihood ``log N(s; mu, diag(sigma^2))``.

        Matches ``context/02-math-model.md`` §Topic 동역학::

            -0.5 * sum[ (s - mu)^2 / sigma^2 + log(2*pi*sigma^2) ]
        """
        sigma2 = self.variance()
        diff = s - self.mu
        return float(
            -0.5 * np.sum(diff * diff / sigma2 + np.log(2.0 * math.pi * sigma2))
        )

    def prediction_error(self, s: np.ndarray) -> float:
        """Mahalanobis-form PE: ``sum( (s - mu)^2 / sigma^2 )``.

        Related to ``-2 * log_likelihood`` up to a constant
        (``context/02-math-model.md`` §Prediction Error).
        """
        sigma2 = self.variance()
        diff = s - self.mu
        return float(np.sum(diff * diff / sigma2))
