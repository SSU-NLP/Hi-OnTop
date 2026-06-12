"""Unit tests for Topic (Welford online update + Gaussian likelihood)."""

from __future__ import annotations

import numpy as np
import pytest

from hi_ontop.topic import Topic


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_welford_matches_offline(rng: np.random.Generator) -> None:
    """Welford online mean/variance converges to ``np.mean`` / ``np.var``."""
    samples = rng.normal(loc=1.5, scale=0.3, size=(100, 16))
    t = Topic(topic_id=0, dim=16, sigma0_sq=0.01, sigma_min_sq=1e-12)
    for s in samples:
        t.update(s)
    np.testing.assert_allclose(t.mu, samples.mean(axis=0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        t.M2 / t.n, samples.var(axis=0, ddof=0), rtol=1e-10, atol=1e-10
    )


def test_cold_start_variance_returns_prior() -> None:
    """While n < 3, variance() equals sigma0_sq * 1 — irrespective of M2."""
    t = Topic(topic_id=0, dim=8, sigma0_sq=0.05, sigma_min_sq=1e-6)
    np.testing.assert_array_equal(t.variance(), np.full(8, 0.05))
    t.update(np.ones(8))
    np.testing.assert_array_equal(t.variance(), np.full(8, 0.05))
    t.update(np.ones(8) * 2)
    np.testing.assert_array_equal(t.variance(), np.full(8, 0.05))


def test_variance_floor_after_cold_start() -> None:
    """After n >= 3 with identical samples (M2 = 0), floor is sigma_min_sq."""
    t = Topic(topic_id=0, dim=4, sigma0_sq=0.05, sigma_min_sq=1e-6)
    for _ in range(3):
        t.update(np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(t.variance(), np.full(4, 1e-6))


def test_log_likelihood_peaks_at_centroid(rng: np.random.Generator) -> None:
    """log_likelihood is strictly greater at the centroid than nearby points."""
    samples = rng.normal(loc=0.0, scale=1.0, size=(50, 8))
    t = Topic(topic_id=0, dim=8, sigma0_sq=0.01, sigma_min_sq=1e-6)
    for s in samples:
        t.update(s)
    ll_at_mu = t.log_likelihood(t.mu)
    ll_elsewhere = t.log_likelihood(t.mu + 2.0)
    assert ll_at_mu > ll_elsewhere


def test_prediction_error_zero_at_centroid() -> None:
    t = Topic(topic_id=0, dim=4, sigma0_sq=0.01, sigma_min_sq=1e-6)
    for _ in range(5):
        t.update(np.array([1.0, 0.5, -0.5, 0.0]))
    assert t.prediction_error(t.mu) == pytest.approx(0.0, abs=1e-10)
    assert t.prediction_error(np.zeros(4)) > 0.0


def test_update_does_not_mutate_input() -> None:
    s = np.array([1.0, 2.0, 3.0, 4.0])
    snapshot = s.copy()
    t = Topic(topic_id=0, dim=4, sigma0_sq=0.01, sigma_min_sq=1e-6)
    t.update(s)
    np.testing.assert_array_equal(s, snapshot)
