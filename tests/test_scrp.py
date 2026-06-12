"""Unit tests for sticky-CRP prior (matches SEM2 `_calculate_unnormed_sCRP`)."""

from __future__ import annotations

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed


def test_first_turn_no_prev() -> None:
    """All-zero counts, no prev: alpha placed at index 0."""
    counts = np.zeros(5, dtype=np.int64)
    prior = sticky_crp_unnormed(counts, prev_k=None, alpha=1.0, lmda=10.0)
    expected = np.zeros(5, dtype=np.float64)
    expected[0] = 1.0
    np.testing.assert_array_equal(prior, expected)


def test_stickiness_on_prev() -> None:
    """Prev cluster gets lmda bonus; new cluster slot gets alpha."""
    counts = np.array([3, 2, 0, 0, 0], dtype=np.int64)
    prior = sticky_crp_unnormed(counts, prev_k=0, alpha=1.0, lmda=10.0)
    expected = np.array([3 + 10, 2, 1, 0, 0], dtype=np.float64)
    np.testing.assert_array_equal(prior, expected)


def test_no_prev_bonus_when_none() -> None:
    counts = np.array([3, 2, 0], dtype=np.int64)
    prior = sticky_crp_unnormed(counts, prev_k=None, alpha=1.0, lmda=10.0)
    expected = np.array([3, 2, 1], dtype=np.float64)
    np.testing.assert_array_equal(prior, expected)


def test_matches_sem2_default_hyperparams() -> None:
    """SEM2 defaults (alfa=10, lmda=1): new-cluster weight should dominate."""
    counts = np.array([5, 3, 1, 0, 0], dtype=np.int64)
    prior = sticky_crp_unnormed(counts, prev_k=1, alpha=10.0, lmda=1.0)
    expected = np.array([5, 3 + 1, 1, 10, 0], dtype=np.float64)
    np.testing.assert_array_equal(prior, expected)


def test_hi_ontop_inverted_hyperparams() -> None:
    """Hi-OnTop (alpha=1, lmda=10): prev cluster should dominate."""
    counts = np.array([5, 3, 1, 0, 0], dtype=np.int64)
    prior = sticky_crp_unnormed(counts, prev_k=1, alpha=1.0, lmda=10.0)
    expected = np.array([5, 3 + 10, 1, 1, 0], dtype=np.float64)
    np.testing.assert_array_equal(prior, expected)


def test_full_capacity_no_new_slot() -> None:
    """If every slot is used (no trailing zero), no alpha added."""
    counts = np.array([5, 3, 2], dtype=np.int64)
    prior = sticky_crp_unnormed(counts, prev_k=None, alpha=1.0, lmda=10.0)
    expected = np.array([5, 3, 2], dtype=np.float64)
    np.testing.assert_array_equal(prior, expected)


def test_input_counts_unchanged() -> None:
    """Caller's counts array must not be mutated."""
    counts = np.array([3, 2, 0], dtype=np.int64)
    snapshot = counts.copy()
    _ = sticky_crp_unnormed(counts, prev_k=0, alpha=1.0, lmda=10.0)
    np.testing.assert_array_equal(counts, snapshot)
