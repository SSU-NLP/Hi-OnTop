"""Behavior tests for HiOnTopSegmenterV32 (Bounded Cosine MAP + sub-linear sCRP count)."""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.sem_core_optimize import HiOnTopSegmenterV3
from hi_ontop.sem_core_optimize_pe import HiOnTopSegmenterV32


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_v32_rejects_invalid_beta():
    import pytest

    with pytest.raises(ValueError):
        HiOnTopSegmenterV32(dim=8, beta=0.0)
    with pytest.raises(ValueError):
        HiOnTopSegmenterV32(dim=8, beta=1.5)


def test_v32_first_assignment_creates_topic_zero():
    rng = np.random.default_rng(0)
    seg = HiOnTopSegmenterV32(dim=16, beta=0.5)
    s = _unit(rng.normal(size=16))
    k, is_boundary = seg.assign(s)
    assert k == 0
    assert is_boundary is False
    assert int(seg.counts[0]) == 1


def test_v32_beta_one_matches_v31_segmentation():
    """``beta=1.0`` reduces to v3.1 exactly: same prior, same likelihood, same path."""
    rng = np.random.default_rng(7)
    seq = [_unit(rng.normal(size=16)) for _ in range(30)]

    seg_v31 = HiOnTopSegmenterV3(dim=16, alpha=10.0, lmda=10.0, tau=50.0, cos_threshold=0.3)
    seg_v32 = HiOnTopSegmenterV32(
        dim=16, alpha=10.0, lmda=10.0, tau=50.0, cos_threshold=0.3, beta=1.0
    )

    path_v31 = [seg_v31.assign(s)[0] for s in seq]
    path_v32 = [seg_v32.assign(s)[0] for s in seq]
    assert path_v31 == path_v32


def test_v32_smaller_beta_creates_more_topics_under_mega_topic_pressure():
    """With dominant cluster + low ``alpha``, sub-linear count opens more
    new topics than raw sCRP. Constructed scenario: many in-cluster samples
    seed C_k high, then we feed off-axis samples and check segmentation."""
    rng = np.random.default_rng(13)
    dim = 32
    in_cluster_mean = _unit(rng.normal(size=dim))

    def _seed_then_test(beta):
        seg = HiOnTopSegmenterV32(
            dim=dim, alpha=1.0, lmda=0.0, tau=10.0, cos_threshold=0.5, beta=beta,
        )
        for _ in range(50):
            noise = rng.normal(size=dim) * 0.05
            seg.assign(_unit(in_cluster_mean + noise))
        # Now push 20 off-axis samples
        new_topics = 0
        for _ in range(20):
            s = _unit(rng.normal(size=dim))
            k, _ = seg.assign(s)
            if k > 0:
                new_topics += max(new_topics, k)
        return len(set(int(c) for c in seg.counts.nonzero()[0]))

    n_topics_b1 = _seed_then_test(1.0)   # raw sCRP
    n_topics_b05 = _seed_then_test(0.25)  # strong squash

    # Sub-linear count should yield at least as many topics, typically more.
    assert n_topics_b05 >= n_topics_b1


def test_v32_assign_updates_counts_and_topics():
    rng = np.random.default_rng(2)
    seg = HiOnTopSegmenterV32(dim=16, beta=0.5)
    for _ in range(10):
        s = _unit(rng.normal(size=16))
        seg.assign(s)
    total = int(seg.counts.sum())
    assert total == 10


def test_v32_predict_topic_does_not_mutate():
    rng = np.random.default_rng(3)
    seg = HiOnTopSegmenterV32(dim=16, beta=0.5)
    for _ in range(8):
        seg.assign(_unit(rng.normal(size=16)))
    counts_before = seg.counts.copy()
    prev_k_before = seg.prev_k
    s = _unit(rng.normal(size=16))
    _ = seg.predict_topic(s)
    assert np.array_equal(seg.counts, counts_before)
    assert seg.prev_k == prev_k_before


def test_sticky_crp_unnormed_beta_default_unchanged():
    """``beta=1.0`` (default) reproduces the original SEM Eq 1 behavior."""
    counts = np.array([3, 0, 0, 0], dtype=np.int64)
    p_default = sticky_crp_unnormed(counts, prev_k=0, alpha=10.0, lmda=2.0)
    p_explicit = sticky_crp_unnormed(counts, prev_k=0, alpha=10.0, lmda=2.0, beta=1.0)
    assert np.allclose(p_default, p_explicit)
    # Manual check: visited[0] = 3 + 2 (lmda), unvisited[1] = 10 (alpha), rest 0
    assert math.isclose(p_default[0], 5.0)
    assert math.isclose(p_default[1], 10.0)


def test_sticky_crp_unnormed_beta_squashes_visited():
    """beta < 1 squashes (C+1)^β on visited topics; un-visited stays 0 + alpha."""
    counts = np.array([8, 0, 0, 0], dtype=np.int64)
    p = sticky_crp_unnormed(counts, prev_k=None, alpha=2.0, lmda=0.0, beta=0.5)
    # Visited[0] = (8 + 1)^0.5 = 3.0
    assert math.isclose(p[0], 3.0)
    # New-cluster slot[1] = 0 + alpha = 2.0
    assert math.isclose(p[1], 2.0)
