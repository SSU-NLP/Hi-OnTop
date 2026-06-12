"""Smoke + behavior tests for HiOnTopSegmenterV3 (Bounded Cosine MAP).

Mirrors the structure of ``test_sem_core.py`` but for the v3 segmenter.
"""

from __future__ import annotations

import numpy as np

from hi_ontop.sem_core_optimize import HiOnTopSegmenterV3
from hi_ontop.topic_optimize import TopicV3


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_topic_v3_update_keeps_unit_norm():
    rng = np.random.default_rng(0)
    t = TopicV3(topic_id=0, dim=16)
    for _ in range(10):
        s = _unit(rng.normal(size=16))
        t.update(s)
        assert np.isclose(np.linalg.norm(t.mu), 1.0, atol=1e-9)


def test_topic_v3_log_likelihood_is_cosine():
    rng = np.random.default_rng(1)
    t = TopicV3(topic_id=0, dim=16)
    for _ in range(5):
        t.update(_unit(rng.normal(size=16)))
    s = _unit(rng.normal(size=16))
    assert np.isclose(t.log_likelihood(s), float(np.dot(t.mu, s)))


def test_segmenter_v3_first_assignment_is_topic_zero():
    rng = np.random.default_rng(2)
    seg = HiOnTopSegmenterV3(dim=8, tau=10.0)
    s = _unit(rng.normal(size=8))
    k, is_boundary = seg.assign(s)
    assert k == 0
    assert is_boundary is False
    assert seg.counts[0] == 1
    assert seg.prev_k == 0


def test_segmenter_v3_segments_orthogonal_clusters_low_sticky():
    """Two strictly orthogonal clusters with very low sticky prior produce
    distinct topic ids. This is a calibration sanity check — real-world
    settings (LoCoMo) need careful tau/lmda tuning, which this test does
    not attempt to be representative of."""
    mu1 = np.zeros(768)
    mu1[0] = 1.0
    mu2 = np.zeros(768)
    mu2[1] = 1.0

    seq = [mu1.copy() for _ in range(3)]
    seq += [mu2.copy() for _ in range(3)]
    seq += [mu1.copy() for _ in range(3)]

    # tau high, sticky very low → cosine signal dominates.
    seg = HiOnTopSegmenterV3(dim=768, tau=100.0, alpha=10.0, lmda=0.1)
    ids = [seg.assign(s)[0] for s in seq]
    assert ids[:3] == [0] * 3
    assert ids[3:6] == [1] * 3
    assert ids[6:] == [0] * 3
    assert len(seg.topics) == 2


def test_segmenter_v3_predict_topic_does_not_mutate():
    rng = np.random.default_rng(4)
    seg = HiOnTopSegmenterV3(dim=32, tau=20.0)
    for _ in range(5):
        seg.assign(_unit(rng.normal(size=32)))
    snapshot_counts = seg.counts.copy()
    snapshot_prev_k = seg.prev_k
    snapshot_topic_n = [t.n for t in seg.topics]

    s = _unit(rng.normal(size=32))
    _ = seg.predict_topic(s)
    assert np.array_equal(seg.counts, snapshot_counts)
    assert seg.prev_k == snapshot_prev_k
    assert [t.n for t in seg.topics] == snapshot_topic_n


def test_segmenter_v3_predict_empty_returns_zero():
    seg = HiOnTopSegmenterV3(dim=8, tau=10.0)
    assert seg.predict_topic(_unit(np.ones(8))) == 0
