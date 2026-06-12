"""Unit tests for HiOnTopSegmenter online MAP loop."""

from __future__ import annotations

import numpy as np

from hi_ontop.sem_core import HiOnTopSegmenter


def test_first_turn_creates_topic_zero() -> None:
    """First assignment always goes to topic 0 and is not a boundary."""
    seg = HiOnTopSegmenter(dim=4)
    k, boundary = seg.assign(np.array([1.0, 0.0, 0.0, 0.0]))
    assert k == 0
    assert boundary is False
    assert seg.counts[0] == 1
    assert seg.prev_k == 0
    assert len(seg.topics) == 1


def test_sticky_topic_persists_under_noise() -> None:
    """High stickiness keeps similar consecutive scenes in the same topic."""
    rng = np.random.default_rng(0)
    seg = HiOnTopSegmenter(dim=8, alpha=1.0, lmda=10.0)
    center = rng.normal(size=8)
    center = center / np.linalg.norm(center)
    for _ in range(10):
        s = center + rng.normal(scale=0.05, size=8)
        s = s / np.linalg.norm(s)
        k, b = seg.assign(s)
        assert k == 0
        assert b is False


def test_distinct_clusters_recovered() -> None:
    """Three well-separated orthogonal-ish clusters each land in their own topic."""
    rng = np.random.default_rng(7)
    dim = 32
    centers = []
    for _ in range(3):
        c = rng.normal(size=dim)
        centers.append(c / np.linalg.norm(c))
    centers[1] = centers[1] - 0.9 * centers[0]
    centers[1] /= np.linalg.norm(centers[1])
    centers[2] = centers[2] - 0.9 * centers[0] - 0.9 * centers[1]
    centers[2] /= np.linalg.norm(centers[2])

    seg = HiOnTopSegmenter(dim=dim, alpha=1.0, lmda=10.0, sigma0_sq=0.01)
    assignments: dict[int, list[int]] = {}
    for cluster_id in range(3):
        for _ in range(8):
            s = centers[cluster_id] + rng.normal(scale=0.02, size=dim)
            s = s / np.linalg.norm(s)
            k, _ = seg.assign(s)
            assignments.setdefault(k, []).append(cluster_id)

    # Each topic's samples should be mostly from a single true cluster
    for k, true_labels in assignments.items():
        majority = max(set(true_labels), key=true_labels.count)
        purity = true_labels.count(majority) / len(true_labels)
        assert purity >= 0.8, f"topic {k}: purity {purity:.2f}, labels {true_labels}"


def test_boundary_flag_on_switch() -> None:
    """``is_boundary`` is True iff topic changes from previous turn."""
    rng = np.random.default_rng(1)
    dim = 8
    c0 = np.zeros(dim)
    c0[0] = 1.0
    c1 = np.zeros(dim)
    c1[1] = 1.0
    seg = HiOnTopSegmenter(dim=dim, alpha=1.0, lmda=10.0)

    # 3 samples near c0 → topic 0
    boundaries = []
    for _ in range(3):
        s = c0 + rng.normal(scale=0.01, size=dim)
        s = s / np.linalg.norm(s)
        _, b = seg.assign(s)
        boundaries.append(b)
    # then 3 samples near c1 → topic 1, first one is a boundary
    for _ in range(3):
        s = c1 + rng.normal(scale=0.01, size=dim)
        s = s / np.linalg.norm(s)
        _, b = seg.assign(s)
        boundaries.append(b)

    # First sample ever: no boundary. Then 2 more non-boundaries in topic 0.
    assert boundaries[0] is False
    assert boundaries[1] is False
    assert boundaries[2] is False
    # Switch to topic 1
    assert boundaries[3] is True
    assert boundaries[4] is False
    assert boundaries[5] is False


def test_reproducibility() -> None:
    """Same input sequence → same output (segmenter has no internal RNG)."""
    rng = np.random.default_rng(123)
    scenes = rng.normal(size=(20, 16))
    scenes = scenes / np.linalg.norm(scenes, axis=1, keepdims=True)

    seg1 = HiOnTopSegmenter(dim=16, alpha=1.0, lmda=10.0)
    seg2 = HiOnTopSegmenter(dim=16, alpha=1.0, lmda=10.0)
    out1 = [seg1.assign(s)[0] for s in scenes]
    out2 = [seg2.assign(s)[0] for s in scenes]
    assert out1 == out2
