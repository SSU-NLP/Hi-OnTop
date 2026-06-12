"""Tests for v3.3.2 — surprise-driven hard PE boundary."""

from __future__ import annotations

import numpy as np
import torch

from hi_ontop.sem_core_v331_rnn import HiOnTopSegmenterV331
from hi_ontop.sem_core_v332_rnn_pe import HiOnTopSegmenterV332


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_v332_disabled_pe_threshold_matches_v331_segmentation():
    """With pe_threshold=1.0 the surprise rule never fires, so v3.3.2's
    assignment trace should match v3.3.1's exactly under the same inputs.
    """
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    seq = [_unit(rng.normal(size=8)) for _ in range(20)]

    torch.manual_seed(0)
    seg_a = HiOnTopSegmenterV331(dim=8, alpha=1.0, lmda=10.0, tau=50.0,
                              cos_threshold=0.5, beta=0.5,
                              rnn_hidden_dim=4, rnn_train_steps=1,
                              rnn_min_history=2)
    torch.manual_seed(0)
    seg_b = HiOnTopSegmenterV332(dim=8, alpha=1.0, lmda=10.0, tau=50.0,
                              cos_threshold=0.5, beta=0.5,
                              pe_threshold=1.0,  # disabled
                              rnn_hidden_dim=4, rnn_train_steps=1,
                              rnn_min_history=2)

    path_a = [seg_a.assign(s)[0] for s in seq]
    path_b = [seg_b.assign(s)[0] for s in seq]
    assert path_a == path_b, f"v3.3.2 pe_threshold=1.0 must equal v3.3.1: {path_a} vs {path_b}"


def test_v332_aggressive_pe_threshold_creates_more_topics():
    """A very low cos-cap (=> high pe_threshold) should force boundaries
    on every novelty spike. The v3.3.2 trace should have at least as many
    distinct topics as v3.3.1's on the same input.
    """
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    seq = [_unit(rng.normal(size=8)) for _ in range(20)]

    torch.manual_seed(0)
    seg_default = HiOnTopSegmenterV331(dim=8, alpha=1.0, lmda=10.0, tau=50.0,
                                    cos_threshold=0.5, beta=0.5,
                                    rnn_hidden_dim=4, rnn_train_steps=1,
                                    rnn_min_history=2)
    torch.manual_seed(0)
    seg_aggressive = HiOnTopSegmenterV332(dim=8, alpha=1.0, lmda=10.0, tau=50.0,
                                        cos_threshold=0.5, beta=0.5,
                                        pe_threshold=0.5,  # cos < 0.5 forces new
                                        rnn_hidden_dim=4, rnn_train_steps=1,
                                        rnn_min_history=2)
    n_a = len({seg_default.assign(s)[0] for s in seq})
    n_b = len({seg_aggressive.assign(s)[0] for s in seq})
    assert n_b >= n_a, f"aggressive PE rule should not yield fewer topics ({n_b} < {n_a})"


def test_v332_predict_topic_does_not_mutate():
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    seg = HiOnTopSegmenterV332(dim=16, pe_threshold=0.5,
                             rnn_hidden_dim=4, rnn_train_steps=1)
    for _ in range(10):
        seg.assign(_unit(rng.normal(size=16)))
    counts_before = seg.counts.copy()
    prev_k_before = seg.prev_k
    topic_ns_before = [t.n for t in seg.topics]
    _ = seg.predict_topic(_unit(rng.normal(size=16)))
    assert np.array_equal(seg.counts, counts_before)
    assert seg.prev_k == prev_k_before
    assert [t.n for t in seg.topics] == topic_ns_before


def test_v332_invalid_pe_threshold_rejected():
    import pytest
    with pytest.raises(ValueError):
        HiOnTopSegmenterV332(dim=8, pe_threshold=-0.1)
    with pytest.raises(ValueError):
        HiOnTopSegmenterV332(dim=8, pe_threshold=1.5)


def test_v332_pe_threshold_zero_forces_new_topic_every_turn():
    """pe_threshold=0.0 means cos < 1.0 forces new (i.e., almost always).
    Each turn should be assigned to a distinct topic until k_max.
    """
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    seg = HiOnTopSegmenterV332(dim=8, pe_threshold=0.0,
                             rnn_hidden_dim=4, rnn_train_steps=1,
                             k_max=64)
    path = [seg.assign(_unit(rng.normal(size=8)))[0] for _ in range(15)]
    # First turn always gets topic 0 (no existing topics; surprise can't fire).
    # Subsequent turns: cos to existing < 1.0 → force new each time.
    assert path[0] == 0
    assert all(path[i] == i for i in range(len(path))), f"got {path}"
