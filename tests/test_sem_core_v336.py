"""v3.3.6 — SEM2-faithful event dynamics invariants.

Pins the codex-locked spec (2026-05-17):

* v3.3.5's ``f_is_trained`` cold-start gating still holds (untrained
  topic ties the fresh slot at ``L0`` → survives via ``+λ``);
* ``rnn_ready`` is a *separate* gate: a trained-but-not-ready topic
  predicts by **persistence** (last observed embedding), never a
  random RNN;
* the per-topic RNN owns its weights (no cross-topic interference) and
  is reproducible (deterministic seeded init) — two runs with the same
  seed produce identical assignments.
"""

from __future__ import annotations

import numpy as np
import pytest

from hi_ontop.sem_core_v336 import HiOnTopSegmenterV336
from hi_ontop.topic_v336 import TopicV336

DIM = 8


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_untrained_ties_fresh_slot_and_survives() -> None:
    """v3.3.5 invariant still holds under v3.3.6 dynamics."""
    seg = HiOnTopSegmenterV336(dim=DIM, alpha=1.0, lmda=10.0)
    a = np.zeros(DIM); a[0] = 1.0
    b = np.zeros(DIM); b[1] = 1.0  # orthogonal → any predictor PE ≈ 1
    k1, _ = seg.assign(a)
    k2, bnd2 = seg.assign(b)
    assert k1 == 0 and k2 == 0  # young untrained prev survives via λ
    assert bnd2 is False


def test_persistence_until_rnn_ready() -> None:
    """A trained topic below ``rnn_ready_min_transitions`` predicts by
    persistence (last embedding), not the RNN."""
    seg = HiOnTopSegmenterV336(
        dim=DIM, alpha=1.0, lmda=10.0, rnn_ready_min_transitions=3
    )
    v1 = _unit(1, 1, 0, 0, 0, 0, 0, 0)
    v2 = _unit(1, 0.9, 0.1, 0, 0, 0, 0, 0)
    seg.assign(v1)
    seg.assign(v2)  # continuation → transition_count[0] == 1 (trained, not ready)
    assert seg._is_trained(0) is True
    assert seg._rnn_ready(0) is False
    # predictor must be persistence: predict_next(use_rnn=False) == last emb
    persist = seg.topics[0].predict_next(use_rnn=False)
    np.testing.assert_allclose(persist, seg.topics[0]._seq[-1])
    # segmenter PE uses persistence here
    pe_seg = seg._pe(0, v1)
    pe_persist = 1.0 - float(
        np.dot(persist / np.linalg.norm(persist), v1)
    )
    assert pe_seg == pytest.approx(pe_persist)


def test_rnn_trained_on_history_changes_prediction() -> None:
    """Once enough transitions accrue the RNN is used and its cached
    prediction reflects the replayed topic history (not persistence)."""
    seg = HiOnTopSegmenterV336(
        dim=DIM, alpha=1.0, lmda=50.0, rnn_ready_min_transitions=2
    )
    base = _unit(1, 0.2, 0, 0, 0, 0, 0, 0)
    for j in range(6):  # stay in one topic (high λ) and accumulate history
        seg.assign(_unit(1.0, 0.2 + 0.02 * j, 0.01 * j, 0, 0, 0, 0, 0))
    assert seg._rnn_ready(0) is True
    rnn_pred = seg.topics[0].predict_next(use_rnn=True)
    persist = seg.topics[0].predict_next(use_rnn=False)
    # RNN prediction is a real (unit-norm) vector distinct from persistence
    assert np.linalg.norm(rnn_pred) > 0.5
    assert not np.allclose(rnn_pred, persist)


def test_per_topic_model_is_independent() -> None:
    """Each topic owns its model — different object identities."""
    seg = HiOnTopSegmenterV336(dim=DIM, alpha=1.0, lmda=0.0)
    seg.assign(_unit(1, 0, 0, 0, 0, 0, 0, 0))
    seg.assign(_unit(0, 1, 0, 0, 0, 0, 0, 0))  # likely a new topic (λ=0)
    if len(seg.topics) >= 2:
        assert seg.topics[0].model is not seg.topics[1].model


def test_reproducible_with_seed() -> None:
    """Same seed + same inputs → identical assignment sequence."""
    rng = np.random.default_rng(42)
    seq = [rng.standard_normal(DIM) for _ in range(25)]
    seq = [v / np.linalg.norm(v) for v in seq]

    def run() -> list[tuple[int, bool]]:
        seg = HiOnTopSegmenterV336(dim=DIM, alpha=1.0, lmda=10.0, seed=7)
        return [seg.assign(v) for v in seq]

    assert run() == run()


def test_invalid_rnn_ready_rejected() -> None:
    with pytest.raises(ValueError):
        HiOnTopSegmenterV336(dim=DIM, rnn_ready_min_transitions=0)
    with pytest.raises(ValueError):
        TopicV336(topic_id=0, dim=DIM, n_epochs=0)


def test_assign_api_contract() -> None:
    seg = HiOnTopSegmenterV336(dim=DIM)
    k, bnd = seg.assign(_unit(1, 0, 0, 0, 0, 0, 0, 0))
    assert isinstance(k, int) and isinstance(bnd, bool)
    assert k == 0 and bnd is False
    assert seg.predict_topic(_unit(1, 0, 0, 0, 0, 0, 0, 0)) == 0
