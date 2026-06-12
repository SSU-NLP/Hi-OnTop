from __future__ import annotations

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.sem_core_v331_rnn import HiOnTopSegmenterV331
from hi_ontop.topic_v331_rnn import TopicV33RNN


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_event_rnn_forward_returns_unit_prediction():
    torch.manual_seed(0)
    model = EventRNN(input_dim=8, hidden_dim=4)
    x = torch.randn(2, 3, 8)
    y = model(x)
    assert tuple(y.shape) == (2, 8)
    assert torch.allclose(torch.linalg.norm(y, dim=-1), torch.ones(2), atol=1e-6)


def test_topic_v331_update_trains_and_tracks_centroid():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    topic = TopicV33RNN(
        topic_id=0,
        dim=8,
        hidden_dim=4,
        lr=1e-3,
        train_steps=1,
        max_context=4,
    )
    s1 = _unit(rng.normal(size=8))
    s2 = _unit(s1 + 0.05 * rng.normal(size=8))
    topic.update(s1)
    topic.update(s2)
    pred = topic.predict_next()
    assert topic.n == 2
    assert np.isfinite(pred).all()
    assert np.isclose(np.linalg.norm(pred), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(topic.mu), 1.0, atol=1e-6)


def test_topic_v331_prediction_error_range_for_unit_vectors():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    topic = TopicV33RNN(topic_id=0, dim=8, hidden_dim=4, train_steps=1)
    seq = [_unit(rng.normal(size=8)) for _ in range(4)]
    for s in seq[:-1]:
        topic.update(s)
    pe = topic.prediction_error(seq[-1])
    score = topic.log_likelihood(seq[-1])
    assert np.isfinite(pe)
    assert 0.0 <= pe <= 2.0
    assert -1.0 <= score <= 1.0


def test_segmenter_v331_assigns_synthetic_sequence_without_nan():
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    seg = HiOnTopSegmenterV331(
        dim=16,
        alpha=10.0,
        lmda=1.0,
        tau=10.0,
        cos_threshold=0.2,
        beta=0.5,
        rnn_hidden_dim=4,
        rnn_train_steps=1,
        rnn_max_context=4,
    )
    path = []
    for _ in range(30):
        k, _ = seg.assign(_unit(rng.normal(size=16)))
        path.append(k)
    assert len(path) == 30
    assert int(seg.counts.sum()) == 30
    assert len(seg.topics) >= 1
    assert np.isfinite(seg.counts.astype(float)).all()


def test_segmenter_v331_predict_topic_does_not_mutate():
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    seg = HiOnTopSegmenterV331(dim=16, rnn_hidden_dim=4, rnn_train_steps=1)
    for _ in range(8):
        seg.assign(_unit(rng.normal(size=16)))
    counts_before = seg.counts.copy()
    prev_k_before = seg.prev_k
    topic_ns_before = [t.n for t in seg.topics]
    _ = seg.predict_topic(_unit(rng.normal(size=16)))
    assert np.array_equal(seg.counts, counts_before)
    assert seg.prev_k == prev_k_before
    assert [t.n for t in seg.topics] == topic_ns_before
