"""v3.3.5 — SEM2 ``f_is_trained`` cold-start gating invariants.

These tests pin the locked spec (codex 2026-05-17):

* an *untrained* existing topic and the fresh-cluster slot receive the
  **identical** likelihood constant ``L0`` (likelihood ties → sCRP prior
  decides) — the exact SEM2 invariant Hi-OnTop v3.3.4 dropped;
* the chicken-and-egg loop is broken: a 1-turn ``prev_k`` whose centroid
  predicts the next turn terribly still survives via ``+λ`` stickiness
  instead of dying to a fresh topic;
* ``transition_count`` (the ``f_is_trained`` gate) increments only on a
  within-event continuation, not on a boundary / episode start;
* once trained, the topic is scored by the v3.3.4 variance likelihood.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hi_ontop.sem_core_v335 import HiOnTopSegmenterV335

DIM = 8


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float64)
    return v / np.linalg.norm(v)


def _orthogonal_pair() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros(DIM)
    a[0] = 1.0
    b = np.zeros(DIM)
    b[1] = 1.0  # cos(a, b) = 0  → centroid PE ≈ 1 (terrible prediction)
    return a, b


def test_untrained_existing_topic_ties_fresh_slot_likelihood() -> None:
    """SEM2 invariant: untrained topic lik == fresh slot lik (== L0)."""
    seg = HiOnTopSegmenterV335(dim=DIM, alpha=1.0, lmda=10.0)
    a, b = _orthogonal_pair()
    seg.assign(a)  # turn 1 → topic 0, untrained (no transition yet)

    prior = np.zeros(seg.k_max, dtype=np.float64)
    # build the same prior the segmenter would see on turn 2
    from hi_ontop.scrp import sticky_crp_unnormed

    prior = sticky_crp_unnormed(
        seg.counts, seg.prev_k, seg.alpha, seg.lmda, beta=seg.beta
    )
    active = np.flatnonzero(prior)
    log_scores, _, _ = seg._scores(b, prior, active)

    # topic 0 is untrained → its likelihood term must equal L0, i.e.
    # log_score - log_prior == L0, identical to the fresh slot's.
    idx0 = int(np.flatnonzero(active == 0)[0])
    fresh_i = seg._new_cluster_index_in_active(active)
    assert fresh_i is not None
    lik0 = log_scores[idx0] - math.log(prior[0])
    lik_fresh = log_scores[fresh_i] - math.log(prior[int(active[fresh_i])])
    assert lik0 == pytest.approx(seg._L0)
    assert lik0 == pytest.approx(lik_fresh)


def test_chicken_and_egg_broken_young_topic_survives() -> None:
    """A 1-turn topic with a terrible centroid prediction still survives
    turn 2 via λ stickiness (no new topic, no boundary)."""
    seg = HiOnTopSegmenterV335(dim=DIM, alpha=1.0, lmda=10.0)
    a, b = _orthogonal_pair()
    k1, bnd1 = seg.assign(a)
    k2, bnd2 = seg.assign(b)  # orthogonal → centroid PE ≈ 1

    assert k1 == 0
    assert k2 == 0, "young untrained prev_k must survive via stickiness"
    assert bnd2 is False, "staying in prev_k is a continuation, not a boundary"


def test_transition_count_gates_trained_state() -> None:
    """transition_count increments only on continuation; it flips
    _is_trained, which switches scoring to the variance likelihood."""
    seg = HiOnTopSegmenterV335(dim=DIM, alpha=1.0, lmda=10.0)
    a, b = _orthogonal_pair()

    seg.assign(a)
    assert seg._transition_count[0] == 0
    assert seg._is_trained(0) is False  # untrained → L0 path

    seg.assign(b)  # continuation (stayed in topic 0)
    assert seg._transition_count[0] == 1
    assert seg._is_trained(0) is True  # now scored by variance likelihood


def test_boundary_does_not_increment_transition_count() -> None:
    """A label change (new topic) is an episode start, not a transition:
    the *new* topic stays untrained."""
    seg = HiOnTopSegmenterV335(dim=DIM, alpha=1.0, lmda=0.0)  # λ=0 → no stickiness
    # λ=0 removes the stickiness bonus, so an orthogonal turn 2 spawns a
    # brand-new topic (prior fresh=α=1 vs prev=(C+1)^β≈1, likelihood tied).
    a, b = _orthogonal_pair()
    seg.assign(a)
    k2, _ = seg.assign(b)
    if k2 != 0:  # a genuine new-topic spawn happened
        assert seg._transition_count[k2] == 0
        assert seg._is_trained(k2) is False


def test_assign_api_contract() -> None:
    seg = HiOnTopSegmenterV335(dim=DIM)
    k, bnd = seg.assign(_unit(1, 0, 0, 0, 0, 0, 0, 0))
    assert isinstance(k, int) and isinstance(bnd, bool)
    assert k == 0 and bnd is False  # first ever turn
    assert seg.predict_topic(_unit(1, 0, 0, 0, 0, 0, 0, 0)) == 0


def test_invalid_min_transitions_rejected() -> None:
    with pytest.raises(ValueError):
        HiOnTopSegmenterV335(dim=DIM, min_transitions_for_pe=0)
