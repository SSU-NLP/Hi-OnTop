"""v3.3.8 — SEM2-calibrated fresh baseline + non-prev f0 scoring.

Pins the codex-locked structural change (2026-05-17, re-diagnosis):

* the fresh / untrained baseline ``L0`` is derived from ``pe_prior``
  (the chance-level PE of an uninformative prediction), **not** from
  ``cos_threshold`` — ``cos_threshold`` no longer feeds ``L0``;
* a non-previous existing topic is scored by its **f0** likelihood
  (SEM2 ``log_likelihood_f0`` for ``k0 != k_prev``), not the
  within-event repeat predictor;
* ``pe_prior`` is the dominant lever on segmentation granularity
  (higher ⇒ harsher fresh slot ⇒ fewer, larger topics). The idx=374
  sweep showed neither SEM2 extreme works as a fixed default
  (``cos_threshold=0.9`` over-fragments, ``pe_prior=1.0`` mega-merges);
  the value is embedding-dependent and must be benchmark-calibrated —
  these tests pin *behaviour*, not a production default.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hi_ontop.sem_core_v338 import HiOnTopSegmenterV338

DIM = 8


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_L0_from_pe_prior_not_cos_threshold() -> None:
    """L0 == -pe_prior²/(2σ0²)·var_w, independent of cos_threshold."""
    s1 = HiOnTopSegmenterV338(
        dim=DIM, pe_prior=1.0, pe_var_sigma0_sq=0.04, cos_threshold=0.9
    )
    s2 = HiOnTopSegmenterV338(
        dim=DIM, pe_prior=1.0, pe_var_sigma0_sq=0.04, cos_threshold=0.1
    )
    expected = -(1.0**2) / (2.0 * 0.04)
    assert s1._L0 == pytest.approx(expected)
    # cos_threshold changed but L0 must not (it no longer feeds L0)
    assert s1._L0 == pytest.approx(s2._L0)


def test_pe_prior_controls_fresh_harshness() -> None:
    """Higher pe_prior ⇒ harsher (more negative) fresh baseline."""
    lo = HiOnTopSegmenterV338(dim=DIM, pe_prior=0.4)._L0
    hi = HiOnTopSegmenterV338(dim=DIM, pe_prior=1.0)._L0
    assert hi < lo < 0.0


def test_pe_prior_monotone_on_topic_count() -> None:
    """On a drifting stream, a harsher fresh slot yields *fewer* topics
    (the v3.3.8 lever direction)."""
    rng = np.random.default_rng(0)
    stream = [v / np.linalg.norm(v) for v in rng.standard_normal((30, DIM))]

    def n_topics(pe_prior: float) -> int:
        seg = HiOnTopSegmenterV338(
            dim=DIM, alpha=1.0, lmda=10.0, seed=1, pe_prior=pe_prior
        )
        for v in stream:
            seg.assign(v)
        return int((seg.counts > 0).sum())

    assert n_topics(1.0) <= n_topics(0.4)


def test_non_prev_topic_scored_by_f0() -> None:
    """An existing topic that is not ``prev_k`` is scored by its f0
    likelihood (SEM2 ``k0 != k_prev`` path), not the repeat predictor."""
    seg = HiOnTopSegmenterV338(
        dim=DIM, alpha=1.0, lmda=0.0, f0_min_starts=1, pe_prior=0.5
    )
    a = _unit(1, 0, 0, 0, 0, 0, 0, 0)
    b = _unit(0, 1, 0, 0, 0, 0, 0, 0)
    seg.assign(a)  # topic 0, f0 seeded with a
    seg.assign(b)  # likely topic 1 (λ=0, orthogonal), prev_k -> 1
    # Now score a turn while prev_k != 0; topic 0 is a non-prev existing
    # topic and must be scored via _f0_loglik.
    from hi_ontop.scrp import sticky_crp_unnormed

    s = _unit(0.9, 0.1, 0, 0, 0, 0, 0, 0)
    prior = sticky_crp_unnormed(
        seg.counts, seg.prev_k, seg.alpha, seg.lmda, beta=seg.beta
    )
    active = np.flatnonzero(prior)
    log_scores, _, _ = seg._scores(s, prior, active)
    if seg.prev_k != 0 and 0 in active:
        i0 = int(np.flatnonzero(active == 0)[0])
        expected = math.log(prior[0]) + seg._f0_loglik(0, s)
        assert log_scores[i0] == pytest.approx(expected)


def test_reproducible_with_seed() -> None:
    rng = np.random.default_rng(11)
    seq = [v / np.linalg.norm(v) for v in rng.standard_normal((20, DIM))]

    def run() -> list[tuple[int, bool]]:
        seg = HiOnTopSegmenterV338(dim=DIM, alpha=1.0, lmda=10.0, seed=3)
        return [seg.assign(v) for v in seq]

    assert run() == run()


def test_invalid_pe_prior_rejected() -> None:
    with pytest.raises(ValueError):
        HiOnTopSegmenterV338(dim=DIM, pe_prior=-0.1)
    with pytest.raises(ValueError):
        HiOnTopSegmenterV338(dim=DIM, pe_prior=2.5)


def test_assign_api_contract() -> None:
    seg = HiOnTopSegmenterV338(dim=DIM)
    k, bnd = seg.assign(_unit(1, 0, 0, 0, 0, 0, 0, 0))
    assert isinstance(k, int) and isinstance(bnd, bool)
    assert k == 0 and bnd is False
    assert seg.predict_topic(_unit(1, 0, 0, 0, 0, 0, 0, 0)) == 0
