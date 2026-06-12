"""Unit tests for v3.3.3-2 / v3.3.4-2 segmenters."""

from __future__ import annotations

import numpy as np
import pytest

from hi_ontop.sem_core_v333_2 import HiOnTopSegmenterV333_2
from hi_ontop.sem_core_v334_2 import HiOnTopSegmenterV334_2


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _two_topic_seq(n_each: int = 5, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = _unit(rng.standard_normal(dim))
    b = _unit(rng.standard_normal(dim))
    seq = []
    for _ in range(n_each):
        seq.append(_unit(a + 0.1 * rng.standard_normal(dim)))
    for _ in range(n_each):
        seq.append(_unit(b + 0.1 * rng.standard_normal(dim)))
    return seq


# ------------------------------------------------------------ v3.3.3-2


def test_v333_2_runs_and_returns_two_topics():
    seg = HiOnTopSegmenterV333_2(dim=16, alpha=10.0, lmda=10.0, cos_threshold=0.7)
    out = [seg.assign(s) for s in _two_topic_seq()]
    ks = [k for k, _ in out]
    assert len(set(ks)) >= 2  # at least two distinct topics
    assert any(b for _, b in out)  # at least one boundary


def test_v333_2_prototype_max_capped():
    seg = HiOnTopSegmenterV333_2(dim=16, f0_proto_max=2)
    rng = np.random.default_rng(1)
    for _ in range(30):
        seg.assign(_unit(rng.standard_normal(16)))
    for protos in seg._f0_protos:
        assert len(protos) <= 2


def test_v333_2_invalid_p_threshold():
    with pytest.raises(ValueError):
        HiOnTopSegmenterV333_2(dim=8, restart_p_threshold=0.0)
    with pytest.raises(ValueError):
        HiOnTopSegmenterV333_2(dim=8, restart_p_threshold=1.0)


# ------------------------------------------------------------ v3.3.4-2


def test_v334_2_runs_and_returns_two_topics():
    seg = HiOnTopSegmenterV334_2(dim=16, alpha=10.0, lmda=10.0, cos_threshold=0.7)
    out = [seg.assign(s) for s in _two_topic_seq()]
    ks = [k for k, _ in out]
    assert len(set(ks)) >= 2


def test_v334_2_shrinkage_blends_toward_prior():
    """With shrink_c high and few samples, σ_eff stays near σ_0²."""
    seg = HiOnTopSegmenterV334_2(
        dim=16, pe_var_shrink_c=100.0, pe_var_sigma0_sq=0.04, pe_var_min_samples=1
    )
    seg._ensure_topic_slot(0)
    seg._pe_var[0] = 0.20  # well above σ_0
    seg._pe_var_count[0] = 2
    eff = seg._sigma_sq_for(0)
    # heavy shrinkage → close to σ_0²
    assert abs(eff - 0.04) < 0.01


def test_v334_2_no_shrinkage_when_c_is_zero():
    seg = HiOnTopSegmenterV334_2(
        dim=16, pe_var_shrink_c=0.0, pe_var_sigma0_sq=0.04, pe_var_min_samples=1
    )
    seg._ensure_topic_slot(0)
    seg._pe_var[0] = 0.20
    seg._pe_var_count[0] = 2
    eff = seg._sigma_sq_for(0)
    assert abs(eff - 0.20) < 1e-6


def test_v334_2_robust_scale_option():
    seg = HiOnTopSegmenterV334_2(
        dim=16, pe_var_robust=True, pe_var_shrink_c=0.0, pe_var_min_samples=1
    )
    rng = np.random.default_rng(2)
    for _ in range(20):
        seg.assign(_unit(rng.standard_normal(16)))
    # robust scale path returns clipped value within bounds
    for k in range(len(seg.topics)):
        v = seg._sigma_sq_for(k)
        assert seg.pe_var_min_sq <= v <= seg.pe_var_max_sq
