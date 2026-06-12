"""Hi-OnTop-CR (`src/hi_ontop/hi_ontop_cr.py`) — 핵심 동작 smoke/property 테스트.

데이터 비의존(synthetic 임베딩). 정식 수치 회귀(AMI 0.401/0.140)는 REPORT/parity 체크가 담당하고,
여기서는 알고리즘 불변식만 잠근다: index 유효성, 결정론, reset 모드, 명백한 경계 검출.
"""
from __future__ import annotations

import numpy as np
import pytest

from hi_ontop.hi_ontop_cr import segment, DEFAULTS, _beta


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _two_topic_emb(n_a=30, n_b=30, dim=384, seed=0, noise=0.25):
    """공통 global 성분(g) + topic 성분(u/v) + noise — 실제 임베딩 구조 모사. seam = n_a.

    de-neut 은 공통 g 를 제거하고 topic 변별만 보므로 실제처럼 g 가 있어야 신호가 의도대로 작동.
    within-topic noise 가 너무 작으면(균질) σ→0 으로 적응 임계치가 과민해지는 합성 artifact 가 생기므로
    실제 발화 수준의 spread(noise≈0.25) 를 준다.
    """
    rng = np.random.default_rng(seed)
    g = _unit(rng.standard_normal(dim))                      # 공통(중립) 방향
    u = _unit(rng.standard_normal(dim)); v = _unit(rng.standard_normal(dim))
    def cluster(topic, m):
        x = 0.7 * g + 0.6 * topic + noise * rng.standard_normal((m, dim))
        return _unit(x)
    return np.vstack([cluster(u, n_a), cluster(v, n_b)]), n_a


@pytest.mark.parametrize("reset", ["commit_refine", "threshold"])
def test_index_validity_and_sorted(reset):
    emb, _ = _two_topic_emb()
    pred = segment(emb, reset=reset)
    assert pred == sorted(pred)
    assert all(0 < i < len(emb) for i in pred)
    assert len(pred) == len(set(pred))


@pytest.mark.parametrize("reset", ["commit_refine", "threshold"])
def test_deterministic(reset):
    emb, _ = _two_topic_emb()
    assert segment(emb, reset=reset) == segment(emb, reset=reset)


@pytest.mark.parametrize("reset", ["commit_refine", "threshold"])
def test_detects_clear_seam(reset):
    """명백한 단일 화제전환은 seam(±tolerance) 근처에 경계가 잡혀야 한다."""
    emb, seam = _two_topic_emb(n_a=40, n_b=40)
    pred = segment(emb, reset=reset)
    assert pred, "경계를 하나도 못 잡음"
    # online localization 은 본질적으로 부정확(REPORT 격차) — seam 근방(±6) 검출이면 충분
    assert min(abs(p - seam) for p in pred) <= 6


def test_reset_modes_can_differ_invalid_raises():
    emb, _ = _two_topic_emb()
    with pytest.raises(ValueError):
        segment(emb, reset="nope")


def test_homogeneous_few_boundaries():
    """단일 화제(균질) 스트림은 경계가 거의 없어야 한다(과분절 방지)."""
    rng = np.random.default_rng(1)
    c = rng.standard_normal(384); c /= np.linalg.norm(c)
    x = c + 0.12 * rng.standard_normal((60, 384))
    emb = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert len(segment(emb, reset="commit_refine")) <= 3


def test_adaptive_beta_monotone_and_clipped():
    """β = clip(A−B·log(1+k/L0)): 짧은 seg→큰 β(→1), 긴 seg→작은 β, [0,1] clip."""
    A, B, L0 = DEFAULTS["A"], DEFAULTS["B"], DEFAULTS["L0"]
    b_short = _beta(1, A, B, L0); b_long = _beta(200, A, B, L0)
    assert b_short >= b_long
    assert 0.0 <= b_long <= b_short <= 1.0


def test_short_segments_higher_beta_mean():
    """run-length 적응이 의도대로: 짧은 k 평균 β > 긴 k 평균 β (sharp/drift 자동판별 근거)."""
    A, B, L0 = DEFAULTS["A"], DEFAULTS["B"], DEFAULTS["L0"]
    short = np.mean([_beta(k, A, B, L0) for k in range(1, 6)])
    long = np.mean([_beta(k, A, B, L0) for k in range(40, 80)])
    assert short > long
