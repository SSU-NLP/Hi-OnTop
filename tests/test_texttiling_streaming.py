"""Unit tests for ``hi_ontop.baselines.StreamingTextTiling``."""

from __future__ import annotations

import random
import statistics

import pytest

from hi_ontop.baselines.texttiling_streaming import (
    StreamingTextTiling,
    _Welford,
    _cosine,
    _tokenize,
)


# ------------------------------------------------------------- helpers (tests)
def _bag(label: str, n: int = 10) -> str:
    """label 한 단어를 n 번 반복한 발화."""
    return " ".join([label] * n)


# ---------------------------------------------------------------- welford ---
def test_welford_matches_offline_mean_std():
    vals = [random.Random(0).random() for _ in range(50)]
    w = _Welford()
    for v in vals:
        w.push(v)
    assert w.n == 50
    assert w.mean == pytest.approx(statistics.fmean(vals), rel=1e-9)
    # statistics.stdev uses n-1; Welford here uses n-1 → 일치
    assert w.std() == pytest.approx(statistics.stdev(vals), rel=1e-9)


def test_welford_zero_one_sample():
    w = _Welford()
    assert w.std() == 0.0
    w.push(3.0)
    assert w.std() == 0.0  # n<2
    w.push(5.0)
    assert w.mean == pytest.approx(4.0)
    assert w.std() == pytest.approx(statistics.stdev([3.0, 5.0]))


# ------------------------------------------------------------ tokenize ---
def test_tokenize_lowercase_and_stopwords():
    out = _tokenize("The QUICK brown Fox.", frozenset({"the"}))
    assert out == ["quick", "brown", "fox"]


def test_tokenize_strips_single_char():
    out = _tokenize("a I you we he is at no", frozenset())
    # 1글자 'a'/'i' 는 길이 필터로 제외, stop=∅ 라도 길이>1 만 남음
    assert "a" not in out and "i" not in out


# --------------------------------------------------------------- cosine ---
def test_cosine_basic():
    from collections import Counter

    assert _cosine(Counter({"x": 1}), Counter({"x": 1})) == pytest.approx(1.0)
    assert _cosine(Counter({"x": 1}), Counter({"y": 1})) == 0.0
    assert _cosine(Counter(), Counter({"x": 1})) == 0.0


# ----------------------------------------------- StreamingTextTiling ---
def test_warmup_no_boundary_on_short_prefix():
    seg = StreamingTextTiling(w=5, k=2, warmup_gaps=3, min_gap=2)
    # 매우 짧은 prefix (3 발화) 동안은 절대 boundary 안 나옴
    for u in ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]:
        assert seg.push(u) == []


def test_no_boundary_when_topic_stays_constant():
    seg = StreamingTextTiling(w=8, k=3, warmup_gaps=3, min_gap=3, c=0.5)
    bs = []
    for _ in range(40):
        bs.extend(seg.push(_bag("food", 8)))
    bs.extend(seg.flush())
    # 단일 단어 반복 → cosine ≈ 1 → depth ≈ 0 → boundary 없어야 함
    assert bs == []


def test_topic_change_detected_near_transition():
    seg = StreamingTextTiling(w=6, k=3, warmup_gaps=2, min_gap=2, c=0.3)
    bs = []
    for _ in range(20):
        bs.extend(seg.push(_bag("food", 6)))
    transition_t = 20  # 21번째 발화부터 sports
    for _ in range(20):
        bs.extend(seg.push(_bag("sports", 6)))
    bs.extend(seg.flush())
    assert bs, "topic 변화에서 boundary 가 하나는 잡혀야 함"
    # 검출된 첫 boundary 가 transition 주변 (±10) 인지
    assert any(abs(b - (transition_t + 1)) <= 10 for b in bs), bs


def test_min_gap_suppression():
    seg = StreamingTextTiling(w=4, k=2, warmup_gaps=2, min_gap=5, c=0.0)
    bs = []
    # 매우 빠르게 topic 이 흔들리는 입력 → boundary 후보 잦음 → min_gap 으로 흡수
    for i in range(30):
        topic = "food" if (i // 3) % 2 == 0 else "sports"
        bs.extend(seg.push(_bag(topic, 4)))
    bs.extend(seg.flush())
    # min_gap=5 → 연속 두 boundary 사이 차이 ≥ 5
    diffs = [b2 - b1 for b1, b2 in zip(bs, bs[1:])]
    assert all(d >= 5 for d in diffs), (bs, diffs)


def test_deterministic_across_runs():
    runs = []
    for _ in range(3):
        seg = StreamingTextTiling(w=6, k=3, warmup_gaps=2, min_gap=2, c=0.3)
        bs = []
        for i in range(40):
            topic = "alpha" if i < 20 else "omega"
            bs.extend(seg.push(_bag(topic, 6)))
        bs.extend(seg.flush())
        runs.append(tuple(bs))
    assert len(set(runs)) == 1, runs


def test_state_keys():
    seg = StreamingTextTiling()
    for _ in range(15):
        seg.push("hello world test")
    s = seg.state()
    for key in (
        "t",
        "n_pseudo_sentence",
        "n_scored_gap",
        "mean_depth",
        "std_depth",
        "left_peak",
        "last_boundary_t",
    ):
        assert key in s
