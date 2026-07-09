"""`hi_ontop.ami_scoring` (AMI 공식 채점기) 단위 테스트.

확정 스펙 (2026-06-15): ±2 tolerant binary F1 + nltk Pk/WD(auto) + Score, per-meeting 평균,
start-turn 정렬(shift 0). 레퍼런스 = `scripts/ami_adaptive_deneut_deploy.py::{ev,tol_f1}` (parity 검증됨).
"""

from __future__ import annotations

import pytest

from hi_ontop import ami_scoring as A


def test_tol_f1_tolerance():
    # ±2 안: 정답 취급
    assert A.tol_f1([5], [5]) == pytest.approx(1.0)
    assert A.tol_f1([5], [6]) == pytest.approx(1.0)   # |6-5|=1 ≤ 2
    assert A.tol_f1([5], [7]) == pytest.approx(1.0)   # |7-5|=2 ≤ 2
    # ±2 밖: 오답
    assert A.tol_f1([5], [8]) == pytest.approx(0.0)   # |8-5|=3 > 2
    assert A.tol_f1([], [3]) == 0.0
    assert A.tol_f1([3], []) == 0.0


def test_tol_f1_precision_recall():
    # gold 2개, pred 3개(2개 맞고 1개 멂) → P=2/3, R=2/2 → F1=2*(2/3)/(2/3+1)
    f = A.tol_f1([10, 20], [10, 20, 40])
    p, r = 2 / 3, 1.0
    assert f == pytest.approx(2 * p * r / (p + r))


def test_score_formula_perfect():
    # 완벽 예측 → F1=1, Pk=0, WD=0 → Score=1.0
    gold = [[0, 1, 0, 0, 1, 0]]
    pred = [[0, 1, 0, 0, 1, 0]]
    r = A.score_meetings(gold, pred)
    assert r["f1"] == pytest.approx(1.0)
    assert r["score"] == pytest.approx(0.5 * r["f1"] + 0.25 * (1 - r["pk"]) + 0.25 * (1 - r["wd"]))
    assert r["n"] == 1


def test_per_meeting_average():
    # 2 meeting 평균 (per-meeting macro)
    golds = [[0, 1, 0], [0, 1, 0, 1, 0]]
    preds = [[0, 1, 0], [0, 1, 0, 0, 0]]
    r = A.score_meetings(golds, preds)
    assert r["n"] == 2
    assert 0.0 <= r["score"] <= 1.0


def test_boundaries_to_pred_shift0():
    # 경계 index → per-turn 0/1, shift 0 (그 자리)
    assert A.boundaries_to_pred(5, [2, 4]) == [0, 0, 1, 0, 1]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        A.score_meetings([[0, 1, 0]], [[0, 1]])
