"""``hi_ontop.dts_scoring`` (공식 DTS 채점기 래퍼) 단위 테스트.

검증:
1. F1 집계가 **per-dialogue 평균** (풀링 아님) — 공식 규약.
2. Score 공식 = 0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD).
3. ``signal_to_pred`` 의 끝-turn(-1) 정렬.
4. 래퍼가 레포 ``SegmentationEvaluation`` 과 동일 값.
"""

from __future__ import annotations

import pytest

from hi_ontop import dts_scoring as S


def test_f1_is_per_dialogue_not_pooled():
    # D1: 완벽(F1=1.0). D2: 경계 2개 중 1개 놓침(P=1,R=0.5→F1=0.6667).
    golds = [[0, 1, 0], [0, 1, 0, 1, 0]]
    preds = [[0, 1, 0], [0, 1, 0, 0, 0]]
    r = S.score_dialogues(golds, preds)
    # per-dialogue 평균 = (1.0 + 0.6667)/2 = 0.8333  (풀링이면 0.80)
    assert r["f1"] == pytest.approx(0.8333, abs=1e-3)
    assert r["f1"] != pytest.approx(0.80, abs=1e-3)
    assert r["n"] == 2


def test_score_formula():
    golds = [[0, 1, 0, 0]]
    preds = [[0, 1, 0, 0]]  # 완벽 → F1=1, Pk=0, WD=0 → Score=1.0
    r = S.score_dialogues(golds, preds)
    assert r["f1"] == pytest.approx(1.0)
    assert r["pk"] == pytest.approx(0.0)
    assert r["wd"] == pytest.approx(0.0)
    assert r["score"] == pytest.approx(0.5 * r["f1"] + 0.25 * (1 - r["pk"]) + 0.25 * (1 - r["wd"]))
    assert r["score"] == pytest.approx(1.0)


def test_signal_to_pred_end_turn_alignment():
    # 스파이크 t=1,3 → 경계 t-1=0,2. 마지막 turn 0.
    delta = [0.0, 0.9, 0.2, 0.95]
    assert S.signal_to_pred(delta, 0.5) == [1, 0, 1, 0]


def test_matches_repo_class_directly():
    golds = [[0, 1, 0, 1, 0], [0, 0, 1, 0]]
    preds = [[0, 1, 0, 0, 0], [0, 1, 0, 0]]
    # 래퍼
    r = S.score_dialogues(golds, preds)
    # 레포 클래스 직접 호출 (동일 force_last_zero 처리)
    ev = S.new_evaluation()
    for yt, yp in zip(golds, preds):
        yt, yp = list(yt), list(yp)
        yt[-1] = 0
        yp[-1] = 0
        ev.add(yt, yp)
    raw = ev.compute()
    assert r["f1"] == pytest.approx(raw["f1(binary)"])
    assert r["pk"] == pytest.approx(raw["pk"])
    assert r["wd"] == pytest.approx(raw["windowdiff"])
    assert r["score"] == pytest.approx(raw["total_score"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        S.score_dialogues([[0, 1, 0]], [[0, 1]])
