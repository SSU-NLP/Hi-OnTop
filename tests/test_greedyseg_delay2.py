"""Unit tests for ``hi_ontop.baselines.GreedySegOnlineDelay2``.

BERT 추론은 무거우므로 tiny model (``sshleifer/tiny-distilbert-base-cased``)
fixture 로 검증. *논리 흐름* 검증 위주 — 점수 절대값 검증 안 함.
"""

from __future__ import annotations

import pytest

from hi_ontop.baselines import GreedySegOnlineDelay2

TINY = "sshleifer/tiny-distilbert-base-cased"


@pytest.fixture(scope="module")
def seg_factory():
    """tiny model 로 빠르게 인스턴스 만드는 factory."""

    def make(**kw):
        defaults = dict(
            backbone=TINY,
            device="cpu",
            max_seq_length=16,
        )
        defaults.update(kw)
        return GreedySegOnlineDelay2(**defaults)

    return make


def test_warmup_empty_until_right_context(seg_factory):
    seg = seg_factory()
    # 첫 jump_step-1+1+window_size = 4 발화 까지는 첫 후보 (i_rel=1) 의 right_sent
    # 미확보 → boundary 0. (window_size=2 이므로 t=4 부터 i_rel=1 평가 가능)
    out = []
    for i, u in enumerate(["a b c", "d e", "f g", "h i"]):
        bs = seg.push(u)
        out.append((i + 1, bs))
    # 4 발화 모두 boundary 0 이어야 함 (max_seg_round=8 까지 누적 전)
    assert all(bs == [] for _, bs in out), out


def test_emits_after_max_seg_round(seg_factory):
    seg = seg_factory()
    # MAX_SEGMENT_ROUND=8, JUMP_STEP=2 → 후보 i_rel ∈ {1,3,5,7}
    # 마지막 후보 평가에 필요한 t = cut_index + 7 + window_size = 0 + 7 + 2 = 9
    bs_all = []
    for i, u in enumerate([f"word{i} word{i}" for i in range(10)]):
        bs_all.extend(seg.push(u))
    # t=9 시점에 i_rel=7 까지 evaluable → segment finalize → boundary 1개 이상
    assert bs_all, "9 발화 입력 후 boundary 1개 이상 기대"


def test_deterministic_same_input(seg_factory):
    runs = []
    for _ in range(2):
        seg = seg_factory()
        bs_all = []
        utts = [f"alpha beta {i}" for i in range(15)]
        for u in utts:
            bs_all.extend(seg.push(u))
        bs_all.extend(seg.flush())
        runs.append(tuple(bs_all))
    assert runs[0] == runs[1], runs


def test_state_keys(seg_factory):
    seg = seg_factory()
    for u in ["hello", "world", "test", "input", "five", "six", "seven", "eight",
              "nine", "ten"]:
        seg.push(u)
    s = seg.state()
    for key in ("t", "cut_index", "n_candidates", "device",
                "bert_forwards", "last_boundary"):
        assert key in s, key
    assert s["t"] == 10
    assert s["device"] == "cpu"
    assert s["bert_forwards"] > 0


def test_flush_handles_partial_segment(seg_factory):
    seg = seg_factory()
    # 5 발화만 — max_seg_round 못 채움. push 만으론 boundary 0.
    for u in ["a", "b", "c", "d", "e"]:
        bs = seg.push(u)
        assert bs == []
    out = seg.flush()
    # flush 가 force_end=True 로 잔여 후보 평가 + argmin → boundary 1개 또는 0개
    # (i_rel=1 만 평가 가능: cut_index=0, i_rel=1, right_hi=min(5, 4)=4 > 3 → OK)
    assert isinstance(out, list)


def test_invalid_device_raises():
    with pytest.raises(ValueError):
        GreedySegOnlineDelay2(device="xpu").push("hi")
