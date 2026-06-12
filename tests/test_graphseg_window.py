"""Unit tests for ``hi_ontop.baselines.GraphSegWindowD``.

GloVe 6B.300d 가 무거우므로 *tiny in-memory embedding fixture* 사용 — random
deterministic vectors. 알고리즘 *논리 흐름* 검증 위주 (점수 절대값 검증 안 함).
"""

from __future__ import annotations

import numpy as np
import pytest

from hi_ontop.baselines.graphseg_window import (
    GraphSegWindowD,
    _sequential_merge,
)


@pytest.fixture(scope="module")
def tiny_glove():
    """deterministic random GloVe-like vocab. dim=8."""
    rng = np.random.default_rng(0)
    words = [
        # cluster A (food)
        "pizza", "pasta", "wine", "cheese", "bread",
        # cluster B (sports)
        "soccer", "basketball", "tennis", "baseball", "marathon",
        # cluster C (tech)
        "computer", "software", "internet", "code", "server",
    ]
    # 3 cluster 의 anchor 벡터에 노이즈 합쳐 cluster 내부 유사도 높게
    anchors = rng.normal(size=(3, 8))
    out: dict[str, np.ndarray] = {}
    for i, w in enumerate(words):
        cl = i // 5
        v = anchors[cl] + 0.15 * rng.normal(size=8)
        out[w] = v.astype(np.float32)
    return out


@pytest.fixture
def seg_factory(tiny_glove, monkeypatch):
    """Tiny-embedding GraphSegWindowD factory. POS filter 없이 단순화."""
    from hi_ontop.baselines import graphseg_window as gw

    def make(**kw):
        defaults = dict(
            window_d=6, sim_threshold=0.0, min_seg_size=2,
            freq_source="none", use_pos_filter=False,
        )
        defaults.update(kw)
        seg = GraphSegWindowD(**defaults)
        # tiny glove 주입 (lazy load 우회)
        seg._glove = tiny_glove
        seg._glove_dim = 8
        # IC table: uniform 1.0
        seg._ic_table = {"__UNK__": 1.0}
        return seg

    return make


# ----------------------------------------------- sequential_merge ----------
def test_sequential_merge_simple_blocks():
    # cliques = 두 5-node clique → 두 segment
    cliques = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    sim = np.zeros((10, 10))
    segs = _sequential_merge(cliques, 10, sim, min_seg=2)
    assert len(segs) == 2
    assert segs[0] == (0, 4)
    assert segs[1] == (5, 9)


def test_sequential_merge_absorbs_singleton():
    # 3 clique: [0,1,2,3], [4] (singleton), [5,6,7]
    cliques = [[0, 1, 2, 3], [5, 6, 7]]  # 4 는 어디에도 없음 (singleton)
    sim = np.zeros((8, 8))
    # 4 는 5 와 비슷 (right side 흡수)
    sim[4, 5] = 0.9; sim[5, 4] = 0.9
    sim[4, 3] = 0.1; sim[3, 4] = 0.1
    segs = _sequential_merge(cliques, 8, sim, min_seg=2)
    # singleton 4 가 right (5..7) 으로 흡수돼야 함
    assert any(s[0] <= 4 <= s[1] and s[1] >= 5 for s in segs)


def test_sequential_merge_min_seg():
    cliques = [[0, 1], [2], [3, 4, 5]]  # 2 는 singleton
    sim = np.ones((6, 6)) * 0.5
    segs = _sequential_merge(cliques, 6, sim, min_seg=2)
    # min_seg=2 인데 [2] 가 singleton → 흡수돼야
    assert all(s[1] - s[0] + 1 >= 1 for s in segs)


# ----------------------------------------- GraphSegWindowD push/flush ---
def test_warmup_empty_under_window(seg_factory):
    seg = seg_factory()
    # window_d=6: 1~5 발화 동안은 평가 없음
    for u in ["pizza pasta", "wine cheese", "bread pizza"]:
        assert seg.push(u) == []


def test_window_eval_triggers_at_window_d(seg_factory):
    seg = seg_factory()
    # 6 발화 — 한 cluster 만 → boundary 없을 가능성 (under-seg)
    for u in ["pizza pasta", "wine cheese", "bread pizza",
              "pasta wine", "cheese bread", "pizza wine"]:
        seg.push(u)
    # 평가는 됐어야 함
    s = seg.state()
    assert s["t"] == 6


def test_two_clusters_detect_boundary(seg_factory):
    seg = seg_factory()
    # 앞 6 = food, 뒤 6 = sports
    utts = (["pizza pasta", "wine cheese", "bread pizza",
             "pasta wine", "cheese bread", "pizza wine"]
            + ["soccer basketball", "tennis baseball", "marathon soccer",
               "basketball tennis", "baseball marathon", "soccer tennis"])
    all_bs = []
    for u in utts:
        all_bs.extend(seg.push(u))
    all_bs.extend(seg.flush())
    # 두 cluster 사이 어딘가에 boundary 가 있어야 함
    assert all_bs, "두 cluster 전환 시 boundary 1개 이상 기대"


def test_deterministic(seg_factory):
    utts = ["pizza pasta", "wine cheese", "bread pizza", "pasta wine",
            "cheese bread", "pizza wine", "soccer basketball", "tennis baseball"]
    runs = []
    for _ in range(2):
        seg = seg_factory()
        bs = []
        for u in utts:
            bs.extend(seg.push(u))
        bs.extend(seg.flush())
        runs.append(tuple(bs))
    assert runs[0] == runs[1], runs


def test_state_keys(seg_factory):
    seg = seg_factory()
    for u in ["pizza", "soccer", "code"]:
        seg.push(u)
    s = seg.state()
    for key in ("t", "n_utts", "last_confirmed_1b", "glove_loaded",
                "glove_vocab"):
        assert key in s, key
    assert s["t"] == 3
    assert s["glove_loaded"] is True
    assert s["glove_vocab"] > 0


def test_invalid_freq_source_raises():
    with pytest.raises(ValueError):
        GraphSegWindowD(freq_source="zipf")
