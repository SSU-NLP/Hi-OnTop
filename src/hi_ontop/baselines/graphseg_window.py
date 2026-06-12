"""GraphSeg window-local maximal clique baseline (bounded-lookahead, delay=d).

원본 GraphSeg (Glavaš et al. 2016, "Unsupervised Text Segmentation Using
Semantic Relatedness Graphs"; 구현 = Dobatymo/graphseg-python) 의 sentence
similarity (IC × GloVe + Hungarian) + Bron-Kerbosch maximal clique +
sequential merge 3-phase 를 *window 안에서만* 적용. boundary 는 window head
위치 (t-d+1) 에 대해서만 확정.

**codex 2026-05-21 검증**: 전역 graph 본질이 양보됨 → 강한 `-online` 명명
**금지**. 정직 명명 = ``GraphSeg-inspired bounded-window`` (short:
``GraphSeg-window-d``). **AUXILIARY only**.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


_DEFAULT_STOP = frozenset(
    """a an and are as at be but by for from has have he her him his i if in
    is it its me my no not of on or our she so that the their them they this
    to was we were what when where which who will with you your""".split()
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_CONTENT_TAGS = frozenset({
    "NN", "NNS", "NNP", "NNPS",       # nouns
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # verbs
    "JJ", "JJR", "JJS",                # adjectives
})


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class GraphSegWindowD:
    """GraphSeg-inspired bounded-window segmenter (delay=d).

    Args:
        window_d: 평가 window 크기 (sentence/utterance 단위). delay 도 d.
        sim_threshold: edge 채택 threshold (sim > τ).
        min_seg_size: sequential merge phase-3 의 최소 segment 크기.
        glove_path: GloVe txt 파일 경로 (lazy load).
        freq_source: word frequency 출처. ``"brown"`` (NLTK brown corpus,
            default) 또는 ``"none"`` (모든 단어 uniform IC=1).
        use_pos_filter: content-word (noun/verb/adj) 만 사용.
        stop_words: stopword set. None → 내장 영문 set.

    Notes:
        push(utterance) → list[int] (1-based boundary indices, lag-emission).
        flush() 는 잔여 utterance 강제 처리.
        per-turn 비용 O(d²) — Bron-Kerbosch 는 sparse window 에서 실용적.
    """

    window_d: int = 10
    sim_threshold: float = 0.25
    min_seg_size: int = 3
    glove_path: str = "benchmarks/glove/glove.6B.300d.txt"
    freq_source: str = "brown"
    use_pos_filter: bool = True
    stop_words: frozenset[str] = field(default_factory=lambda: _DEFAULT_STOP)

    # internal state -----------------------------------------------------------
    _utts: list[str] = field(default_factory=list, init=False)
    _utt_tokens: list[list[str]] = field(default_factory=list, init=False)  # content-filtered tokens per utt
    _utt_word_ic: list[dict[str, float]] = field(default_factory=list, init=False)  # {word: ic_weight}
    _t: int = field(default=0, init=False)
    _last_confirmed_1b: int = field(default=0, init=False)  # 마지막으로 확정된 segment 시작 1-based idx (=1 초기)
    _glove: Optional[dict[str, np.ndarray]] = field(default=None, init=False)
    _glove_dim: int = field(default=300, init=False)
    _ic_table: Optional[dict[str, float]] = field(default=None, init=False)
    _neural_sec: float = field(default=0.0, init=False)  # cumulative GloVe vector-op time
    # Preprocess = tokenize·POS·stopword·GloVe lookup·IC map·vector stack (everything
    # *before* the similarity-graph computation). Cosine matrix·clique·merge stay
    # outside this counter — those are the segmentation decision logic.
    _preprocess_sec: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.freq_source not in ("brown", "none"):
            raise ValueError(f"freq_source must be 'brown' or 'none', got {self.freq_source!r}")

    # ---- API -----------------------------------------------------------------
    def push(self, utterance: str) -> list[int]:
        """발화 1개 입력. 이번 호출에서 새로 확정된 boundary utterance index
        (1-based) list 반환."""
        self._t += 1
        self._utts.append(utterance)
        self._ensure_resources()
        toks, ic_map = self._prepare_utt(utterance)
        self._utt_tokens.append(toks)
        self._utt_word_ic.append(ic_map)
        emitted: list[int] = []
        # window 가 가득 차면 평가
        if self._t >= self.window_d:
            self._evaluate_window(emitted)
        return emitted

    def flush(self) -> list[int]:
        """대화 종료. 잔여 utterance 로 마지막 partial window 한 번 더 평가
        + 남은 window 안의 모든 boundary 확정."""
        emitted: list[int] = []
        if self._t < 2:
            return emitted
        self._evaluate_window(emitted, finalize=True)
        return emitted

    def state(self) -> dict:
        return dict(
            t=self._t,
            n_utts=len(self._utts),
            last_confirmed_1b=self._last_confirmed_1b,
            glove_loaded=self._glove is not None,
            glove_vocab=len(self._glove) if self._glove else 0,
        )

    # ---- internals -----------------------------------------------------------
    def _ensure_resources(self) -> None:
        if self._glove is None:
            self._glove, self._glove_dim = _load_glove(self.glove_path)
        if self._ic_table is None:
            self._ic_table = _build_ic_table(self.freq_source)

    def _prepare_utt(self, text: str) -> tuple[list[str], dict[str, float]]:
        # All of _prepare_utt is preprocess (tokenization → POS → stopword →
        # GloVe-membership lookup → IC-weight lookup). Wrap the whole thing
        # so we don't have to chase every branch.
        _pp0 = time.perf_counter()
        toks_all = _tokenize(text)
        if self.use_pos_filter and toks_all:
            import nltk
            tags = nltk.pos_tag(toks_all)
            _t0 = time.perf_counter()
            kept = [w for (w, t) in tags
                    if t in _CONTENT_TAGS and w not in self.stop_words
                    and len(w) > 1 and w in self._glove]
            self._neural_sec += time.perf_counter() - _t0
        else:
            _t0 = time.perf_counter()
            kept = [w for w in toks_all
                    if w not in self.stop_words and len(w) > 1
                    and w in self._glove]
            self._neural_sec += time.perf_counter() - _t0
        # ic_map: 단어별 IC weight (1회 lookup 후 cache)
        ic_map: dict[str, float] = {}
        for w in set(kept):
            ic_map[w] = self._ic_table.get(w, self._ic_table.get("__UNK__", 1.0))
        self._preprocess_sec += time.perf_counter() - _pp0
        return kept, ic_map

    def _utt_pair_similarity(self, i: int, j: int) -> float:
        """utt_i, utt_j (0-based, into buffers) 간 IC × GloVe cosine
        + Hungarian 매칭."""
        ti = self._utt_tokens[i]
        tj = self._utt_tokens[j]
        if not ti or not tj:
            return 0.0
        # 비용행렬: 1 - ic_weighted_cosine
        n, m = len(ti), len(tj)
        # neural-forward: GloVe vector lookup + stack + cosine ops (legacy timer)
        _t0 = time.perf_counter()
        # Preprocess: GloVe vector LOOKUP (fetch d-dim vectors per kept token).
        # The lookup is the last "preprocess" step per Hi-OnTop's column definition;
        # the cosine matrix that follows belongs to the similarity-graph segmentation.
        _pp0 = time.perf_counter()
        vec_i = np.stack([self._glove[w] for w in ti])  # (n, d)
        vec_j = np.stack([self._glove[w] for w in tj])  # (m, d)
        self._preprocess_sec += time.perf_counter() - _pp0
        # cosine matrix (n × m) — similarity graph edge weights (segmentation logic)
        nn = vec_i / (np.linalg.norm(vec_i, axis=1, keepdims=True) + 1e-12)
        mm = vec_j / (np.linalg.norm(vec_j, axis=1, keepdims=True) + 1e-12)
        cos = nn @ mm.T  # (n, m)
        self._neural_sec += time.perf_counter() - _t0
        # IC weight per pair = min(ic_i, ic_j) — Glavaš 식 변형 (보존적)
        ic_i = np.array([self._utt_word_ic[i][w] for w in ti])
        ic_j = np.array([self._utt_word_ic[j][w] for w in tj])
        w_mat = np.minimum(ic_i[:, None], ic_j[None, :])
        # cost = -ic_weighted_cosine (Hungarian = 최소화)
        cost = -(cos * w_mat)
        # Hungarian — rectangular matrix 도 지원
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        matched_score = float(-cost[ri, ci].sum())
        # normalize by min(n, m) of IC-weighted potential
        norm = float(w_mat.max(axis=0).sum() + 1e-12) if m <= n else \
            float(w_mat.max(axis=1).sum() + 1e-12)
        sim = matched_score / norm if norm > 0 else 0.0
        return max(0.0, min(1.0, sim))

    def _evaluate_window(self, emitted: list[int], finalize: bool = False) -> None:
        """현 window 의 모든 utterance 쌍 similarity → graph → Bron-Kerbosch
        → sequential merge → boundary 확정."""
        if finalize:
            # 대화 종료: 잔여 [last_confirmed_1b .. t] 전체 (이미 확정 안 된 구간)
            window_start_1b = max(1, self._last_confirmed_1b)
            window_end_1b = self._t
        else:
            # 일반: 가장 최근 window_d 발화 (head ~ tail)
            window_end_1b = self._t
            window_start_1b = max(1, window_end_1b - self.window_d + 1)
        # 이미 다 확정된 경우 skip
        if window_start_1b > self._t:
            return
        n_w = window_end_1b - window_start_1b + 1
        if n_w < 2:
            return
        # window 안 sim matrix
        sim = np.zeros((n_w, n_w), dtype=float)
        for a in range(n_w):
            for b in range(a + 1, n_w):
                i = window_start_1b - 1 + a  # 0-based into buffers
                j = window_start_1b - 1 + b
                s = self._utt_pair_similarity(i, j)
                sim[a, b] = s
                sim[b, a] = s
        # graph
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n_w))
        for a in range(n_w):
            for b in range(a + 1, n_w):
                if sim[a, b] > self.sim_threshold:
                    G.add_edge(a, b, weight=sim[a, b])
        # Bron-Kerbosch maximal cliques
        cliques = [sorted(c) for c in nx.find_cliques(G)]
        # sequential merge 3-phase
        segments = _sequential_merge(cliques, n_w, sim, self.min_seg_size)
        # segments → boundary indices (window-local, then offset)
        # segments = list of (start_a, end_a)  (0-based, inclusive)
        # boundary BEFORE segment[i] (i ≥ 1) → utt index = window_start_1b + start_a
        new_bounds_1b: list[int] = []
        for seg in segments[1:]:  # 첫 segment 는 boundary 없음
            start_a = seg[0]
            new_bounds_1b.append(window_start_1b + start_a)
        # window 안에서 발견된 boundary 를 *비가역* 으로 emit (이전에 확정 안 된 것만).
        # 'delay' 의 의미 = window=d 의 lookahead 가 확보돼야 평가 시작. boundary 가
        # window 내 어디에 있는지는 알고리즘에 맡김. 새로 발견된 것만 추가.
        for b in new_bounds_1b:
            if b > self._last_confirmed_1b and 2 <= b <= self._t:
                emitted.append(b)
                self._last_confirmed_1b = b


# -------------------------------------------------------- helpers ----------
_GLOVE_CACHE: dict[str, tuple[dict[str, np.ndarray], int]] = {}


def _load_glove(path: str) -> tuple[dict[str, np.ndarray], int]:
    """GloVe txt → dict[word → vec]. 메모리 공유 cache.

    Streams line-by-line (~990MB 6B.300d 도 peak ~700MB only,
    splitlines() 안 함). 7-8GB RAM 환경 + swap 압박 시에도 OK.
    """
    if path in _GLOVE_CACHE:
        return _GLOVE_CACHE[path]
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"GloVe 파일 없음: {p}. 다운로드 안내는 methods/README.md 참조.")
    vec: dict[str, np.ndarray] = {}
    dim: int | None = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 3:
                continue
            word = parts[0]
            try:
                v = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if dim is None:
                dim = v.size
            elif v.size != dim:
                continue
            vec[word] = v
    if dim is None:
        raise RuntimeError(f"GloVe 파싱 실패: {p}")
    _GLOVE_CACHE[path] = (vec, dim)
    return vec, dim


_IC_CACHE: dict[str, dict[str, float]] = {}


def _build_ic_table(source: str) -> dict[str, float]:
    """word → IC weight. brown corpus 빈도 기반 (codex 권고 NLTK brown)."""
    if source in _IC_CACHE:
        return _IC_CACHE[source]
    if source == "none":
        table = {"__UNK__": 1.0}
        _IC_CACHE[source] = table
        return table
    from nltk.corpus import brown
    counts: Counter = Counter()
    for w in brown.words():
        w = w.lower()
        if _TOKEN_RE.fullmatch(w):
            counts[w] += 1
    total = sum(counts.values())
    table: dict[str, float] = {}
    for w, c in counts.items():
        p = c / total
        table[w] = -math.log(p)
    # UNK ≈ rare-end IC
    if counts:
        table["__UNK__"] = -math.log(1.0 / total)
    else:
        table["__UNK__"] = 1.0
    _IC_CACHE[source] = table
    return table


def _sequential_merge(
    cliques: list[list[int]], n: int, sim: np.ndarray, min_seg: int
) -> list[tuple[int, int]]:
    """GraphSeg sequential merge 3-phase (window-local 변형).

    Args:
        cliques: maximal cliques (node id list).
        n: window 크기.
        sim: n×n similarity matrix.
        min_seg: 최소 segment 크기.

    Returns:
        segments: [(start_a, end_a)] (0-based, inclusive), contiguous in [0, n-1].
    """
    if n == 0:
        return []
    # node → clique 멤버십 set (가장 큰 clique 우선 배정)
    assign: list[int] = [-1] * n
    cliques_sorted = sorted(cliques, key=lambda c: -len(c))
    for cid, c in enumerate(cliques_sorted):
        for v in c:
            if assign[v] == -1:
                assign[v] = cid
    # Phase 1: 인접 same-clique → 같은 segment
    # contiguous run of identical assign[v] (and v that is in non-singleton clique)
    # Phase 2: singleton (clique 미배정 = assign=-1, 또는 1-node clique) 흡수
    # Phase 3: min_seg 미만 segment 병합
    # 구현: 먼저 contiguous run 으로 segment 만들고 후처리.
    segments: list[list[int]] = []
    for v in range(n):
        if not segments:
            segments.append([v])
            continue
        prev_seg = segments[-1]
        prev_v = prev_seg[-1]
        if assign[v] != -1 and assign[v] == assign[prev_v]:
            prev_seg.append(v)
        else:
            segments.append([v])
    # Phase 2: singleton segment (len==1) 를 양쪽 중 더 유사한 segment 로 흡수
    changed = True
    while changed:
        changed = False
        for k in range(len(segments)):
            if len(segments[k]) == 1:
                v = segments[k][0]
                left_sim = sim[v, segments[k - 1][-1]] if k - 1 >= 0 else -1.0
                right_sim = sim[v, segments[k + 1][0]] if k + 1 < len(segments) else -1.0
                if left_sim < 0 and right_sim < 0:
                    continue
                if left_sim >= right_sim and k - 1 >= 0:
                    segments[k - 1].append(v)
                else:
                    segments[k + 1].insert(0, v)
                segments.pop(k)
                changed = True
                break
    # Phase 3: min_seg 미만 segment 를 더 유사한 이웃으로 흡수
    changed = True
    while changed:
        changed = False
        for k in range(len(segments)):
            if len(segments[k]) < min_seg and len(segments) > 1:
                left_idx = segments[k - 1][-1] if k - 1 >= 0 else None
                right_idx = segments[k + 1][0] if k + 1 < len(segments) else None
                anchor = segments[k][0]
                left_sim = sim[anchor, left_idx] if left_idx is not None else -1.0
                right_sim = sim[anchor, right_idx] if right_idx is not None else -1.0
                if left_sim >= right_sim and k - 1 >= 0:
                    segments[k - 1].extend(segments[k])
                else:
                    segments[k + 1] = segments[k] + segments[k + 1]
                segments.pop(k)
                changed = True
                break
    return [(seg[0], seg[-1]) for seg in segments]
