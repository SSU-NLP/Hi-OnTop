"""Streaming TextTiling (online, prefix-causal, per-turn O(w)).

원본 TextTiling (Hearst 1997) 의 block-cosine + depth-score idea 를 유지하되,
**streaming** 화: pseudo-sentence 가 닫힐 때마다 block TF 만 incremental 로
관리하고, depth threshold 를 Welford running mean/std 로 유지한다. 매 발화
당 한 번의 `push()` 호출에서 *그 발화 직전 gap* 의 boundary 여부만 즉시
판정 (미래 발화 미관측). 이름·점수 모두 NLTK 원본과 다른 별도 baseline
(`TextTiling-online-streaming`); decision-log 2026-05-20 참조.

핵심 결정 (codex:rescue 위임 2026-05-20):

- block size ``k=6``, pseudo-sentence size ``w=10`` (원본 SuperDialseg 설정).
- depth = ``max(0, left_peak_so_far − score)`` (right peak 는 causal 로 알 수
  없으므로 one-sided).
- threshold cutoff = ``mean + c · std`` (depth 가 클수록 valley → boundary;
  c 기본 0.5).
- close-boundary suppression: 마지막 채택 boundary 와 거리 ``< min_gap``
  (utterance 단위, 기본 4) 면 reject.
- warmup: depth 표본이 ``warmup_gaps`` 미만이면 threshold 미확정 → False.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

_DEFAULT_STOP = frozenset(
    """a an and are as at be but by for from has have he her him his i if in
    is it its me my no not of on or our she so that the their them they this
    to was we were what when where which who will with you your""".split()
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _tokenize(text: str, stop: frozenset[str]) -> list[str]:
    toks = (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
    return [t for t in toks if t not in stop and len(t) > 1]


def _block_sum(blocks: list[Counter], i: int, k: int) -> Counter:
    """sum of TF counters blocks[i : i+k]."""
    acc: Counter = Counter()
    for c in blocks[i : i + k]:
        acc.update(c)
    return acc


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[w] * b[w] for w in a if w in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


@dataclass
class StreamingTextTiling:
    """Causal/streaming TextTiling segmenter.

    Args:
        w: pseudo-sentence token size (원본 default 10).
        k: block size (원본 default 6).
        c: cutoff = mean(depth) + c·std(depth). 기본 0.5.
        min_gap: 마지막 채택 boundary 와 최소 utterance 거리. 기본 4.
        warmup_gaps: depth 표본이 이 수 미만이면 boundary 보류. 기본 3.
        stop_words: stopword set. None → 내장 영문 stopword.
    """

    w: int = 10
    k: int = 6
    c: float = 0.5
    min_gap: int = 4
    warmup_gaps: int = 3
    stop_words: frozenset[str] = field(default_factory=lambda: _DEFAULT_STOP)

    # internal state
    _tok_buf: list[str] = field(default_factory=list, init=False)
    _utts_since_close: int = field(default=0, init=False)
    _ps_owner: list[int] = field(default_factory=list, init=False)  # utt-index that closed each ps
    _ps_tf: list[Counter] = field(default_factory=list, init=False)
    _scores: list[float] = field(default_factory=list, init=False)  # score(g_i)
    _evaluated_until: int = field(default=-1, init=False)  # 마지막 평가한 gap idx
    _left_peak: float = field(default=0.0, init=False)
    _welf: _Welford = field(default_factory=_Welford, init=False)
    _t: int = field(default=0, init=False)  # 누적 utterance 수 (1-based after push)
    _last_boundary_t: int = field(default=-(10**9), init=False)
    _neural_sec: float = field(default=0.0, init=False)  # no neural component → always 0.0
    _preprocess_sec: float = field(default=0.0, init=False)  # tokenize + stopword + BoW (Counter)

    # ------------------------------------------------------------------ api
    def push(self, utterance: str) -> list[int]:
        """발화 1개를 받고, 이번 호출에서 **새로 확정된** boundary utterance
        index (1-based) 목록 반환.

        causal 이지만 lag 가 있다: gap 의 score 는 right block (k pseudo-
        sentence) 이 모두 닫혀야 평가 가능하므로, 한 boundary 가 "확정"
        되는 시점은 그 boundary 의 utterance 가 들어온 시점보다 늦다. 따라서
        ``push(u_t)`` 가 반환하는 list 의 원소는 *t 이하의 과거 utterance
        index* 일 수 있다. 호출자는 모든 push 반환을 누적하면 대화 종료
        시점에 완전한 boundary 집합을 얻는다.
        """
        self._t += 1
        # Preprocess: 토큰화·불용어 제거 → bag-of-words (Counter) 누적.
        # 결정 로직(block-cosine depth + threshold) 전 단계 비용 전부를 묶음.
        _pp0 = time.perf_counter()
        toks = _tokenize(utterance, self.stop_words)
        new_boundaries: list[int] = []

        # 1) 토큰 누적 + pseudo-sentence close (Counter 생성 = BoW 벡터)
        for tok in toks:
            self._tok_buf.append(tok)
            if len(self._tok_buf) >= self.w:
                self._close_pseudo_sentence()
        self._preprocess_sec += time.perf_counter() - _pp0

        # 2) 평가 가능한 새 gap 들을 처리 + boundary candidate 결정
        j = len(self._ps_tf) - 1  # 최신 pseudo-sentence index
        # gap g_i 는 ps_{i-1}, ps_i 사이; score 는 ps_{i-k} .. ps_{i+k-1} 필요
        # → 최신 평가 가능 gap idx = j - k + 1 (j 가 right block 끝, i = j - k + 1)
        new_last = j - self.k + 1
        if new_last > self._evaluated_until:
            for gi in range(self._evaluated_until + 1, new_last + 1):
                if gi < self.k:  # left block 부족
                    continue
                left = _block_sum(self._ps_tf, gi - self.k, self.k)
                right = _block_sum(self._ps_tf, gi, self.k)
                s = _cosine(left, right)
                self._scores.append(s)
                # depth = one-sided causal: left_peak_so_far − s
                depth = max(0.0, self._left_peak - s)
                self._welf.push(depth)
                if s > self._left_peak:
                    self._left_peak = s
                if (
                    self._welf.n >= self.warmup_gaps
                    and depth > self._welf.mean + self.c * self._welf.std()
                ):
                    owner_t = self._ps_owner[gi]
                    if owner_t - self._last_boundary_t >= self.min_gap:
                        self._last_boundary_t = owner_t
                        new_boundaries.append(owner_t)
            self._evaluated_until = new_last

        return new_boundaries

    def flush(self) -> list[int]:
        """대화 종료 신호. 남은 토큰 (< w) 으로 마지막 pseudo-sentence 강제
        close 한 뒤, 평가 가능한 남은 gap 을 모두 처리. 반환은 같은 의미의
        boundary utterance index list.
        """
        if self._tok_buf:
            self._ps_tf.append(Counter(self._tok_buf))
            self._ps_owner.append(self._t)
            self._tok_buf = []
        # 남은 gap 들을 평가
        j = len(self._ps_tf) - 1
        new_last = j - self.k + 1
        out: list[int] = []
        for gi in range(self._evaluated_until + 1, new_last + 1):
            if gi < self.k:
                continue
            left = _block_sum(self._ps_tf, gi - self.k, self.k)
            right = _block_sum(self._ps_tf, gi, self.k)
            s = _cosine(left, right)
            depth = max(0.0, self._left_peak - s)
            self._welf.push(depth)
            if s > self._left_peak:
                self._left_peak = s
            if (
                self._welf.n >= self.warmup_gaps
                and depth > self._welf.mean + self.c * self._welf.std()
            ):
                owner_t = self._ps_owner[gi]
                if owner_t - self._last_boundary_t >= self.min_gap:
                    self._last_boundary_t = owner_t
                    out.append(owner_t)
        self._evaluated_until = new_last
        return out

    def state(self) -> dict:
        """진단용 snapshot."""
        return {
            "t": self._t,
            "n_pseudo_sentence": len(self._ps_tf),
            "n_scored_gap": self._welf.n,
            "mean_depth": self._welf.mean,
            "std_depth": self._welf.std(),
            "left_peak": self._left_peak,
            "last_boundary_t": self._last_boundary_t,
        }

    # ----------------------------------------------------------- internals
    def _close_pseudo_sentence(self) -> None:
        tf: Counter = Counter(self._tok_buf[: self.w])
        del self._tok_buf[: self.w]
        self._ps_tf.append(tf)
        self._ps_owner.append(self._t)
