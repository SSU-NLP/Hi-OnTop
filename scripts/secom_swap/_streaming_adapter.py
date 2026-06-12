"""Shared adapter for streaming/online segmenters → SeCom contract.

Each Hi-OnTop baseline segmenter (TextTiling-streaming, GreedySeg-delay2,
GraphSeg-window, CSM-online) exposes a per-session streaming API:

    seg = SegmenterClass(**kwargs)
    for utt in session:
        new_boundaries = seg.push(utt)   # boundaries that became confirmed now
    final_boundaries = seg.flush()       # boundaries confirmed at end of stream

This module converts that stream into SeCom's expected output —
``segments: List[List[str]]`` — and times each session for latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StreamingSegmentLatency:
    n_sessions: int = 0
    n_exchanges: int = 0
    push_sec: float = 0.0
    flush_sec: float = 0.0
    total_sec: float = 0.0
    neural_sec: float = 0.0       # legacy = pure neural-forward (kept for back-compat)
    preprocess_sec: float = 0.0   # everything before the segmentation decision
                                  # (tokenization, BoW, POS, GloVe lookup, neural fwd, …)
    per_turn: list[float] = field(default_factory=list)

    @property
    def ms_per_exchange(self) -> float:
        return self.total_sec * 1000 / max(1, self.n_exchanges)

    @property
    def ms_per_exchange_neural(self) -> float:
        return self.neural_sec * 1000 / max(1, self.n_exchanges)

    @property
    def ms_per_exchange_logic(self) -> float:
        return (self.total_sec - self.neural_sec) * 1000 / max(1, self.n_exchanges)

    @property
    def ms_per_exchange_preprocess(self) -> float:
        return self.preprocess_sec * 1000 / max(1, self.n_exchanges)

    @property
    def ms_per_exchange_segdecision(self) -> float:
        return (self.total_sec - self.preprocess_sec) * 1000 / max(1, self.n_exchanges)

    def asdict(self) -> dict[str, Any]:
        import numpy as np
        return {
            "n_sessions": self.n_sessions,
            "n_exchanges": self.n_exchanges,
            "push_sec": self.push_sec,
            "flush_sec": self.flush_sec,
            "total_sec": self.total_sec,
            "neural_sec": self.neural_sec,
            "preprocess_sec": self.preprocess_sec,
            "ms_per_exchange": self.ms_per_exchange,
            "ms_per_exchange_segment_only": self.ms_per_exchange,
            "ms_per_exchange_neural": self.ms_per_exchange_neural,
            "ms_per_exchange_logic": self.ms_per_exchange_logic,
            "ms_per_exchange_preprocess": self.ms_per_exchange_preprocess,
            "ms_per_exchange_segdecision": self.ms_per_exchange_segdecision,
            "per_turn_p50_ms": float(np.median(self.per_turn)) * 1000 if self.per_turn else 0.0,
            "per_turn_p95_ms": float(np.percentile(self.per_turn, 95)) * 1000 if self.per_turn else 0.0,
            "per_turn_max_ms": float(np.max(self.per_turn)) * 1000 if self.per_turn else 0.0,
        }


def session_segments_from_boundaries(session: list[str], boundary_indices_1based: list[int]) -> list[list[str]]:
    """Convert 1-based boundary utterance indices to SeCom segments list.

    Boundary at index i means utterance i is the LAST of its segment (or
    equivalently, a new segment STARTS at i+1). This matches the convention
    used by TextTiling-streaming / GreedySeg / GraphSeg in this repo.
    """
    n = len(session)
    if not session:
        return []
    # unique + clip to valid range, sorted
    bnd = sorted({b for b in boundary_indices_1based if 1 <= b <= n})
    # ensure final segment is bounded by n (close last segment)
    if not bnd or bnd[-1] != n:
        bnd = bnd + [n]
    segments: list[list[str]] = []
    prev = 0
    for b in bnd:
        seg = session[prev:b]
        if seg:
            segments.append(seg)
        prev = b
    return segments


def run_streaming_segmenter(
    sessions: list[list[str]],
    factory: Callable[[], Any],
) -> tuple[list[list[str]], StreamingSegmentLatency]:
    """Run a streaming segmenter (push/flush API) over all sessions.

    Args:
        sessions: list of sessions; each is a list of utterance strings.
        factory: zero-arg callable returning a fresh streaming segmenter
            (must expose ``push(str) -> list[int]`` and ``flush() -> list[int]``).

    Returns:
        (segments, latency) — flat segments across all sessions + timing.
    """
    lat = StreamingSegmentLatency()
    segments: list[list[str]] = []
    t_total = time.perf_counter()
    for sess in sessions:
        if not sess:
            continue
        lat.n_sessions += 1
        lat.n_exchanges += len(sess)
        seg = factory()
        confirmed: list[int] = []
        t_push = time.perf_counter()
        for utt in sess:
            tt = time.perf_counter()
            new_b = seg.push(utt)
            lat.per_turn.append(time.perf_counter() - tt)
            confirmed.extend(new_b)
        lat.push_sec += time.perf_counter() - t_push
        t_flush = time.perf_counter()
        final = seg.flush()
        lat.flush_sec += time.perf_counter() - t_flush
        confirmed.extend(final)
        lat.neural_sec += float(getattr(seg, "_neural_sec", 0.0))
        # Preprocess = all work before the segmentation decision (tokenization,
        # stopword/POS filter, BoW, GloVe lookup, neural fwd). Baselines that
        # already split this explicitly expose _preprocess_sec; for those where
        # the only preprocess IS the neural forward (GreedySeg/CSM/Hi-OnTop), we
        # fall back to _neural_sec so the semantic is consistent.
        lat.preprocess_sec += float(
            getattr(seg, "_preprocess_sec",
                    getattr(seg, "_neural_sec", 0.0))
        )
        segments.extend(session_segments_from_boundaries(sess, confirmed))
    lat.total_sec = time.perf_counter() - t_total
    return segments, lat
