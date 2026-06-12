"""SeCom segmentation backend adapter for v4.1.3.

Wraps :class:`hi_ontop.hi_ontop.HiOnTop` to satisfy SeCom's
``segment(sessions: List[List[str]]) -> List[List[str]]`` contract.

SeCom (Microsoft, 2024) uses an LLM (default ``gpt-4o-mini``) to segment
each session of a conversation. This adapter swaps that LLM call for our
online O(1)-per-turn graded-boundary segmenter, keeping every downstream
step (compress, retrieve, chat, eval) identical.

Design choices
--------------
- **Encoder**: a sentence-transformers model whose output dimension matches
  the segmenter ``dim``. We L2-normalize embeddings (cosine basis) since
  v4.1.x's :math:`\\delta = 1 - \\cos` likelihood assumes unit norm.
- **Per-session reset**: a fresh :class:`HiOnTop` is built for
  every session, matching SeCom's per-session LLM call (no cross-session
  state leak).
- **Boundary placement**: a boundary at turn *t* starts a new segment with
  exchange *t* (turn *t* belongs to the NEW segment). This is the standard
  contract — same as SeCom's LLM output where "num_exchanges = N" means
  the next segment starts at exchange N.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from hi_ontop.hi_ontop import HiOnTop


@dataclass
class SegmentLatency:
    """Per-call timing for a SeCom-style ``segment()`` invocation."""

    n_sessions: int = 0
    n_exchanges: int = 0
    encode_sec: float = 0.0
    segment_sec: float = 0.0  # pure assign() loop (excludes encode)
    total_sec: float = 0.0
    per_turn: list[float] = field(default_factory=list)  # per-exchange assign-only

    @property
    def sec_per_exchange(self) -> float:
        return self.total_sec / max(1, self.n_exchanges)

    def asdict(self) -> dict[str, Any]:
        return {
            "n_sessions": self.n_sessions,
            "n_exchanges": self.n_exchanges,
            "encode_sec": self.encode_sec,
            "segment_sec": self.segment_sec,
            "total_sec": self.total_sec,
            "sec_per_exchange": self.sec_per_exchange,
            "per_turn_p50": float(np.median(self.per_turn)) if self.per_turn else 0.0,
            "per_turn_p95": (
                float(np.percentile(self.per_turn, 95)) if self.per_turn else 0.0
            ),
            "per_turn_max": float(np.max(self.per_turn)) if self.per_turn else 0.0,
        }


class HiOnTopSecomSegmenter:
    """Drop-in replacement for SeCom's ``segmentor`` + ``segment()``.

    Usage::

        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(
            "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        )
        seg = HiOnTopSecomSegmenter(encoder, dim=768, delta_star=0.30)
        segments = seg.segment(sessions)  # same shape as SeCom.segment

    Args:
        encoder: a sentence-transformers model exposing
            ``encode(list[str], normalize_embeddings=bool) -> np.ndarray``.
        dim: embedding dimension (must match ``encoder``'s output).
        delta_star: v4.1.x boundary threshold. Encoder-dependent — must be
            calibrated on a held-out set for the chosen encoder.
        hiontop_kwargs: extra kwargs forwarded to :class:`HiOnTop`.
        encode_batch_size: batch size for the encoder (default 32).
        normalize: whether to L2-normalize encoder outputs (default True).
    """

    def __init__(
        self,
        encoder: Any,
        dim: int = 768,
        delta_star: float = 0.30,
        hiontop_kwargs: dict[str, Any] | None = None,
        encode_batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.encoder = encoder
        self.dim = dim
        self.delta_star = delta_star
        self.hiontop_kwargs = dict(hiontop_kwargs or {})
        self.encode_batch_size = encode_batch_size
        self.normalize = normalize
        self.last_latency: SegmentLatency | None = None
        # boundary-strength rollup over all segment() calls
        self.boundary_strength_total: dict[str, int] = {
            "very_weak": 0,
            "weak": 0,
            "normal": 0,
            "strong": 0,
        }

    def _make_segmenter(self) -> HiOnTop:
        kwargs = {"dim": self.dim, "delta_star": self.delta_star}
        kwargs.update(self.hiontop_kwargs)
        return HiOnTop(**kwargs)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode exchanges to ``(n, dim)`` float64 vectors.

        Supports two encoder contracts:

        - ``sentence-transformers`` ``SentenceTransformer`` — the rich kwarg
          signature (batch size, explicit ``normalize_embeddings``).
        - :mod:`hi_ontop.embedding` encoders (``QueryEncoder`` / ``APIEncoder``,
          the latter backed by the Crts ``/v1/embeddings`` API) — these
          expose ``encode(list[str]) -> L2-normalized ndarray`` and reject
          the SentenceTransformer kwargs.
        """
        items = list(texts)
        try:
            vecs = self.encoder.encode(
                items,
                batch_size=self.encode_batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
        except TypeError:
            # hi_ontop.embedding encoder — single-arg encode(), already L2-normalized.
            vecs = self.encoder.encode(items)
        return np.asarray(vecs, dtype=np.float64)

    def segment(self, sessions: list[list[str]]) -> list[list[str]]:
        """SeCom-compatible segmentation.

        Args:
            sessions: list of sessions; each session is a list of
                exchange strings ("[human]: ... [bot]: ...").

        Returns:
            Flat list of segments (across all sessions). Each segment is a
            list of consecutive exchange strings. Total exchange count
            preserved — every input exchange goes into exactly one segment.
        """
        lat = SegmentLatency()
        t_total = time.perf_counter()

        segments: list[list[str]] = []
        for exchanges in sessions:
            if not exchanges:
                continue
            lat.n_sessions += 1
            lat.n_exchanges += len(exchanges)

            t_enc = time.perf_counter()
            vecs = self._encode(exchanges)
            lat.encode_sec += time.perf_counter() - t_enc

            seg = self._make_segmenter()
            current: list[str] = []
            t_seg_start = time.perf_counter()
            for vec, ex in zip(vecs, exchanges):
                t_turn = time.perf_counter()
                _, is_boundary = seg.assign(vec)
                lat.per_turn.append(time.perf_counter() - t_turn)
                if is_boundary and current:
                    segments.append(current)
                    current = []
                current.append(ex)
            if current:
                segments.append(current)
            lat.segment_sec += time.perf_counter() - t_seg_start

            # roll up boundary-strength histogram
            band = seg.boundary_strength()
            for k, v in band.items():
                self.boundary_strength_total[k] += v

        lat.total_sec = time.perf_counter() - t_total
        self.last_latency = lat
        return segments
