"""Hi-OnTop causal cosine dialogue segmentation; see PAPER_ALIGNMENT.md."""
from __future__ import annotations

import numpy as np


class HiOnTop:
    """Online dialogue topic segmenter — causal-window graded boundary.

    Args:
        dim: embedding dimension (kept for API parity / validation).
        delta_star: boundary threshold delta*. A turn opens a new segment
            when delta_eff >= delta_star. Encoder/dataset-dependent — must
            be calibrated on a held-out set.
        ctx_window: m, number of past utterances in the causal context
            vector. Default 2 — the canonical TIAGE-calibrated config
            that produced the reported v4.1.x scores.
        ctx_decay: rho, geometric decay of the causal window (more recent
            utterances weighted higher).
        ctx_blend_a: a, blend weight delta_eff = a*delta_prev +
            (1-a)*delta_ctx.

    Public per-turn readouts (set after each :meth:`assign`):
        last_delta_eff, last_graded_score, last_is_boundary.
    """

    def __init__(
        self,
        dim: int,
        delta_star: float = 0.5594,
        ctx_window: int = 2,
        ctx_decay: float = 0.7,
        ctx_blend_a: float = 0.5,
        warmup: str = "paper",
    ) -> None:
        if warmup not in ("paper", "legacy"):
            raise ValueError("warmup must be paper or legacy")
        self.warmup = warmup
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if not 0.0 < delta_star <= 2.0:
            raise ValueError(f"delta_star must be in (0, 2], got {delta_star}")
        if ctx_window < 1:
            raise ValueError(f"ctx_window must be >= 1, got {ctx_window}")
        if not 0.0 < ctx_decay <= 1.0:
            raise ValueError(f"ctx_decay must be in (0, 1], got {ctx_decay}")
        if not 0.0 <= ctx_blend_a <= 1.0:
            raise ValueError(f"ctx_blend_a must be in [0, 1], got {ctx_blend_a}")

        self.dim = dim
        self.delta_star = delta_star
        self.ctx_window = ctx_window
        self.ctx_decay = ctx_decay
        self.ctx_blend_a = ctx_blend_a

        # streaming state
        self._prev_s: np.ndarray | None = None        # s_{t-1}
        self._recent: list[np.ndarray] = []           # causal window (past)
        self._topic_id: int = -1                      # current segment id
        self._n_turns: int = 0

        # per-turn readouts
        self.last_delta_eff: float = 0.0
        self.last_graded_score: float = 0.0
        self.last_is_boundary: bool = False

        self._history: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # distance terms
    # ------------------------------------------------------------------

    def _delta_prev(self, s: np.ndarray) -> float | None:
        """delta_prev = 1 - cos(s_{t-1}, s). None before the first turn."""
        if self._prev_s is None:
            return None
        a, b = self._prev_s, s
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return None
        return 1.0 - float(np.dot(a, b) / (na * nb))

    def _delta_ctx(self, s: np.ndarray) -> float | None:
        """delta_ctx = 1 - cos(c_{t-1}, s), c_{t-1} = normalize(
        sum rho^{i-1} s_{t-i}) over the causal window (recent weighted
        higher). None before any past turn."""
        if self.warmup == "paper" and len(self._recent) < self.ctx_window:
            return self._delta_prev(s)
        if not self._recent:
            return None
        win = self._recent[-self.ctx_window:]            # past only, <= m
        c = np.zeros_like(win[-1], dtype=np.float64)
        for i, vec in enumerate(reversed(win)):          # i=0 -> s_{t-1}
            c += (self.ctx_decay ** i) * vec
        nc = float(np.linalg.norm(c))
        ns = float(np.linalg.norm(s))
        if nc <= 1e-12 or ns <= 1e-12:
            return None
        return 1.0 - float(np.dot(c, s) / (nc * ns))

    def _delta_eff(self, s: np.ndarray) -> float:
        """delta_eff = a*delta_prev + (1-a)*delta_ctx (causal-window blend).

        Returns 0.0 before s_{t-1} exists (stream start) — matches
        v4.1.3's ``_delta_eff`` fallback so turn-0 graded score is 0.
        """
        d_prev = self._delta_prev(s)
        if d_prev is None:
            return 0.0
        d_ctx = self._delta_ctx(s)
        a = self.ctx_blend_a
        if d_ctx is None:
            return d_prev
        return a * d_prev + (1.0 - a) * d_ctx

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        """Consume one utterance embedding, return ``(topic_id, is_boundary)``.

        ``topic_id`` is a monotonically increasing segment id. ``is_boundary``
        is True when this turn opens a new segment.
        """
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (self.dim,) or not np.isfinite(s).all() or np.linalg.norm(s) <= 1e-12:
            raise ValueError("embedding must be a finite nonzero vector of the configured dimension")
        s = s / np.linalg.norm(s)
        delta_eff = self._delta_eff(s)
        graded = delta_eff / self.delta_star if self.delta_star > 0 else 0.0

        if self._topic_id < 0:
            # turn 0 — open the first segment, never a boundary
            is_boundary = False
            self._topic_id = 0
        else:
            is_boundary = delta_eff >= self.delta_star
            if is_boundary:
                self._topic_id += 1

        self.last_delta_eff = delta_eff
        self.last_graded_score = graded
        self.last_is_boundary = is_boundary
        self._history.append({
            "turn": self._n_turns,
            "topic_id": self._topic_id,
            "is_boundary": is_boundary,
            "delta_eff": delta_eff,
            "graded_score": graded,
        })
        self._n_turns += 1

        # update streaming state
        s_copy = np.array(s, dtype=np.float64, copy=True)
        self._prev_s = s_copy
        self._recent.append(s_copy)
        if len(self._recent) > self.ctx_window:
            del self._recent[0]

        return self._topic_id, is_boundary

    # ------------------------------------------------------------------
    # readouts
    # ------------------------------------------------------------------

    def history(self) -> list[dict[str, object]]:
        """Per-turn record list: turn, topic_id, is_boundary, delta_eff,
        graded_score."""
        return list(self._history)

    def graded_scores(self) -> list[float]:
        """Per-turn graded_score sequence (delta_eff / delta*)."""
        return [float(h["graded_score"]) for h in self._history]

    def boundary_strength(self) -> dict[str, int]:
        """Histogram of boundary-strength bands (Ben-Yakov & Henson 2018)."""
        out = {"very_weak": 0, "weak": 0, "normal": 0, "strong": 0}
        for h in self._history:
            g = float(h["graded_score"])
            if g < 0.7:
                out["very_weak"] += 1
            elif g < 1.0:
                out["weak"] += 1
            elif g < 1.3:
                out["normal"] += 1
            else:
                out["strong"] += 1
        return out
