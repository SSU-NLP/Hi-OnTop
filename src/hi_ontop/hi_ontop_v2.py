"""Hi-OnTop-v2 — adaptive (distribution-relative) boundary threshold.

v2 keeps Hi-OnTop's entire ``delta_eff`` boundary signal unchanged and only
replaces the **fixed** threshold ``delta*`` with a **distribution-adaptive**
one, in the spirit of TextTiling's classic ``mu - c*sigma`` cutoff
(Hearst 1997): a boundary fires when ``delta_eff`` is unusually high relative
to the conversation's own ``delta_eff`` statistics.

    boundary(t)  <=>  delta_eff(t) >= mu_t + c * sigma_t

where ``mu_t``/``sigma_t`` are the mean/std of the ``delta_eff`` values. Since
Hi-OnTop boundaries are *high* ``delta_eff`` (cohesion drop), the sign is
``+c*sigma`` (flag the unusually-high ones), the mirror of TextTiling's
``mu - c*sigma`` (whose depth score points the other way).

Two estimators:
- ``online`` (default, preserves the 0-look-ahead identity): a **running**
  ``mu``/``sigma`` over the ``delta_eff`` seen so far (past + current only),
  with a short ``warmup`` before any boundary may fire (until the variance
  estimate stabilizes the threshold falls back to ``delta_star``).
- ``offline``: ``mu``/``sigma`` over the whole sequence (TextTiling-classic;
  needs the full document → not streaming, comparison/baseline only).

Why: a single global ``delta*`` is too strict for one conversation and too
loose for another (different encoders/topics shift the ``delta_eff`` scale).
``mu + c*sigma`` self-calibrates per conversation. The only knob is ``c``
(small constant); it is *not* a per-domain magic number.

Lineage: this is a calibration variant of [[hi-ontop]] (the ``delta_eff``
signal is byte-identical); the earlier lexical-overlap experiment lives in
[[hi-ontop-lex]] (non-promoted, 2026-05-23). See decision-log 2026-06-09.
"""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.hi_ontop import HiOnTop


class HiOnTopV2(HiOnTop):
    """Hi-OnTop with an adaptive ``mu + c*sigma`` boundary threshold.

    All :class:`HiOnTop` args behave identically (they shape the unchanged
    ``delta_eff``). ``delta_star`` is used only as the warm-up fallback.

    Args:
        c: threshold spread multiplier — ``thr = mu + c*sigma``. Larger c =
            stricter (fewer boundaries). Small constant, encoder-ish, not a
            per-domain magic number.
        warmup: number of ``delta_eff`` samples to accumulate before the
            adaptive threshold is trusted; before that, fall back to the
            fixed ``delta_star``.

    Per-turn readouts (in addition to :class:`HiOnTop`'s):
        last_threshold — the adaptive threshold used at the last turn.
    """

    def __init__(
        self,
        dim: int,
        delta_star: float = 0.5594,
        ctx_window: int = 2,
        ctx_decay: float = 0.7,
        ctx_blend_a: float = 0.5,
        c: float = 1.0,
        warmup: int = 5,
    ) -> None:
        super().__init__(dim, delta_star, ctx_window, ctx_decay, ctx_blend_a)
        if c < 0.0:
            raise ValueError(f"c must be >= 0, got {c}")
        if warmup < 1:
            raise ValueError(f"warmup must be >= 1, got {warmup}")
        self.c = c
        self.warmup = warmup
        # running stats over real delta_eff (turn >= 1)
        self._d_n = 0
        self._d_sum = 0.0
        self._d_sumsq = 0.0
        self.last_threshold: float = delta_star

    def _running_threshold(self) -> float:
        """mu + c*sigma over delta_eff seen so far; delta_star until warmup."""
        if self._d_n < self.warmup:
            return self.delta_star
        mu = self._d_sum / self._d_n
        var = max(0.0, self._d_sumsq / self._d_n - mu * mu)
        return mu + self.c * math.sqrt(var)

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        s = np.asarray(s, dtype=np.float64)
        delta_eff = self._delta_eff(s)
        has_prev = self._prev_s is not None      # turn 0 delta_eff is a sentinel 0

        if has_prev:                              # accumulate running stats first
            self._d_n += 1
            self._d_sum += delta_eff
            self._d_sumsq += delta_eff * delta_eff

        thr = self._running_threshold()
        graded = delta_eff / thr if thr > 0 else 0.0

        if self._topic_id < 0:
            is_boundary = False
            self._topic_id = 0
        else:
            is_boundary = has_prev and delta_eff >= thr
            if is_boundary:
                self._topic_id += 1

        self.last_delta_eff = delta_eff
        self.last_threshold = thr
        self.last_graded_score = graded
        self.last_is_boundary = is_boundary
        self._history.append({
            "turn": self._n_turns, "topic_id": self._topic_id,
            "is_boundary": is_boundary, "delta_eff": delta_eff,
            "threshold": thr, "graded_score": graded,
        })
        self._n_turns += 1

        s_copy = np.array(s, dtype=np.float64, copy=True)
        self._prev_s = s_copy
        self._recent.append(s_copy)
        if len(self._recent) > self.ctx_window:
            del self._recent[0]
        return self._topic_id, is_boundary


def _otsu_threshold(vals: np.ndarray, nbins: int = 64) -> float:
    """Otsu (1979) parameter-free threshold: the cut that maximizes between-
    class variance of the value histogram (splits low/high into 2 classes)."""
    vals = vals[np.isfinite(vals)]
    if vals.size < 2 or vals.min() == vals.max():
        return float(vals.max()) + 1.0 if vals.size else 1.0
    hist, edges = np.histogram(vals, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = vals.size
    sum_all = float((hist * centers).sum())
    wB = 0.0; sumB = 0.0; best_var = -1.0; best_thr = centers[0]
    for i in range(nbins):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += hist[i] * centers[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > best_var:
            best_var = between; best_thr = float(edges[i + 1])
    return best_thr


def adaptive_boundaries(seq, c: float = 1.0, mode: str = "online",
                        warmup: int = 5, fallback: float = 0.5594) -> list[int]:
    """Boundary labels for a ``delta_eff`` sequence via ``mu + c*sigma``.

    Mirrors ``run_encoder_comparison.boundaries`` indexing (compares seq[i]
    for i in 1..n-1, trailing 0) so it is drop-in for the eval harness.

    Args:
        seq: per-turn ``delta_eff`` list (seq[0] is the turn-0 sentinel ~0).
        mode: ``online`` (running, causal) or ``offline`` (whole-sequence).
    """
    n = len(seq)
    if n <= 1:
        return [0] * n
    real = [float(v) for v in seq[1:]]            # turn>=1 are real delta_eff
    out = []
    if mode == "offline":
        arr = np.asarray(real, dtype=np.float64)
        thr = float(arr.mean() + c * arr.std())
        out = [1 if seq[i] >= thr else 0 for i in range(1, n)]
    elif mode == "otsu":
        thr = _otsu_threshold(np.asarray(real, dtype=np.float64))
        out = [1 if seq[i] >= thr else 0 for i in range(1, n)]
    elif mode == "online":
        s = sq = 0.0
        cnt = 0
        for i in range(1, n):
            cnt += 1; s += seq[i]; sq += seq[i] * seq[i]
            if cnt < warmup:
                thr = fallback
            else:
                mu = s / cnt
                thr = mu + c * math.sqrt(max(0.0, sq / cnt - mu * mu))
            out.append(1 if seq[i] >= thr else 0)
    elif mode == "online_nf":
        # no fixed fallback — adaptive mu+c*sigma from the 2nd sample on.
        s = sq = 0.0
        cnt = 0
        for i in range(1, n):
            cnt += 1; s += seq[i]; sq += seq[i] * seq[i]
            if cnt < 2:
                out.append(0); continue
            mu = s / cnt
            thr = mu + c * math.sqrt(max(0.0, sq / cnt - mu * mu))
            out.append(1 if seq[i] >= thr else 0)
    elif mode == "ewma":
        # EWMA mean/var (local-adaptive); span via `warmup` (>=2).
        alpha = 2.0 / (warmup + 1.0)
        mu = ev = 0.0
        init = False
        for i in range(1, n):
            x = seq[i]
            if not init:
                mu, ev, init = x, 0.0, True
                out.append(0); continue
            thr = mu + c * math.sqrt(max(0.0, ev))
            out.append(1 if x >= thr else 0)
            d = x - mu
            mu += alpha * d
            ev = (1 - alpha) * (ev + alpha * d * d)
    else:
        raise ValueError(f"mode must be online/online_nf/ewma/offline, got {mode}")
    return out + [0]
