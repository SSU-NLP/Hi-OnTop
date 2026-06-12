"""Hi-OnTop-v3.2 (experimental) — SEM-faithful event-centroid segmenter with
content-adaptive observation units (B+C fusion).

Status: experimental candidate for the v3 = *granularity-adaptive* Hi-OnTop
line. Competes with v3.1 (explicit content-block unit selection). The
winner of the AMI-recovery + DTS-no-regression bench-off is promoted to the
canonical ``v3``. See ``context/06-decision-log.md`` (2026-06-08) and
``context/methodology/hi-ontop-v3.2.md``.

Motivation
----------
On meeting speech (AMI) the turn-level Hi-OnTop signal collapses: 31% of
turns are 1-word backchannels, within-topic adjacent cosine median 0.167,
boundary-separation AUC 0.567 (~chance). Two coupled fixes, both grounded
in SEM:

1. **Content-adaptive observation unit (the "B" half).** A single turn
   embedding is too content-poor to compare (a lone "yeah" is dissimilar to
   everything). Accumulate incoming turns into a *pending unit* until its
   embedding **stabilizes** — i.e. folding in the next turn moves the unit
   vector by less than ``stab_eps`` cosine — subject to a small token
   floor. Short backchannels are absorbed into a content-rich unit; rich
   turns (clean text dialogue) stabilize immediately so unit == turn and
   resolution is preserved. No fixed time/word magic number — the stop rule
   is an *encoder* property (how much text it needs to be stable), not a
   domain constant.

2. **Event-centroid comparison + sticky prior (the "C" half).** Instead of
   comparing a unit to the previous 1-2 units (fixed window ``delta_ctx``),
   compare it to the running centroid ``mu_k`` of the *current event*
   (= SEM's current event model ``f_k`` sufficient statistic). A boundary
   opens when the unit's prediction error ``1 - cos(u, mu_k)`` exceeds
   ``delta*`` after a sCRP sticky margin that resists switching on a single
   blip. On a boundary the centroid resets; otherwise the unit is folded in
   (token-mass weighted). Segment length therefore *emerges* from the
   sequential MAP decision rather than a fixed block size.

SEM lineage
-----------
Direct reduced-form of SEM's "compare new observation to current event
model, break on prediction error tempered by a sticky prior". ``mu_k`` is
the Hi-OnTop analogue of ``f_k.update(x_t)``; the sticky margin is the sCRP
stickiness ``alpha``. Recorded SEM-3step PASS in the decision log: the
backchannel/short-utterance regime is outside SEM2's input-domain
assumptions (rich scene/paragraph observations), so its absence is
non-implementation, not deliberate exclusion; the added machinery is a
direct realization of existing SEM mechanisms, not a new component.

Relationship to delta_ctx
-------------------------
``mu_k`` generalizes Hi-OnTop's fixed causal window ``delta_ctx`` (window
``m``) to a *variable, segment-spanning, reset-at-boundary* context:
``m -> |current event|`` with a hard reset each boundary.
"""

from __future__ import annotations

import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class HiOnTopV3Event:
    """Event-centroid segmenter with content-adaptive units (v3.2 exp).

    Streaming, online (past-only), O(1) amortized per turn.

    Args:
        dim: embedding dimension.
        delta_star: boundary threshold on the unit prediction error
            ``1 - cos(unit, mu_k)``. Encoder/dataset-calibrated, label-free
            (percentile of the held-out delta distribution).
        stab_eps: unit-stability stop. A pending unit closes once folding in
            the next turn changes its (normalized) vector by less than
            ``stab_eps`` in cosine distance, i.e. ``1 - cos(u_before,
            u_after) < stab_eps``. Encoder property (default 0.05).
        min_tokens: token floor before the stability test can fire — guards
            against closing a unit on a single short turn. Encoder property.
        alpha: sCRP stickiness. Sticky margin added to ``delta_star`` for
            the current event, scaled by event commitment (see below). 0.0
            disables stickiness.
        max_tokens: hard cap on unit size so a monotone stream still closes
            units (safety; not a resolution knob).

    Per-unit readouts (set after each closed unit in :meth:`feed`):
        last_unit_delta, last_unit_tokens, last_is_boundary.
    """

    def __init__(
        self,
        dim: int,
        delta_star: float = 0.66,
        stab_eps: float = 0.05,
        min_tokens: int = 12,
        alpha: float = 0.0,
        max_tokens: int = 400,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if not 0.0 <= delta_star <= 2.0:
            raise ValueError(f"delta_star must be in [0, 2], got {delta_star}")
        if not 0.0 <= stab_eps < 1.0:
            raise ValueError(f"stab_eps must be in [0, 1), got {stab_eps}")
        if min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {min_tokens}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        self.dim = dim
        self.delta_star = delta_star
        self.stab_eps = stab_eps
        self.min_tokens = min_tokens
        self.alpha = alpha
        self.max_tokens = max_tokens

        # pending unit accumulator (token-mass weighted sum of turn embeddings)
        self._u_sum = np.zeros(dim, dtype=np.float64)
        self._u_tok = 0
        # current event centroid accumulator (token-mass weighted sum of units)
        self._e_sum = np.zeros(dim, dtype=np.float64)
        self._e_units = 0
        self._topic_id = -1
        self._history: list[dict[str, object]] = []

        self.last_unit_delta: float = 0.0
        self.last_unit_tokens: int = 0
        self.last_is_boundary: bool = False

    # ------------------------------------------------------------------
    # unit accumulation
    # ------------------------------------------------------------------

    def _would_stabilize(self, s: np.ndarray, tokens: int) -> bool:
        """True if the pending unit is content-stable enough to close.

        Requires the token floor, then checks that folding ``s`` in barely
        moves the (normalized) pending vector.
        """
        if self._u_tok < self.min_tokens:
            return False
        if self._u_tok + tokens >= self.max_tokens:
            return True
        before = _norm(self._u_sum)
        after = _norm(self._u_sum + tokens * s)
        return (1.0 - float(before @ after)) < self.stab_eps

    def _close_unit(self) -> tuple[np.ndarray, int]:
        u = _norm(self._u_sum)
        tok = self._u_tok
        self._u_sum = np.zeros(self.dim, dtype=np.float64)
        self._u_tok = 0
        return u, tok

    # ------------------------------------------------------------------
    # event-centroid boundary decision
    # ------------------------------------------------------------------

    def _decide(self, u: np.ndarray, tok: int) -> bool:
        """Boundary decision for a closed unit vs the current event centroid."""
        if self._e_units == 0:                      # first unit of stream
            self._e_sum = tok * u.copy()
            self._e_units = 1
            self._topic_id = 0
            self.last_unit_delta = 0.0
            return False
        mu = _norm(self._e_sum)
        delta = 1.0 - float(u @ mu)
        # sCRP sticky margin: commitment grows (saturating) with event size
        margin = self.alpha * (self._e_units / (self._e_units + 1.0))
        is_b = delta >= (self.delta_star + margin)
        self.last_unit_delta = delta
        if is_b:
            self._e_sum = tok * u.copy()            # reset event to this unit
            self._e_units = 1
            self._topic_id += 1
        else:
            self._e_sum = self._e_sum + tok * u     # fold into current event
            self._e_units += 1
        return is_b

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def feed(self, s: np.ndarray, tokens: int) -> tuple[int, bool] | None:
        """Consume one ``(turn embedding, token count)``.

        Returns ``(topic_id, is_boundary)`` **only when a unit closes** at
        this turn; returns ``None`` while the turn is still being buffered
        into the pending unit. Callers that need a per-turn label should
        treat buffered turns as continuing the current segment (no boundary).
        """
        s = np.asarray(s, dtype=np.float64)
        tokens = max(1, int(tokens))
        # close the *current* pending unit first if adding this turn would
        # leave it stable (i.e. the pending content is already self-contained)
        if self._u_tok > 0 and self._would_stabilize(s, tokens):
            u, tok = self._close_unit()
            is_b = self._decide(u, tok)
            self.last_unit_tokens = tok
            self.last_is_boundary = is_b
            self._history.append(
                {"delta": self.last_unit_delta, "tokens": tok,
                 "topic_id": self._topic_id, "is_boundary": is_b})
            # start a fresh pending unit with the current turn
            self._u_sum = tokens * s
            self._u_tok = tokens
            return self._topic_id, is_b
        # otherwise buffer this turn into the pending unit
        self._u_sum = self._u_sum + tokens * s
        self._u_tok += tokens
        return None

    def flush(self) -> tuple[int, bool] | None:
        """Close any remaining pending unit at end of stream."""
        if self._u_tok == 0:
            return None
        u, tok = self._close_unit()
        is_b = self._decide(u, tok)
        self.last_unit_tokens = tok
        self.last_is_boundary = is_b
        self._history.append(
            {"delta": self.last_unit_delta, "tokens": tok,
             "topic_id": self._topic_id, "is_boundary": is_b})
        return self._topic_id, is_b

    def history(self) -> list[dict[str, object]]:
        return list(self._history)
