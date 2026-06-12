"""Hi-OnTop-Lex — Hi-OnTop + TextTiling-style lexical-overlap correction.

Hi-OnTop-Lex keeps the entire :class:`~hi_ontop.hi_ontop.HiOnTop` semantic
boundary mechanism unchanged and adds a single *auxiliary local
observation feature*: a TextTiling-style word-frequency overlap signal.

Motivation
----------
Hi-OnTop's known structural failure mode (see ``hi_ontop.py`` docstring and
``context/methodology/hi_ontop.md``): when wording stays similar but the
topic shifts, the adjacent-embedding cosine distance ``delta_eff`` stays
low and the boundary is missed. Conversely, lexically very cohesive
stretches can be over-segmented by embedding noise. A lexical-overlap
term corrects ``delta_eff`` in both directions.

Algorithm
---------
For an L2-normalized utterance-embedding stream paired with the raw
utterance text::

    delta_base(t) = a*delta_prev(t) + (1-a)*delta_ctx(t)   # = HiOnTop delta_eff
    L_{t-1}       = sum_{i=1..m_lex} rho_lex^{i-1} * subtf(u_{t-i})
    overlap(t)    = cos_tf( L_{t-1}, subtf(u_t) )           # word-freq overlap
    lexdist(t)    = 1 - overlap(t)
    r_t           = min(1, min(n_ctx, n_t) / min_tokens)    # short-turn gate
    delta_eff_v2  = clip_[0,2]( delta_base + w_lex * r_t * (lexdist - mu_lex) )
    boundary(t)  <=>  delta_eff_v2(t) >= delta*

``subtf`` is a sublinear (``1 + log count``) term-frequency vector over
stopword-filtered content tokens. ``lexdist`` is *centered* by the
train-set median ``mu_lex`` so the correction is a residual: lexically
more-divergent-than-usual raises ``delta_eff``, more-cohesive-than-usual
lowers it. ``r_t`` down-weights the lexical term for short utterances
where the TF vectors are unreliable.

The lexical context window (``m_lex``, ``rho_lex``) mirrors Hi-OnTop's
causal embedding window (``m``, ``rho``) — same geometric-decay,
past-only, O(m)/turn structure, zero look-ahead. Unlike streaming
TextTiling's block-cosine (which lags until the right block closes),
``overlap(t)`` is available immediately at turn ``t``.

SEM lineage
-----------
The lexical term is *not* a SEM core mechanism — SEM models event
boundaries via structured scene dynamics, not word-level lexical
cohesion. It is recorded as a domain-specific auxiliary observation
feature that corrects the embedding-cosine observation when paraphrase /
generic wording masks a topic shift. The online / causal / local-MAP
structure of Hi-OnTop is preserved. See ``context/06-decision-log.md``
(2026-05-23) and ``context/methodology/hi-ontop-lex.md``.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from hi_ontop.baselines.texttiling_streaming import _DEFAULT_STOP, _tokenize
from hi_ontop.hi_ontop import HiOnTop


def _subtf(counts: Counter) -> dict[str, float]:
    """Sublinear term-frequency vector: tf(w) = 1 + log(count(w))."""
    return {w: 1.0 + math.log(n) for w, n in counts.items() if n > 0}


def _cos_tf(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity of two sparse TF dicts. 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    dot = sum(a[w] * b[w] for w in a if w in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


class HiOnTopLex(HiOnTop):
    """Hi-OnTop with a TextTiling-style lexical-overlap correction.

    All :class:`HiOnTop` arguments behave identically (they shape the
    unchanged ``delta_base`` term). Lexical arguments:

    Args:
        w_lex: lexical-residual weight. ``delta_eff_v2 = delta_base +
            w_lex * r_t * (lexdist - mu_lex)``. 0.0 -> reduces to Hi-OnTop.
        m_lex: lexical causal-window size (number of past utterances).
        rho_lex: geometric decay of the lexical window.
        mu_lex: train-set median of ``lexdist`` — centers the residual.
            Must be calibrated per corpus/encoder; 0.0 = no centering.
        min_tokens: short-turn gate denominator for the confidence ``r_t``.
        stop_words: stopword set (default: shared English stopword set).

    Per-turn readouts (in addition to the :class:`HiOnTop` ones):
        last_delta_base, last_lex_overlap, last_lexdist, last_r.
    """

    def __init__(
        self,
        dim: int,
        delta_star: float = 0.5594,
        ctx_window: int = 2,
        ctx_decay: float = 0.7,
        ctx_blend_a: float = 0.5,
        w_lex: float = 0.15,
        m_lex: int = 2,
        rho_lex: float = 0.7,
        mu_lex: float = 0.0,
        min_tokens: int = 3,
        stop_words: frozenset[str] | None = None,
    ) -> None:
        super().__init__(dim, delta_star, ctx_window, ctx_decay, ctx_blend_a)
        if w_lex < 0.0:
            raise ValueError(f"w_lex must be >= 0, got {w_lex}")
        if m_lex < 1:
            raise ValueError(f"m_lex must be >= 1, got {m_lex}")
        if not 0.0 < rho_lex <= 1.0:
            raise ValueError(f"rho_lex must be in (0, 1], got {rho_lex}")
        if not 0.0 <= mu_lex <= 1.0:
            raise ValueError(f"mu_lex must be in [0, 1], got {mu_lex}")
        if min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {min_tokens}")

        self.w_lex = w_lex
        self.m_lex = m_lex
        self.rho_lex = rho_lex
        self.mu_lex = mu_lex
        self.min_tokens = min_tokens
        self.stop_words = stop_words if stop_words is not None else _DEFAULT_STOP

        # lexical streaming state: raw token-count Counter per past turn
        self._lex_recent: list[Counter] = []

        # per-turn lexical readouts
        self.last_delta_base: float = 0.0
        self.last_lex_overlap: float = 0.0
        self.last_lexdist: float = 0.0
        self.last_r: float = 0.0

    # ------------------------------------------------------------------
    # lexical term
    # ------------------------------------------------------------------

    def _lex_terms(self, cur_counts: Counter) -> tuple[float, float, float]:
        """Return ``(overlap, lexdist, r_t)`` for the current turn.

        ``overlap`` = cos of the decayed lexical context vector and the
        current sublinear-TF vector. ``r_t`` is the short-turn confidence
        gate. Before any past turn exists, returns ``(0.0, 0.0, 0.0)`` so
        the lexical correction is inert.
        """
        win = self._lex_recent[-self.m_lex:]
        if not win or not cur_counts:
            return 0.0, 0.0, 0.0

        ctx_vec: dict[str, float] = {}
        for i, counts in enumerate(reversed(win)):       # i=0 -> u_{t-1}
            wgt = self.rho_lex ** i
            for w, v in _subtf(counts).items():
                ctx_vec[w] = ctx_vec.get(w, 0.0) + wgt * v

        overlap = _cos_tf(ctx_vec, _subtf(cur_counts))
        lexdist = 1.0 - overlap

        n_t = sum(cur_counts.values())
        n_ctx = sum(sum(c.values()) for c in win)
        r_t = min(1.0, min(n_ctx, n_t) / self.min_tokens)
        return overlap, lexdist, r_t

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def assign(self, s: np.ndarray, text: str = "") -> tuple[int, bool]:  # type: ignore[override]
        """Consume one ``(embedding, utterance text)`` pair.

        Returns ``(topic_id, is_boundary)``. ``text`` is required for the
        lexical term; an empty string disables the lexical correction for
        that turn (``delta_eff_v2 = delta_base``).
        """
        s = np.asarray(s, dtype=np.float64)
        first_turn = self._prev_s is None      # no s_{t-1} yet
        delta_base = self._delta_eff(s)

        cur_counts = Counter(_tokenize(text, self.stop_words)) if text else Counter()
        overlap, lexdist, r_t = self._lex_terms(cur_counts)

        if first_turn:
            # stream start — no s_{t-1}; keep HiOnTop turn-0 behaviour (delta=0)
            delta_eff = 0.0
        else:
            corr = self.w_lex * r_t * (lexdist - self.mu_lex)
            # clip to the cosine-distance range [0, 2]; at w_lex=0 this is a
            # no-op so delta_eff_v2 == HiOnTop delta_eff exactly (verified).
            delta_eff = float(np.clip(delta_base + corr, 0.0, 2.0))

        graded = delta_eff / self.delta_star if self.delta_star > 0 else 0.0

        if self._topic_id < 0:
            is_boundary = False
            self._topic_id = 0
        else:
            is_boundary = delta_eff >= self.delta_star
            if is_boundary:
                self._topic_id += 1

        self.last_delta_base = delta_base
        self.last_lex_overlap = overlap
        self.last_lexdist = lexdist
        self.last_r = r_t
        self.last_delta_eff = delta_eff
        self.last_graded_score = graded
        self.last_is_boundary = is_boundary
        self._history.append({
            "turn": self._n_turns,
            "topic_id": self._topic_id,
            "is_boundary": is_boundary,
            "delta_eff": delta_eff,
            "delta_base": delta_base,
            "lex_overlap": overlap,
            "lexdist": lexdist,
            "r": r_t,
            "graded_score": graded,
        })
        self._n_turns += 1

        # update streaming state — embedding window (HiOnTop) + lexical window
        s_copy = np.array(s, dtype=np.float64, copy=True)
        self._prev_s = s_copy
        self._recent.append(s_copy)
        if len(self._recent) > self.ctx_window:
            del self._recent[0]
        self._lex_recent.append(cur_counts)
        if len(self._lex_recent) > self.m_lex:
            del self._lex_recent[0]

        return self._topic_id, is_boundary
