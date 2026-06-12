"""v4.2.3-exp — dual-channel calibrated z-score energy PE.

Experimental fork of v4.1.1 (``sem_core_v411.HiOnTopSegmenterV411``) using
**two encoders** for the scene-vector representation, combined via a
calibrated z-score energy:

```
z_topic = δ_topic / δ*_topic        # mpnet 채널 (topic content)
z_flow  = δ_flow  / δ*_flow         # DSE-BERT 채널 (dialogue flow)
r       = √(w_topic·z_topic² + w_flow·z_flow²)
boundary ⇔ r ≥ 1
```

where each ``δ_*`` is computed via v4.1.1's causal-window δ_eff
machinery (``a·δ_prev + (1−a)·δ_ctx``) but with channel-specific
hyperparameters (m, ρ, a) and channel-specific δ*.

**Why "exp" suffix**: v4.2.3-exp is an experimental ablation, NOT a
default candidate. v4.2.2 smoke (2026-05-20) showed mpnet vs DSE-BERT
have complementary strengths (TIAGE = DSE-favoured natural dialogue,
Dialseg711/SuperDialseg = mpnet-favoured formal topic shifts). v4.2.3
tests whether the **calibrated energy combination** recovers both.

**SEM 계승 3-step (CONDITIONAL PASS, codex 2026-05-20 분석)**:
1. SEM2's ``LinearEvent`` has no multi-view encoder PE. v4.2.3 frames
   the dual channel as "SEM2 residual likelihood reduced to a
   calibrated two-feature PE", NOT as a new dynamics model.
2. sCRP / Bayes / local-MAP / prior-corrected B0 are unchanged —
   the combined energy ``r`` is the new scalar δ_eff, and V411's
   ``δ_eff < δ*`` decision becomes ``r < 1`` (with internal
   ``delta_star = 1.0`` and ``σδ² = sigma_delta_c`` so the likelihood
   ``L(r) = -r²/(2σδ²)`` evaluates to L(1) = -8.0 at boundary, exactly
   matching v4.1.1's L(δ*) at its own δ* — softness ratio preserved.
3. Zacks 2007 EST is a *conceptual* link only (multi-feature PE);
   performance evidence requires this v4.2.3-exp smoke.

**NOT a retrieval rule violation** (CLAUDE.md retrieval §):
segmentation-stage PE on dual encoder is upstream of query-time
retrieval; the importance score path is untouched.

**Construction**: the segmenter consumes ``(s_topic, s_flow)`` pairs.
Each step caches the computed ``r`` in ``self._cached_r``, then the
inherited V411 ``assign(s_topic)`` runs its sCRP/MAP scoring with the
overridden ``_delta_eff`` returning that cached ``r``. The topic
channel (mpnet) drives topic identity (sCRP counts, f0 centroids,
restart logic); the flow channel (DSE) only contributes to ``r``.

**State buffers** (parallel to V411's ``_prev_s``/``_recent``):
- ``_prev_s_flow``: previous-turn flow embedding (DSE)
- ``_recent_flow``: flow channel causal window (capped by ``m_flow``)

**Defaults** (from v4.2.2 smoke train-best per encoder):
- topic (mpnet): m=2, ρ=0.7, a=0.5, δ*=0.5594
- flow  (DSE):   m=2, ρ=0.5, a=0.0, δ*=0.4569
- weights:       w_topic=0.75, w_flow=0.25 (mpnet 우선 — Dialseg711/
  SuperDialseg 에서 mpnet 압승 정량 근거).

**Risks to monitor** (codex 2026-05-20):
- false re-entry from generic opener ("so", "anyway", ...)
- over-segment (flow channel too sensitive on short turns)
- noise correlation ρ(z_topic, z_flow) — high ρ ⇒ dual channel adds
  no info, only confidence
- both-high FP / both-low FN failure modes
"""

from __future__ import annotations

import math
import numpy as np

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411

# v4.2.2 smoke (2026-05-20) train-calibration defaults per encoder.
TOPIC_DEFAULTS = dict(
    encoder_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    m=2, rho=0.7, a=0.5, delta_star=0.5594,
)
FLOW_DEFAULTS = dict(
    encoder_name="aws-ai/dse-bert-base",
    m=2, rho=0.5, a=0.0, delta_star=0.4569,
)


class HiOnTopSegmenterV423Exp(HiOnTopSegmenterV411):
    """Dual-channel PE segmenter (v4.1.1 algorithm + multi-view scene rep).

    Args:
        dim: shared embedding dimension (both encoders MUST be 768d
            here — mpnet and DSE-BERT both are; mismatched dim is not
            supported).
        m_topic / rho_topic / a_topic / delta_star_topic:
            topic-channel (mpnet) causal-window hyperparameters from
            v4.1.1 train calibration (TIAGE-train best).
        m_flow / rho_flow / a_flow / delta_star_flow:
            flow-channel (DSE-BERT) causal-window hyperparameters from
            v4.2.2 train calibration.
        w_topic / w_flow: combination weights (sum need not equal 1 —
            r normalisation already happened via δ*). Default
            (0.75, 0.25) reflects smoke result that topic content
            dominates on formal benchmarks.
        **v411_kwargs: forwarded to V411 (alpha, lmda, beta, etc.).
            Note: ``delta_star`` is forced to 1.0 internally and
            ``ctx_window/ctx_decay/ctx_blend_a`` are taken from the
            topic-channel parameters (for parent compatibility — they
            drive V411's ``_recent``/``_delta_ctx`` on the topic
            channel; flow channel state is separate).

    Construction notes:
        - Internal ``delta_star = 1.0`` and ``σδ² = sigma_delta_c``
          (default 0.0625), so V411's ``_delta_loglik(r)`` evaluates
          ``L(1) = -8.0`` at boundary — identical magnitude to
          ``L(δ*)`` in v4.1.1 with δ*=0.5594 σδ_c=0.0625.
        - ``encoder_name_topic`` / ``encoder_name_flow`` are stored for
          run-time identity logging only; the segmenter itself does
          not load encoders (caller provides ``(s_topic, s_flow)``
          pairs via ``assign``).
    """

    def __init__(
        self,
        dim: int,
        m_topic: int = TOPIC_DEFAULTS["m"],
        rho_topic: float = TOPIC_DEFAULTS["rho"],
        a_topic: float = TOPIC_DEFAULTS["a"],
        delta_star_topic: float = TOPIC_DEFAULTS["delta_star"],
        m_flow: int = FLOW_DEFAULTS["m"],
        rho_flow: float = FLOW_DEFAULTS["rho"],
        a_flow: float = FLOW_DEFAULTS["a"],
        delta_star_flow: float = FLOW_DEFAULTS["delta_star"],
        w_topic: float = 0.75,
        w_flow: float = 0.25,
        encoder_name_topic: str | None = TOPIC_DEFAULTS["encoder_name"],
        encoder_name_flow: str | None = FLOW_DEFAULTS["encoder_name"],
        **v411_kwargs,
    ) -> None:
        if delta_star_topic <= 0.0 or delta_star_flow <= 0.0:
            raise ValueError("delta_star_topic/flow must be > 0 (z-score divisor)")
        if w_topic < 0.0 or w_flow < 0.0:
            raise ValueError("weights must be non-negative")
        if w_topic + w_flow <= 0.0:
            raise ValueError("at least one weight must be > 0")

        # Force parent's delta_star to 1.0 (normalized r) and use topic-
        # channel (m, ρ, a) for parent's _recent / _delta_prev / _delta_ctx
        # buffer governance. Caller-supplied delta_star (in v411_kwargs)
        # would be misleading; strip and warn.
        if "delta_star" in v411_kwargs:
            import warnings
            warnings.warn(
                "delta_star is forced to 1.0 in v4.2.3-exp (normalized r). "
                "Ignoring supplied value.",
                RuntimeWarning,
                stacklevel=2,
            )
            v411_kwargs.pop("delta_star")
        for k in ("ctx_window", "ctx_decay", "ctx_blend_a"):
            if k in v411_kwargs:
                import warnings
                warnings.warn(
                    f"v4.2.3-exp drives topic channel from m_topic/rho_topic/"
                    f"a_topic; ignoring supplied {k}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                v411_kwargs.pop(k)

        super().__init__(
            dim=dim,
            delta_star=1.0,
            ctx_window=m_topic,
            ctx_decay=rho_topic,
            ctx_blend_a=a_topic,
            **v411_kwargs,
        )

        # Channel-specific HPs
        self.m_topic = m_topic
        self.rho_topic = rho_topic
        self.a_topic = a_topic
        self.delta_star_topic = delta_star_topic
        self.m_flow = m_flow
        self.rho_flow = rho_flow
        self.a_flow = a_flow
        self.delta_star_flow = delta_star_flow
        self.w_topic = w_topic
        self.w_flow = w_flow
        self.encoder_name_topic = encoder_name_topic
        self.encoder_name_flow = encoder_name_flow

        # Flow channel state (parallel to V411's _prev_s / _recent on topic)
        self._prev_s_flow: np.ndarray | None = None
        self._recent_flow: list[np.ndarray] = []

        # Most recent computed r (cached for _delta_eff override)
        self._cached_r: float = 0.0

        # Diagnostic accumulators (set by assign each step; readable for
        # smoke logging — corr(z_topic, z_flow), both-high bucket, etc.)
        self.last_z_topic: float | None = None
        self.last_z_flow: float | None = None
        self.last_r: float | None = None

    # ------------------------------------------------------------------
    # Flow-channel δ helpers (mirror V411 _delta_prev / _delta_ctx)
    # ------------------------------------------------------------------

    def _delta_prev_flow(self, s_flow: np.ndarray) -> float | None:
        if self._prev_s_flow is None:
            return None
        a, b = self._prev_s_flow, s_flow
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return None
        return 1.0 - float(np.dot(a, b) / (na * nb))

    def _delta_ctx_flow(self, s_flow: np.ndarray) -> float | None:
        if not self._recent_flow:
            return None
        win = self._recent_flow[-self.m_flow:]
        c = np.zeros_like(win[-1], dtype=np.float64)
        for i, vec in enumerate(reversed(win)):
            c += (self.rho_flow ** i) * vec
        nc = float(np.linalg.norm(c))
        ns = float(np.linalg.norm(s_flow))
        if nc <= 1e-12 or ns <= 1e-12:
            return None
        return 1.0 - float(np.dot(c, s_flow) / (nc * ns))

    def _delta_flow_adj(self, s_flow: np.ndarray) -> float | None:
        """Flow-channel δ_adj = a_flow·δ_prev_flow + (1-a_flow)·δ_ctx_flow."""
        dp = self._delta_prev_flow(s_flow)
        if dp is None:
            return None
        dc = self._delta_ctx_flow(s_flow)
        if dc is None:
            return dp
        return self.a_flow * dp + (1.0 - self.a_flow) * dc

    def _delta_topic_adj(self, s_topic: np.ndarray) -> float | None:
        """Topic-channel δ_adj using parent's _delta_prev/_delta_ctx
        (which read _prev_s / _recent and the parent's ctx_* params,
        set to topic-channel values in __init__)."""
        dp = self._delta_prev(s_topic)
        if dp is None:
            return None
        dc = self._delta_ctx(s_topic)
        if dc is None:
            return dp
        return self.ctx_blend_a * dp + (1.0 - self.ctx_blend_a) * dc

    # ------------------------------------------------------------------
    # Energy combination (the v4.2.3 core)
    # ------------------------------------------------------------------

    def _combined_r(
        self, s_topic: np.ndarray, s_flow: np.ndarray
    ) -> tuple[float, float | None, float | None]:
        """Return (r, z_topic, z_flow) for the current step.

        - Stream start (no prev): r = 0.0 (no boundary).
        - Only topic available: r = z_topic.
        - Only flow available: r = z_flow (weighted by w_flow alone? —
          conservative: return z_flow without down-weighting since the
          weight blend is only meaningful when both present).
        - Both: r = sqrt(w_topic·z_topic² + w_flow·z_flow²).
        """
        d_topic = self._delta_topic_adj(s_topic)
        d_flow = self._delta_flow_adj(s_flow)
        z_t = (d_topic / self.delta_star_topic) if d_topic is not None else None
        z_f = (d_flow / self.delta_star_flow) if d_flow is not None else None
        if z_t is None and z_f is None:
            return 0.0, None, None
        if z_t is None:
            return float(z_f), None, float(z_f)
        if z_f is None:
            return float(z_t), float(z_t), None
        r = math.sqrt(self.w_topic * z_t * z_t + self.w_flow * z_f * z_f)
        return float(r), float(z_t), float(z_f)

    # ------------------------------------------------------------------
    # Override _delta_eff: V411 calls this in _scores; we substitute r.
    # ------------------------------------------------------------------

    def _delta_eff(self, k: int, s: np.ndarray) -> float:  # noqa: ARG002
        """v4.2.3 override: return the cached normalized energy r.

        The arguments ``k, s`` are ignored — the cached value was set by
        :meth:`assign` for the current step (s == s_topic from caller).
        """
        return self._cached_r

    # ------------------------------------------------------------------
    # Public API: assign takes a (s_topic, s_flow) pair
    # ------------------------------------------------------------------

    def assign_pair(
        self,
        s_topic: np.ndarray,
        s_flow: np.ndarray,
    ) -> tuple[int, bool]:
        """Dual-channel assign. Returns (topic_id, is_boundary).

        Caller must encode each turn with both encoders and pass the
        L2-normalized 768d vectors. Both arrays MUST share dim
        (default 768d for mpnet + DSE-BERT).
        """
        # 1. Compute combined r BEFORE super().assign updates buffers
        r, z_t, z_f = self._combined_r(s_topic, s_flow)
        self._cached_r = r
        self.last_z_topic = z_t
        self.last_z_flow = z_f
        self.last_r = r

        # 2. Run V411's sCRP/MAP/topic-state machinery with s_topic.
        #    This will: read _prev_s/_recent (topic), call _delta_eff
        #    (overridden → returns r), update topic state + _prev_s +
        #    _recent at the end.
        topic_id, is_boundary = super().assign(s_topic)

        # 3. Update flow-channel state (parallel to parent's topic update)
        sf_copy = np.array(s_flow, dtype=np.float64, copy=True)
        self._prev_s_flow = sf_copy
        self._recent_flow.append(sf_copy)
        if len(self._recent_flow) > self.m_flow:
            del self._recent_flow[0]

        return topic_id, is_boundary

    def assign(self, s):  # noqa: D401
        """v4.2.3-exp requires assign_pair(s_topic, s_flow). Single-vector
        assign is unsupported (would lose the dual channel)."""
        raise NotImplementedError(
            "HiOnTopSegmenterV423Exp requires assign_pair(s_topic, s_flow); "
            "single-vector assign() is not defined for the dual-channel variant."
        )
