"""v4.2.4-exp — DSE-BERT 가 v4.1.1 의 δ_model (RNN slot) 자리 대체.

Experimental fork of v4.1.1 (``sem_core_v411.HiOnTopSegmenterV411``). v4.1.1
의 식 구조 (``δ_eff² = η·δ_adj² + (1−η)·δ_model²``) 를 그대로 유지하되,
δ_model 의 산출처를 학습형 RNN PE 에서 **frozen pretrained DSE-BERT 의
causal-window PE** 로 교체.

```
δ_adj(t)   = a·δ_prev(t) + (1−a)·δ_ctx(t)              # mpnet (v4.1.1 그대로)
δ_model(t) = a_dse·δ_prev_dse(t) + (1−a_dse)·δ_ctx_dse(t)   # ★ DSE 로 RNN 대체
δ_eff²(t)  = η·δ_adj² + (1−η)·δ_model²
boundary ⇔ δ_eff < δ*                                   # δ* = 0.5594 (mpnet, 그대로)
```

**왜 "exp" suffix**: η < 1 일 때 raw scale 의 두 δ 를 그대로 섞으므로
``δ*=0.5594`` (mpnet 기준 calibration) 가 mixed δ_eff 에 그대로 맞으리란
보장이 없음. η sweep 으로 적합 weight 탐색 + train calibration 별도
검증 필요.

**v4.2.3-exp 와 차이**:
- v4.2.3: dual-channel calibrated energy ``r = √(w_t·z_t² + w_f·z_f²)``,
  각 채널 자기 δ\* 로 normalize. internal δ\*=1.0 trick.
- v4.2.4: v4.1.1 의 ``η·δ_adj² + (1−η)·δ_model²`` 형태 그대로. 두 δ 가 raw scale.
  δ\* = mpnet 의 0.5594 그대로. normalization 미적용.

**SEM 계승**:
- v4.1.1 본체는 RNN 을 보존하되 η=1 auto-skip (RNN 학습 환경 부족 진단).
- codex 2026-05-18 P2: "frozen-pretrained-f 제안 → 현 A3 유지, C 천장측정
  후 재검토". v4.2.4-exp 는 그 P2 후속 후보의 실현. SEM2 ``LinearEvent``
  의 learned dynamics 자리에 frozen pretrained encoder 의 causal-window
  PE 를 끼워넣은 형태.
- 3-step PASS conditional: SEM2 의 dynamics 자리 자체는 존재 (learned-f),
  v4.2.4 는 그것을 frozen-f 로 대체. sCRP / Bayes / local-MAP / B0 무변경.
  δ_eff 식 구조 (η blend) 도 v4.1.1 그대로. ⚠ 새로운 mechanism 이라기
  보다 "RNN 못 배우는 환경의 frozen 대체" framing 유지 필수.

**Operational notes**:
- ``use_rnn`` 은 강제로 False (parent 의 RNN compute 회피). ``eta_prev``
  는 default 0.5 — η=1 이면 v4.1.1 sanity (mpnet 만), η=0 이면 DSE PE
  단독 (mpnet δ_eff 자리에 DSE PE 가 raw 로 들어감, δ\* scale mismatch
  위험).
- DSE channel HP default = v4.2.2 train-best (m_dse=2, ρ_dse=0.5, a_dse=0.0).
- mpnet channel HP = parent 의 ctx_window/ctx_decay/ctx_blend_a (v4.1.1
  default: m=2, ρ=0.7, a=0.5).
- δ\*, σδ² 모두 parent (mpnet 기준) 그대로.

**Risks**:
- raw scale mismatch: mpnet δ ≈ 0.4~0.6, DSE δ ≈ 0.3~0.5. η blend 가 raw
  로 일어나서 effective weight 가 η 와 일치 안 할 수 있음.
- η sweep + 필요시 train re-calibration 으로 보정.
"""

from __future__ import annotations

import math
import numpy as np

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411

# v4.2.2 smoke train-best for DSE channel
DSE_DEFAULTS = dict(
    encoder_name="aws-ai/dse-bert-base",
    m=2, rho=0.5, a=0.0,
)


class HiOnTopSegmenterV424Exp(HiOnTopSegmenterV411):
    """DSE-BERT 가 δ_model 자리를 채우는 v4.1.1 확장.

    Args:
        dim: 768 (mpnet, DSE 둘 다 BERT-base).
        eta_prev: η ∈ [0, 1]. **default 0.5** (RNN 자리에 DSE 가 들어가니
            blend 활성화). η=1 = mpnet 단독 (v4.1.1 sanity), η=0 = DSE
            단독 (단 δ\* scale mismatch 위험).
        m_dse / rho_dse / a_dse: DSE channel causal-window HP (default
            v4.2.2 TIAGE-train best for DSE).
        encoder_name_dse: 라벨링 (실제 로드는 caller 측).
        **v411_kwargs: V411 default 상속. ``use_rnn`` 은 강제로 False
            로 override (RNN compute 회피).

    Operational:
        Caller 가 ``assign_pair(s_topic, s_dse)`` 로 두 encoder 의 normalized
        768d 벡터를 매 turn 마다 넘김. parent V411 의 모든 state (sCRP
        counts, f0 centroids, etc.) 는 s_topic (mpnet) 으로 운영됨.
        s_dse 는 ``_prev_s_dse`` / ``_recent_dse`` buffer 에 별도 보관.
    """

    def __init__(
        self,
        dim: int,
        eta_prev: float = 0.5,
        m_dse: int = DSE_DEFAULTS["m"],
        rho_dse: float = DSE_DEFAULTS["rho"],
        a_dse: float = DSE_DEFAULTS["a"],
        encoder_name_dse: str | None = DSE_DEFAULTS["encoder_name"],
        **v411_kwargs,
    ) -> None:
        if not 0.0 <= eta_prev <= 1.0:
            raise ValueError(f"eta_prev must be in [0, 1], got {eta_prev}")
        if m_dse < 1:
            raise ValueError(f"m_dse must be >= 1, got {m_dse}")
        if not 0.0 < rho_dse <= 1.0:
            raise ValueError(f"rho_dse must be in (0, 1], got {rho_dse}")
        if not 0.0 <= a_dse <= 1.0:
            raise ValueError(f"a_dse must be in [0, 1], got {a_dse}")

        # use_rnn 은 강제로 False (RNN compute / training 회피 — DSE 가 대체)
        v411_kwargs.pop("use_rnn", None)
        super().__init__(dim=dim, eta_prev=eta_prev, use_rnn=False, **v411_kwargs)

        self.m_dse = m_dse
        self.rho_dse = rho_dse
        self.a_dse = a_dse
        self.encoder_name_dse = encoder_name_dse

        # DSE channel state (parallel to V411's _prev_s / _recent)
        self._prev_s_dse: np.ndarray | None = None
        self._recent_dse: list[np.ndarray] = []

        # Cached δ_model (DSE-based) computed by assign_pair for the current turn
        self._cached_d_model: float = 0.0

        # Diagnostic (smoke logging)
        self.last_d_adj_mpnet: float | None = None
        self.last_d_model_dse: float | None = None
        self.last_d_eff: float | None = None

    # ------------------------------------------------------------------
    # DSE-channel δ helpers (mirror V411 _delta_prev / _delta_ctx)
    # ------------------------------------------------------------------

    def _delta_prev_dse(self, s_dse: np.ndarray) -> float | None:
        if self._prev_s_dse is None:
            return None
        a, b = self._prev_s_dse, s_dse
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return None
        return 1.0 - float(np.dot(a, b) / (na * nb))

    def _delta_ctx_dse(self, s_dse: np.ndarray) -> float | None:
        if not self._recent_dse:
            return None
        win = self._recent_dse[-self.m_dse:]
        c = np.zeros_like(win[-1], dtype=np.float64)
        for i, vec in enumerate(reversed(win)):
            c += (self.rho_dse ** i) * vec
        nc = float(np.linalg.norm(c))
        ns = float(np.linalg.norm(s_dse))
        if nc <= 1e-12 or ns <= 1e-12:
            return None
        return 1.0 - float(np.dot(c, s_dse) / (nc * ns))

    def _delta_model_dse(self, s_dse: np.ndarray) -> float | None:
        """δ_model 의 DSE 산출. = a_dse·δ_prev_dse + (1-a_dse)·δ_ctx_dse.

        Returns None at stream start (no prev/recent). After turn 2, both
        available.
        """
        dp = self._delta_prev_dse(s_dse)
        if dp is None:
            return None
        dc = self._delta_ctx_dse(s_dse)
        if dc is None:
            return dp
        return self.a_dse * dp + (1.0 - self.a_dse) * dc

    # ------------------------------------------------------------------
    # Override _delta_eff: substitute DSE δ_model for RNN PE
    # ------------------------------------------------------------------

    def _delta_eff(self, k: int, s: np.ndarray) -> float:  # noqa: ARG002
        """v4.1.1 의 δ_eff 식 그대로, ``d_model`` 만 cached DSE PE.

        - stream start (no mpnet prev): d_model_dse 만 (있으면), 아니면 0.
        - η = 1.0: pure mpnet (= v4.1.1 sanity).
        - η < 1.0: √(η·d_adj² + (1-η)·d_model²).
        """
        d_prev = self._delta_prev(s)
        d_model = self._cached_d_model

        if d_prev is None:
            # 첫 step — mpnet 정보 없음. d_model 만 (DSE) 사용 (있으면).
            self.last_d_adj_mpnet = None
            self.last_d_eff = d_model
            return d_model

        d_ctx = self._delta_ctx(s)
        a = self.ctx_blend_a
        d_adj = d_prev if d_ctx is None else a * d_prev + (1.0 - a) * d_ctx
        self.last_d_adj_mpnet = d_adj

        eta = self.eta_prev
        if eta >= 1.0:
            # v4.1.1 sanity (mpnet only)
            self.last_d_eff = d_adj
            return d_adj
        # η blend with DSE d_model
        d_eff = math.sqrt(eta * d_adj * d_adj + (1.0 - eta) * d_model * d_model)
        self.last_d_eff = d_eff
        return d_eff

    # ------------------------------------------------------------------
    # Public API: pair assign
    # ------------------------------------------------------------------

    def assign_pair(
        self,
        s_topic: np.ndarray,
        s_dse: np.ndarray,
    ) -> tuple[int, bool]:
        """Dual-encoder assign. mpnet drives topic state, DSE feeds δ_model slot.

        Returns: (topic_id, is_boundary).
        """
        # 1. Cache DSE δ_model for the upcoming _delta_eff call
        dm = self._delta_model_dse(s_dse)
        self._cached_d_model = 0.0 if dm is None else float(dm)
        self.last_d_model_dse = dm

        # 2. Run V411 sCRP/MAP using s_topic (mpnet)
        topic_id, is_boundary = super().assign(s_topic)

        # 3. Update DSE channel state
        sd_copy = np.array(s_dse, dtype=np.float64, copy=True)
        self._prev_s_dse = sd_copy
        self._recent_dse.append(sd_copy)
        if len(self._recent_dse) > self.m_dse:
            del self._recent_dse[0]

        return topic_id, is_boundary

    def assign(self, s):  # noqa: D401
        """v4.2.4-exp 는 assign_pair(s_topic, s_dse) 필요. 단일 벡터 미지원."""
        raise NotImplementedError(
            "HiOnTopSegmenterV424Exp requires assign_pair(s_topic, s_dse); "
            "single-vector assign() is undefined for this dual-encoder variant."
        )
