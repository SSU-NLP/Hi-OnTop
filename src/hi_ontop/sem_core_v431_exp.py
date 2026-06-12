r"""v4.3.1-exp — DialoGPT-small surprisal 이 v4.1.1 의 δ_model (RNN slot) 자리 대체.

Experimental fork of v4.1.1 (``sem_core_v411.HiOnTopSegmenterV411``). v4.1.1
의 식 구조 (``δ_eff² = η·δ_adj² + (1−η)·δ_model²``) 를 그대로 유지하되,
δ_model 의 산출처를 **학습형 RNN PE → frozen DialoGPT-small 의 causal
surprisal** 로 교체.

```
δ_adj(t)   = a·δ_prev(t) + (1−a)·δ_ctx(t)              # mpnet (v4.1.1 그대로)
δ_model(t) = mean_{w ∈ u_t}  −log P_LM(w | u_{t−m..t−1}, u_t<w)   # ★ DialoGPT
δ_eff²(t)  = η·δ_adj² + (1−η)·δ_model²
boundary ⇔ δ_eff < δ*                                   # δ* = 0.5594 (mpnet, 그대로)
```

**왜 "exp" suffix**: raw NLL 의 scale (≈ 2–8 nats/token) 이 mpnet δ 의
scale (≈ 0.4–0.6) 과 매우 다름. δ\* = 0.5594 (mpnet 기준 calibration) 가
mixed δ_eff 에 그대로 맞으리란 보장이 없음. η sweep + 별도 δ\*
re-calibration 검증 필요.

**v4.2.4-exp 와 차이**: v4.2.4 는 DSE-BERT (frozen sentence encoder) 의
causal-window cosine 으로 δ_model 을 만듦 — 즉 "embedding-space PE".
v4.3.1 은 **token-level autoregressive surprisal** 로 δ_model 을 만듦 —
"진짜 next-utterance prediction" 의 직접 구현. cosine 매칭 단 한 번도
거치지 않음. retrieval 규칙 (cosine 금지) 도 자연 준수.

**SEM 계승**:
- SEM2 ``LinearEvent`` 의 learned dynamics slot 에 frozen pretrained
  encoder 를 끼워넣는 형태 (v4.2.4 가 DSE 로 한 패턴 그대로). 단,
  대체 encoder 가 sentence-similarity 가 아닌 **causal LM surprisal**.
- prediction-by-production (Pickering & Garrod, 2013) 의 *심리언어학적
  근거* 와 SEM2 dynamics slot 이 만나는 지점. v4.1.1 에서 RNN 이 학습
  환경 부족으로 비활성된 자리를, 사전학습 대화 LM 의 surprisal 이 채움.
- 3-step PASS conditional: SEM2 의 dynamics slot 자체는 존재 (learned-f).
  v4.3.1 는 그것을 frozen LM-surprisal 로 대체. sCRP / Bayes / local-MAP
  / B0 무변경. δ_eff 식 구조 (η blend) 도 v4.1.1 그대로. likelihood
  `L(δ) = −δ²/(2σδ²)` 도 그대로 (단 δ scale 의 적합성은 caveat).
- ⚠ Framing: "다음 발화 생성적 확률 구현" 이라고 약하게 인용. CSM 의
  NSP-supervised next-utterance prediction 과 구분되는 점은 (a) frozen
  zero-shot (b) sCRP / local-MAP / context model 안에서 *PE channel* 로
  쓰임 (c) cosine-based retrieval / ranking 미사용.

**Operational notes**:
- ``use_rnn`` 은 강제로 False (parent 의 RNN compute 회피).
- caller 가 ``assign_pair(s_topic, nll_t)`` 로 매 turn 의 mpnet 임베딩과
  precomputed DialoGPT mean-token NLL 을 함께 넘김. NLL 계산은 비용이
  크므로 **외부 precompute → pkl 캐시 → segmenter 는 scalar 만 소비**
  형태 (v4.2.4 가 DSE 임베딩 캐시 쓴 패턴 그대로).
- 첫 turn (no prior context) 의 NLL 은 None / 0.0 처리. caller 가 0.0
  으로 넘기면 됨 (parent 의 stream-start fallback 과 자연 정합).
"""

from __future__ import annotations

import math
import numpy as np

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411

# DialoGPT-small default settings (참고용 라벨; 실제 forward 는 외부 precompute 가 담당)
DIALOGPT_DEFAULTS = dict(
    lm_name="microsoft/DialoGPT-small",
    context_window=5,
    alternate_roles=True,
    aggregate="mean_token",
)


class HiOnTopSegmenterV431Exp(HiOnTopSegmenterV411):
    r"""DialoGPT-small surprisal 이 δ_model 자리를 채우는 v4.1.1 확장.

    Args:
        dim: 768 (mpnet).
        eta_prev: η ∈ [0, 1]. **default 0.5** (RNN 자리에 LM surprisal 이
            들어가니 blend 활성화). η=1 = mpnet 단독 (v4.1.1 sanity),
            η=0 = LM surprisal 단독 (δ\* scale mismatch 위험 매우 큼).
        lm_name / context_window_lm / alternate_roles / lm_aggregate:
            라벨링 전용 (실제 LM forward 는 caller / precompute script
            측). 재현성 / methodology doc 정합용.
        **v411_kwargs: V411 default 상속. ``use_rnn`` 은 강제로 False.

    Operational:
        Caller 가 ``assign_pair(s_topic, nll_t)`` 로 매 turn 의 mpnet
        임베딩과 **precomputed** DialoGPT mean-token NLL (scalar) 을
        넘김. 첫 turn 은 nll_t = 0.0 (또는 None) 으로 처리.
    """

    def __init__(
        self,
        dim: int,
        eta_prev: float = 0.5,
        lm_name: str | None = DIALOGPT_DEFAULTS["lm_name"],
        context_window_lm: int = DIALOGPT_DEFAULTS["context_window"],
        alternate_roles: bool = DIALOGPT_DEFAULTS["alternate_roles"],
        lm_aggregate: str = DIALOGPT_DEFAULTS["aggregate"],
        **v411_kwargs,
    ) -> None:
        if not 0.0 <= eta_prev <= 1.0:
            raise ValueError(f"eta_prev must be in [0, 1], got {eta_prev}")
        if context_window_lm < 1:
            raise ValueError(f"context_window_lm must be >= 1, got {context_window_lm}")
        if lm_aggregate not in ("mean_token", "sum_token"):
            raise ValueError(
                f"lm_aggregate must be 'mean_token' or 'sum_token', got {lm_aggregate}"
            )

        # use_rnn 강제 False (RNN compute / training 회피 — LM surprisal 이 대체)
        v411_kwargs.pop("use_rnn", None)
        super().__init__(dim=dim, eta_prev=eta_prev, use_rnn=False, **v411_kwargs)

        self.lm_name = lm_name
        self.context_window_lm = context_window_lm
        self.alternate_roles = alternate_roles
        self.lm_aggregate = lm_aggregate

        # Cached δ_model (LM surprisal) for the current turn, set by assign_pair
        self._cached_d_model: float = 0.0

        # Diagnostic
        self.last_d_adj_mpnet: float | None = None
        self.last_d_model_lm: float | None = None
        self.last_d_eff: float | None = None

    # ------------------------------------------------------------------
    # Override _delta_eff: substitute LM surprisal for RNN PE
    # ------------------------------------------------------------------

    def _delta_eff(self, k: int, s: np.ndarray) -> float:  # noqa: ARG002
        """v4.1.1 의 δ_eff 식 그대로, ``d_model`` 만 cached LM surprisal.

        - stream start (no mpnet prev): d_model 만 (있으면), 아니면 0.
        - η = 1.0: pure mpnet (= v4.1.1 sanity, LM 정보 무시).
        - η < 1.0: √(η·d_adj² + (1-η)·d_model²).
        """
        d_prev = self._delta_prev(s)
        d_model = self._cached_d_model

        if d_prev is None:
            # 첫 step — mpnet 정보 없음. d_model 만 (LM) 사용.
            self.last_d_adj_mpnet = None
            self.last_d_eff = d_model
            return d_model

        d_ctx = self._delta_ctx(s)
        a = self.ctx_blend_a
        d_adj = d_prev if d_ctx is None else a * d_prev + (1.0 - a) * d_ctx
        self.last_d_adj_mpnet = d_adj

        eta = self.eta_prev
        if eta >= 1.0:
            self.last_d_eff = d_adj
            return d_adj
        d_eff = math.sqrt(eta * d_adj * d_adj + (1.0 - eta) * d_model * d_model)
        self.last_d_eff = d_eff
        return d_eff

    # ------------------------------------------------------------------
    # Public API: pair assign
    # ------------------------------------------------------------------

    def assign_pair(
        self,
        s_topic: np.ndarray,
        nll_t: float | None,
    ) -> tuple[int, bool]:
        """Mpnet 임베딩 + precomputed LM surprisal scalar.

        Args:
            s_topic: mpnet normalized 768d 벡터 (이번 turn).
            nll_t: u_t 의 mean-token NLL (causal-window prior context
                기준). 첫 turn 또는 unavailable 시 None / 0.0.

        Returns:
            (topic_id, is_boundary).
        """
        dm = 0.0 if nll_t is None else float(nll_t)
        self._cached_d_model = dm
        self.last_d_model_lm = dm

        topic_id, is_boundary = super().assign(s_topic)
        return topic_id, is_boundary

    def assign(self, s):  # noqa: D401
        """v4.3.1-exp 는 assign_pair(s_topic, nll_t) 필요. 단일 벡터 미지원."""
        raise NotImplementedError(
            "HiOnTopSegmenterV431Exp requires assign_pair(s_topic, nll_t); "
            "single-vector assign() is undefined for this LM-surprisal variant."
        )
