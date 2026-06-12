r"""v4.3.2-exp — frozen Sentence-T5 + learned NextEmbedHead 가 δ_model 자리 대체.

Experimental fork of v4.1.1 (``sem_core_v411.HiOnTopSegmenterV411``). v4.1.1
의 식 구조 (``δ_eff² = η·δ_adj² + (1−η)·δ_model²``) 그대로 유지하되,
δ_model 의 산출처를 **frozen Sentence-T5 위에 DailyDialog 로 학습한
NextEmbedHead 의 예측 오차** 로 교체.

```
δ_adj(t)   = a·δ_prev(t) + (1−a)·δ_ctx(t)              # mpnet (v4.1.1 그대로)

\hat{s}_t  = head_θ(s_st5[t-m..t-1])                  # ★ learned next-embedding
δ_model(t) = 1 − cos_ST5(\hat{s}_t, s_st5[t])         # prediction error in ST5 space

δ_eff²(t)  = η·δ_adj² + (1−η)·δ_model²
boundary ⇔ δ_eff < δ*                                  # δ* = 0.5594 (mpnet)
```

**왜 "exp" suffix**: NextEmbedHead 의 head architecture / DailyDialog
학습 분포 / out-of-domain generalization (TIAGE/Dialseg711/SuperDialseg)
모두 미검증. δ_model raw scale 은 ST5 embedding-space cosine distance
이므로 mpnet δ_adj scale (0.4~0.6) 과 *원리적으로 유사* — 다만 head 가
under/over-fit 될 경우 distribution shift 가능.

**v4.3.1-exp 와 차이**: v4.3.1 (DialoGPT) 은 token-level autoregressive
surprisal (raw NLL, scale 2~8) — "다음 token 확률" 정의. v4.3.2 는
embedding-space prediction error (scale ≈ 0.3~0.7) — "다음 *발화 의미
벡터* 예측" 정의. 둘 다 'next-utterance prediction' 이지만 *공간이
다름*. CSM (NSP-supervised cosine) 과의 구분은 다음과 같다:

- CSM: 학습형 응집성 score (positive/negative ranking, hinge loss).
  → "응집적인가" 의 binary-discriminative 학습.
- v4.3.2: 학습형 *generative-style* next-embedding regression
  (cosine loss). → "다음 *벡터 가 무엇인가*" 의 continuous prediction.
- 게다가 boundary 는 sCRP + local-MAP 가 결정. NextEmbedHead 의 cosine
  score 가 직접 boundary rule 이 되는 게 아님.

**SEM 계승**:
- SEM2 ``LinearEvent`` 의 learned dynamics slot 에 frozen-encoder +
  learned-head 형태로 끼움. 정확히는 *SEM2 본래의 자리 그 자체* (SEM2
  도 dynamics 가 학습형 신경망). v4.3.2 는 SEM2 와 가장 가까운 구현일
  수 있음. 다만 *per-event* learning (SEM2) → *corpus-wide pretrained*
  (v4.3.2) 라는 변형 존재.
- 3-step PASS conditional: SEM2 의 dynamics slot 존재, v4.3.2 는 *학습
  scope* 만 corpus-wide 로 확장. sCRP / Bayes / local-MAP / B0 무변경.
  δ_eff 식 (η blend) 도 v4.1.1 그대로. likelihood `L(δ) = −δ²/(2σδ²)`
  도 그대로. δ scale 도 cosine-distance 기반이라 mpnet 과 자연 양립.
- ⚠ Framing: "DailyDialog 로 next-utterance regressor 를 frozen ST5
  위에 학습, SEM2 dynamics 자리에 주입" 으로 기술. CSM 류 (NSP
  supervised cosine ranking) 와는 학습 목적 함수가 다름을 명시.

**Operational notes**:
- ``use_rnn`` 은 강제로 False (parent 의 RNN compute 회피).
- Caller 는 ``assign_pair(s_topic, delta_model_t)`` 사용. δ_model 은
  precompute script (`precompute_v432_delta.py`) 가 NextEmbedHead 로
  계산해 캐시된 scalar.
- 첫 turn (no prior ST5 context) 의 δ_model 은 0.0 (caller 전달).
- 단일 벡터 ``assign`` 은 ``NotImplementedError``.
"""

from __future__ import annotations

import math
import numpy as np

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411

V432_DEFAULTS = dict(
    encoder_st5="sentence-transformers/sentence-t5-base",
    context_window=5,
    head_arch="mlp",
    head_hidden=1024,
    loss="cosine",
    train_corpus="DailyDialog",
)


class HiOnTopSegmenterV432Exp(HiOnTopSegmenterV411):
    r"""Learned next-embedding head (ST5 frozen) 이 δ_model 자리를 채우는 v4.1.1 확장.

    Args:
        dim: 768 (mpnet topic state). ST5 channel 의 d 는 head 내부에서만 사용.
        eta_prev: η ∈ [0, 1]. **default 0.5**. η=1 → v4.1.1 sanity.
        encoder_st5 / context_window_st5 / head_arch / head_hidden /
            train_corpus: 라벨링 전용 (실제 head forward 는 precompute 측).
        **v411_kwargs: V411 default 상속. ``use_rnn`` 은 강제로 False.

    Operational:
        Caller 는 ``assign_pair(s_topic, delta_model_t)`` 로 매 turn 의
        mpnet 임베딩과 **precomputed** δ_model (scalar, = 1−cos(\\hat{s},
        s) in ST5 space) 을 넘김.
    """

    def __init__(
        self,
        dim: int,
        eta_prev: float = 0.5,
        encoder_st5: str | None = V432_DEFAULTS["encoder_st5"],
        context_window_st5: int = V432_DEFAULTS["context_window"],
        head_arch: str = V432_DEFAULTS["head_arch"],
        head_hidden: int = V432_DEFAULTS["head_hidden"],
        train_corpus: str | None = V432_DEFAULTS["train_corpus"],
        calibrated: bool = False,
        delta_star_adj: float = 0.5594,
        delta_star_model: float | None = None,
        **v411_kwargs,
    ) -> None:
        if not 0.0 <= eta_prev <= 1.0:
            raise ValueError(f"eta_prev must be in [0, 1], got {eta_prev}")
        if context_window_st5 < 1:
            raise ValueError(f"context_window_st5 must be >= 1, got {context_window_st5}")
        if calibrated:
            if delta_star_adj <= 0.0:
                raise ValueError(f"delta_star_adj must be > 0, got {delta_star_adj}")
            if delta_star_model is None or delta_star_model <= 0.0:
                raise ValueError("calibrated=True requires delta_star_model > 0")

        v411_kwargs.pop("use_rnn", None)
        # In calibrated mode, force parent's δ* = 1.0 (z-score blend already scaled).
        # In raw mode, parent's δ* comes from v411_kwargs (default 0.5594).
        if calibrated:
            if "delta_star" in v411_kwargs:
                import warnings
                warnings.warn(
                    "delta_star is forced to 1.0 in v4.3.2 calibrated mode "
                    "(z-score blend). Use delta_star_adj for the mpnet z-score divisor.",
                    RuntimeWarning, stacklevel=2,
                )
                v411_kwargs.pop("delta_star")
            super().__init__(dim=dim, eta_prev=eta_prev, use_rnn=False,
                             delta_star=1.0, **v411_kwargs)
        else:
            super().__init__(dim=dim, eta_prev=eta_prev, use_rnn=False, **v411_kwargs)

        self.encoder_st5 = encoder_st5
        self.context_window_st5 = context_window_st5
        self.head_arch = head_arch
        self.head_hidden = head_hidden
        self.train_corpus = train_corpus
        self.calibrated = calibrated
        self.delta_star_adj = delta_star_adj
        self.delta_star_model = delta_star_model

        self._cached_d_model: float = 0.0

        # Diagnostic
        self.last_d_adj_mpnet: float | None = None
        self.last_d_model_st5: float | None = None
        self.last_d_eff: float | None = None
        self.last_z_adj: float | None = None
        self.last_z_model: float | None = None

    # ------------------------------------------------------------------
    # Override _delta_eff: substitute ST5-head prediction error for RNN PE
    # ------------------------------------------------------------------

    def _delta_eff(self, k: int, s: np.ndarray) -> float:  # noqa: ARG002
        """δ_eff 산출.

        Raw mode (default, calibrated=False):
            v4.1.1 식 그대로. δ_eff² = η·δ_adj² + (1-η)·δ_model².
            Parent 의 δ* (default 0.5594) 와 비교.

        Calibrated mode (calibrated=True, v4.2.3 pattern):
            z_adj   = δ_adj / δ*_adj      (δ*_adj = mpnet TIAGE-train, 0.5594)
            z_model = δ_model / δ*_model  (caller 가 train 셋에서 별도 산출)
            z_eff² = η·z_adj² + (1-η)·z_model²
            Internal δ* = 1.0 (parent forced). boundary ⇔ z_eff < 1.

        Raw mode 의 scale mismatch (δ_model << δ_adj → 과합병) 를 calibrated
        mode 가 해소. 두 채널을 자기 δ* 로 normalize 한 후 blend.
        """
        d_prev = self._delta_prev(s)
        d_model = self._cached_d_model

        if d_prev is None:
            # Stream start — mpnet 정보 없음
            self.last_d_adj_mpnet = None
            if self.calibrated:
                z_model = d_model / self.delta_star_model if self.delta_star_model else 0.0
                self.last_z_model = z_model
                self.last_d_eff = z_model
                return z_model
            self.last_d_eff = d_model
            return d_model

        d_ctx = self._delta_ctx(s)
        a = self.ctx_blend_a
        d_adj = d_prev if d_ctx is None else a * d_prev + (1.0 - a) * d_ctx
        self.last_d_adj_mpnet = d_adj
        eta = self.eta_prev

        if self.calibrated:
            z_adj = d_adj / self.delta_star_adj
            z_model = d_model / self.delta_star_model
            self.last_z_adj = z_adj
            self.last_z_model = z_model
            if eta >= 1.0:
                self.last_d_eff = z_adj
                return z_adj
            z_eff = math.sqrt(eta * z_adj * z_adj + (1.0 - eta) * z_model * z_model)
            self.last_d_eff = z_eff
            return z_eff

        # Raw mode (legacy)
        if eta >= 1.0:
            self.last_d_eff = d_adj
            return d_adj
        d_eff = math.sqrt(eta * d_adj * d_adj + (1.0 - eta) * d_model * d_model)
        self.last_d_eff = d_eff
        return d_eff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_pair(
        self,
        s_topic: np.ndarray,
        delta_model_t: float | None,
    ) -> tuple[int, bool]:
        """Mpnet 임베딩 + precomputed ST5-head prediction-error δ.

        Args:
            s_topic: mpnet normalized 768d 벡터 (이번 turn).
            delta_model_t: 1 − cos(\\hat{s}_t, s_st5[t]). 첫 turn / context
                부족 시 None / 0.0.

        Returns:
            (topic_id, is_boundary).
        """
        dm = 0.0 if delta_model_t is None else float(delta_model_t)
        self._cached_d_model = dm
        self.last_d_model_st5 = dm

        topic_id, is_boundary = super().assign(s_topic)
        return topic_id, is_boundary

    def assign(self, s):  # noqa: D401
        """v4.3.2-exp 는 assign_pair(s_topic, delta_model_t) 필요. 단일 벡터 미지원."""
        raise NotImplementedError(
            "HiOnTopSegmenterV432Exp requires assign_pair(s_topic, delta_model_t); "
            "single-vector assign() is undefined for this learned-head variant."
        )
