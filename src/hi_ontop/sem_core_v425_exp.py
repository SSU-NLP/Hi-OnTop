r"""v4.2.5-exp — finetuned BERT-base coherence model (CSM) 이 δ_model 자리 대체.

Experimental fork of v4.1.1 (``sem_core_v411.HiOnTopSegmenterV411``). v4.1.1
의 식 구조 (``δ_eff² = η·δ_adj² + (1−η)·δ_model²``) 그대로 유지하되,
δ_model 의 산출처를 **NSP-supervised CSM (Xing & Carenini 2021; lxing532/
Dialogue-Topic-Segmenter)** 의 incoherence probability 로 교체.

```
δ_adj(t)   = a·δ_prev(t) + (1−a)·δ_ctx(t)              # mpnet (v4.1.1 그대로)

# CSM 채널 (★ RNN 자리 대체):
p_coh(t)   = softmax(CoherenceNet([CLS] u_{t-1} [SEP] u_t [SEP]))[0]
δ_model(t) = 1 − p_coh(t)                              # incoherence prob

# Blend (v4.1.1 식 그대로 또는 calibrated z-blend)
δ_eff²(t)  = η·δ_adj² + (1−η)·δ_model²    (raw, default)
           = η·z_adj² + (1−η)·z_model²    (calibrated, z = δ/δ*)
boundary ⇔ δ_eff < δ*                                   # raw: 0.5594, z: 1.0
```

**왜 "exp" suffix**: CSM 학습 corpus (DailyDialog NSP triplet) 와
TIAGE/Dialseg711/SuperDialseg 의 domain shift, 그리고 raw incoherence
prob 의 scale (≈0~1 boundary 에서 큰 값) 이 mpnet δ_adj 와 scale
mismatch — calibrated z-blend 권장.

**v4.2.4 (frozen DSE) / v4.3.2 (continuous regression) 와 차이**:

| | v4.2.4 (DSE) | v4.3.2 (TX head) | **v4.2.5 (CSM)** |
|---|---|---|---|
| δ_model 산출 | 1 − cos(s_{t-1}, s_t) (DSE 임베딩) | 1 − cos(\hat{s}_t, s_t) (학습된 head) | 1 − softmax(decoder([CLS]))[0] |
| 학습 데이터 | 없음 (frozen) | DailyDialog cosine regression | DailyDialog NSP triplet (positive/negative ranking) |
| 학습 목적 | — | continuous (cosine loss) | **discriminative (marginal ranking, margin=1)** |
| Cosine retrieval 사용? | δ 정의에 cosine | δ 정의에 cosine | 0회 (softmax decoder) |
| Output 형태 | similarity (작을수록 같음) | similarity (작을수록 같음) | **probability** (작을수록 incoherent → 큰 boundary) |

→ v4.2.5 는 **discriminative NSP ranking** 방향의 head. v4.3.2 의 약점
(regression-to-mean) 을 *원리적으로* 회피할 수 있음 (positive/negative
margin 학습이 generic response collapse 방지).

**SEM 계승 3-step (PASS, conditional)**:
1. SEM2 ``LinearEvent`` 의 learned dynamics slot 에 frozen-encoder +
   learned-head. v4.3.2 와 같은 패턴 ([[v4.2.4]] 의 frozen-f 확장).
2. 충돌 없음: sCRP / Bayes / local-MAP / B0 무변경. δ_eff 식 (η blend)
   도 v4.1.1 그대로. likelihood ``L(δ) = −δ²/(2σδ²)`` 도 그대로.
3. decision-log: 2026-05-21 entry.

⚠ Framing: "frozen BERT-base + DailyDialog 로 NSP-supervised learned
coherence head 가 SEM2 dynamics slot 의 learned-f 자리에." CSM 의
*discriminative ranking* 학습이 *continuous regression* (v4.3.2) 의
한계를 회피하는지 직접 비교 데이터.

**Operational notes**:
- ``use_rnn`` 강제 False (parent 의 RNN compute 회피).
- caller 는 ``assign_pair(s_topic, delta_model_t)`` 로 매 turn 의 mpnet
  embedding + precomputed CSM δ_model (= 1 − coherent_prob) 를 넘김.
- 첫 turn (no prev) 의 δ_model 은 0.0.
- calibrated 모드: z_adj = δ_adj / δ*_adj, z_model = δ_model / δ*_model,
  internal δ\* = 1.0 (parent forced). v4.2.3 / v4.3.2 패턴.
"""

from __future__ import annotations

import math
import numpy as np

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411

V425_DEFAULTS = dict(
    encoder_bert="bert-base-uncased",
    coherence_model="CSM",  # lxing532/Dialogue-Topic-Segmenter
    train_corpus="DailyDialog",
    max_length=128,
)


class HiOnTopSegmenterV425Exp(HiOnTopSegmenterV411):
    r"""CSM (finetuned BERT-base coherence head) 이 δ_model 자리를 채우는 v4.1.1 확장.

    Args:
        dim: 768 (mpnet topic state).
        eta_prev: η ∈ [0, 1]. **default 0.5**. η=1 → v4.1.1 sanity.
        encoder_bert / coherence_model / train_corpus / max_length:
            라벨링 전용 (실제 inference 는 precompute 측).
        calibrated: v4.2.3 패턴 z-blend 활성. default False.
        delta_star_adj: mpnet z-score divisor (default 0.5594).
        delta_star_model: CSM z-score divisor (caller 가 train/test 에서
            산출, calibrated=True 일 때 필수).
        **v411_kwargs: V411 default 상속. ``use_rnn`` 강제 False.

    Operational:
        Caller 는 ``assign_pair(s_topic, delta_model_t)`` 로 매 turn 의
        mpnet 임베딩과 **precomputed** CSM δ_model (= 1 − coherent_prob)
        를 넘김.
    """

    def __init__(
        self,
        dim: int,
        eta_prev: float = 0.5,
        encoder_bert: str | None = V425_DEFAULTS["encoder_bert"],
        coherence_model: str | None = V425_DEFAULTS["coherence_model"],
        train_corpus: str | None = V425_DEFAULTS["train_corpus"],
        max_length: int = V425_DEFAULTS["max_length"],
        calibrated: bool = False,
        delta_star_adj: float = 0.5594,
        delta_star_model: float | None = None,
        **v411_kwargs,
    ) -> None:
        if not 0.0 <= eta_prev <= 1.0:
            raise ValueError(f"eta_prev must be in [0, 1], got {eta_prev}")
        if calibrated:
            if delta_star_adj <= 0.0:
                raise ValueError(f"delta_star_adj must be > 0, got {delta_star_adj}")
            if delta_star_model is None or delta_star_model <= 0.0:
                raise ValueError("calibrated=True requires delta_star_model > 0")

        v411_kwargs.pop("use_rnn", None)
        if calibrated:
            if "delta_star" in v411_kwargs:
                import warnings
                warnings.warn(
                    "delta_star forced to 1.0 in v4.2.5 calibrated mode "
                    "(z-score blend). Use delta_star_adj for mpnet z-score divisor.",
                    RuntimeWarning, stacklevel=2,
                )
                v411_kwargs.pop("delta_star")
            super().__init__(dim=dim, eta_prev=eta_prev, use_rnn=False,
                             delta_star=1.0, **v411_kwargs)
        else:
            super().__init__(dim=dim, eta_prev=eta_prev, use_rnn=False, **v411_kwargs)

        self.encoder_bert = encoder_bert
        self.coherence_model = coherence_model
        self.train_corpus = train_corpus
        self.max_length = max_length
        self.calibrated = calibrated
        self.delta_star_adj = delta_star_adj
        self.delta_star_model = delta_star_model

        self._cached_d_model: float = 0.0

        # Diagnostic
        self.last_d_adj_mpnet: float | None = None
        self.last_d_model_csm: float | None = None
        self.last_d_eff: float | None = None
        self.last_z_adj: float | None = None
        self.last_z_model: float | None = None

    def _delta_eff(self, k: int, s: np.ndarray) -> float:  # noqa: ARG002
        """δ_eff 산출. Raw or calibrated z-blend (v4.3.2 와 동일 로직)."""
        d_prev = self._delta_prev(s)
        d_model = self._cached_d_model

        if d_prev is None:
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

        if eta >= 1.0:
            self.last_d_eff = d_adj
            return d_adj
        d_eff = math.sqrt(eta * d_adj * d_adj + (1.0 - eta) * d_model * d_model)
        self.last_d_eff = d_eff
        return d_eff

    def assign_pair(
        self,
        s_topic: np.ndarray,
        delta_model_t: float | None,
    ) -> tuple[int, bool]:
        """Mpnet 임베딩 + precomputed CSM incoherence prob.

        Args:
            s_topic: mpnet normalized 768d 벡터 (이번 turn).
            delta_model_t: 1 − p_coherent([CLS] u_{t-1} [SEP] u_t [SEP]).
                첫 turn / context 부족 시 None / 0.0.

        Returns:
            (topic_id, is_boundary).
        """
        dm = 0.0 if delta_model_t is None else float(delta_model_t)
        self._cached_d_model = dm
        self.last_d_model_csm = dm

        topic_id, is_boundary = super().assign(s_topic)
        return topic_id, is_boundary

    def assign(self, s):  # noqa: D401
        """v4.2.5-exp 는 assign_pair(s_topic, delta_model_t) 필요. 단일 벡터 미지원."""
        raise NotImplementedError(
            "HiOnTopSegmenterV425Exp requires assign_pair(s_topic, delta_model_t); "
            "single-vector assign() is undefined for this CSM-head variant."
        )
