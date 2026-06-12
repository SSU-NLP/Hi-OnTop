"""Online MAP segmenter — v3.3.4 강화안 (v3.3.4-2).

v3.3.4 (per-topic cosine PE variance σ_k² 로 Gaussian-style likelihood) 의
약점을 두 가지 보정한다.

1. **σ_k² 가 작은 topic 에서 불안정** → Bayesian shrinkage 로 global prior
   ``σ_0²`` 와 EMA estimate 를 융합한다.

   $σ_{k,eff}^2 = (n_k/(n_k+c)) σ_k^2 + (c/(n_k+c)) σ_0^2$

   ``c = pe_var_shrink_c`` (default 8) 가 prior strength.

2. **Robust scale option** (default OFF) — outlier turn 한두 개가 EMA 분산을
   부풀려 topic 을 과도하게 넓히는 것을 막기 위해 absolute deviation EMA
   스케일을 옵션으로 제공한다 (Welford EMA 와 별도 EMA, 추가 비용 O(1)).

학습 추가 없음. EMA 1회 + cosine 1회만 추가 (v3.3.4 와 동일 latency).

Retrieval 측 turn-level rerank 는 STM embedding 저장이 필요해 본 버전에서는
함께 묶지 않고 segmentation 변경만 (REPORT 한계로 명시).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


class HiOnTopSegmenterV334_2:
    """v3.3.4 + per-topic σ² shrinkage + optional robust scale."""

    def __init__(
        self,
        dim: int,
        alpha: float = 100.0,
        lmda: float = 10.0,
        tau: float = 50.0,
        cos_threshold: float = 0.9,
        beta: float = 0.25,
        pe_threshold: float = 1.0,
        k_max: int = 256,
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 3,
        rnn_max_context: int = 8,
        rnn_min_history: int = 2,
        # v3.3.4 carryover
        pe_var_decay: float = 0.95,
        pe_var_min_samples: int = 5,
        pe_var_sigma0_sq: float = 0.04,
        pe_var_min_sq: float = 1e-4,
        pe_var_max_sq: float = 0.25,
        var_likelihood_weight: float = 1.0,
        hard_pe_fallback: bool = False,
        # v3.3.4-2 only
        pe_var_shrink_c: float = 8.0,
        pe_var_robust: bool = False,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if rnn_train_steps < 0:
            raise ValueError(f"rnn_train_steps must be >= 0, got {rnn_train_steps}")
        if not 0.0 <= pe_threshold <= 1.0:
            raise ValueError(f"pe_threshold must be in [0, 1], got {pe_threshold}")
        if not 0.0 < pe_var_decay < 1.0:
            raise ValueError(f"pe_var_decay must be in (0, 1), got {pe_var_decay}")
        if pe_var_min_samples < 1:
            raise ValueError(f"pe_var_min_samples must be >= 1, got {pe_var_min_samples}")
        if not pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq:
            raise ValueError(
                "expect pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq, got "
                f"{pe_var_min_sq}, {pe_var_sigma0_sq}, {pe_var_max_sq}"
            )
        if var_likelihood_weight <= 0:
            raise ValueError(f"var_likelihood_weight must be > 0, got {var_likelihood_weight}")
        if pe_var_shrink_c < 0:
            raise ValueError(f"pe_var_shrink_c must be >= 0, got {pe_var_shrink_c}")

        self.dim = dim
        self.alpha = alpha
        self.lmda = lmda
        self.tau = tau
        self.cos_threshold = cos_threshold
        self.beta = beta
        self.pe_threshold = pe_threshold
        self.k_max = k_max
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_lr = rnn_lr
        self.rnn_train_steps = rnn_train_steps
        self.rnn_max_context = rnn_max_context
        self.rnn_min_history = rnn_min_history

        self.pe_var_decay = pe_var_decay
        self.pe_var_rho = 1.0 - pe_var_decay
        self.pe_var_min_samples = pe_var_min_samples
        self.pe_var_sigma0_sq = pe_var_sigma0_sq
        self.pe_var_min_sq = pe_var_min_sq
        self.pe_var_max_sq = pe_var_max_sq
        self.var_likelihood_weight = var_likelihood_weight
        self.hard_pe_fallback = hard_pe_fallback
        self.pe_var_shrink_c = pe_var_shrink_c
        self.pe_var_robust = pe_var_robust

        self.model = EventRNN(input_dim=dim, hidden_dim=rnn_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rnn_lr)
        self.topics: list[TopicV33RNN] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        self._pe_mean: list[float] = []
        self._pe_var: list[float] = []          # Welford-EMA squared deviation
        self._pe_abs_dev: list[float] = []      # robust EMA |PE - mean|
        self._pe_var_count: list[int] = []

    def _new_topic(self) -> TopicV33RNN:
        return TopicV33RNN(
            topic_id=len(self.topics),
            dim=self.dim,
            hidden_dim=self.rnn_hidden_dim,
            lr=self.rnn_lr,
            train_steps=self.rnn_train_steps,
            min_rnn_history=self.rnn_min_history,
            model=self.model,
            optimizer=self.optimizer,
        )

    def _ensure_topic_slot(self, k: int) -> None:
        while len(self.topics) <= k:
            self.topics.append(self._new_topic())
            self._pe_mean.append(0.0)
            self._pe_var.append(self.pe_var_sigma0_sq)
            self._pe_abs_dev.append(math.sqrt(self.pe_var_sigma0_sq))
            self._pe_var_count.append(0)

    def _raw_sigma_sq(self, k: int) -> float:
        if self.pe_var_robust:
            # 1.4826 * MAD ≈ Gaussian σ. Use abs deviation EMA.
            scale = 1.4826 * self._pe_abs_dev[k]
            return float(np.clip(scale * scale, self.pe_var_min_sq, self.pe_var_max_sq))
        return float(np.clip(self._pe_var[k], self.pe_var_min_sq, self.pe_var_max_sq))

    def _sigma_sq_for(self, k: int) -> float:
        """Shrinkage-stabilised σ_eff².

        Cold-start (count < min_samples) uses σ_0² directly. Otherwise blend
        ``raw_σ²`` with ``σ_0²`` weighted by ``n_k/(n_k+c)``.
        """
        n = self._pe_var_count[k]
        if n < self.pe_var_min_samples:
            return self.pe_var_sigma0_sq
        raw = self._raw_sigma_sq(k)
        c = self.pe_var_shrink_c
        if c <= 0:
            return raw
        eta = n / (n + c)
        eff = eta * raw + (1.0 - eta) * self.pe_var_sigma0_sq
        return float(np.clip(eff, self.pe_var_min_sq, self.pe_var_max_sq))

    def _calibrated_loglik(self, pe: float, sigma_sq: float) -> float:
        return -(pe * pe) / (2.0 * sigma_sq)

    def _new_cluster_score_calibrated(self) -> float:
        pe0 = 1.0 - self.cos_threshold
        return self.var_likelihood_weight * self._calibrated_loglik(
            pe0, self.pe_var_sigma0_sq
        )

    def _surprise_forces_new(self, s: np.ndarray) -> bool:
        if (
            not self.hard_pe_fallback
            or not self.topics
            or self.pe_threshold >= 1.0
        ):
            return False
        cos_max = -float("inf")
        threshold = 1.0 - self.pe_threshold
        for topic in self.topics:
            c = topic.cosine_score(s)
            if c > cos_max:
                cos_max = c
            if cos_max >= threshold:
                return False
        return True

    def _new_cluster_index_in_active(self, active: np.ndarray) -> int | None:
        for i, k in enumerate(active):
            if int(k) >= len(self.topics):
                return i
        return None

    def _scores(
        self, s: np.ndarray, prior: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, list[float | None]]:
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        per_pe: list[float | None] = [None] * active.shape[0]
        for i, k in enumerate(active):
            k_int = int(k)
            log_prior = math.log(prior[k_int])
            if k_int >= len(self.topics):
                log_scores[i] = log_prior + self._new_cluster_score_calibrated()
                continue
            pe = self.topics[k_int].prediction_error(s)
            sigma_sq = self._sigma_sq_for(k_int)
            log_lik = self.var_likelihood_weight * self._calibrated_loglik(pe, sigma_sq)
            log_scores[i] = log_prior + log_lik
            per_pe[i] = pe
        return log_scores, per_pe

    def _update_pe_stats(self, k: int, pe: float) -> None:
        rho = self.pe_var_rho
        old_mean = self._pe_mean[k]
        new_mean = (1.0 - rho) * old_mean + rho * pe
        new_var = (1.0 - rho) * self._pe_var[k] + rho * (pe - old_mean) * (pe - new_mean)
        new_abs = (1.0 - rho) * self._pe_abs_dev[k] + rho * abs(pe - new_mean)
        self._pe_mean[k] = new_mean
        self._pe_var[k] = float(np.clip(new_var, self.pe_var_min_sq, self.pe_var_max_sq))
        self._pe_abs_dev[k] = float(max(new_abs, math.sqrt(self.pe_var_min_sq)))
        self._pe_var_count[k] += 1

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, per_pe = self._scores(s, prior, active)

        forced = self._surprise_forces_new(s)
        if forced:
            new_idx = self._new_cluster_index_in_active(active)
            chosen_idx = new_idx if new_idx is not None else int(np.argmax(log_scores))
        else:
            chosen_idx = int(np.argmax(log_scores))
        k = int(active[chosen_idx])

        self._ensure_topic_slot(k)
        pe_for_chosen = per_pe[chosen_idx]
        if pe_for_chosen is not None:
            self._update_pe_stats(k, pe_for_chosen)

        self.topics[k].update(s)
        self.counts[k] += 1

        is_boundary = self.prev_k is not None and k != self.prev_k
        self.prev_k = k
        return k, is_boundary

    def predict_topic(self, s: np.ndarray) -> int:
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, _ = self._scores(s, prior, active)

        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            if new_idx is not None:
                return int(active[new_idx])
        chosen_idx = int(np.argmax(log_scores))
        return int(active[chosen_idx])
