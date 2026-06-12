"""Online MAP segmenter — v3.3.3-2 강화안 (v3.3.3-3).

v3.3.3-2 의 LoCoMo iter1 sanity 결과 (acc ≈동일, R-mh@k -0.016, T1var 47→72):
restart 분기가 너무 자주 발화하여 한 topic 안 multi-hop evidence 가 episode
단위로 분산되고 segmentation 결정이 RNN init seed 에 흔들리는 문제. 본
버전은 다음 보수화로 그 회귀를 흡수하는 것이 목표.

1. **Restart 발화 보수화**:
   - posterior threshold ↑ (`restart_p_threshold` default `0.5`)
   - posterior margin (`restart_prob_margin` default `0.15`): `p_rst - p_rep > margin`
   - minimum-span hysteresis (`episode_min_span` default `4`): 직전 restart 후
     최소 turn 간격 안에서는 restart 비활성

2. **Prototype softening**: top-M `max` (seed-민감) → mean + logmeanexp 혼합
   - `f0sim = (1-γ) cos(s, mean(P)) + γ · logmeanexp_p cos(s, p)`
   - `γ = f0_proto_weight` (default `0.25`).

3. (Episode atomicity 의 retrieval-layer 통합은 별도 iter3 에서 STM 측 변경
   포함하여 다룸. 본 버전은 segmentation 만 변경.)

학습 추가 없음. RNN train_steps 와 latency 은 v3.3.3 / v3.3.3-2 와 동일.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


def _logmeanexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    s = sum(math.exp(v - m) for v in values)
    return m + math.log(s / len(values))


class HiOnTopSegmenterV333_3:
    """v3.3.3-2 + restart hysteresis + prototype softening."""

    def __init__(
        self,
        dim: int,
        alpha: float = 100.0,
        lmda: float = 10.0,
        tau: float = 50.0,
        cos_threshold: float = 0.9,
        beta: float = 0.25,
        pe_threshold: float = 0.5,
        k_max: int = 256,
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 3,
        rnn_max_context: int = 8,
        rnn_min_history: int = 2,
        # v3.3.3 carryover
        f0_tau: float | None = None,
        f0_min_starts: int = 2,
        # v3.3.3-2 carryover
        f0_proto_max: int = 4,
        # v3.3.3-3 only
        restart_p_threshold: float = 0.5,
        restart_prob_margin: float = 0.15,
        episode_min_span: int = 4,
        f0_proto_weight: float = 0.25,
        restart_pe_min: float = 0.0,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if rnn_train_steps < 0:
            raise ValueError(f"rnn_train_steps must be >= 0, got {rnn_train_steps}")
        if not 0.0 <= pe_threshold <= 1.0:
            raise ValueError(f"pe_threshold must be in [0, 1], got {pe_threshold}")
        if f0_proto_max < 1:
            raise ValueError(f"f0_proto_max must be >= 1, got {f0_proto_max}")
        if not 0.0 < restart_p_threshold < 1.0:
            raise ValueError(
                f"restart_p_threshold must be in (0, 1), got {restart_p_threshold}"
            )
        if not 0.0 <= restart_prob_margin < 1.0:
            raise ValueError(
                f"restart_prob_margin must be in [0, 1), got {restart_prob_margin}"
            )
        if episode_min_span < 1:
            raise ValueError(f"episode_min_span must be >= 1, got {episode_min_span}")
        if not 0.0 <= f0_proto_weight <= 1.0:
            raise ValueError(
                f"f0_proto_weight must be in [0, 1], got {f0_proto_weight}"
            )

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

        self.f0_tau = tau if f0_tau is None else f0_tau
        self.f0_min_starts = f0_min_starts
        self.f0_proto_max = f0_proto_max
        self.f0_proto_weight = f0_proto_weight

        self.restart_p_threshold = restart_p_threshold
        self.restart_prob_margin = restart_prob_margin
        self.episode_min_span = episode_min_span
        self.restart_pe_min = restart_pe_min

        self.model = EventRNN(input_dim=dim, hidden_dim=rnn_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rnn_lr)
        self.topics: list[TopicV33RNN] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        # f0 prototype state.
        self._f0_protos: list[list[np.ndarray]] = []
        self._f0_means: list[np.ndarray] = []
        self._f0_counts: list[int] = []

        # Episode hysteresis: per-topic, turns since last episode-start.
        self._episode_age: list[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            self._f0_protos.append([])
            self._f0_means.append(np.zeros(self.dim, dtype=np.float64))
            self._f0_counts.append(0)
            self._episode_age.append(0)

    def _surprise_forces_new(self, s: np.ndarray) -> bool:
        if not self.topics or self.pe_threshold >= 1.0:
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

    def _f0_score(self, k: int, s: np.ndarray) -> float:
        """Softened f0 score: (1-γ) cos(s, mean) + γ logmeanexp_p cos(s, p).

        Cold-start (count < f0_min_starts) → topic centroid mu fallback.
        """
        if self._f0_counts[k] < self.f0_min_starts or not self._f0_protos[k]:
            ref = self.topics[k].mu
            n = float(np.linalg.norm(ref))
            if n <= 1e-12:
                return 0.0
            return float(np.dot(ref / n, s))
        # Mean prototype.
        mean = self._f0_means[k]
        nm = float(np.linalg.norm(mean))
        cos_mean = float(np.dot(mean / max(nm, 1e-12), s)) if nm > 1e-12 else 0.0
        # logmeanexp over individual prototypes (numerical stable softmax of cos).
        cos_vals = [float(np.dot(p, s)) for p in self._f0_protos[k]]
        soft_max = _logmeanexp(cos_vals)
        gamma = self.f0_proto_weight
        return (1.0 - gamma) * cos_mean + gamma * soft_max

    def _update_f0(self, k: int, s: np.ndarray) -> None:
        self._f0_counts[k] += 1
        s_unit = s / max(float(np.linalg.norm(s)), 1e-12)
        s_unit = s_unit.astype(np.float64)
        # Update running mean (un-normalized; normalized at score time).
        n = self._f0_counts[k]
        self._f0_means[k] = self._f0_means[k] + (s_unit - self._f0_means[k]) / n
        # Append / merge into prototype list.
        protos = self._f0_protos[k]
        if len(protos) < self.f0_proto_max:
            protos.append(s_unit)
            return
        best_i, best_v = 0, -2.0
        for i, p in enumerate(protos):
            v = float(np.dot(p, s_unit))
            if v > best_v:
                best_v, best_i = v, i
        merged = (protos[best_i] + s_unit) / 2.0
        nrm = float(np.linalg.norm(merged))
        if nrm > 1e-12:
            merged = merged / nrm
        protos[best_i] = merged

    # ------------------------------------------------------------------
    # Score / branch
    # ------------------------------------------------------------------

    def _scores_with_repeat_restart(
        self, s: np.ndarray, prior: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        repeat_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)
        restart_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)
        for i, k in enumerate(active):
            k_int = int(k)
            log_prior = math.log(prior[k_int])
            if k_int >= len(self.topics):
                log_scores[i] = log_prior + self.tau * self.cos_threshold
                continue
            repeat_lik = self.tau * self.topics[k_int].log_likelihood(s)
            repeat = log_prior + repeat_lik
            repeat_scores[i] = repeat
            if k_int == self.prev_k:
                base = max((self.counts[k_int] + 1) ** self.beta, 1e-300)
                log_prior_no_sticky = math.log(base)
                f0_lik = self.f0_tau * self._f0_score(k_int, s)
                restart = log_prior_no_sticky + f0_lik
                restart_scores[i] = restart
                pe_repeat = self.topics[k_int].prediction_error(s)
                if pe_repeat >= self.restart_pe_min:
                    log_scores[i] = max(repeat, restart)
                else:
                    log_scores[i] = repeat
            else:
                log_scores[i] = repeat
        return log_scores, repeat_scores, restart_scores

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, repeat_scores, restart_scores = self._scores_with_repeat_restart(
            s, prior, active
        )

        forced = self._surprise_forces_new(s)
        if forced:
            new_idx = self._new_cluster_index_in_active(active)
            chosen_idx = new_idx if new_idx is not None else int(np.argmax(log_scores))
        else:
            chosen_idx = int(np.argmax(log_scores))
        k = int(active[chosen_idx])

        # Posterior odds + margin + hysteresis for same-label restart.
        is_restart = False
        if (
            self.prev_k is not None
            and k == self.prev_k
            and not math.isinf(restart_scores[chosen_idx])
            and not math.isinf(repeat_scores[chosen_idx])
        ):
            r = restart_scores[chosen_idx]
            p = repeat_scores[chosen_idx]
            m = max(r, p)
            er = math.exp(r - m)
            ep = math.exp(p - m)
            denom = er + ep
            p_rst = er / denom
            p_rep = ep / denom
            age = self._episode_age[k] if k < len(self._episode_age) else 0
            if (
                p_rst > self.restart_p_threshold
                and (p_rst - p_rep) > self.restart_prob_margin
                and age >= self.episode_min_span
            ):
                is_restart = True

        self._ensure_topic_slot(k)
        self.topics[k].update(s)
        self.counts[k] += 1

        is_label_change = self.prev_k is not None and k != self.prev_k
        is_boundary = is_label_change or is_restart

        if is_boundary or self.counts[k] == 1:
            self._update_f0(k, s)
            self._episode_age[k] = 0
        else:
            self._episode_age[k] += 1

        self.prev_k = k
        return k, is_boundary

    def predict_topic(self, s: np.ndarray) -> int:
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, _, _ = self._scores_with_repeat_restart(s, prior, active)

        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            if new_idx is not None:
                return int(active[new_idx])
        chosen_idx = int(np.argmax(log_scores))
        return int(active[chosen_idx])
