"""Online MAP segmenter — v3.3.3 강화안 (v3.3.3-2).

v3.3.3 (SEM2 ``log_likelihood_f0`` restart 분기 복원) 의 두 가지 약점을
보정한다.

1. **f0 centroid 가 cosine surrogate 환경에서 RNN repeat prediction 을 거의
   못 이김** → boundary-start embedding 의 top-M prototype list 를 유지하고
   ``f0_score = max_p cos(s, p)`` 로 평가한다. 같은 topic 의 시작 패턴이
   여러 개 일 수 있다는 가정을 SEM2 ``new_token`` 정신과 일치시킨다.

2. **hard ``restart_pe_threshold`` gate + ``restart_margin=0`` 이 실제로
   restart 분기를 거의 발화 안 시킴** → posterior odds 로 재정식화. 같은
   label 안에서 ``p_rst = e^{S_rst} / (e^{S_rep}+e^{S_rst}) > ρ_rst`` 이면
   restart 로 인정 (default ρ_rst=0.35).

학습/메모리 비용은 v3.3.3 와 동일 (RNN train_steps 그대로, prototype 갱신
은 cosine 평균 + L2 norm 만). 추가 EMA 없음.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


class HiOnTopSegmenterV333_2:
    """v3.3.3 + boundary-start prototype f0 + posterior-odds restart."""

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
        # v3.3.3-2 only
        f0_proto_max: int = 4,
        restart_p_threshold: float = 0.35,
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
        if not 0.0 <= restart_pe_min <= 2.0:
            raise ValueError(
                f"restart_pe_min must be in [0, 2], got {restart_pe_min}"
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
        self.restart_p_threshold = restart_p_threshold
        self.restart_pe_min = restart_pe_min

        self.model = EventRNN(input_dim=dim, hidden_dim=rnn_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rnn_lr)
        self.topics: list[TopicV33RNN] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        # f0 prototypes: per-topic list of unit-norm boundary-start embeddings.
        self._f0_protos: list[list[np.ndarray]] = []
        self._f0_counts: list[int] = []

        # Diagnostic counters (sweep-result interpretation; see methodology).
        self._n_assigns: int = 0                  # total assign() calls
        self._n_label_change: int = 0             # k != prev_k boundaries
        self._n_same_label_restart: int = 0       # k == prev_k & posterior-odds fired
        self._n_repeat_wins: int = 0              # k == prev_k & repeat ≥ restart at MAP
        self._n_restart_wins: int = 0             # k == prev_k & restart > repeat at MAP
        self._n_forced_new: int = 0               # _surprise_forces_new triggered

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
            self._f0_counts.append(0)

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
        """Max cosine over topic ``k``'s boundary-start prototypes.

        Cold-start (count < f0_min_starts) falls back to the topic centroid
        ``mu``. SEM2 ``log_likelihood_f0`` 의 cosine surrogate.
        """
        if self._f0_counts[k] < self.f0_min_starts or not self._f0_protos[k]:
            ref = self.topics[k].mu
            n = float(np.linalg.norm(ref))
            if n <= 1e-12:
                return 0.0
            return float(np.dot(ref / n, s))
        best = -1.0
        for p in self._f0_protos[k]:
            v = float(np.dot(p, s))
            if v > best:
                best = v
        return best

    def _update_f0(self, k: int, s: np.ndarray) -> None:
        """Append ``s`` (unit-norm) as a new boundary-start prototype.

        Capped at ``f0_proto_max`` — when full, replace the prototype with
        the highest cosine to ``s`` (i.e. merge into the closest existing
        cluster) by averaging + re-normalising.
        """
        self._f0_counts[k] += 1
        s_unit = s / max(float(np.linalg.norm(s)), 1e-12)
        protos = self._f0_protos[k]
        if len(protos) < self.f0_proto_max:
            protos.append(s_unit.astype(np.float64))
            return
        # Merge into nearest prototype.
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
    # Score / branch selection
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
                # Posterior odds: same-label restart is judged by softmax,
                # not a hard PE gate. ``restart_pe_min`` is an optional safety
                # floor (default 0 → off).
                pe_repeat = self.topics[k_int].prediction_error(s)
                if pe_repeat >= self.restart_pe_min:
                    log_scores[i] = max(repeat, restart)
                else:
                    log_scores[i] = repeat
            else:
                log_scores[i] = repeat
        return log_scores, repeat_scores, restart_scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # Posterior-odds restart trigger (only when label stays same).
        is_restart = False
        if (
            self.prev_k is not None
            and k == self.prev_k
            and not math.isinf(restart_scores[chosen_idx])
            and not math.isinf(repeat_scores[chosen_idx])
        ):
            r = restart_scores[chosen_idx]
            p = repeat_scores[chosen_idx]
            # Numerically stable softmax for two scores.
            m = max(r, p)
            er = math.exp(r - m)
            ep = math.exp(p - m)
            p_rst = er / (er + ep)
            if p_rst > self.restart_p_threshold:
                is_restart = True

        self._ensure_topic_slot(k)
        self.topics[k].update(s)
        self.counts[k] += 1

        is_label_change = self.prev_k is not None and k != self.prev_k
        is_boundary = is_label_change or is_restart
        if is_boundary or self.counts[k] == 1:
            self._update_f0(k, s)

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
