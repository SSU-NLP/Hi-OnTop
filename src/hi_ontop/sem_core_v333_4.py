"""Online MAP segmenter — v3.3.3-x new series (v3.3.3-4).

iter1/iter2 (v3.3.3-2 / v3.3.3-3) 의 segmentation-only 변경이 prefill 에 효과
없음을 확인하고 폐기. v3.3.3-4 는 codex iter3 권장에 따라 **minimal segmenter +
retrieval atomicity** 방향으로 전환:

1. Segmenter 자체는 v3.3.3 baseline 에 가깝게 보수화:
   - `restart_p_threshold=0.5`, `restart_prob_margin=0.15`, `episode_min_span=8`
   - `f0_proto_weight=0.0`, `f0_proto_max=1` (single f0 centroid; prototype list 비활성)
2. **(topic_id, episode_id) emit**: same-label restart 또는 label-change 시
   해당 topic 의 episode_id 를 증가. retrieval atomic unit = episode (SEM2
   `new_token` 직접 대응).

본 segmenter 는 episode_id 만 추가; retrieval 측 변경 (episode rerank +
dormant LTM safety) 은 orchestrator/MemoryWindow 측에서 처리.

학습 추가 없음, latency 동일.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hi_ontop.event_rnn import EventRNN
from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v331_rnn import TopicV33RNN


class HiOnTopSegmenterV333_4:
    """v3.3.3 + episode_id emit + 보수화된 same-label restart."""

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
        # Boundary 보수화 (v3.3.3-3 에서 가져옴)
        restart_p_threshold: float = 0.5,
        restart_prob_margin: float = 0.15,
        episode_min_span: int = 8,
        restart_pe_min: float = 0.0,
        # f0 prototype (default = single centroid, prototype 비활성)
        f0_proto_max: int = 1,
        f0_proto_weight: float = 0.0,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
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
        if f0_proto_max < 1:
            raise ValueError(f"f0_proto_max must be >= 1, got {f0_proto_max}")
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

        # f0 state.
        self._f0_protos: list[list[np.ndarray]] = []
        self._f0_means: list[np.ndarray] = []
        self._f0_counts: list[int] = []

        # Episode state (per topic).
        self._episode_id: list[int] = []          # current episode_id for each topic
        self._episode_age: list[int] = []         # turns since last episode boundary

        # PE running stats per topic (for importance v2: SEM-internal salience).
        # Updated each assign(): running mean + max of PE = 1 - cos(s, ŝ_k).
        self._pe_mean: list[float] = []
        self._pe_max: list[float] = []
        self._pe_count: list[int] = []

        # Diagnostic counters.
        self._n_assigns: int = 0
        self._n_label_change: int = 0
        self._n_same_label_restart: int = 0
        self._n_repeat_wins: int = 0
        self._n_restart_wins: int = 0
        self._n_hysteresis_blocked: int = 0

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
            self._episode_id.append(0)
            self._episode_age.append(0)
            self._pe_mean.append(0.0)
            self._pe_max.append(0.0)
            self._pe_count.append(0)

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
        if self._f0_counts[k] < self.f0_min_starts or not self._f0_protos[k]:
            ref = self.topics[k].mu
            n = float(np.linalg.norm(ref))
            if n <= 1e-12:
                return 0.0
            return float(np.dot(ref / n, s))
        # Default: single centroid (proto_weight=0).
        mean = self._f0_means[k]
        nm = float(np.linalg.norm(mean))
        cos_mean = float(np.dot(mean / max(nm, 1e-12), s)) if nm > 1e-12 else 0.0
        if self.f0_proto_weight <= 0.0 or self.f0_proto_max <= 1:
            return cos_mean
        # Optional logmeanexp soft mix (kept for ablation; default off).
        cos_vals = [float(np.dot(p, s)) for p in self._f0_protos[k]]
        m = max(cos_vals)
        soft_max = m + math.log(sum(math.exp(v - m) for v in cos_vals) / len(cos_vals))
        return (1.0 - self.f0_proto_weight) * cos_mean + self.f0_proto_weight * soft_max

    def _update_f0(self, k: int, s: np.ndarray) -> None:
        self._f0_counts[k] += 1
        s_unit = (s / max(float(np.linalg.norm(s)), 1e-12)).astype(np.float64)
        n = self._f0_counts[k]
        self._f0_means[k] = self._f0_means[k] + (s_unit - self._f0_means[k]) / n
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
    # Score
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, s: np.ndarray) -> tuple[int, bool, int]:
        """Return ``(topic_id, is_boundary, episode_id)``.

        ``episode_id`` is per-topic and increases on every boundary event
        (label-change OR same-label restart).
        """
        self._n_assigns += 1
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
            if r > p:
                self._n_restart_wins += 1
            else:
                self._n_repeat_wins += 1
            age = self._episode_age[k] if k < len(self._episode_age) else 0
            posterior_ok = (
                p_rst > self.restart_p_threshold
                and (p_rst - p_rep) > self.restart_prob_margin
            )
            if posterior_ok and age >= self.episode_min_span:
                is_restart = True
            elif posterior_ok and age < self.episode_min_span:
                self._n_hysteresis_blocked += 1

        self._ensure_topic_slot(k)
        # PE = 1 - cos(s, predicted_next). Use *before* topic.update(s) (predictor
        # vs incoming embedding). For brand-new topics (count==0) skip — predictor
        # hasn't learned anything yet, PE meaningless.
        if self.counts[k] >= 1:
            pe = float(self.topics[k].prediction_error(s))
            pe = max(0.0, min(2.0, pe))  # bounded
            n_prev = self._pe_count[k]
            n_new = n_prev + 1
            self._pe_mean[k] = (self._pe_mean[k] * n_prev + pe) / n_new
            self._pe_max[k] = max(self._pe_max[k], pe)
            self._pe_count[k] = n_new
        self.topics[k].update(s)
        self.counts[k] += 1

        is_label_change = self.prev_k is not None and k != self.prev_k
        is_boundary = is_label_change or is_restart

        if is_label_change:
            self._n_label_change += 1
        if is_restart:
            self._n_same_label_restart += 1

        if is_boundary or self.counts[k] == 1:
            self._update_f0(k, s)
            self._episode_id[k] += 1  # bump episode counter
            self._episode_age[k] = 0
        else:
            self._episode_age[k] += 1

        self.prev_k = k
        return k, is_boundary, self._episode_id[k]

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

    # ------------------------------------------------------------------
    # SEM-internal salience exports (for importance v2 — 2026-05-11)
    # ------------------------------------------------------------------

    def pe_stats(self) -> dict[int, dict[str, float]]:
        """Per-topic PE running stats. Used by ``compute_importance_v2``.

        Returns: ``{topic_id: {"mean": float, "max": float, "count": int}}``.
        """
        out: dict[int, dict[str, float]] = {}
        for k in range(len(self.topics)):
            if self._pe_count[k] > 0:
                out[k] = {
                    "mean": float(self._pe_mean[k]),
                    "max": float(self._pe_max[k]),
                    "count": int(self._pe_count[k]),
                }
        return out

    def boundary_counts(self) -> dict[int, int]:
        """Per-topic episode boundary count (= number of episodes started).

        Equals ``_episode_id[k] + 1`` for topics with ≥1 assignment.
        """
        return {
            k: int(self._episode_id[k] + 1)
            for k in range(len(self.topics))
            if self.counts[k] > 0
        }
