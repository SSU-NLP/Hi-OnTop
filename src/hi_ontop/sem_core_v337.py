"""Online MAP segmenter — v3.3.6 + SEM2 ``map_variance`` σ² (v3.3.7).

v3.3.6 fixed the predictor (persistence + per-topic replay, no random RNN),
breaking the hard 3-turn ceiling. But the idx=374 trace exposed a residual
*boundary-placement* defect: #14→#15 (a literal "habits / habit-loop"
continuation) was split, while #12+#13 (WWII-radio vs non-fiction-recs,
across a real session change) were wrongly merged.

codex (2026-05-17) traced the dominant cause to the variance calibration,
not the predictor: v3.3.4's ``pe_var_min_samples=5`` gate kept every topic
at the fixed cold-start ``σ0²=0.04`` (no topic survives 5 continuations
before a restart freezes ``transition_count``), so a perfectly normal
conversational PE (≈0.55–0.61 — same topic, shifted sub-question) yields
``-PE²/(2·0.04) ≈ -4.7`` and loses to the lenient fresh baseline ``L0``.

SEM2 has no such gate. ``SEM/sem/event_models.py`` applies
``map_variance`` (Gelman scaled-inverse-χ² posterior mode) from the
**2nd** prediction error onward, with a scaled-inverse-χ² *prior*
(``var_df0``, ``var_scale0 = get_prior_scale(df, mode)``):

    σ²  =  (ν₀·var₀ + n·v) / (ν₀ + n + 2)        (n ≥ 2)
    var₀ = σ²_prior · (ν₀ + 2) / ν₀              (prior mode == σ²_prior)

v3.3.7 keeps every v3.3.6 term (TopicV336 dynamics, ``f_is_trained``
gating, f0/restart, deterministic seed) and only:

* replaces the EMA + min-samples σ² with the SEM2 posterior mode,
  effective from the 2nd PE sample (the ``pe_var_min_samples`` gate is
  removed);
* feeds the σ² sample buffer on **continuations *and* same-label
  restarts** (codex processing C): a restart is still a PE observation
  of the topic, so its σ² may progress even while ``transition_count``
  (the ``f_is_trained`` gate) is frozen by the restart.
"""

from __future__ import annotations

import math

import numpy as np

from hi_ontop.scrp import sticky_crp_unnormed
from hi_ontop.topic_v336 import TopicV336


class HiOnTopSegmenterV337:
    """v3.3.6 dynamics + SEM2 scaled-inv-χ² posterior σ_k² (n≥2)."""

    def __init__(
        self,
        dim: int,
        alpha: float = 1.0,
        lmda: float = 10.0,
        tau: float = 50.0,  # legacy CLI compat — unused
        cos_threshold: float = 0.9,
        beta: float = 0.25,
        pe_threshold: float = 1.0,
        k_max: int = 256,
        rnn_hidden_dim: int = 32,
        rnn_lr: float = 1e-3,
        rnn_train_steps: int = 3,  # legacy CLI compat — unused
        rnn_max_context: int = 8,  # legacy CLI compat — unused
        rnn_min_history: int = 2,  # legacy CLI compat — superseded
        # variance prior (SEM2 scaled-inv-χ²)
        pe_var_sigma0_sq: float = 0.04,  # prior MODE of the PE variance
        pe_var_df0: float = 1.0,  # ν₀ — SEM2 var_df0 default
        pe_var_min_sq: float = 1e-4,
        pe_var_max_sq: float = 0.25,
        pe_var_window: int = 256,  # cap on retained PE samples / topic
        pe_var_min_samples: int = 2,  # legacy CLI compat — fixed at 2 (SEM2)
        pe_var_decay: float = 0.95,  # legacy CLI compat — unused
        var_likelihood_weight: float = 1.0,
        hard_pe_fallback: bool = False,
        # v3.3.5 cold-start gating
        min_transitions_for_pe: int = 1,
        # v3.3.3 f0 / restart branch
        restart_pe_threshold: float = 0.5,
        restart_margin: float = 0.0,
        f0_min_starts: int = 2,
        # v3.3.6 SEM2-faithful event dynamics
        rnn_n_epochs: int = 10,
        rnn_ready_min_transitions: int = 3,
        rnn_max_history: int = 64,
        seed: int = 0,
    ) -> None:
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if rnn_lr <= 0:
            raise ValueError(f"rnn_lr must be > 0, got {rnn_lr}")
        if not 0.0 <= pe_threshold <= 1.0:
            raise ValueError(f"pe_threshold must be in [0, 1], got {pe_threshold}")
        if pe_var_df0 <= 0:
            raise ValueError(f"pe_var_df0 must be > 0, got {pe_var_df0}")
        if not pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq:
            raise ValueError(
                "expect pe_var_min_sq < pe_var_sigma0_sq <= pe_var_max_sq, got "
                f"{pe_var_min_sq}, {pe_var_sigma0_sq}, {pe_var_max_sq}"
            )
        if pe_var_window < 2:
            raise ValueError(f"pe_var_window must be >= 2, got {pe_var_window}")
        if var_likelihood_weight <= 0:
            raise ValueError(
                f"var_likelihood_weight must be > 0, got {var_likelihood_weight}"
            )
        if min_transitions_for_pe < 1:
            raise ValueError(
                f"min_transitions_for_pe must be >= 1, got {min_transitions_for_pe}"
            )
        if not 0.0 <= restart_pe_threshold <= 2.0:
            raise ValueError(
                f"restart_pe_threshold must be in [0, 2], got {restart_pe_threshold}"
            )
        if f0_min_starts < 1:
            raise ValueError(f"f0_min_starts must be >= 1, got {f0_min_starts}")
        if rnn_n_epochs < 1:
            raise ValueError(f"rnn_n_epochs must be >= 1, got {rnn_n_epochs}")
        if rnn_ready_min_transitions < 1:
            raise ValueError(
                "rnn_ready_min_transitions must be >= 1, got "
                f"{rnn_ready_min_transitions}"
            )
        if rnn_max_history < 2:
            raise ValueError(f"rnn_max_history must be >= 2, got {rnn_max_history}")

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

        # SEM2 scaled-inverse-χ² prior: pick var0 so the prior mode
        # (n=0) equals pe_var_sigma0_sq  →  var0 = mode·(ν0+2)/ν0.
        self.pe_var_sigma0_sq = pe_var_sigma0_sq
        self.pe_var_df0 = pe_var_df0
        self._var0 = pe_var_sigma0_sq * (pe_var_df0 + 2.0) / pe_var_df0
        self.pe_var_min_sq = pe_var_min_sq
        self.pe_var_max_sq = pe_var_max_sq
        self.pe_var_window = pe_var_window
        self.var_likelihood_weight = var_likelihood_weight
        self.hard_pe_fallback = hard_pe_fallback

        self.min_transitions_for_pe = min_transitions_for_pe
        self.restart_pe_threshold = restart_pe_threshold
        self.restart_margin = restart_margin
        self.f0_min_starts = f0_min_starts

        self.rnn_n_epochs = rnn_n_epochs
        self.rnn_ready_min_transitions = rnn_ready_min_transitions
        self.rnn_max_history = rnn_max_history
        self.seed = seed

        pe0 = 1.0 - cos_threshold
        self._L0 = var_likelihood_weight * self._calibrated_loglik(
            pe0, pe_var_sigma0_sq
        )

        self.topics: list[TopicV336] = []
        self.counts: np.ndarray = np.zeros(k_max, dtype=np.int64)
        self.prev_k: int | None = None

        self._transition_count: list[int] = []
        # SEM2-style: retained scalar PE samples per topic (the
        # ``prediction_errors`` buffer, capped like ``variance_window``).
        self._pe_samples: list[list[float]] = []
        self._f0_centroids: list[np.ndarray] = []
        self._f0_counts: list[int] = []
        self._pe_run_mean: list[float] = []
        self._pe_run_max: list[float] = []
        self._pe_run_count: list[int] = []
        self._n_boundaries: list[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_topic(self) -> TopicV336:
        return TopicV336(
            topic_id=len(self.topics),
            dim=self.dim,
            hidden_dim=self.rnn_hidden_dim,
            lr=self.rnn_lr,
            n_epochs=self.rnn_n_epochs,
            max_history=self.rnn_max_history,
            seed=self.seed,
        )

    def _ensure_topic_slot(self, k: int) -> None:
        while len(self.topics) <= k:
            self.topics.append(self._new_topic())
            self._transition_count.append(0)
            self._pe_samples.append([])
            self._f0_centroids.append(np.zeros(self.dim, dtype=np.float64))
            self._f0_counts.append(0)
            self._pe_run_mean.append(0.0)
            self._pe_run_max.append(0.0)
            self._pe_run_count.append(0)
            self._n_boundaries.append(0)

    def _is_trained(self, k: int) -> bool:
        return self._transition_count[k] >= self.min_transitions_for_pe

    def _rnn_ready(self, k: int) -> bool:
        return self._transition_count[k] >= self.rnn_ready_min_transitions

    def _pe(self, k: int, s: np.ndarray) -> float:
        return self.topics[k].prediction_error(s, use_rnn=self._rnn_ready(k))

    def _sigma_sq_for(self, k: int) -> float:
        """SEM2 ``map_variance`` posterior mode of the PE variance.

        ``σ² = (ν₀·var₀ + n·v) / (ν₀ + n + 2)`` for ``n ≥ 2`` samples
        (``v`` = population variance of the topic's PE samples); the
        scaled-inverse-χ² prior alone (mode == ``pe_var_sigma0_sq``)
        for ``n < 2``. Clipped to ``[min_sq, max_sq]``.
        """
        samples = self._pe_samples[k]
        n = len(samples)
        if n < 2:
            sigma_sq = self.pe_var_sigma0_sq
        else:
            v = float(np.var(samples))  # population variance (SEM2 np.var)
            nu0 = self.pe_var_df0
            sigma_sq = (nu0 * self._var0 + n * v) / (nu0 + n + 2.0)
        return float(np.clip(sigma_sq, self.pe_var_min_sq, self.pe_var_max_sq))

    def _record_pe(self, k: int, pe: float) -> None:
        """Append a PE sample to topic ``k``'s buffer (capped, FIFO)."""
        buf = self._pe_samples[k]
        buf.append(float(pe))
        if len(buf) > self.pe_var_window:
            del buf[0]

    def _calibrated_loglik(self, pe: float, sigma_sq: float) -> float:
        return -(pe * pe) / (2.0 * sigma_sq)

    def _trained_loglik(self, k: int, pe: float) -> float:
        return self.var_likelihood_weight * self._calibrated_loglik(
            pe, self._sigma_sq_for(k)
        )

    def _f0_dir(self, k: int) -> np.ndarray | None:
        if self._f0_counts[k] < self.f0_min_starts:
            return None
        ref = self._f0_centroids[k]
        n = float(np.linalg.norm(ref))
        if n <= 1e-12:
            return None
        return ref / n

    def _update_f0(self, k: int, s: np.ndarray) -> None:
        self._f0_counts[k] += 1
        delta = s - self._f0_centroids[k]
        self._f0_centroids[k] = self._f0_centroids[k] + delta / self._f0_counts[k]
        nrm = float(np.linalg.norm(self._f0_centroids[k]))
        if nrm > 1e-12:
            self._f0_centroids[k] = self._f0_centroids[k] / nrm

    def _surprise_forces_new(self, s: np.ndarray) -> bool:
        if (
            not self.hard_pe_fallback
            or not self.topics
            or self.pe_threshold >= 1.0
        ):
            return False
        threshold = 1.0 - self.pe_threshold
        for i, topic in enumerate(self.topics):
            if topic.cosine_score(s, use_rnn=self._rnn_ready(i)) >= threshold:
                return False
        return True

    def _new_cluster_index_in_active(self, active: np.ndarray) -> int | None:
        for i, k in enumerate(active):
            if int(k) >= len(self.topics):
                return i
        return None

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _scores(
        self, s: np.ndarray, prior: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_scores = np.empty(active.shape[0], dtype=np.float64)
        repeat_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)
        restart_scores = np.full(active.shape[0], -np.inf, dtype=np.float64)

        for i, k in enumerate(active):
            k_int = int(k)
            log_prior = math.log(prior[k_int])

            if k_int >= len(self.topics):  # fresh cluster slot
                log_scores[i] = log_prior + self._L0
                continue

            if self._is_trained(k_int):
                pe = self._pe(k_int, s)
                repeat_lik = self._trained_loglik(k_int, pe)
            else:
                repeat_lik = self._L0  # SEM2: untrained → fixed prior const

            repeat = log_prior + repeat_lik
            repeat_scores[i] = repeat

            if k_int == self.prev_k and self._is_trained(k_int):
                pe_repeat = self._pe(k_int, s)
                if pe_repeat > self.restart_pe_threshold:
                    base = max(
                        (self.counts[k_int] + 1.0) ** self.beta, 1e-300
                    )
                    log_prior_no_sticky = math.log(base)
                    f0_dir = self._f0_dir(k_int)
                    if f0_dir is None:
                        restart_lik = self._L0
                    else:
                        pe_f0 = 1.0 - float(np.dot(f0_dir, s))
                        restart_lik = self._trained_loglik(k_int, pe_f0)
                    restart = log_prior_no_sticky + restart_lik
                    restart_scores[i] = restart
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
        log_scores, repeat_scores, restart_scores = self._scores(s, prior, active)

        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            chosen_idx = (
                new_idx if new_idx is not None else int(np.argmax(log_scores))
            )
        else:
            chosen_idx = int(np.argmax(log_scores))
        k = int(active[chosen_idx])

        is_restart = False
        if (
            self.prev_k is not None
            and k == self.prev_k
            and not math.isinf(restart_scores[chosen_idx])
            and restart_scores[chosen_idx]
            > repeat_scores[chosen_idx] + self.restart_margin
        ):
            is_restart = True

        self._ensure_topic_slot(k)

        is_label_change = self.prev_k is not None and k != self.prev_k
        is_boundary = is_label_change or is_restart
        is_continuation = (
            self.prev_k is not None and k == self.prev_k and not is_restart
        )

        # σ² sample buffer: feed on continuation AND same-label restart
        # (codex processing C — a restart is still a PE observation of
        # topic k, so σ² may progress even though transition_count, the
        # f_is_trained gate, is frozen by the restart).
        if is_continuation or is_restart:
            pe = float(self._pe(k, s))
            self._record_pe(k, pe)
            if is_continuation:
                pe_clamped = max(0.0, min(2.0, pe))
                n_prev = self._pe_run_count[k]
                n_new = n_prev + 1
                self._pe_run_mean[k] = (
                    self._pe_run_mean[k] * n_prev + pe_clamped
                ) / n_new
                self._pe_run_max[k] = max(self._pe_run_max[k], pe_clamped)
                self._pe_run_count[k] = n_new

        self.topics[k].update(s)
        self.counts[k] += 1

        if is_continuation:
            self._transition_count[k] += 1

        if is_boundary or self.counts[k] == 1:
            self._update_f0(k, s)
            self._n_boundaries[k] += 1

        self.prev_k = k
        return k, is_boundary

    def predict_topic(self, s: np.ndarray) -> int:
        if not self.topics:
            return 0
        prior = sticky_crp_unnormed(
            self.counts, self.prev_k, self.alpha, self.lmda, beta=self.beta
        )
        active = np.flatnonzero(prior)
        log_scores, _, _ = self._scores(s, prior, active)
        if self._surprise_forces_new(s):
            new_idx = self._new_cluster_index_in_active(active)
            if new_idx is not None:
                return int(active[new_idx])
        return int(active[int(np.argmax(log_scores))])

    # ------------------------------------------------------------------
    # Importance v2 readouts
    # ------------------------------------------------------------------

    def pe_stats(self) -> dict[int, dict[str, float]]:
        return {
            k: {
                "mean": float(self._pe_run_mean[k]),
                "max": float(self._pe_run_max[k]),
                "count": int(self._pe_run_count[k]),
            }
            for k in range(len(self.topics))
            if self._pe_run_count[k] > 0
        }

    def boundary_counts(self) -> dict[int, int]:
        return {
            k: int(self._n_boundaries[k])
            for k in range(len(self.topics))
            if self.counts[k] > 0
        }
