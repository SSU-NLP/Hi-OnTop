"""v4.1.2-topicctx (ABLATION CANDIDATE) — topic-membership-aware causal ctx.

v4.1.1 (`sem_core_v411`) 의 고정 길이 ctx_window 를 **현 topic 멤버십
필터**로 대체하는 변형. v4.1.1 본체 무수정(subclass), 알고리즘 수식
중 ctx 인덱스 필터만 교체:

  현 v4.1.1: c_{t-1} = Σ_{i=1..m} ρ^{i-1} · s_{t-i}   (fixed-window m)
  TopicCur : c_{t-1} = Σ_{j ∈ P_cur} ρ^{off(j)} · s_j
              where P_cur = { j<t : topic_id(j) == topic_id(t-1) }
              off(j) = temporal offset from t-1 (most recent = 0)
  TopicPrev: P_cur ∪ { j<t : topic_id(j) == prev_distinct_topic }

ρ-감쇠는 temporal-offset 그대로(최근 가중). 완전 causal (assign 끝난
과거 topic id 만 참조). cold-start (현 topic 에 t-1 만) → ctx = s_{t-1}
단일 → δ_ctx=δ_prev → δ_adj 자동 δ_prev 회귀 (의도된 동작).
ctx_max_len = 안전캡 (무한 성장 방지, 기본 64).

codex 결정 2026-05-20: v4.1.2 default 승격 전 ablation 후보. 1차
TopicCur, 2차 TopicPrev. δ*=0.5594 고정 smoke + TIAGE-train 재calib
본평가 분리. v4.1.1 본체 무수정 + assign 시그니처 보존.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411

CtxPolicy = Literal["window", "topic_cur", "topic_prev"]


class HiOnTopSegmenterV412TopicCtx(HiOnTopSegmenterV411):
    """v4.1.1 + topic-aware ctx. policy='window' 면 v4.1.1 와 동일."""

    def __init__(
        self,
        *args,
        ctx_policy: CtxPolicy = "topic_cur",
        ctx_max_len: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ctx_policy not in ("window", "topic_cur", "topic_prev"):
            raise ValueError(
                f"ctx_policy must be window|topic_cur|topic_prev, "
                f"got {ctx_policy!r}")
        if ctx_max_len < 2:
            raise ValueError(f"ctx_max_len must be >= 2, got {ctx_max_len}")
        self.ctx_policy = ctx_policy
        self.ctx_max_len = ctx_max_len
        # past-turn topic history: (s_copy, topic_id)
        self._topic_hist: list[tuple[np.ndarray, int]] = []

    def _delta_ctx(self, s: np.ndarray) -> float | None:
        # window policy → v4.1.1 parent 동작 그대로
        if self.ctx_policy == "window":
            return super()._delta_ctx(s)
        if not self._topic_hist:
            return None

        prev_k = self._topic_hist[-1][1]  # topic_id(t-1)
        if self.ctx_policy == "topic_cur":
            keep_ids: set[int] = {prev_k}
        else:  # topic_prev
            # 가장 최근 distinct topic id (없으면 None)
            prev_distinct: int | None = None
            for _, kk in reversed(self._topic_hist):
                if kk != prev_k:
                    prev_distinct = kk
                    break
            keep_ids = {prev_k}
            if prev_distinct is not None:
                keep_ids.add(prev_distinct)

        # 필터: keep_ids 안에 든 과거 turn 만, 시간 순(과거→최근) 유지
        filtered = [v for v, kk in self._topic_hist if kk in keep_ids]
        if not filtered:
            return None

        # ρ-감쇠: temporal offset (가장 최근 = i=0)
        c = np.zeros_like(filtered[-1], dtype=np.float64)
        for i, vec in enumerate(reversed(filtered)):
            c += (self.ctx_decay ** i) * vec
        nc = float(np.linalg.norm(c))
        ns = float(np.linalg.norm(s))
        if nc <= 1e-12 or ns <= 1e-12:
            return None
        return 1.0 - float(np.dot(c, s) / (nc * ns))

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        # parent 가 scoring(_delta_ctx 호출 포함) → state 갱신 → 결정 반환
        k, is_boundary = super().assign(s)
        # commit: 이 turn 의 (s, k) 를 topic history 에 append (다음 turn 의 ctx 가 이걸 봄)
        s_copy = np.array(s, dtype=np.float64, copy=True)
        self._topic_hist.append((s_copy, k))
        if len(self._topic_hist) > self.ctx_max_len:
            del self._topic_hist[0]
        return k, is_boundary
