"""v4.1.3 — graded boundary score (output-only).

[[v4.1.1]] 의 thin subclass. **algorithm 변경 0** — assign 의 결정 흐름
(sCRP / Bayes / local-MAP / prior-corrected B0) 그대로. 추가는
**graded boundary score 출력만**:

`graded_score = δ_eff / δ*` 의 1-D 연속값을 매 turn 노출. Ben-Yakov &
Henson 2018 의 hippocampal boundary response 가 graded profile 임을 보인
신경과학적 발견과 동형. Downstream uncertainty-aware consumer (예:
SeCom memory) 가 직접 활용 가능 — `score < 0.7` 약한 경계 (보류),
`score > 1.3` 강한 경계 (즉시 commit), `0.7 ≤ score ≤ 1.3` 모호
(deferred decision).

### Re-entry 기능 제거 (2026-05-21)

초기 revision 에서 `is_reentry` / sub-type flag 등 re-entry detection 을
포함했으나, 측정 결과:
- v4.1.1 default (`f0_min_starts=2`) 에선 re-entry mechanism 봉인 → flag
  항상 False, 무의미
- Analysis mode (`f0_min_starts=1`) 활성 시 Score 가 폭락 (-6.9 ~ -19.6pp)
  — segmenter 가 옛 topic_id 재사용 → label-change count 감소

→ **사용자 결정 (2026-05-21): re-entry 관련 attribute / method 모두 제거**.
v4.1.3 = pure graded boundary score output 으로 단순화. paper claim 도
"graded boundary score" 한 가지로 단순화.

re-entry mechanism 자체는 [[v4.2.1]] 클래스로 ablation 보존 (별도 사용 안 함).

**SEM 계승 3-step (trivially PASS)**:
1. SEM2 의 Gaussian likelihood 는 본질적으로 graded scalar. 표면 노출만.
2. 충돌 없음 — read-only attribute.
3. decision-log 2026-05-21.
"""

from __future__ import annotations

import numpy as np

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411


class HiOnTopSegmenterV413(HiOnTopSegmenterV411):
    """v4.1.1 + graded boundary score output (output-only).

    Public attributes set after each :meth:`assign` call:

    - ``last_delta_eff``: float, last turn 의 chosen topic 에 대한 δ_eff (raw).
    - ``last_graded_score``: float, ``last_delta_eff / delta_star`` ∈ [0, ~3].
      < 1 = no-boundary 직전, = 1 = boundary 임계, > 1 = boundary 강.
    - ``last_is_boundary``: bool (v4.1.1 의 return 값과 동일).

    Full history via :meth:`history` / :meth:`graded_scores` /
    :meth:`boundary_strength`.

    Algorithm 자체는 v4.1.1 그대로 — 본 클래스는 attribute 노출만 추가.

    Boundary strength bands (Ben-Yakov & Henson 2018 inspired):

    ============= =================================================
    graded_score   해석
    ============= =================================================
    < 0.7          매우 약한 경계 — downstream consumer 보류 권장
    0.7 ≤ x < 1.0  약한 경계 — repeat 우세 영역 (boundary off)
    1.0 ≤ x < 1.3  경계 (정상)
    ≥ 1.3          강한 경계 — 즉시 commit 권장
    ============= =================================================

    Args:
        Inherited from V411.
    """

    def __init__(self, *args, **kwargs) -> None:
        # V411 default 그대로 (`f0_min_starts=2` 등). Algorithm 변경 0.
        super().__init__(*args, **kwargs)

        # Per-turn cache (reset at the start of each assign)
        self._delta_effs_this_turn: dict[int, float] = {}

        # Public state — last-turn readouts
        self.last_delta_eff: float | None = None
        self.last_graded_score: float | None = None
        self.last_is_boundary: bool = False

        # Per-turn history (append on each assign)
        self._history: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Override _delta_eff to record every (k, δ_eff) called this turn.
    # V411 _scores calls _delta_eff for each active topic; the chosen
    # one is identified after assign returns.
    # ------------------------------------------------------------------

    def _delta_eff(self, k: int, s: np.ndarray) -> float:
        d = super()._delta_eff(k, s)
        self._delta_effs_this_turn[k] = float(d)
        return d

    # ------------------------------------------------------------------
    # Override assign: keep parent's return contract, add public readouts.
    # ------------------------------------------------------------------

    def assign(self, s: np.ndarray) -> tuple[int, bool]:
        # 1. reset per-turn cache, capture prev_k BEFORE super updates it
        self._delta_effs_this_turn = {}
        prev_k_before = self.prev_k

        # 2. run v4.1.1 sCRP/MAP
        topic_id, is_boundary = super().assign(s)

        # 3. derive per-turn readouts.
        # Graded score 의 의미: "이 turn 이 직전 topic 의 연속이라기엔
        # 얼마나 멀리 떨어져 있나" = δ_eff(prev_k) / δ*.
        # - prev_k 있으면 (turn ≥ 2): δ_eff(prev_k) 사용. boundary 결정
        #   의 핵심 신호. fresh-cluster slot 은 V411 의 _scores 에서
        #   _delta_eff 가 호출되지 않으므로 prev_k 값만 신뢰 가능.
        # - prev_k None (turn 1): δ_eff 정의 안 됨 → 0.
        if prev_k_before is not None and prev_k_before in self._delta_effs_this_turn:
            chosen_d_eff = self._delta_effs_this_turn[prev_k_before]
        elif topic_id in self._delta_effs_this_turn:
            # fallback: 의외 케이스 (예: 단 1 active topic 환경).
            chosen_d_eff = self._delta_effs_this_turn[topic_id]
        else:
            chosen_d_eff = 0.0
        graded = chosen_d_eff / self.delta_star if self.delta_star > 0 else 0.0

        self.last_delta_eff = chosen_d_eff
        self.last_graded_score = graded
        self.last_is_boundary = bool(is_boundary)

        # 4. record in history
        self._history.append({
            "turn": len(self._history),
            "topic_id": int(topic_id),
            "is_boundary": bool(is_boundary),
            "delta_eff": float(chosen_d_eff),
            "graded_score": float(graded),
            "topic_count": int(self.counts[topic_id]),
        })

        return topic_id, is_boundary

    # ------------------------------------------------------------------
    # Public readouts
    # ------------------------------------------------------------------

    def history(self) -> list[dict[str, object]]:
        """Per-turn record list. Each entry has::

            {
                "turn": int,            # 0-indexed turn position
                "topic_id": int,        # assigned topic
                "is_boundary": bool,    # v4.1.1 return flag
                "delta_eff": float,     # raw δ_eff for the chosen topic
                "graded_score": float,  # δ_eff / δ*
                "topic_count": int,     # cumulative count (incl this turn)
            }
        """
        return list(self._history)

    def graded_scores(self) -> list[float]:
        """Per-turn graded_score sequence (δ_eff / δ*)."""
        return [float(h["graded_score"]) for h in self._history]

    def boundary_strength(self) -> dict[str, int]:
        """Histogram of boundary strength bands (Ben-Yakov & Henson framing).

        Bands:
            very_weak (<0.7), weak (0.7~1.0), normal (1.0~1.3), strong (≥1.3).
        """
        scores = [float(h["graded_score"]) for h in self._history]
        out = {"very_weak": 0, "weak": 0, "normal": 0, "strong": 0}
        for s in scores:
            if s < 0.7:
                out["very_weak"] += 1
            elif s < 1.0:
                out["weak"] += 1
            elif s < 1.3:
                out["normal"] += 1
            else:
                out["strong"] += 1
        return out
