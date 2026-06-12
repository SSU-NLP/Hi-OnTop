"""v4.2.1 — first-turn topic prediction (``f0_min_starts=1`` default).

New MAJOR-minor line on top of v4.1.1 (``sem_core_v411.HiOnTopSegmenterV411``,
causal-window δ identity segmenter). The ONLY algorithmic delta is the
default value of ``f0_min_starts``: 2 (v4.1.1) → **1** (v4.2.1).

Motivation (codex 2026-05-18/19, decision-log 2026-05-19): v4.1.1's
``_f0_dir()`` returns ``None`` until a topic has ``f0_min_starts=2``
recorded starts, so a topic seen exactly once falls back to the
prior-free constant ``_l0()`` on the non-prev / restart paths — its
re-entry likelihood is never opened. SEM2 (``SEM/sem/event_models.py``
``LinearEvent.update_f0`` → ``f0_is_trained=True``) makes f0 usable from
the **first** start observation. v4.2.1 restores that SEM2-faithful
cold-start: a topic's episode-start centroid is consulted from its first
appearance ("predict the topic from the first turn").

SEM inheritance 3-step:
1. **In SEM2?** Yes — ``update_f0`` flips ``f0_is_trained`` on the first
   start; v4.1.1's ``f0_min_starts=2`` gate is a Hi-OnTop-only deviation.
2. **Conflict?** None. ``f0_min_starts`` changes only *when* the f0
   start-likelihood becomes available; the δ_eff / sCRP / Bayes /
   local-MAP / prior-corrected fresh-baseline structure is unchanged.
   The dominant 2nd-turn-onward δ_prev/δ_ctx decision is untouched.
3. decision-log 2026-05-19.

NOT a retrieval-cosine rule (CLAUDE.md retrieval § is query-time only;
segmentation-PE f0 centroid similarity is out of that rule's scope —
context/methodology/v3.3.9.md).

Everything else (causal-window δ, η=1 identity, RNN retained via
``use_rnn`` and auto-skipped, prior-corrected B0, fixed σδ²=c·δ*²) is
inherited from v4.1.1 verbatim. ``delta_star`` is still corpus/encoder
dependent and MUST be re-estimated per dataset (no test leakage).

Expected effect (codex, hypothesis-level): not a main-line F1 jump —
it stabilises topic identity only on the early re-entry / restart
subset; risk is over-eager false re-entry from a generic opener
(monitored via collapse / pred_rate / ARI guards).
"""

from __future__ import annotations

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411


class HiOnTopSegmenterV421(HiOnTopSegmenterV411):
    """v4.1.1 with ``f0_min_starts`` default flipped 2 → 1.

    Thin subclass: the v4.1.1 algorithm is the single source of truth;
    v4.2.1 only changes the f0 cold-start availability default so a
    topic's episode-start centroid is consulted from its first turn
    (SEM2-faithful ``f0_is_trained`` on first start). Pass an explicit
    ``f0_min_starts=2`` to recover exact v4.1.1 behaviour (the ablation
    pair used in the 2026-05-19 experiment).
    """

    def __init__(self, *args, f0_min_starts: int = 1, **kwargs) -> None:
        super().__init__(*args, f0_min_starts=f0_min_starts, **kwargs)
