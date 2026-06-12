"""Sticky Chinese Restaurant Process prior (SEM Eq 1).

Hi-OnTop uses SEM2's sticky-CRP verbatim with hyperparameters reversed
(alpha=1.0, lambda=10.0) for conversation topic persistence. See
``context/01-hi-ontop-design.md`` §2 and ``context/00-sem-paper.md`` §2.
"""

from __future__ import annotations

import numpy as np


def sticky_crp_unnormed(
    counts: np.ndarray,
    prev_k: int | None,
    alpha: float,
    lmda: float,
    beta: float = 1.0,
) -> np.ndarray:
    """Unnormalized sticky-CRP prior (SEM Eq 1) with optional sub-linear count.

    Default ``beta=1.0`` reproduces SEM Eq 1 verbatim:

    .. math::
        \\Pr(e_n = k \\mid e_{1:n-1}) \\propto
            \\begin{cases}
                C_k + \\lambda\\,\\mathbb{I}[e_{n-1}=k] & k \\leq K \\\\
                \\alpha & k = K+1
            \\end{cases}

    With ``beta < 1.0`` (Hi-OnTop v3.2 P0 modification) the visited-topic
    contribution is squashed to ``(C_k + 1) ** beta`` before adding the
    stickiness bonus, breaking the ``log C_k`` accumulation that drives
    mega-topic collapse on long conversations:

    .. math::
        \\Pr(e_n = k) \\propto (C_k + 1)^{\\beta} +
            \\lambda\\,\\mathbb{I}[e_{n-1}=k]
        \\quad (k \\leq K)

    Args:
        counts: Prior assignment counts per cluster, shape ``(max_k,)``.
            The new-cluster slot is the first index whose count is 0.
        prev_k: Most recent assignment, or ``None`` on the first step.
        alpha: Concentration parameter (new-cluster probability).
        lmda: Stickiness parameter (bonus for staying in previous cluster).
        beta: Sub-linear count exponent. ``1.0`` = original SEM (no
            squash). ``0.5`` = sqrt, ``0.25`` = stronger squash. Applied
            only to already-visited topics; un-visited slots stay 0 so
            the new-cluster slot logic is unchanged.

    Returns:
        Unnormalized prior as ``float64`` array, same shape as ``counts``.
        Not normalized — the caller takes :math:`\\log` directly.
    """
    prior = counts.astype(np.float64, copy=True)
    if beta != 1.0:
        visited = counts > 0
        if visited.any():
            prior[visited] = (prior[visited] + 1.0) ** beta
    n_visited = int(np.count_nonzero(counts))
    if n_visited < counts.shape[0]:
        prior[n_visited] += alpha
    if prev_k is not None:
        prior[prev_k] += lmda
    return prior
