"""Shared GRU event model for Hi-OnTop v3.3.1 (A+B variant).

Single ``nn.Module`` shared across topics by the segmenter. Per-topic state
is just a hidden vector ``h_k`` carried by
:class:`hi_ontop.topic_v331_rnn.TopicV33RNN`. Mirrors SEM2's "library of event
models" structure (one network with per-event hidden state) rather than
per-event Keras instances.

Public API:
    step(x, h_prev)  — one GRU step, returns updated hidden ``(B, hidden)``.
    predict(h)       — unit-norm next-embedding prediction ``(B, dim)``.
    forward(x)       — convenience: run ``step`` over a 3-D tensor
                       ``(B, T, dim)`` and return ``predict(h_T)``. Kept for
                       backward compatibility with smoke tests that pass a
                       sequence batch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EventRNN(nn.Module):
    """GRUCell-based next-embedding predictor.

    Args:
        input_dim: Embedding dimension.
        hidden_dim: GRU hidden size. Kept small because one model serves
            all topics in a segmenter.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, input_dim)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One GRU step.

        ``x`` shape: ``(B, input_dim)``.
        ``h`` shape: ``(B, hidden_dim)``.
        Returns ``(B, hidden_dim)``.
        """
        if x.ndim != 2 or h.ndim != 2:
            raise ValueError(
                f"expected 2D tensors, got x={tuple(x.shape)} h={tuple(h.shape)}"
            )
        return self.cell(x, h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Unit-normalized next-embedding prediction.

        ``h`` shape: ``(B, hidden_dim)``. Returns ``(B, input_dim)``.
        """
        return F.normalize(self.head(h), dim=-1, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequence-level convenience wrapper.

        ``x`` shape: ``(B, T, input_dim)``. Runs ``T`` GRU steps from a
        zero hidden state and predicts from the final hidden. Returns
        ``(B, input_dim)``. Used only by the smoke / batched check; the
        segmenter's per-turn path uses :meth:`step` + :meth:`predict`.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, T, dim), got shape {tuple(x.shape)}")
        b, t, _ = x.shape
        h = torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype)
        for step_i in range(t):
            h = self.cell(x[:, step_i, :], h)
        return self.predict(h)
