"""Topic state for Hi-OnTop v3.3.1 (A+B variant).

Each topic carries:
    - centroid ``mu`` (running mean, unit-norm)
    - sample count ``n``
    - per-topic GRU hidden state ``h``
    - cached next-embedding prediction (refreshed only on ``update``)
    - last (h_in, x) pair so the segmenter can backprop through that step

The shared :class:`hi_ontop.event_rnn.EventRNN` and its optimizer live on the
segmenter, *not* on the topic. ``TopicV33RNN.update`` keeps a legacy
standalone path: if no model is injected at construction time, the topic
allocates its own model+optimizer (kept for unit tests and backward
compatibility).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from hi_ontop.event_rnn import EventRNN


class TopicV33RNN:
    """Topic with running centroid + GRU hidden state + cached prediction.

    Args:
        topic_id: Integer identifier.
        dim: Feature dimension.
        hidden_dim: Shared GRU hidden size (must match segmenter model).
        lr: Optimizer learning rate (used only in standalone mode when
            ``model`` is None).
        train_steps: Online SGD steps per assigned turn (used in
            standalone mode; segmenter overrides via its own value).
        max_context: Legacy hyperparameter from the per-topic-GRU
            variant — accepted for backward CLI compatibility, ignored
            here (per-topic hidden state replaces the deque history).
        min_rnn_history: Below this many assigned turns,
            ``predict_next`` returns the centroid instead of the cached
            RNN prediction. Avoids using a barely-trained prediction
            for cold topics.
        model: Optional shared :class:`EventRNN`. If None, a per-topic
            model is allocated (legacy / standalone path).
        optimizer: Optional shared optimizer paired with ``model``. If
            None and ``model`` is given, a per-topic Adam is created
            over ``model.parameters()`` (still updates shared weights).
    """

    def __init__(
        self,
        topic_id: int,
        dim: int,
        *,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        train_steps: int = 1,
        max_context: int = 8,  # noqa: ARG002 — kept for CLI/test back-compat
        min_rnn_history: int = 2,
        model: EventRNN | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if train_steps < 0:
            raise ValueError(f"train_steps must be >= 0, got {train_steps}")
        if min_rnn_history < 1:
            raise ValueError(f"min_rnn_history must be >= 1, got {min_rnn_history}")

        self.topic_id = topic_id
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.train_steps = train_steps
        self.min_rnn_history = min_rnn_history

        if model is None:
            self.model = EventRNN(input_dim=dim, hidden_dim=hidden_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self._owns_model = True
        else:
            if model.input_dim != dim or model.hidden_dim != hidden_dim:
                raise ValueError(
                    "shared model dims do not match topic dims: "
                    f"model=({model.input_dim},{model.hidden_dim}) topic=({dim},{hidden_dim})"
                )
            self.model = model
            self.optimizer = (
                optimizer
                if optimizer is not None
                else torch.optim.Adam(model.parameters(), lr=lr)
            )
            self._owns_model = False

        self.mu: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.n: int = 0
        self.h: torch.Tensor = torch.zeros(1, hidden_dim, dtype=torch.float32)
        self._prev_h_in: torch.Tensor | None = None
        self._prev_x: torch.Tensor | None = None
        self._cached_pred: np.ndarray = np.zeros(dim, dtype=np.float64)

    def predict_next(self) -> np.ndarray:
        """Cached unit-norm prediction (or centroid for cold-start).

        ``update`` refreshes the cache; this getter is O(1).
        """
        if self.n == 0:
            return np.zeros(self.dim, dtype=np.float64)
        if self.n < self.min_rnn_history:
            return self.mu.copy()
        return self._cached_pred

    def cosine_score(self, s: np.ndarray) -> float:
        return float(np.dot(self.predict_next(), s))

    def prediction_error(self, s: np.ndarray) -> float:
        return float(1.0 - self.cosine_score(s))

    def log_likelihood(self, s: np.ndarray) -> float:
        return self.cosine_score(s)

    def update(self, s: np.ndarray) -> None:
        """Train shared model on the previous transition, step GRU forward,
        cache the new prediction, then update the centroid.

        Calls ``self.model`` (shared or owned) and ``self.optimizer``.
        """
        s_tensor = torch.from_numpy(np.asarray(s, dtype=np.float32)).unsqueeze(0)

        # Training: replay the previous (h_in, x) pair, target is current s.
        if (
            self.train_steps > 0
            and self._prev_h_in is not None
            and self._prev_x is not None
        ):
            self.model.train()
            target = F.normalize(s_tensor, dim=-1, eps=1e-12)
            for _ in range(self.train_steps):
                self.optimizer.zero_grad(set_to_none=True)
                h_redo = self.model.step(self._prev_x, self._prev_h_in)
                pred = self.model.predict(h_redo)
                loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        # State forward (no grad). Save (h_in, x) for the next training step.
        self.model.eval()
        with torch.no_grad():
            prev_h_in = self.h.clone()
            prev_x = s_tensor.clone()
            h_new = self.model.step(s_tensor, self.h)
            pred_new = self.model.predict(h_new)
        self.h = h_new.detach()
        self._prev_h_in = prev_h_in.detach()
        self._prev_x = prev_x.detach()
        self._cached_pred = pred_new[0].cpu().numpy().astype(np.float64)

        # Centroid update (numpy, Welford-style, then unit-norm).
        s64 = np.asarray(s, dtype=np.float64)
        self.n += 1
        delta = s64 - self.mu
        self.mu = self.mu + delta / self.n
        norm = float(np.linalg.norm(self.mu))
        if norm > 1e-12:
            self.mu = self.mu / norm
