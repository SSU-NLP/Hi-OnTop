"""Topic state for Hi-OnTop v3.3.6 — SEM2-faithful event dynamics.

v3.3.5 (and earlier v3.3.x) used :class:`hi_ontop.topic_v331_rnn.TopicV33RNN`:
a *shared* ``EventRNN`` trained by a single-pair 3-step online SGD, with a
``centroid`` fallback below ``rnn_min_history``. The idx=374 per-turn trace
showed this collapses: the moment a topic became "trained" and switched
from the centroid predictor (cos 0.40–0.84) to the shared GRU
(cos 0.03–0.53, near-random), its repeat likelihood went catastrophic and
the topic died — a hard 3-turn ceiling.

SEM2 (``SEM/sem/event_models.py``) does *not* do this:

* ``predict_next`` for an **untrained** model returns the **identity**
  (the last input — a persistence prediction), never a random network.
* ``estimate`` retrains over the event's **entire** ``training_pairs``
  history (batch sampling, ``n_epochs`` epochs), not one pair × 3 steps.
* each event effectively owns its weights (SEM2 snapshots/restores
  ``model_weights`` per event), so other events do not corrupt it.

``TopicV336`` restores all three:

* **own** :class:`EventRNN` + optimizer per topic (no cross-topic
  interference);
* ``predict_next`` returns the last observed embedding (persistence)
  until the RNN is *ready*; the segmenter decides readiness via
  ``transition_count`` and passes ``use_rnn``;
* ``update`` replays the topic's full embedding sequence for
  ``n_epochs`` epochs (full-sequence batch; deterministic for
  reproducibility) before refreshing the cached prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from hi_ontop.event_rnn import EventRNN


class TopicV336:
    """Per-topic GRU event model with persistence cold-start.

    Args:
        topic_id: Integer identifier.
        dim: Embedding dimension.
        hidden_dim: GRU hidden size.
        lr: Optimizer learning rate.
        n_epochs: Full-history replay epochs per :meth:`update`
            (SEM2 ``estimate`` default is 10).
        max_history: Cap on retained embeddings per topic. Replay uses
            the most recent ``max_history`` turns (keeps update O(H·E)
            bounded on very long single-topic spans).
    """

    def __init__(
        self,
        topic_id: int,
        dim: int,
        *,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        n_epochs: int = 10,
        max_history: int = 64,
        seed: int = 0,
    ) -> None:
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}")
        if max_history < 2:
            raise ValueError(f"max_history must be >= 2, got {max_history}")

        self.topic_id = topic_id
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.max_history = max_history
        self.seed = seed

        # Per-topic EventRNN + optimizer are LAZILY constructed — only when
        # the GRU is actually trained (``update(train_rnn=True)``). With
        # η=1 / RNN disabled (v4.1.x default) they are never built, which
        # avoids a torch RNN module construction per topic — profiled at
        # ~77% of assign() wall time when eagerly constructed (2026-05-22).
        self.model: EventRNN | None = None
        self.optimizer: object | None = None

        self.mu: np.ndarray = np.zeros(dim, dtype=np.float64)
        self.n: int = 0
        self._seq: list[np.ndarray] = []  # observed embeddings, in order
        self._cached_pred: np.ndarray = np.zeros(dim, dtype=np.float64)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _persistence(self) -> np.ndarray:
        """SEM2 untrained ``predict_next``: the last observed input."""
        if not self._seq:
            return np.zeros(self.dim, dtype=np.float64)
        return self._seq[-1]

    def predict_next(self, use_rnn: bool) -> np.ndarray:
        """Next-embedding prediction.

        ``use_rnn=False`` → persistence (last observed embedding).
        ``use_rnn=True``  → cached GRU prediction over the topic history.
        """
        if self.n == 0:
            return np.zeros(self.dim, dtype=np.float64)
        if not use_rnn:
            return self._persistence()
        return self._cached_pred

    def cosine_score(self, s: np.ndarray, *, use_rnn: bool) -> float:
        pred = self.predict_next(use_rnn)
        nrm = float(np.linalg.norm(pred))
        if nrm <= 1e-12:
            return 0.0
        return float(np.dot(pred / nrm, s))

    def prediction_error(self, s: np.ndarray, *, use_rnn: bool) -> float:
        return float(1.0 - self.cosine_score(s, use_rnn=use_rnn))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily build the per-topic EventRNN + optimizer on first GRU use.

        Deterministic per-topic init for reproducibility (CLAUDE.md): seed
        by (seed, topic_id). Global RNG is snapshot/restored so other
        randomness is unperturbed — identical behavior to eager construction.
        """
        if self.model is not None:
            return
        _rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.seed * 100_003 + self.topic_id)
            self.model = EventRNN(input_dim=self.dim, hidden_dim=self.hidden_dim)
        finally:
            torch.random.set_rng_state(_rng_state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _retrain(self) -> None:
        """Replay the full (capped) history: predict x_{i+1} from the GRU
        run over x_{0..i}, cosine loss, ``n_epochs`` epochs."""
        seq = self._seq[-self.max_history :]
        if len(seq) < 2:
            return
        self._ensure_model()
        x = torch.from_numpy(
            np.asarray(seq, dtype=np.float32)
        )  # (L, dim)
        inp = x[:-1].unsqueeze(1)  # (L-1, 1, dim) — one step per pair
        tgt = F.normalize(x[1:], dim=-1, eps=1e-12)  # (L-1, dim)

        self.model.train()
        for _ in range(self.n_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            h = torch.zeros(inp.shape[0], self.hidden_dim, dtype=torch.float32)
            # Sequential GRU roll: hidden carries within the topic span so
            # the prediction of x_{i+1} conditions on x_{0..i}.
            preds = []
            h_run = torch.zeros(1, self.hidden_dim, dtype=torch.float32)
            for i in range(x.shape[0] - 1):
                h_run = self.model.step(x[i : i + 1], h_run)
                preds.append(self.model.predict(h_run))
            pred = torch.cat(preds, dim=0)  # (L-1, dim)
            loss = (1.0 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def _refresh_cache(self) -> None:
        """Recompute the cached next prediction from the full history."""
        self.model.eval()
        with torch.no_grad():
            h = torch.zeros(1, self.hidden_dim, dtype=torch.float32)
            for emb in self._seq[-self.max_history :]:
                x = torch.from_numpy(
                    np.asarray(emb, dtype=np.float32)
                ).unsqueeze(0)
                h = self.model.step(x, h)
            pred = self.model.predict(h)
        self._cached_pred = pred[0].cpu().numpy().astype(np.float64)

    def update(self, s: np.ndarray, *, train_rnn: bool = True) -> None:
        """Append ``s``, (optionally) retrain over the topic history +
        refresh the cached prediction, update the centroid (diagnostic).

        ``train_rnn=False`` skips the GRU retrain/refresh (the expensive
        part, ``n_epochs`` replay) while keeping ``_seq``/centroid — used
        when the segmenter does not consume the learned prediction
        (η=1 / RNN disabled), a pure ~4× speedup with identical output.
        """
        s64 = np.asarray(s, dtype=np.float64)
        self._seq.append(s64)
        self.n += 1

        if train_rnn and self.n >= 2:
            self._retrain()
            self._refresh_cache()

        # Centroid kept only as a diagnostic readout (not a predictor).
        delta = s64 - self.mu
        self.mu = self.mu + delta / self.n
        nrm = float(np.linalg.norm(self.mu))
        if nrm > 1e-12:
            self.mu = self.mu / nrm
