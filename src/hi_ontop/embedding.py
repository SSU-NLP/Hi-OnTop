"""Query embedders — local (sentence-transformers) and OpenAI-compatible API.

Scene vector ``s_n = normalize(encoder(query_n))`` per
``context/01-hi-ontop-design.md`` §1 and ``02-math-model.md`` §쿼리 임베딩.

Two backends with the same interface (``.dim``, ``.encode()``):

* :class:`QueryEncoder` — local ``sentence-transformers`` (default
  ``multi-qa-mpnet-base-dot-v1``, dim 768). The model is trained for
  dot-product retrieval; L2-normalizing the outputs (which we do via
  ``normalize_embeddings=True``) makes dot product equivalent to cosine
  similarity, matching Hi-OnTop's downstream sCRP / RAG math. Heavy deps
  imported lazily inside
  ``__init__`` so the module stays importable for type-checking and tests.
* :class:`APIEncoder` — OpenAI-compatible ``/v1/embeddings`` HTTP client
  (works against OpenRouter, OpenAI, vLLM-served embeddings, etc.). API
  responses are L2-normalized client-side to match the local backend's
  contract regardless of provider behavior. Embedding dimension is probed
  with one short request at construction (``HIONTOP_EMBEDDING_DIM`` env
  overrides if probing is undesired).

:func:`make_encoder` is the entry-point factory: env-driven backend
selection (``HIONTOP_EMBEDDING_BACKEND=local|api``) so call sites can stay
agnostic.

Thread safety: Both backends serialize the underlying call via an
internal lock. PyTorch MPS is process-singleton (concurrent forward
crashes), and the OpenAI SDK's per-client connection pool is safer
serialized for our small batch sizes. Phase 4 ThreadPool, RAG, and
Hi-OnTop preload/handle_turn can call ``encode()`` from multiple threads
safely.
"""

from __future__ import annotations

import os
import threading
from typing import Protocol

import numpy as np

LOCAL_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
LOCAL_DIM = 768

# Back-compat aliases (older imports may still reference BGE_*).
BGE_MODEL_NAME = LOCAL_MODEL_NAME
BGE_DIM = LOCAL_DIM

DEFAULT_API_MODEL = "openai/text-embedding-3-small"
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
API_PROBE_TEXT = "dim probe"
API_BATCH_SIZE = 64


class _Encoder(Protocol):
    """Structural contract relied on by the segmenter."""

    dim: int

    def encode(self, text: str | list[str]) -> np.ndarray: ...


class QueryEncoder:
    """L2-normalized local embeddings for Hi-OnTop scene vectors.

    Default model: ``multi-qa-mpnet-base-dot-v1`` (dim 768, trained on
    QA-style retrieval). Outputs are L2-normalized so dot product
    equals cosine similarity — matches Hi-OnTop's downstream math.

    Args:
        device: ``"cuda"`` / ``"mps"`` / ``"cpu"`` or ``None`` (auto-detect).
            Auto-detect priority: cuda → mps (Apple Silicon) → cpu.
            Same weights on every backend; inference dispatch only.
            Numerical jitter ≈1e-5 between kernels — no impact on
            segmentation/RAG.
        model_name: ``sentence-transformers`` model id.
    """

    def __init__(
        self,
        device: str | None = None,
        model_name: str = LOCAL_MODEL_NAME,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension() or LOCAL_DIM
        self._lock = threading.Lock()

    def encode(self, text: str | list[str]) -> np.ndarray:
        """Encode one or many strings to L2-normalized vectors.

        Thread-safe (serialized via an internal lock — see module docstring).

        Args:
            text: A single string or a list of strings.

        Returns:
            If ``text`` is ``str`` → shape ``(dim,)``.
            If ``text`` is ``list[str]`` → shape ``(n, dim)``.
        """
        with self._lock:
            if isinstance(text, str):
                out = self._model.encode([text], normalize_embeddings=True)
                return np.asarray(out[0])
            return np.asarray(self._model.encode(text, normalize_embeddings=True))


class APIEncoder:
    """OpenAI-compatible ``/v1/embeddings`` client (OpenRouter, OpenAI, …).

    Mirrors :class:`QueryEncoder`'s interface (``.dim`` set at init,
    ``.encode()`` returning L2-normalized vectors with the same shape
    convention). API responses are normalized client-side so callers can
    swap backends without touching downstream cosine-similarity math.

    Args:
        model: Provider-qualified model id (e.g.
            ``openai/text-embedding-3-small`` for OpenRouter).
        api_key: Auth key. Falls back to ``HIONTOP_EMBEDDING_API_KEY`` then
            ``OPENAI_API_KEY``.
        base_url: API root. Falls back to ``HIONTOP_EMBEDDING_BASE_URL`` then
            ``OPENAI_BASE_URL`` then ``DEFAULT_API_BASE_URL``.
        dim: Embedding dimension. If ``None``, probed with one request.
        batch_size: Max inputs per request. Larger lists are chunked.
        dimensions: Optional ``dimensions`` param for providers that
            support truncated outputs (e.g. OpenAI text-embedding-3 family).
    """

    def __init__(
        self,
        model: str = DEFAULT_API_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        dim: int | None = None,
        batch_size: int = API_BATCH_SIZE,
        dimensions: int | None = None,
    ) -> None:
        from openai import OpenAI

        api_key = (
            api_key
            or os.environ.get("HIONTOP_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        base_url = (
            base_url
            or os.environ.get("HIONTOP_EMBEDDING_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_API_BASE_URL
        )
        if not api_key:
            raise RuntimeError(
                "APIEncoder requires an API key — set HIONTOP_EMBEDDING_API_KEY "
                "or OPENAI_API_KEY (see .env / .env.example)."
            )

        self.model = model
        self.base_url = base_url
        self.batch_size = max(1, int(batch_size))
        self.dimensions = dimensions
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._lock = threading.Lock()

        self.device = "api"
        self.model_name = model

        if dim is None:
            probe = self._raw_embed([API_PROBE_TEXT])
            self.dim = int(probe.shape[1])
        else:
            self.dim = int(dim)

    def _raw_embed(self, texts: list[str]) -> np.ndarray:
        """Call the embeddings endpoint, return un-normalized (n, dim) array."""
        kwargs: dict[str, object] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(**kwargs)
        # Provider may not preserve input order; sort by ``index``.
        rows = sorted(resp.data, key=lambda d: d.index)
        return np.asarray([r.embedding for r in rows], dtype=np.float64)

    def _embed_batched(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            chunks.append(self._raw_embed(texts[i : i + self.batch_size]))
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.dim))

    def encode(self, text: str | list[str]) -> np.ndarray:
        """Encode one or many strings to L2-normalized vectors.

        Thread-safe (serialized via an internal lock).

        Args:
            text: A single string or a list of strings.

        Returns:
            If ``text`` is ``str`` → shape ``(dim,)``.
            If ``text`` is ``list[str]`` → shape ``(n, dim)``.
        """
        with self._lock:
            if isinstance(text, str):
                vecs = self._embed_batched([text])
                return _l2_normalize(vecs)[0]
            if not text:
                return np.empty((0, self.dim))
            vecs = self._embed_batched(list(text))
            return _l2_normalize(vecs)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (zero rows kept as-is to avoid div-by-zero)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def make_encoder(
    device: str | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> _Encoder:
    """Factory that picks ``local`` or ``api`` backend.

    Resolution order:
        1. ``backend`` arg
        2. ``HIONTOP_EMBEDDING_BACKEND`` env var (``local`` | ``api``)
        3. default: ``local``

    Args:
        device: Forwarded to local backend; ignored by API backend.
        backend: ``"local"`` or ``"api"``. ``None`` = read env / default.
        model: Override model name. Local default ``BGE_MODEL_NAME``;
            API default ``DEFAULT_API_MODEL`` (also overridable via
            ``HIONTOP_EMBEDDING_MODEL``).

    Returns:
        An object exposing ``.dim`` and ``.encode()``.
    """
    backend = (backend or os.environ.get("HIONTOP_EMBEDDING_BACKEND") or "local").lower()
    if backend == "api":
        api_model = (
            model
            or os.environ.get("HIONTOP_EMBEDDING_MODEL")
            or DEFAULT_API_MODEL
        )
        dim_env = os.environ.get("HIONTOP_EMBEDDING_DIM")
        dim = int(dim_env) if dim_env else None
        dimensions_env = os.environ.get("HIONTOP_EMBEDDING_DIMENSIONS")
        dimensions = int(dimensions_env) if dimensions_env else None
        return APIEncoder(model=api_model, dim=dim, dimensions=dimensions)
    if backend == "local":
        local_model = model or os.environ.get("HIONTOP_EMBEDDING_MODEL") or BGE_MODEL_NAME
        return QueryEncoder(device=device, model_name=local_model)
    raise ValueError(
        f"Unknown HIONTOP_EMBEDDING_BACKEND={backend!r} (expected 'local' or 'api')"
    )
