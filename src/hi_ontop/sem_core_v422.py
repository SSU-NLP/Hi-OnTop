"""v4.2.2 — scene vector encoder swap (bge-base-en-v1.5 → aws-ai/dse-bert-base).

New MAJOR-minor line on top of v4.1.1 (``sem_core_v411.HiOnTopSegmenterV411``,
causal-window δ identity segmenter). **The algorithm is identical to
v4.1.1.** The ONLY change is the scene vector encoder used to produce
``s_t``: ``baai/bge-base-en-v1.5`` (v4.1.1 default) → **``aws-ai/dse-bert-base``**
(this version). Both are BERT-base (768d) — dim unchanged.

DSE-BERT (AWS AI Labs, "Dialogue Sentence Embedder") is bert-base-uncased
continued-pretrained on dialogue corpora for sentence-level embedding. It
is the same encoder family that ``segment.py`` of lxing532's CSM repo
hardcodes for CM eval, so swapping it in here aligns Hi-OnTop's scene-vector
space with a dialogue-tuned representation.

⚠ NOT a "RNN replacement". v4.1.1 already auto-skips the RNN at
``eta_prev=1`` (default) — ``_uses_rnn = use_rnn and eta_prev < 1.0`` —
so the dynamics-prediction path is already inactive. v4.2.2 swaps the
upstream embedding source, not the in-segmenter dynamics. The δ_eff /
sCRP / Bayes / local-MAP / prior-corrected fresh-baseline structure is
unchanged.

SEM inheritance 3-step (CLAUDE.md SEM 계승 §):
1. **In SEM2?** SEM2 (``nicktfranklin/SEM2``) uses pretrained text
   embeddings as scene vectors; the choice of encoder is not part of
   SEM2's algorithmic core. Switching encoder while keeping the
   downstream event-model intact is a representation-space change, not
   a new mechanism.
2. **Conflict?** None. sCRP / Bayes / local MAP / prior-corrected B0
   are agnostic to the choice of normalized 768d embedding. δ_eff is
   computed in the same way; only its empirical scale shifts because
   cosine geometry differs across encoders.
3. **decision-log 2026-05-20.**

NOT a retrieval-cosine violation (CLAUDE.md retrieval §): retrieval is
query-time STM/LTM ordering by importance; scene-vector encoder choice
is upstream of segmentation, not part of retrieval.

Operational requirements (these MUST be done before reporting v4.2.2 vs
v4.1.1 numbers — otherwise the comparison is meaningless):
- Set ``HIONTOP_EMBEDDING_MODEL=aws-ai/dse-bert-base`` in ``.env``
  (``HIONTOP_EMBEDDING_BACKEND=api`` if served via Crts; ``=local`` if
  loaded via sentence-transformers / HuggingFace locally).
- **Re-calibrate ``delta_star``** on the target corpus's train split
  with the DSE-BERT encoder (``scripts/calibrate_v411_delta_star.py``
  or its v4.2.2-named equivalent). The v4.1.1 default
  ``delta_star=0.5557`` is TIAGE-train + bge-base specific and will not
  transfer.
- ``σδ² = c · δ*²`` follows from the recalibrated ``δ*`` automatically.

Everything else (causal-window δ, η=1 identity, RNN retained via
``use_rnn`` and auto-skipped, prior-corrected B0, fixed ``σδ²=c·δ*²``,
``f0_min_starts=2``) is inherited from v4.1.1 verbatim. To stack with
v4.2.1's ``f0_min_starts=1``, pass that kwarg explicitly.

Expected effect (hypothesis-level): unknown ahead of measurement.
DSE-BERT is dialogue-tuned and may yield a cleaner cosine geometry for
turn-pair similarity → better δ_prev/δ_ctx separation → better
segmentation. But the gain depends on whether bge-base was already
sufficient for these short utterances; could equally be a wash. Decide
after smoke ablation on DialSeg-711 / TIAGE / SuperDialseg train splits.
"""

from __future__ import annotations

from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411

EXPECTED_ENCODER = "aws-ai/dse-bert-base"


class HiOnTopSegmenterV422(HiOnTopSegmenterV411):
    """v4.1.1 algorithm pinned to ``aws-ai/dse-bert-base`` scene encoder.

    Thin subclass: algorithm is the single source of truth in v4.1.1;
    v4.2.2 only fixes the upstream encoder identity by convention and
    optional sanity logging. The encoder itself is selected by the
    orchestrator via ``HIONTOP_EMBEDDING_MODEL``; this class does not
    instantiate one.

    Pass ``encoder_name`` to record the encoder identity in logs /
    experiment metadata (purely informational). If it does not match
    ``EXPECTED_ENCODER`` a warning is logged so that mislabelled runs
    are caught before reporting.

    ``delta_star`` MUST be re-calibrated for DSE-BERT — the v4.1.1
    default (0.5557, bge-base on TIAGE train) does not transfer.
    """

    def __init__(
        self,
        *args,
        encoder_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder_name = encoder_name
        if encoder_name is not None and encoder_name != EXPECTED_ENCODER:
            import warnings

            warnings.warn(
                f"v4.2.2 expects encoder={EXPECTED_ENCODER!r}, "
                f"got {encoder_name!r}. Result label may be misleading.",
                RuntimeWarning,
                stacklevel=2,
            )
