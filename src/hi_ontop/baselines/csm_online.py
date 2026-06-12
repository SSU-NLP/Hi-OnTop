"""CSM-online-delay2: streaming variant of lxing532 / Dialogue-Topic-Segmenter.

Original CSM (Xing et al., Dialogue-Topic-Segmenter) inference:

  1. encode each adjacent (utt_i, utt_{i+1}) pair via CoherenceNet (DSE-BERT
     backbone + small MLP head trained as Coherence Model).
  2. compute depth_score per gap using TextTiling-style left/right peak
     (offline: looks at ALL future scores).
  3. threshold = global ``mean + alpha * std`` of all depth scores.
  4. emit boundaries where depth > threshold.

This implementation makes it streaming:

  - ``push(utt)`` per turn: as soon as we see ``utt_t`` we can score the
    similarity between ``utt_{t-1}`` and ``utt_t`` (one BERT forward).
  - Depth-score is computed with **delay=k** right context: when the
    similarity sequence has at least *k* future values, the depth for the
    earlier gap can be approximated using past-full + future-k peak.
  - Threshold is maintained online with Welford running mean+std over the
    depth scores seen so far. Same alpha hyper-parameter as the original
    (default 0.0; positive → fewer boundaries).
  - min_gap suppression after emit.

This honors the original score formula and ckpt; only the *thresholding
horizon* is replaced by a streaming approximation. Suitable as an
``online`` row in the segmentation comparison.

Ckpt path defaults to our local trained checkpoint
(`outputs/runs/_misc/cpt_277000.pth`). The backbone is hard-coded to
``aws-ai/dse-bert-base`` to match the original CM mode.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_DEFAULT_CKPT = "methods/CSM/cpt_277000.pth"


def _ensure_csm_import():
    """Add external/Dialogue-Topic-Segmenter to sys.path so we can reuse
    ``CoherenceNet`` from there. We import it lazily to avoid pulling
    transformers at module-import time.

    Path resolution: this file is at ``src/hi_ontop/baselines/csm_online.py``,
    so the repo root is ``parents[3]``.
    """
    # Allow CSM_REPO_DIR env override (matches Hi-OnTop .env convention).
    env_dir = os.environ.get("CSM_REPO_DIR")
    if env_dir:
        csm_dir = Path(env_dir).expanduser().resolve()
    else:
        csm_dir = (
            Path(__file__).resolve().parents[3]
            / "external" / "Dialogue-Topic-Segmenter"
        )
    if not csm_dir.exists():
        raise FileNotFoundError(
            f"CSM repo not found at {csm_dir}. Set CSM_REPO_DIR env."
        )
    s = str(csm_dir)
    if s not in sys.path:
        sys.path.insert(0, s)


@dataclass
class _Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


@dataclass
class CSMOnlineDelay2:
    """Streaming CSM with delay-2 right context.

    Args:
        ckpt_path: trained CoherenceNet checkpoint (default = our local).
        backbone: DSE-BERT id (default ``aws-ai/dse-bert-base``).
        device: ``cpu`` | ``cuda`` | ``auto``.
        alpha: threshold = mean_depth + alpha * std_depth. Lower → more
            boundaries. Default 0.0 (= threshold at running mean).
        delay: right-context width when computing depth (default 2).
        warmup_gaps: hold all boundaries until at least this many depth
            samples have been observed. Default 3.
        min_gap: minimum utterance distance between consecutive emitted
            boundaries. Default 2.
        max_seq_length: tokenizer max length (original 128).
    """

    ckpt_path: str = _DEFAULT_CKPT
    # Our CSM ckpt was trained with bert-base-uncased (per CSM_ENCODER in .env),
    # NOT DSE-BERT that the upstream segment.py hardcodes for CM mode.
    # Matching the trained backbone here is essential for non-collapsed scores
    # (smoke test confirmed: DSE-BERT backbone → softmax saturates to 1.0).
    backbone: str = "bert-base-uncased"
    device: str = "cpu"
    alpha: float = 0.0
    delay: int = 2
    warmup_gaps: int = 3
    min_gap: int = 2
    max_seq_length: int = 128

    # internal state ------------------------------------------------------
    _utts: list[str] = field(default_factory=list, init=False)
    _t: int = field(default=0, init=False)
    _scores: list[float] = field(default_factory=list, init=False)  # sim(s_{i-1}, s_i)
    _depths: dict[int, float] = field(default_factory=dict, init=False)  # gap idx → depth
    _evaluated_until: int = field(default=-1, init=False)
    _emitted: list[int] = field(default_factory=list, init=False)
    _last_boundary_t: int = field(default=-(10**9), init=False)
    _welf: _Welford = field(default_factory=_Welford, init=False)
    _model: object = field(default=None, init=False)
    _tokenizer: object = field(default=None, init=False)
    _device_resolved: Optional[str] = field(default=None, init=False)
    _neural_sec: float = field(default=0.0, init=False)  # cumulative coherence-forward time

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _ensure_csm_import()
        import torch
        from transformers import AutoModel, AutoTokenizer
        from model_utils import CoherenceNet  # type: ignore

        if self.device == "auto":
            self._device_resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device_resolved = self.device

        bert = AutoModel.from_pretrained(self.backbone)
        net = CoherenceNet(bert, self._device_resolved)
        ckpt_full = Path(self.ckpt_path)
        if not ckpt_full.is_absolute():
            ckpt_full = Path(__file__).resolve().parents[3] / self.ckpt_path
        state = torch.load(str(ckpt_full), map_location=self._device_resolved)
        net.load_state_dict(state)
        net.eval()
        net.to(self._device_resolved)
        self._model = net
        self._tokenizer = AutoTokenizer.from_pretrained(self.backbone)

    # ------------------------------------------------------------------
    def _coherence_score(self, sent1: str, sent2: str) -> float:
        """One coherence-model forward → continuous coherence score.

        Matches the SuperDialseg paper baseline
        (``benchmarks/superdialseg/src/super_dialseg/models/csm/modeling_csm.py:90``):
        ``sigmoid(logits[:, 0])`` — i.e. read class-0 logit only and pass
        through sigmoid. The ckpt ``cpt_277000.pth`` (lxing532
        CoherenceNet) outputs a [1, 2] logit tensor; the paper baseline
        reuses it as a single-class scorer via sigmoid on class-0.

        Note: in MTB+ (``"[human]: ... [bot]: ..."`` role-marked format)
        the classifier saturates near 1.0 for nearly every pair, which
        previously motivated a raw-logit-diff hack. On DTS-bundle plain
        text the sigmoid signal is non-saturated and stable, so we now
        match the paper formulation. (Earlier raw-diff hack was a
        domain-specific bypass — see decision-log 2026-05-24.)
        """
        import torch
        tok = self._tokenizer(
            sent1, sent2, padding="max_length",
            max_length=self.max_seq_length, truncation=True, return_tensors="pt",
        )
        _t0 = time.perf_counter()
        with torch.no_grad():
            tok = {k: v.to(self._device_resolved) for k, v in tok.items()}
            bert_out = self._model.bert(**tok)
            cls = bert_out.last_hidden_state[:, 0, :]
            logits = self._model.coherence_decoder(cls)  # [1, 2]
        # paper: sigmoid(logits[:, 0]) → coherence score in (0, 1).
        out = float(torch.sigmoid(logits[0, 0]).cpu().item())
        self._neural_sec += time.perf_counter() - _t0
        return out

    # ------------------------------------------------------------------
    def push(self, utterance: str) -> list[int]:
        """Append one utterance, emit any newly-confirmed boundaries (1-based)."""
        self._ensure_loaded()
        self._utts.append(utterance)
        self._t = len(self._utts)
        new_boundaries: list[int] = []

        # When we receive utt_t (t >= 2), we can score the gap between
        # utt_{t-1} and utt_t. gap_index = t-2 (0-based among gaps).
        if self._t >= 2:
            gap_idx = self._t - 2
            s = self._coherence_score(self._utts[-2], self._utts[-1])
            assert gap_idx == len(self._scores), \
                f"gap_idx mismatch: {gap_idx} vs {len(self._scores)}"
            self._scores.append(s)

        # We can compute depth for gap g once we have right-context of
        # length `delay`. So evaluate depths for any gap up to len(_scores) - delay - 1.
        new_last = len(self._scores) - self.delay - 1
        if new_last > self._evaluated_until:
            for gi in range(self._evaluated_until + 1, new_last + 1):
                depth = self._depth_at(gi)
                self._depths[gi] = depth
                self._welf.push(depth)
                # threshold check
                if (
                    self._welf.n >= self.warmup_gaps
                    and depth > self._welf.mean + self.alpha * self._welf.std()
                ):
                    # gap gi is between 0-based utt[gi] and utt[gi+1].
                    # New segment starts at utt[gi+1] = 1-based index gi+2.
                    # Gold convention (parse_defdts_dialogue): boundary index
                    # = 1-based index of the NEW segment's first utt.
                    # (Earlier this emitted gi+1 = LAST utt of OLD segment,
                    # causing systematic off-by-one vs gold; fixed 2026-05-24.)
                    owner_t = gi + 2
                    if owner_t - self._last_boundary_t >= self.min_gap:
                        self._last_boundary_t = owner_t
                        new_boundaries.append(owner_t)
            self._evaluated_until = new_last
        return new_boundaries

    # ------------------------------------------------------------------
    def _depth_at(self, gi: int) -> float:
        """One-sided depth (TextTiling-style left/right peak)."""
        s_center = self._scores[gi]
        # left peak
        left_peak = s_center
        for j in range(gi - 1, -1, -1):
            if self._scores[j] >= left_peak:
                left_peak = self._scores[j]
            else:
                break
        # right peak (bounded by delay window)
        right_peak = s_center
        for j in range(gi + 1, min(len(self._scores), gi + 1 + self.delay)):
            if self._scores[j] >= right_peak:
                right_peak = self._scores[j]
            else:
                break
        return 0.5 * (left_peak + right_peak - 2.0 * s_center)

    # ------------------------------------------------------------------
    def flush(self) -> list[int]:
        """End-of-stream: evaluate any remaining gaps with whatever right
        context is available (no future to wait for)."""
        out: list[int] = []
        new_last = len(self._scores) - 1
        for gi in range(self._evaluated_until + 1, new_last + 1):
            depth = self._depth_at(gi)
            self._depths[gi] = depth
            self._welf.push(depth)
            if (
                self._welf.n >= self.warmup_gaps
                and depth > self._welf.mean + self.alpha * self._welf.std()
            ):
                # See comment in push(): owner_t = 1-based new-segment start
                # (gi+2), not last-old (gi+1). off-by-one fixed 2026-05-24.
                owner_t = gi + 2
                if owner_t - self._last_boundary_t >= self.min_gap:
                    self._last_boundary_t = owner_t
                    out.append(owner_t)
        self._evaluated_until = new_last
        return out

    # ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "t": self._t,
            "n_scores": len(self._scores),
            "n_depths": len(self._depths),
            "n_boundaries": self._last_boundary_t,
            "device": self._device_resolved,
            "depth_mean": self._welf.mean,
            "depth_std": self._welf.std(),
        }
