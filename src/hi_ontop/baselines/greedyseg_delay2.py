"""GreedySeg bounded-lookahead online variant (delay=2).

원본 SuperDialseg ``GreedySegmenter`` (Coldog2333/SuperDialseg
``models/greedyseg/modeling_greedyseg.py``) 의 BERT cosine + argmin greedy
selection 을 *streaming 입력 + 출력 lag* 으로 변환. score 공식·HP·encoder·
greedy 선택은 원본 그대로 보존. boundary emit 만 right-context (WINDOW_SIZE
=2 미래 발화) 확보 후로 지연.

codex:rescue 2026-05-20/21 검증 = "본질 보존, 강한 online 명명 OK".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GreedySegOnlineDelay2:
    """GreedySeg-online-delay2 — *strict prefix-causal 아님*.

    원본 alg: 매 segment 마다 ``i ∈ {1, 3, 5, 7}`` 에서 (cut_index..cut_index+i)
    block 을 center 로 두고 ``larger = max(cos(left, center), cos(center,
    right))`` 평가. ``argmin larger`` 위치를 boundary 로 비가역 채택.
    delay=2 = ``right_sent`` (WINDOW_SIZE=2 미래 발화) 가 buffer 에 도착해야
    score 계산 가능. emit list 의 원소는 *과거 utterance index* 일 수 있음.

    Args:
        backbone: HuggingFace model id (default ``bert-base-uncased``).
        window_size: left/right context size (원본 2).
        jump_step: 후보 평가 간격 — i mod jump_step == jump_step-1 에서만 평가 (원본 2).
        max_seg_round: segment 당 후보 누적 최대치. argmin pick 트리거 (원본 8).
        sim_threshold: ``left``/``right`` 미정 (segment 시작/끝) 시 대체 score (원본 0.6).
        max_seq_length: BERT max token length (원본 50).
        device: "auto" | "cuda" | "mps" | "cpu". resolve_device() 적용.

    Notes:
        push() 반환 list 의 boundary index 는 1-based ("utts[i] 부터 새 segment").
        flush() 는 잔여 segment 강제 종료 (argmin so-far).
    """

    backbone: str = "bert-base-uncased"
    window_size: int = 2
    jump_step: int = 2
    max_seg_round: int = 8
    sim_threshold: float = 0.6
    max_seq_length: int = 50
    device: str = "auto"

    # internal state -----------------------------------------------------------
    _utts: list[str] = field(default_factory=list, init=False)
    _t: int = field(default=0, init=False)
    _cut_index: int = field(default=0, init=False)  # 0-based; 현 segment 의 첫 utt idx
    _candidates: list[tuple[int, float]] = field(default_factory=list, init=False)
    # (i_rel, larger). i_rel = position within current segment (0-based offset from cut_index)
    _evaluated_i: set[int] = field(default_factory=set, init=False)
    _device_resolved: Optional[str] = field(default=None, init=False)
    _tokenizer: object = field(default=None, init=False)
    _model: object = field(default=None, init=False)
    _bert_forwards: int = field(default=0, init=False)
    _last_boundary: int = field(default=0, init=False)
    _neural_sec: float = field(default=0.0, init=False)  # cumulative BERT-forward time

    def __post_init__(self) -> None:
        # device 문자열 형식만 sanity-check (실제 torch 가용성은 lazy)
        if self.device not in ("auto", "cuda", "mps", "cpu"):
            raise ValueError(
                f"device must be one of (auto, cuda, mps, cpu), got {self.device!r}")

    # ---- API -----------------------------------------------------------------
    def push(self, utterance: str) -> list[int]:
        """발화 1개 입력. 이번 호출에서 새로 확정된 boundary utterance index
        (1-based) list 반환. 빈 list = 결정 안 됨 (right-context 부족 등)."""
        self._utts.append(utterance)
        self._t = len(self._utts)
        emitted: list[int] = []
        # 새로 평가 가능한 후보들 평가 → segment finalize 가능하면 반복
        while True:
            self._eval_new_candidates()
            if not self._try_finalize_segment(emitted):
                break
        return emitted

    def flush(self) -> list[int]:
        """대화 종료. 잔여 segment 의 argmin so-far 로 boundary 강제 emit
        (없으면 빈 list)."""
        emitted: list[int] = []
        self._eval_new_candidates(force_end=True)
        # 현 segment 의 누적 후보가 있으면 argmin 채택
        if self._candidates:
            i_best, _ = min(self._candidates, key=lambda x: x[1])
            self._commit_boundary(i_best, emitted)
        return emitted

    def state(self) -> dict:
        return dict(
            t=self._t, cut_index=self._cut_index,
            n_candidates=len(self._candidates),
            device=self._device_resolved,
            bert_forwards=self._bert_forwards,
            last_boundary=self._last_boundary,
        )

    # ---- internals -----------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from hi_ontop.baselines._device import enable_mps_fallback, resolve_device
        if self.device in ("auto", "mps"):
            enable_mps_fallback()
        self._device_resolved = resolve_device(self.device)
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self._model = AutoModel.from_pretrained(self.backbone)
        self._model.eval()
        self._model.to(self._device_resolved)
        self._torch = torch  # short alias

    def _encode(self, text: str) -> np.ndarray:
        """Text → [CLS] embedding (numpy, 1-D)."""
        self._ensure_model()
        torch = self._torch
        toks = self._tokenizer(
            text or ".", return_tensors="pt", truncation=True,
            max_length=self.max_seq_length, padding=False)
        toks = {k: v.to(self._device_resolved) for k, v in toks.items()}
        _t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._model(**toks)
        # [CLS] = last_hidden_state[:, 0, :]
        cls = out.last_hidden_state[0, 0].detach().to("cpu").float().numpy()
        self._neural_sec += time.perf_counter() - _t0
        self._bert_forwards += 1
        return cls

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _evaluable_positions(self, force_end: bool = False) -> list[int]:
        """현 segment 내 평가 가능한 i_rel (0-based, i_rel mod jump_step ==
        jump_step-1). right_sent 가 buffer 에 있어야 함."""
        out: list[int] = []
        # i_rel = 1, 3, 5, 7  (when jump_step=2, max_seg_round=8)
        for i_rel in range(self.jump_step - 1, self.max_seg_round, self.jump_step):
            if i_rel in self._evaluated_i:
                continue
            # center = utts[cut_index .. cut_index + i_rel]  (i_rel+1 발화)
            # right_sent = utts[cut_index + i_rel + 1 .. cut_index + i_rel + window_size]
            need_t = self._cut_index + i_rel + 1 + self.window_size
            if need_t > self._t and not force_end:
                break  # 이후 후보는 더 못함
            out.append(i_rel)
        return out

    def _eval_new_candidates(self, force_end: bool = False) -> None:
        for i_rel in self._evaluable_positions(force_end=force_end):
            self._evaluated_i.add(i_rel)
            larger = self._score_at(i_rel, force_end=force_end)
            self._candidates.append((i_rel, larger))

    def _score_at(self, i_rel: int, force_end: bool = False) -> float:
        ci = self._cut_index
        # left_sent = utts[max(0, ci - window_size) : ci]
        left_lo = max(0, ci - self.window_size)
        left_sent = " ".join(self._utts[left_lo:ci])
        center_sent = " ".join(self._utts[ci : ci + i_rel + 1])
        right_hi = min(self._t, ci + i_rel + 1 + self.window_size)
        right_sent = " ".join(self._utts[ci + i_rel + 1 : right_hi])
        left_val = self._cos(self._encode(left_sent), self._encode(center_sent)) \
            if left_sent.strip() else self.sim_threshold
        right_val = self._cos(self._encode(center_sent), self._encode(right_sent)) \
            if right_sent.strip() else self.sim_threshold
        return max(left_val, right_val)

    def _try_finalize_segment(self, emitted: list[int]) -> bool:
        """max_seg_round 후보 모두 평가됐거나, buffer 가 segment 끝 (=
        cut_index + max_seg_round + window_size 이상) 에 도달했으면 argmin
        채택 후 segment 전진. 그 외엔 False."""
        # 평가 가능 후보 전수 = i_rel ∈ {jump_step-1, ..., max_seg_round-1, step=jump_step}
        n_total = len(range(self.jump_step - 1, self.max_seg_round, self.jump_step))
        all_evaluated = len(self._evaluated_i) == n_total
        if not all_evaluated:
            return False
        # argmin
        i_best, _ = min(self._candidates, key=lambda x: x[1])
        self._commit_boundary(i_best, emitted)
        return True

    def _commit_boundary(self, i_rel: int, emitted: list[int]) -> None:
        # boundary 위치 = cut_index + i_rel 의 *다음* utterance 부터 새 segment
        # → 1-based boundary index = (cut_index + i_rel) + 1 + 1 = cut_index + i_rel + 2? No.
        # cut_index 는 0-based "현 segment 의 첫 utt index". i_rel = segment 내 offset.
        # boundary AFTER utt(cut_index + i_rel) (0-based) ⇒ next segment starts at
        # cut_index + i_rel + 1 (0-based) ⇒ 1-based 새 segment 시작 utt idx = cut_index + i_rel + 2.
        new_seg_start_1b = self._cut_index + i_rel + 2  # 1-based
        if 2 <= new_seg_start_1b <= self._t:
            emitted.append(new_seg_start_1b)
            self._last_boundary = new_seg_start_1b
        self._cut_index = self._cut_index + i_rel + 1  # 0-based, 다음 segment 첫 utt
        self._candidates = []
        self._evaluated_i = set()
