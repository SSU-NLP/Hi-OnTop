"""Device-agnostic PyTorch helpers for online baselines.

3-device 지원 (cuda / mps / cpu) 패턴. ``auto`` 일 때 cuda → mps → cpu
우선순위. mps 사용 시 일부 op kernel 없을 수 있어 환경변수로 CPU fallback
활성화 (반드시 torch / transformers import *이전* 호출).

codex 2026-05-21 권고 반영.
"""

from __future__ import annotations

import os


_VALID = ("auto", "cuda", "mps", "cpu")


def enable_mps_fallback() -> None:
    """``PYTORCH_ENABLE_MPS_FALLBACK=1`` 설정. mps device 사용 시 torch
    import *이전* 호출 필수. 이미 set 이면 no-op.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(arg: str = "auto") -> str:
    """``--device`` 인자 → 실제 device 문자열. *torch 를 늦게 import* 하여
    enable_mps_fallback() 의 import-순서 제약을 깨지 않음.
    """
    if arg not in _VALID:
        raise ValueError(f"--device must be one of {_VALID}, got {arg!r}")
    import torch  # lazy
    if arg != "auto":
        if arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda 요청했으나 cuda 사용 불가")
        if arg == "mps" and not (torch.backends.mps.is_available()
                                 and torch.backends.mps.is_built()):
            raise RuntimeError("--device mps 요청했으나 mps 사용 불가")
        return arg
    # auto: cuda → mps → cpu
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"
