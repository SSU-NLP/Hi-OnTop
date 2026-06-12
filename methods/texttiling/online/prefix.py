#!/usr/bin/env python3
"""TextTiling **online** (prefix-causal, past-only) — AUXILIARY.

Hi-OnTop 수정본. 단일 진실원천 = ``scripts/run_texttiling_prefix.py``
(검증 완료). 여기서는 코드 중복 없이 그 러너를 그대로 실행하는
얇은 진입점일 뿐 (methods/ 정리 목적). 인자 그대로 전달.

  python methods/texttiling/online.py [--name ... --datasets ... ...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent.parent.parent
          / "scripts" / "run_texttiling_prefix.py")

if __name__ == "__main__":
    sys.argv = [str(SCRIPT)] + sys.argv[1:]
    runpy.run_path(str(SCRIPT), run_name="__main__")
