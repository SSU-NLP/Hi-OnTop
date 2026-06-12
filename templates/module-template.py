"""
<모듈 한 줄 설명>

Related docs:
- context/01-hi-ontop-design.md #<섹션>
- context/02-math-model.md #<섹션>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Example:
    """예시 데이터 클래스."""

    field: str


def example_fn(arg: str) -> str:
    """한 줄 설명.

    Args:
        arg: 설명.

    Returns:
        설명.
    """
    return arg