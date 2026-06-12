"""Prepare Long-MT-Bench+ in SeCom's expected JSONL format.

Saves to ``benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl`` so SeCom's
``segment.py`` / ``retrieve.py`` / ``chat.py`` can read it as-is.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = load_dataset("panzs19/Long-MT-Bench-Plus", split="test")
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for sample in data:
            # SeCom expects: conversation_id, sessions (List[List[str]]),
            # questions (List[str]), answers (List[str])
            row = {
                "conversation_id": sample["conversation_id"],
                "sessions": sample["sessions"],
                "questions": sample["questions"],
                "answers": sample["answers"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} conversations to {OUT}")


if __name__ == "__main__":
    main()
