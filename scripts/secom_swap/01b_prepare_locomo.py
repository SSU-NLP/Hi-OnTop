"""Prepare LoCoMo in SeCom's expected JSONL format.

LoCoMo (`benchmarks/locomo/locomo10.json`) → the same schema SeCom's
`segment.py` / `retrieve.py` / `chat.py` consume:
``{conversation_id, sessions: List[List[str]], questions, answers}``.

LoCoMo specifics
----------------
- Each conversation's ``conversation`` dict holds ``session_N`` lists of
  turn dicts ``{speaker, dia_id, text, ...}``. Sessions are ordered by the
  integer ``N``. Each turn → one exchange string ``"Speaker: text"``.
  Image-only turns (empty ``text``) fall back to ``blip_caption``.
- QA: category-5 questions are *adversarial* — their gold is in
  ``adversarial_answer`` (``answer`` is null). All other categories use
  ``answer``. Everything is stringified.

Output: ``benchmarks/SeCom/experiment/data/locomo/locomo.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "benchmarks/locomo/locomo10.json"
OUT = REPO_ROOT / "benchmarks/SeCom/experiment/data/locomo/locomo.jsonl"


def session_keys(conv: dict) -> list[str]:
    """`session_N` keys sorted by integer N (excludes `*_date_time`)."""
    keys = [
        k for k in conv
        if k.startswith("session_") and not k.endswith("date_time")
        and isinstance(conv[k], list)
    ]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def turn_to_str(turn: dict) -> str:
    speaker = turn.get("speaker", "Unknown")
    text = (turn.get("text") or "").strip()
    if not text:
        cap = (turn.get("blip_caption") or "").strip()
        text = f"[shares an image] {cap}" if cap else "[non-text turn]"
    return f"{speaker}: {text}"


def qa_answer(q: dict) -> str:
    """Gold answer. Category-5 (adversarial) uses `adversarial_answer`."""
    ans = q.get("answer")
    if ans is None:
        ans = q.get("adversarial_answer")
    return "" if ans is None else str(ans)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(SRC.read_text())
    n_conv = n_sess = n_turn = n_qa = 0
    with OUT.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(data):
            conv = sample["conversation"]
            sessions: list[list[str]] = []
            for sk in session_keys(conv):
                sess = [turn_to_str(t) for t in conv[sk]]
                if sess:
                    sessions.append(sess)
                    n_turn += len(sess)
            qa = sample.get("qa", [])
            row = {
                "conversation_id": str(sample.get("sample_id", idx)),
                "sessions": sessions,
                "questions": [str(q["question"]) for q in qa],
                "answers": [qa_answer(q) for q in qa],
                "qa_categories": [q.get("category") for q in qa],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_conv += 1
            n_sess += len(sessions)
            n_qa += len(qa)
    print(f"wrote {n_conv} conversations to {OUT}")
    print(f"  sessions={n_sess}  turns={n_turn}  questions={n_qa}")


if __name__ == "__main__":
    main()
