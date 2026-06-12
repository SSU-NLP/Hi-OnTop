"""Re-segment ONLY the sessions where SeCom's strict JSON parser failed.

Detection heuristic: a session was likely a JSON-parse-failure if SeCom's
fallback (chunks of 3 exchanges) was used → most of its segments have
exactly 3 exchanges, last segment ≤ 3, and the count matches
ceil(n_turns / 3).

For each detected session, re-call gpt-4o-mini and parse with a lenient
regex (``"num_exchanges"\s*:\s*(\d+)``) instead of strict ``json.loads``.

Writes a patched JSONL that mirrors the input with the re-segmented sessions
substituted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/SeCom"))


_RE_NUM_EX = re.compile(r'"num_exchanges"\s*:\s*(\d+)')
_RE_SEG_BLOCK = re.compile(r"<segmentation>([\s\S]*?)</segmentation>")


def _looks_like_fallback(session_segments: list[list[str]], n_exchanges: int) -> bool:
    """Heuristic: SeCom's fixed-3 fallback produces ceil(n/3) segments,
    each exactly 3 except possibly the last."""
    expected_n = math.ceil(n_exchanges / 3)
    if len(session_segments) != expected_n:
        return False
    for i, seg in enumerate(session_segments):
        if i < len(session_segments) - 1:
            if len(seg) != 3:
                return False
        else:
            if len(seg) > 3:
                return False
    return True


def _parse_lenient(response: str, n_exchanges: int, exchanges: list[str]) -> list[list[str]] | None:
    """Try strict JSON first, then regex fallback that just extracts num_exchanges."""
    # Find <segmentation> block
    blocks = _RE_SEG_BLOCK.findall(response)
    if not blocks:
        return None
    body = blocks[0]
    nums = [int(x) for x in _RE_NUM_EX.findall(body)]
    if not nums:
        return None
    if sum(nums) != n_exchanges:
        # Try to coerce: if the sum differs by 1-2, accept and let last segment absorb
        diff = n_exchanges - sum(nums)
        if abs(diff) <= 2 and nums:
            nums[-1] += diff
            if any(n <= 0 for n in nums):
                return None
        else:
            return None
    out: list[list[str]] = []
    prev = 0
    for n in nums:
        if n <= 0:
            return None
        out.append(exchanges[prev : prev + n])
        prev += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--segment_model", default="openai/gpt-4o-mini")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing")
    os.environ.setdefault("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", ""))

    from secom.utils import OpenAILLM
    llm = OpenAILLM(args.segment_model)
    prompt_template = (
        REPO_ROOT
        / "benchmarks/SeCom/secom/instructions/segment_with_exchange_number.md"
    ).read_text()

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))

    n_fixed = 0
    n_still_failed = 0
    n_sessions_total = 0

    for sample in data:
        sessions = sample["sessions"]
        # SeCom's segment() flattens all sessions into one list of segments,
        # in session order. We need to recover which segments belong to which
        # session via cumulative exchange counts.
        all_segs = sample["segments"]
        idx = 0  # into all_segs
        new_all_segs: list[list[str]] = []
        for sess_idx, sess in enumerate(sessions):
            n_sessions_total += 1
            target_n_ex = len(sess)
            # consume segments from all_segs until cumulative exchange count matches
            consumed = []
            cum = 0
            while idx < len(all_segs) and cum < target_n_ex:
                consumed.append(all_segs[idx])
                cum += len(all_segs[idx])
                idx += 1
            if cum != target_n_ex:
                # mismatch — give up on this session, keep as-is
                new_all_segs.extend(consumed)
                continue

            if not _looks_like_fallback(consumed, target_n_ex):
                # successful parse originally → keep
                new_all_segs.extend(consumed)
                continue

            # likely fallback — re-call LLM and parse leniently
            exchanges_str = "".join(
                f"[Exchange {i}]: {ex}\n\n" for i, ex in enumerate(sess)
            )
            prompt = prompt_template.format(text_to_be_segmented=exchanges_str)
            t0 = time.perf_counter()
            try:
                resp = llm(prompt, max_tokens=2048)
            except Exception as e:
                print(f"  conv {sample['conversation_id']} sess {sess_idx}: LLM err {e}", flush=True)
                new_all_segs.extend(consumed)
                n_still_failed += 1
                continue
            parsed = _parse_lenient(resp, target_n_ex, sess)
            dt = time.perf_counter() - t0
            if parsed is None:
                print(f"  conv {sample['conversation_id']} sess {sess_idx}: lenient parse FAILED ({dt:.1f}s, kept fallback)", flush=True)
                new_all_segs.extend(consumed)
                n_still_failed += 1
            else:
                print(f"  conv {sample['conversation_id']} sess {sess_idx}: re-segmented {target_n_ex} ex → {len(parsed)} segs ({dt:.1f}s)", flush=True)
                new_all_segs.extend(parsed)
                n_fixed += 1
        sample["segments"] = new_all_segs

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nfixed {n_fixed}/{n_sessions_total} sessions, "
          f"{n_still_failed} still failed (kept fallback). saved -> {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
