"""Chat baselines without retrieval: Zero History (no context) and Full History.

Two reference rows in SeCom Table 1:
- **Zero History**: chat LLM gets ONLY the question (lower bound — what the
  LLM knows without any memory access).
- **Full History**: chat LLM gets the FULL conversation history concatenated
  (upper bound IF the model can attend to all of it).

Reads MTB+ data jsonl (with sessions + questions + answers). Writes a
chat.jsonl in the same format as 07_chat.py so 08_eval.py can consume it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


ZERO_PROMPT = """Answer the following question briefly and accurately.

[Question]
{question}

Answer:"""

FULL_PROMPT = """Below is the full conversation history that may be relevant to the question. Use it to answer briefly and accurately.

[Conversation history]
{context}

[Question]
{question}

Answer:"""


def build_full_context(sessions: list[list[str]]) -> str:
    """Concatenate all sessions/turns into a single context block."""
    parts = []
    for i, sess in enumerate(sessions):
        parts.append(f"<Session {i}>")
        parts.extend(sess)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True,
                    help="MTB+ jsonl (with sessions, questions, answers, conversation_id)")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--mode", choices=["zero", "full"], required=True)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not key or not base_url:
        sys.exit("OPENAI_API_KEY / OPENAI_BASE_URL missing in .env")

    import openai
    client = openai.OpenAI(api_key=key, base_url=base_url)

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"n_conv: {len(data)}, mode={args.mode}, model={args.model}", flush=True)

    def call(prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                content = r.choices[0].message.content or ""
                return content, time.perf_counter() - t0
            except Exception as e:
                if attempt == 2:
                    return f"[ERROR] {str(e)[:200]}", time.perf_counter() - t0
                time.sleep(2 * (attempt + 1))
        return "", 0.0

    # Build retrieved_texts to mirror 07_chat.py output (for downstream eval).
    # In zero mode: empty string per question.
    # In full mode: full history per question.
    for sample in data:
        questions = sample["questions"]
        if args.mode == "zero":
            sample["retrieved_texts"] = [""] * len(questions)
        else:
            ctx = build_full_context(sample["sessions"])
            sample["retrieved_texts"] = [ctx] * len(questions)

    total_calls = sum(len(s["questions"]) for s in data)
    print(f"total chat calls: {total_calls}", flush=True)

    results = []
    pbar = tqdm(total=total_calls, desc=f"chat-{args.mode}")
    chat_times = []
    for sample in data:
        questions = sample["questions"]
        contexts = sample["retrieved_texts"]
        if args.mode == "zero":
            prompts = [ZERO_PROMPT.format(question=q) for q in questions]
        else:
            prompts = [
                FULL_PROMPT.format(context=ctx, question=q)
                for ctx, q in zip(contexts, questions)
            ]
        preds = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(call, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                i = futures[fut]
                content, dt = fut.result()
                preds[i] = content
                chat_times.append(dt)
                pbar.update(1)
        sample["predictions"] = preds
        results.append(sample)
        with open(args.save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pbar.close()

    # Also write a retrieved.metrics.json sibling so 08_eval.py picks up
    # Context Length numbers correctly (n_ex_avg = avg # turns provided as
    # context, n_token_avg = approx via word count *1.3 chars/token).
    n_ex_list = []
    n_tok_list = []
    for sample in data:
        if args.mode == "zero":
            n_ex_list.append(0.0)
            n_tok_list.append(0.0)
        else:
            n_turns = sum(len(s) for s in sample["sessions"])
            ctx = build_full_context(sample["sessions"])
            # crude token count: ~4 chars / token
            n_tok = len(ctx) / 4
            n_ex_list.append(n_turns)
            n_tok_list.append(n_tok)
    metrics = {
        "n_ex_avg": sum(n_ex_list) / len(n_ex_list),
        "n_token_avg": sum(n_tok_list) / len(n_tok_list),
        "topk": -1,
        "mode": args.mode,
        "n_conv": len(data),
    }
    sidecar = Path(args.save_path).parent / "retrieved.metrics.json"
    sidecar.write_text(json.dumps(metrics, indent=2))
    print(f"\nchat done. avg latency = {sum(chat_times)/len(chat_times):.2f}s", flush=True)
    print(f"context_length sidecar: n_ex_avg={metrics['n_ex_avg']:.1f}, "
          f"n_token_avg={metrics['n_token_avg']:.0f}", flush=True)


if __name__ == "__main__":
    main()
