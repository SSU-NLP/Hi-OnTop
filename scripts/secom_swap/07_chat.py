"""Chat (response generation) given retrieved memory.

Per SeCom's eval contract: for each (question, retrieved_text) pair, ask
the chat LLM to answer using the retrieved context. Writes
``sample["predictions"]: List[str]`` aligned with ``sample["questions"]``.

LLM = ``openai/gpt-4o-mini`` via Crts (configurable).
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


CHAT_PROMPT = """Below is a conversation history excerpt that may be relevant to the question. Use it to answer briefly and accurately.

[Conversation history]
{context}

[Question]
{question}

Answer:"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--no_think",
        choices=["reasoning_effort", "chat_template", "off"],
        default="reasoning_effort",
        help=(
            "How to disable hybrid-thinking. reasoning_effort: pass "
            "reasoning_effort='none' (Crts-served Qwen, gpt-4o family). "
            "chat_template: pass chat_template_kwargs.enable_thinking=False "
            "(self-hosted vLLM Qwen3). off: pass nothing (non-thinking models)."
        ),
    )
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY") or "dummy"
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        sys.exit("OPENAI_BASE_URL missing (.env or env override)")

    # thinking-disable kwargs for chat.completions.create — endpoint-dependent.
    think_kwargs: dict = {}
    if args.no_think == "reasoning_effort":
        think_kwargs["reasoning_effort"] = "none"
    elif args.no_think == "chat_template":
        think_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    import openai
    client = openai.OpenAI(api_key=key, base_url=base_url)

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"n_conv: {len(data)}, model: {args.model}", flush=True)

    # GPT-5 family compat: max_tokens unsupported (→ max_completion_tokens),
    # only temperature=1, top_p=1, reasoning_effort∈{minimal,low,medium,high}.
    is_gpt5_family = any(tag in args.model.lower()
                          for tag in ("gpt-5", "gpt-5.5"))

    def call(prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                if is_gpt5_family:
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max(args.max_tokens, 2000),
                        reasoning_effort="minimal",
                    )
                else:
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        # Hybrid-thinking models otherwise spend the whole token
                        # budget on <think> and return an empty/truncated answer.
                        **think_kwargs,
                    )
                content = r.choices[0].message.content or ""
                return content, time.perf_counter() - t0
            except Exception as e:
                if attempt == 2:
                    return f"[ERROR] {str(e)[:200]}", time.perf_counter() - t0
                time.sleep(2 * (attempt + 1))
        return "", 0.0

    total_calls = sum(len(s["questions"]) for s in data)
    print(f"total chat calls: {total_calls}", flush=True)

    results = []
    pbar = tqdm(total=total_calls, desc="chat")
    chat_times = []
    for sample in data:
        questions = sample["questions"]
        contexts = sample["retrieved_texts"]
        assert len(questions) == len(contexts)
        prompts = [
            CHAT_PROMPT.format(context=ctx, question=q)
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

    print(f"\nchat done. avg latency = {sum(chat_times)/len(chat_times):.2f}s", flush=True)


if __name__ == "__main__":
    main()
