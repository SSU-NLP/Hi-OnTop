"""Paper-aligned evaluation: BLEU, Rouge1/2/L, BERTScore, GPT4Score + Context Length.

Aligns with SeCom (Pan et al., ICLR 2025) Table 1 / Table 3 metrics:

- **QA Performance**:
  - GPT4Score: gpt-4 judge scoring prediction vs gold (Crts ``openai/gpt-4o``)
  - BLEU, Rouge1, Rouge2, RougeL: via HuggingFace ``evaluate`` (same as
    SeCom's ``evaluate_sim``)
  - BERTScore (P/R/F): via ``bert_score`` (roberta-large, CPU)
- **Context Length** (already collected at retrieve stage):
  - # Turns (n_ex_avg), # Tokens (n_token_avg) from ``retrieved.metrics.json``

Also keeps the SQuAD-style QA F1 + Subspan EM for reference (extra columns
in the JSON; they may be cited as auxiliary metrics).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/SeCom/experiment"))

from metrics import evaluate_match  # noqa: E402  (qa_f1 + subspan_em, kept as aux)


# ---------------------------------------------------------------------------
# GPT4Score (LLM judge)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an evaluator scoring a model's answer to a question about a long conversation. Compare the model's prediction to the ground-truth answer.

Scoring rubric (return a single integer 1-10):
- 10: Prediction conveys all key facts of the gold answer, no contradictions.
- 7-9: Mostly correct, minor missing detail or phrasing difference.
- 4-6: Partially correct; some key info missing or hedged.
- 1-3: Largely wrong, misses or contradicts the gold answer.

Reply with ONLY a single integer on the first line, no explanation.

[Question]
{question}

[Gold answer]
{gold}

[Model prediction]
{pred}

[Score (1-10)]"""


def gpt4_score(client, judge_model: str, q: str, gold: str, pred: str, max_retries: int = 3) -> int:
    prompt = JUDGE_PROMPT.format(question=q.strip(), gold=gold.strip(), pred=(pred or "").strip())
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            txt = (r.choices[0].message.content or "").strip()
            m = re.search(r"\b([1-9]|10)\b", txt)
            if m:
                return int(m.group(1))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  judge error (final attempt): {str(e)[:100]}", flush=True)
                return 0
            time.sleep(2 * (attempt + 1))
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True,
                    help="chat.jsonl with predictions + answers")
    ap.add_argument("--save_path", required=True)
    ap.add_argument(
        "--retrieved_metrics",
        default=None,
        help="optional retrieved.metrics.json (for Context Length n_ex/n_token)."
        " If omitted, infer from sibling retrieved.metrics.json next to chat.jsonl.",
    )
    ap.add_argument("--judge_model", default="openai/gpt-4o",
                    help="Judge LLM for GPT4Score (paper uses GPT-4; gpt-4o "
                    "is the modern equivalent. Use --judge_model openai/gpt-4o-mini for ~10x cheaper.)")
    ap.add_argument("--judge_workers", type=int, default=8)
    ap.add_argument("--skip_judge", action="store_true",
                    help="Skip GPT4Score (saves $ but loses paper headline metric)")
    ap.add_argument("--skip_bertscore", action="store_true")
    args = ap.parse_args()

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))
    preds, gts, questions = [], [], []
    for sample in data:
        for p, g, q in zip(sample["predictions"], sample["answers"], sample["questions"]):
            preds.append(p or "")
            gts.append(g)
            questions.append(q)
    print(f"n_qa_pairs: {len(preds)}", flush=True)

    out: dict = {
        "n_conv": len(data),
        "n_qa": len(preds),
        "load_path": args.load_path,
        "judge_model": args.judge_model if not args.skip_judge else None,
    }

    # 1) Context Length
    if args.retrieved_metrics is None:
        sibling = Path(args.load_path).parent / "retrieved.metrics.json"
        if sibling.exists():
            args.retrieved_metrics = str(sibling)
    if args.retrieved_metrics and Path(args.retrieved_metrics).exists():
        rm = json.loads(Path(args.retrieved_metrics).read_text())
        out["n_turns"] = rm.get("n_ex_avg")
        out["n_tokens"] = rm.get("n_token_avg")
        if out["n_turns"] is not None and out["n_tokens"] is not None:
            print(f"context_length: n_turns={out['n_turns']:.2f}, n_tokens={out['n_tokens']:.1f}", flush=True)

    # 2) BLEU + Rouge1/2/L/Lsum
    print("loading bleu + rouge", flush=True)
    import evaluate
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    preds_t = [(p.lstrip("\n").split("\n")[0].strip() or " ") for p in preds]
    gts_t = [(g.lstrip("\n").split("\n")[0].strip() or " ") for g in gts]
    bleu_res = bleu.compute(predictions=preds_t, references=gts_t)
    rouge_res = rouge.compute(predictions=preds_t, references=gts_t)
    out["bleu"] = float(bleu_res["bleu"]) * 100
    out["rouge1"] = float(rouge_res["rouge1"]) * 100
    out["rouge2"] = float(rouge_res["rouge2"]) * 100
    out["rougeL"] = float(rouge_res["rougeL"]) * 100
    out["rougeLsum"] = float(rouge_res["rougeLsum"]) * 100
    print(f"bleu={out['bleu']:.2f}, rouge1={out['rouge1']:.2f}, "
          f"rouge2={out['rouge2']:.2f}, rougeL={out['rougeL']:.2f}", flush=True)

    # 3) BERTScore
    if not args.skip_bertscore:
        print("computing bertscore (roberta-large CPU)", flush=True)
        from bert_score import score as bert_score
        P, R, F = bert_score(preds_t, gts_t, lang="en", verbose=False, batch_size=16)
        out["bertscore_p"] = float(P.mean()) * 100
        out["bertscore_r"] = float(R.mean()) * 100
        out["bertscore_f1"] = float(F.mean()) * 100
        print(f"bertscore_f1={out['bertscore_f1']:.2f}", flush=True)

    # 4) QA F1 + Subspan EM (auxiliary; SeCom evaluate_match)
    m = evaluate_match(preds, gts, truncate_pred=True)
    out["qa_f1_score"] = m["qa_f1_score"]
    out["best_subspan_em"] = m["best_subspan_em"]
    print(f"qa_f1={out['qa_f1_score']:.2f}, subspan_em={out['best_subspan_em']:.2f}", flush=True)

    # 5) GPT4Score (LLM judge)
    if not args.skip_judge:
        load_dotenv(REPO_ROOT / ".env")
        key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not key or not base_url:
            print("[warn] OPENAI_API_KEY/OPENAI_BASE_URL missing - skipping GPT4Score", flush=True)
        else:
            import openai
            client = openai.OpenAI(api_key=key, base_url=base_url)
            print(f"GPT4Score via {args.judge_model} ({len(preds)} pairs, {args.judge_workers} workers)", flush=True)
            scores = [0] * len(preds)
            with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
                futs = {
                    ex.submit(gpt4_score, client, args.judge_model, q, g, p): i
                    for i, (q, g, p) in enumerate(zip(questions, gts, preds))
                }
                for f in tqdm(as_completed(futs), total=len(futs), desc="judge"):
                    i = futs[f]
                    scores[i] = f.result()
            valid = [s for s in scores if s > 0]
            out["gpt4_score_mean_1_10"] = sum(valid) / max(1, len(valid))
            out["gpt4_score_x10"] = sum(valid) / max(1, len(valid)) * 10  # scale to 100
            out["gpt4_score_n_valid"] = len(valid)
            out["gpt4_score_n_failed"] = len(scores) - len(valid)
            out["gpt4_scores_raw"] = scores
            print(f"GPT4Score: {out['gpt4_score_mean_1_10']:.2f} / 10 "
                  f"(×10 = {out['gpt4_score_x10']:.2f}), valid {len(valid)}/{len(scores)}", flush=True)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    out_save = {k: v for k, v in out.items() if k != "gpt4_scores_raw"}
    Path(args.save_path).write_text(json.dumps(out_save, indent=2))
    print(f"\nsaved -> {args.save_path}")


if __name__ == "__main__":
    main()
