"""SeCom baseline segmentation (LLM-based, configurable Crts model).

Mirrors SeCom's ``experiment/segment.py`` but adds per-conversation latency
recording so we can compare against v4.1.3. Uses the same env-var pattern
as ``scripts/run_plainprompt_online.py`` (the canonical Hi-OnTop-on-Crts entry):

    OPENAI_API_KEY  — Crts chat key (e.g. lab-... with chat scope)
    OPENAI_BASE_URL — https://api.ssunlp.co.kr/v1

Default segment model = ``qwen/qwen3.5-9b`` (matches Hi-OnTop's prior usage).
Override via --segment_model (e.g. ``qwen/qwen3.5-27b`` for stronger,
``openai/gpt-4o-mini`` if Crts-side permission exists).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/SeCom"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--load_path",
        default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"),
    )
    ap.add_argument(
        "--save_path",
        default=str(
            REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/gpt4oseg_mtbp.jsonl"
        ),
    )
    ap.add_argument(
        "--latency_path",
        default=str(
            REPO_ROOT
            / "outputs/experiments/2026-05-21_v413_secom_swap/latency_baseline.json"
        ),
    )
    ap.add_argument(
        "--config_path",
        default=str(REPO_ROOT / "benchmarks/SeCom/secom/configs/mpnet.yaml"),
    )
    ap.add_argument(
        "--segment_model",
        default="qwen/qwen3.5-9b",
        help="Crts chat slug used for LLM segmentation (overrides config).",
    )
    ap.add_argument(
        "--no_think",
        choices=["reasoning_effort", "chat_template", "off"],
        default="reasoning_effort",
        help=(
            "Disable hybrid-thinking. reasoning_effort: Crts-served Qwen / "
            "gpt-4o. chat_template: self-hosted vLLM Qwen3. off: non-thinking."
        ),
    )
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    # Hi-OnTop canonical pattern (run_plainprompt_online.py): OPENAI_API_KEY +
    # OPENAI_BASE_URL must be in env. SeCom's OpenAILLM uses OPENAI_API_BASE
    # (a slightly different name) — mirror both for compatibility.
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        sys.exit(
            "OPENAI_API_KEY / OPENAI_BASE_URL missing — set them in .env "
            "(same pattern as scripts/run_plainprompt_online.py)."
        )
    os.environ.setdefault("OPENAI_API_BASE", os.environ["OPENAI_BASE_URL"])

    # Import after env is set (SeCom OpenAILLM reads env at construction).
    from secom import SeCom

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latency_path).parent.mkdir(parents=True, exist_ok=True)

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"n_conv: {len(data)}")

    # Build a stripped config so SeCom doesn't load LLMLingua-2 / mpnet at
    # construction (we only need segmentor here; compress/retrieve are later stages).
    from omegaconf import OmegaConf
    cfg_full = OmegaConf.load(args.config_path)
    seg_only_cfg = REPO_ROOT / "benchmarks/SeCom/secom/configs/_seg_only_tmp.yaml"
    OmegaConf.save({"segmentor": cfg_full.segmentor}, seg_only_cfg)
    secom = SeCom(config_path=str(seg_only_cfg))
    # Override the segment LLM with the user's chosen Crts slug.
    from secom.utils import OpenAILLM
    secom.segment_model = args.segment_model
    secom.segmentor = OpenAILLM(args.segment_model)
    print(f"SeCom segmentor model: {secom.segment_model}")

    # Qwen3.5 (hybrid-thinking) burns the whole token budget on <think> and
    # returns empty/truncated segmentation. SeCom's OpenAILLM (benchmarks/,
    # read-only) does not expose reasoning control — inject it by wrapping the
    # OpenAI client's create(). Mechanism is endpoint-dependent (see --no_think):
    # Crts-served Qwen honors reasoning_effort; self-hosted vLLM Qwen3 honors
    # chat_template_kwargs.enable_thinking. Non-thinking models → off.
    # GPT-5 family: max_tokens / temperature 0.7 / top_p unsupported → translate.
    is_gpt5 = "gpt-5" in args.segment_model.lower()
    if args.no_think != "off" or is_gpt5:
        _cc = secom.segmentor.client.chat.completions
        _orig_create = _cc.create

        def _create_patched(*a, **kw):  # noqa: ANN002,ANN003
            if is_gpt5:
                # GPT-5 requires max_completion_tokens; only temp=1, top_p=1 supported;
                # reasoning_effort must be one of {minimal, low, medium, high}.
                if "max_tokens" in kw:
                    kw["max_completion_tokens"] = max(kw.pop("max_tokens"), 2000)
                kw.pop("temperature", None)
                kw.pop("top_p", None)
                kw.pop("seed", None)
                kw.setdefault("reasoning_effort", "minimal")
                return _orig_create(*a, **kw)
            if args.no_think == "reasoning_effort":
                kw.setdefault("reasoning_effort", "none")
            else:  # chat_template
                eb = dict(kw.get("extra_body") or {})
                eb.setdefault("chat_template_kwargs", {"enable_thinking": False})
                kw["extra_body"] = eb
            return _orig_create(*a, **kw)

        _cc.create = _create_patched
    print(f"no_think mode: {args.no_think} (is_gpt5={is_gpt5})")

    results = []
    per_conv_latency = []

    for idx, sample in enumerate(tqdm(data, desc="seg-baseline")):
        n_ex = sum(len(s) for s in sample["sessions"])
        t0 = time.perf_counter()
        try:
            segments = secom.segment(sample["sessions"])
        except Exception as e:
            print(f"  conv {idx} FAILED: {e}", flush=True)
            segments = []
        t = time.perf_counter() - t0
        sample["segments"] = segments
        results.append(sample)
        lat = {
            "conversation_id": sample["conversation_id"],
            "n_sessions": len(sample["sessions"]),
            "n_exchanges": n_ex,
            "n_segments": len(segments),
            "total_sec": t,
            "sec_per_exchange": t / max(1, n_ex),
        }
        per_conv_latency.append(lat)
        print(
            f"  conv {idx} ({sample['conversation_id']}): "
            f"{n_ex} ex → {len(segments)} segs, {t:.1f}s, "
            f"{lat['sec_per_exchange']*1000:.0f}ms/ex",
            flush=True,
        )
        # incremental save
        with open(args.save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ex_total = sum(lat["n_exchanges"] for lat in per_conv_latency)
    n_seg_total = sum(lat["n_segments"] for lat in per_conv_latency)
    total_sec = sum(lat["total_sec"] for lat in per_conv_latency)
    summary = {
        "method": "secom_baseline_llm",
        "segment_model": secom.segment_model,
        "n_conv": len(per_conv_latency),
        "n_exchanges": n_ex_total,
        "n_segments": n_seg_total,
        "avg_exchanges_per_segment": n_ex_total / max(1, n_seg_total),
        "total_sec": total_sec,
        "ms_per_exchange": total_sec * 1000 / max(1, n_ex_total),
        "per_conv": per_conv_latency,
    }
    with open(args.latency_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nbaseline latency -> {args.latency_path}")
    print(
        f"baseline: {n_seg_total} segments / {n_ex_total} exchanges, "
        f"{summary['ms_per_exchange']:.1f} ms/exchange"
    )


if __name__ == "__main__":
    main()
