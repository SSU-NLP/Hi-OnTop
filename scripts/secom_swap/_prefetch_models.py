"""Pre-download LLMLingua-2 (~2GB) and roberta-large (~1.4GB) for bertscore.

Run once before the heavy pipeline so the download doesn't fail mid-run.
"""

from __future__ import annotations

import sys
import time


def main() -> None:
    t0 = time.perf_counter()
    print("=== LLMLingua-2 download ===", flush=True)
    try:
        from llmlingua import PromptCompressor
        comp = PromptCompressor(
            "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        # tiny smoke
        out = comp.compress_prompt("Hello world this is a test sentence.", rate=0.5)
        print(f"  smoke OK: {out['compressed_prompt'][:60]!r}", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        sys.exit(1)
    t1 = time.perf_counter()
    print(f"  elapsed {t1 - t0:.1f}s\n", flush=True)

    print("=== bertscore roberta-large download ===", flush=True)
    try:
        from bert_score import score as bert_score
        P, R, F = bert_score(
            ["The quick brown fox jumps"],
            ["A fast brown fox leaps"],
            lang="en",
            verbose=False,
        )
        print(f"  smoke OK: P={float(P[0]):.3f} R={float(R[0]):.3f} F={float(F[0]):.3f}",
              flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        sys.exit(1)
    t2 = time.perf_counter()
    print(f"  elapsed {t2 - t1:.1f}s\n", flush=True)
    print(f"=== ALL DONE in {t2 - t0:.1f}s ===")


if __name__ == "__main__":
    main()
