#!/usr/bin/env python3
"""RoBERTa online vs offline 비교 — extended smoke.

dts_result.md 의 RoBERTa online ($^\\sharp$) 행 검증. CPU 에서 부분
subset 으로 빠르게 (online ≈ offline) 검증. GPU 없이 50 dialog × 3
benches = ~150 dialog (≈ 1500 turn). RoBERTa-base CPU forward ~ 100-300 ms
per window → 5-15 min 예상.

방법:
- 같은 subset 으로 offline (eval_set) + online (eval_online) 둘 다 평가
- Score 차이 보고 (Δ = online − offline)
- limit 작아도 offline-vs-online apples-to-apples 비교 가능

데이터: Def-DTS bundle test, 각 bench 의 첫 --limit dialog (deterministic).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, RobertaForTokenClassification

REPO = Path(__file__).resolve().parent.parent

# offline + online 함수 import (분기 방지)
sys.path.insert(0, str(REPO / "methods" / "RoBERTa" / "offline"))
sys.path.insert(0, str(REPO / "methods" / "RoBERTa" / "online"))
from train import eval_set, load_dialogs  # noqa: E402
from segment import eval_online, DEFAULT_MODEL_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50,
                    help="per-bench dialog limit (first N).")
    ap.add_argument("--name", default="2026-05-24_roberta_online_vs_offline")
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--sliding_window", type=int, default=20)
    ap.add_argument("--max_utt_len", type=int, default=25)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = REPO / "outputs" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[device] {device}  ckpt={args.model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = RobertaForTokenClassification.from_pretrained(
        args.model_dir).to(device)
    cfg = dict(model_dir=args.model_dir, sliding_window=args.sliding_window,
               max_utt_len=args.max_utt_len, max_seq_len=args.max_seq_len)

    results = {}
    for ds in ("tiage", "dialseg711", "superseg"):
        full = load_dialogs(ds, "test")
        dialogs = full[: args.limit]
        n_turns = sum(len(u) for u, _ in dialogs)
        print(f"\n=== {ds}  {len(dialogs)}/{len(full)} dial · {n_turns} turn ===",
              flush=True)

        t0 = time.perf_counter()
        r_off = eval_set(model, tok, dialogs, cfg, device)
        t_off = time.perf_counter() - t0
        print(f"  offline: Pk={r_off['pk']:.4f} WD={r_off['wd']:.4f} "
              f"F1={r_off['f1']:.4f} Score={r_off['score']:.4f}  "
              f"({t_off:.1f}s)", flush=True)

        t0 = time.perf_counter()
        r_on = eval_online(model, tok, dialogs, cfg, device)
        t_on = time.perf_counter() - t0
        print(f"  online : Pk={r_on['pk']:.4f} WD={r_on['wd']:.4f} "
              f"F1={r_on['f1']:.4f} Score={r_on['score']:.4f}  "
              f"({t_on:.1f}s)", flush=True)

        d_score = r_on["score"] - r_off["score"]
        print(f"  Δ(online − offline) = {d_score:+.4f}", flush=True)
        results[ds] = dict(
            n_dial=len(dialogs), n_dial_full=len(full), n_turns=n_turns,
            offline=r_off, online=r_on,
            delta_score=d_score, t_offline_s=t_off, t_online_s=t_on)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    L = ["# RoBERTa online vs offline — extended smoke",
         "",
         f"setup: ckpt = `{args.model_dir}`, device {device}, "
         f"per-bench limit = {args.limit} dial (first-N deterministic, "
         "sliding_window={args.sliding_window}, max_utt_len={args.max_utt_len}).",
         "",
         "| 벤치 | n dial | n turn | Pk_off | F1_off | Score_off | "
         "Pk_on | F1_on | Score_on | Δ Score |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in ("tiage", "dialseg711", "superseg"):
        r = results[ds]
        L.append(f"| {ds} | {r['n_dial']} | {r['n_turns']} | "
                 f"{r['offline']['pk']:.4f} | {r['offline']['f1']:.4f} | "
                 f"{r['offline']['score']:.4f} | "
                 f"{r['online']['pk']:.4f} | {r['online']['f1']:.4f} | "
                 f"{r['online']['score']:.4f} | "
                 f"{r['delta_score']:+.4f} |")
    L += ["",
          f"메모: limit={args.limit} 의 subset. 전수 GPU 평가는 별도 (현 머신 CUDA too-old)."]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE -> {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
