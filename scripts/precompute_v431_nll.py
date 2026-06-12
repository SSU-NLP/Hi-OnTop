#!/usr/bin/env python3
"""v4.3.1 precompute: DialoGPT-small mean-token NLL per turn.

For each (dataset, split) in {tiage, dialseg711, superseg} × {test}:
  - load dialogs (SuperDialseg format)
  - for each turn t:
      context = u_{max(0, t-m)..t-1}                     # causal window m=5
      target  = u_t
      nll_t   = mean_{w ∈ tokens(u_t)} −log P_LM(w | context, u_t<w)
      (첫 turn t=0 → context 없음 → nll_0 = 0.0)
  - cache as pickle: list[np.ndarray], one per dialog, shape=(n_turns,)

Format choice — **DialoGPT 의 학습 분포에 맞춘 EOS-concat**:
  full_text = "<|endoftext|>".join(u_{t-m..t-1}) + "<|endoftext|>" + u_t + "<|endoftext|>"
  loss positions = u_t 의 token 위치 (context 부분은 -100 mask)
이는 DialoGPT (Zhang+ 2020) 가 실제로 학습된 포맷. transformers 의
``apply_chat_template`` 은 ChatML 등 학습 분포 밖 포맷을 삽입할 위험이
있어 surprisal 의 의미를 흐림 → 본 script 는 raw EOS-concat 사용.
(decision-log 2026-05-21 entry 참조.)

Output: outputs/runs/_misc/sds_nll_{dataset}_{split}_microsoft_DialoGPT-small.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

LM_NAME = "microsoft/DialoGPT-small"
DEFAULT_CONTEXT_WINDOW = 5
DEFAULT_MAX_LEN = 768  # left-truncate; DialoGPT-small max = 1024


def load_dialogs(dataset: str, split: str):
    raw = json.loads((SDS / dataset / f"segmentation_file_{split}.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        if len(utts) >= 2:
            out.append(utts)
    return out


def build_lm(device: str):
    tok = AutoTokenizer.from_pretrained(LM_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LM_NAME).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def turn_nlls(
    utts: list[str],
    tok,
    model,
    device: str,
    ctx_window: int,
    max_len: int,
) -> np.ndarray:
    """Mean-token NLL per turn (length n). nll[0] = 0.0 (no context)."""
    eos = tok.eos_token
    eos_ids = tok.encode(eos, add_special_tokens=False)  # DialoGPT: [50256]
    nlls = np.zeros(len(utts), dtype=np.float64)

    for t in range(1, len(utts)):
        ctx_utts = utts[max(0, t - ctx_window): t]
        target_utt = utts[t]

        # Tokenize separately so we know target boundary
        ctx_text = "".join(u + eos for u in ctx_utts)  # each turn + EOS
        ctx_ids = tok.encode(ctx_text, add_special_tokens=False)
        tgt_ids = tok.encode(target_utt + eos, add_special_tokens=False)

        if not tgt_ids:
            continue

        # Left-truncate ctx if total exceeds max_len (keep all target tokens)
        budget = max_len - len(tgt_ids)
        if budget < 1:
            # Target alone too long → truncate target tail
            tgt_ids = tgt_ids[: max_len - 1]
            ctx_ids = ctx_ids[-1:] if ctx_ids else []
        else:
            ctx_ids = ctx_ids[-budget:]

        input_ids = ctx_ids + tgt_ids
        if len(input_ids) < 2:
            continue

        labels = [-100] * len(ctx_ids) + list(tgt_ids)
        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        labels_t = torch.tensor([labels], dtype=torch.long, device=device)

        out = model(input_ids=input_t, labels=labels_t)
        # HF CausalLM loss = mean -log P over non-(-100) positions
        nlls[t] = float(out.loss.item())

    return nlls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--ctx-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / mps (default: auto)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing cache")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v4.3.1 NLL] device={device}, lm={LM_NAME}, "
          f"ctx_window={args.ctx_window}, max_len={args.max_len}")

    tok, model = build_lm(device)
    CACHE.mkdir(parents=True, exist_ok=True)
    safe_lm = LM_NAME.replace("/", "_")

    for ds in args.datasets:
        cp = CACHE / f"sds_nll_{ds}_{args.split}_{safe_lm}.pkl"
        if cp.exists() and not args.force:
            print(f"  [skip] {ds}: {cp.name} exists (use --force to redo)")
            continue

        dialogs = load_dialogs(ds, args.split)
        print(f"  [compute] {ds} {args.split}: n_dial={len(dialogs)}")
        t0 = time.perf_counter()
        all_nlls = []
        for i, utts in enumerate(dialogs):
            nlls = turn_nlls(utts, tok, model, device,
                             args.ctx_window, args.max_len)
            all_nlls.append(nlls)
            if (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    {i+1}/{len(dialogs)} ({elapsed:.0f}s, "
                      f"~{elapsed/(i+1):.2f}s/dial)")
        dt = time.perf_counter() - t0
        with open(cp, "wb") as fh:
            pickle.dump(all_nlls, fh)
        all_vals = np.concatenate(all_nlls)
        print(f"  [done] {ds}: {dt:.0f}s, mean_nll={all_vals.mean():.3f}, "
              f"std={all_vals.std():.3f}, min={all_vals.min():.3f}, "
              f"max={all_vals.max():.3f} → {cp.name}")


if __name__ == "__main__":
    main()
