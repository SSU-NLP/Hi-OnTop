#!/usr/bin/env python3
"""v4.2.5 precompute: CSM (CoherenceNet) incoherence prob per turn.

For each (dataset, split) in {tiage, dialseg711, superseg} × {test}:
  - load dialogs (SuperDialseg format)
  - for each turn t:
      if t == 0: δ_model = 0.0  (no prev utterance)
      else:
        enc = tokenizer(u_{t-1}, u_t, max_length=128, truncation=True, padding=True)
        h_cls = bert(enc).last_hidden_state[:, 0, :]
        logits = coherence_decoder(h_cls)
        p_coh = softmax(logits)[0]
        δ_model[t] = 1 − p_coh
  - cache as pickle list[np.ndarray (n_turns,)]

Output: outputs/runs/_misc/sds_v425delta_{ds}_test_{tag}.pkl
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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
# CSM external repo
EXTERNAL = REPO / "external" / "Dialogue-Topic-Segmenter"
sys.path.insert(0, str(EXTERNAL))

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

DEFAULT_ENCODER = "bert-base-uncased"
DEFAULT_TAG = "csm_bert_base"
DEFAULT_MAX_LEN = 128


def load_dialogs(dataset, split):
    raw = json.loads((SDS / dataset / f"segmentation_file_{split}.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        if len(utts) >= 2:
            out.append(utts)
    return out


def load_csm(ckpt_path: Path, encoder_name: str, device: str):
    """CoherenceNet (external repo) load + ckpt state_dict 적용."""
    from model_utils import CoherenceNet  # external/Dialogue-Topic-Segmenter

    bert = AutoModel.from_pretrained(encoder_name)
    dev = torch.device(device)
    model = CoherenceNet(bert, dev).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[csm] loaded {ckpt_path.name}  encoder={encoder_name}  params={n_params:,}")
    return model


@torch.no_grad()
def coherence_probs(
    pairs: list[tuple[str, str]],
    model,
    tokenizer,
    device: str,
    max_length: int,
    batch_size: int = 32,
) -> np.ndarray:
    """List of (u_prev, u_curr) → np.ndarray of p_coherent (n,)."""
    probs = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        prevs = [p[0] for p in chunk]
        currs = [p[1] for p in chunk]
        enc = tokenizer(
            prevs, currs,
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        out = model.bert(**enc)
        h_cls = out.last_hidden_state[:, 0, :]  # (B, 768)
        logits = model.coherence_decoder(h_cls)  # (B, 2)
        sm = F.softmax(logits, dim=-1)
        probs.append(sm[:, 0].cpu().numpy())  # col 0 = coherent
    return np.concatenate(probs, axis=0) if probs else np.array([])


def compute_deltas(
    dialogs: list[list[str]],
    model, tokenizer, device, max_length, batch_size=32,
) -> list[np.ndarray]:
    """δ_model per turn per dialog. 첫 turn = 0.0."""
    out = []
    # Flatten all pairs across dialogs for batched inference
    all_pairs, sizes = [], []
    for utts in dialogs:
        n = len(utts)
        if n < 2:
            sizes.append(0)
            continue
        for t in range(1, n):
            all_pairs.append((utts[t - 1], utts[t]))
        sizes.append(n - 1)
    if not all_pairs:
        return [np.zeros(len(d), dtype=np.float64) for d in dialogs]

    probs = coherence_probs(all_pairs, model, tokenizer, device, max_length, batch_size)
    idx = 0
    for utts, k in zip(dialogs, sizes):
        n = len(utts)
        deltas = np.zeros(n, dtype=np.float64)
        if k > 0:
            deltas[1:] = 1.0 - probs[idx: idx + k]
            idx += k
        out.append(deltas)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CACHE / "cpt_277000.pth"),
                    help="CSM ckpt path (state_dict)")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help="BERT backbone (must match train, default bert-base-uncased)")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="output filename suffix")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v4.2.5 δ] device={device}, ckpt={args.ckpt}, tag={args.tag}")

    model = load_csm(Path(args.ckpt), args.encoder, device)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    CACHE.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        out_path = CACHE / f"sds_v425delta_{ds}_{args.split}_{args.tag}.pkl"
        if out_path.exists() and not args.force:
            print(f"  [skip] {ds}: {out_path.name} exists (--force to redo)")
            continue
        dialogs = load_dialogs(ds, args.split)
        print(f"  [compute-δ] {ds} {args.split}: n_dial={len(dialogs)}, "
              f"n_pair={sum(max(0, len(d)-1) for d in dialogs)}")
        t0 = time.perf_counter()
        deltas = compute_deltas(dialogs, model, tokenizer, device,
                                args.max_length, args.batch_size)
        dt = time.perf_counter() - t0
        with open(out_path, "wb") as fh:
            pickle.dump(deltas, fh)
        all_vals = np.concatenate([d[1:] for d in deltas if len(d) > 1])
        print(f"  [done] {ds}: {dt:.0f}s, mean_δ={all_vals.mean():.4f}, "
              f"std={all_vals.std():.4f}, min={all_vals.min():.4f}, "
              f"max={all_vals.max():.4f} → {out_path.name}")


if __name__ == "__main__":
    main()
