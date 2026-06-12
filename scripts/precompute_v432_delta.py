#!/usr/bin/env python3
"""v4.3.2 precompute: ST5 encode + trained NextEmbedHead 로 δ_model per turn.

For each (dataset, split) in {tiage, dialseg711, superseg} × {test}:
  1. load dialogs (SuperDialseg format)
  2. encode all utterances with **frozen Sentence-T5** (cache per dataset)
  3. load trained NextEmbedHead checkpoint
  4. for each turn t:
       if t == 0: δ_model = 0.0  (no context)
       else:
         ctx = s_st5[max(0, t-m)..t-1]  (left-padded with zeros if k<m)
         \hat{s}_t = head(ctx)
         δ_model[t] = 1 − cos(\hat{s}_t, s_st5[t])
  5. cache as pickle list[np.ndarray (n_turns,)]

Output: outputs/runs/_misc/sds_v432delta_{ds}_test_{tag}.pkl
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.next_embed_head import (  # noqa: E402
    NextEmbedHeadMLP, pack_causal_window, make_head,
)

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

ENC_ST5 = "sentence-transformers/sentence-t5-base"
DEFAULT_TAG = "st5_m5_mlp1024"


def load_dialogs(dataset, split):
    raw = json.loads((SDS / dataset / f"segmentation_file_{split}.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        if len(utts) >= 2:
            out.append(utts)
    return out


def encode_cached_st5(ds, split, dialogs, device):
    safe = ENC_ST5.replace("/", "_")
    cp = CACHE / f"sds_emb_{ds}_{split}_{safe}.pkl"
    if cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    print(f"  [encode-st5] {ds} {split}: n_dial={len(dialogs)} (first run)")
    from hi_ontop.embedding import QueryEncoder
    enc = QueryEncoder(device=device, model_name=ENC_ST5)
    out = [np.asarray(enc.encode(utts), dtype=np.float32) for utts in dialogs]
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump(out, fh)
    return out


def load_head(tag: str, device: str):
    ckpt_path = CACHE / f"next_embed_head_{tag}.pt"
    if not ckpt_path.exists():
        raise SystemExit(
            f"ckpt missing: {ckpt_path}\n"
            f"  run: uv run python scripts/train_next_embed_head.py --tag {tag}"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    head_type = cfg.get("head_type", "mlp")  # 옛 ckpt 호환 default
    model = make_head(
        head_type,
        emb_dim=cfg["emb_dim"],
        context_window=cfg["context_window"],
        hidden_dim=cfg.get("hidden_dim", 1024),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 1),
        dropout=cfg.get("dropout", 0.1),
    ).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"[head] loaded {ckpt_path.name}  "
          f"(type={head_type}, epoch {ckpt.get('best_epoch','?')}, "
          f"valid_loss {ckpt.get('best_valid_loss','?'):.4f})")
    return model, cfg


@torch.no_grad()
def predict_deltas(emb_per_dialog, model, m, device):
    """δ_model per turn per dialog. 첫 turn = 0.0."""
    d = emb_per_dialog[0].shape[1] if emb_per_dialog else 768
    out = []
    for emb in emb_per_dialog:
        n = emb.shape[0]
        deltas = np.zeros(n, dtype=np.float64)
        if n < 2:
            out.append(deltas)
            continue

        # Batch over turns of this dialog
        ctxs = []
        targets = []
        for t in range(1, n):
            start = max(0, t - m)
            win = emb[start:t]  # (k, d)
            k = win.shape[0]
            if k < m:
                pad = np.zeros((m - k, d), dtype=emb.dtype)
                win = np.concatenate([pad, win], axis=0)
            ctxs.append(win)
            targets.append(emb[t])
        ctx_t = torch.from_numpy(np.stack(ctxs, axis=0)).float().to(device)
        tgt_t = torch.from_numpy(np.stack(targets, axis=0)).float().to(device)
        pred = model(ctx_t)  # (n-1, d), L2-normalized
        cos = (pred * tgt_t).sum(dim=-1).cpu().numpy()  # (n-1,)
        deltas[1:] = 1.0 - cos
        out.append(deltas)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="NextEmbedHead checkpoint tag (= filename suffix)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v4.3.2 δ] device={device}, head_tag={args.tag}")

    model, cfg = load_head(args.tag, device)
    m = cfg["context_window"]

    CACHE.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        out_path = CACHE / f"sds_v432delta_{ds}_{args.split}_{args.tag}.pkl"
        if out_path.exists() and not args.force:
            print(f"  [skip] {ds}: {out_path.name} exists (use --force)")
            continue

        dialogs = load_dialogs(ds, args.split)
        embs = encode_cached_st5(ds, args.split, dialogs, device)
        print(f"  [compute-δ] {ds} {args.split}: n_dial={len(dialogs)}")
        t0 = time.perf_counter()
        deltas = predict_deltas(embs, model, m, device)
        dt = time.perf_counter() - t0
        with open(out_path, "wb") as fh:
            pickle.dump(deltas, fh)
        all_vals = np.concatenate([d[1:] for d in deltas if len(d) > 1])
        print(f"  [done] {ds}: {dt:.0f}s, mean_δ={all_vals.mean():.4f}, "
              f"std={all_vals.std():.4f}, min={all_vals.min():.4f}, "
              f"max={all_vals.max():.4f} → {out_path.name}")


if __name__ == "__main__":
    main()
