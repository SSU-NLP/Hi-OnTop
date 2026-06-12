#!/usr/bin/env python3
"""진단: v4.1.x δ*=0.5557 이 bge 값인가 mpnet 값인가.

TIAGE train (anno) 에서 pure δ_prev best-F1 threshold (= δ*, a=1.0
prev-cos calibration) 를 두 인코더로 각각 산출해 0.5557 과 대조한다.
decision-log 기록: 0.5557 @ cos_th 0.4443, F1 0.437, n_conv=300.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "benchmarks" / "tiage" / "data" / "personachat" / "anno"

ENCODERS = {
    "mpnet (multi-qa-mpnet-base-dot-v1)": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "bge   (bge-base-en-v1.5)": "BAAI/bge-base-en-v1.5",
}


def load_train():
    raw = json.loads((DATA / "train" / "anno_train.json").read_text())
    dialogs = [[(t[0], t[1]) for t in d] for d in raw.values()]
    return dialogs


def best_delta_star(dialogs, model_name):
    model = SentenceTransformer(model_name)
    sig, gt = [], []
    for d in dialogs:
        texts = [t[0] for t in d]
        labs = [t[1] for t in d]
        emb = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False)
        for t in range(1, len(emb)):
            cp = float(np.dot(emb[t - 1], emb[t]))  # already L2-normalized
            sig.append(1.0 - cp)                    # δ_prev
            gt.append(labs[t] == "1")               # GT shift at turn t
    sig, gt = np.array(sig), np.array(gt)
    best_f1, best_th = 0.0, 0.0
    for th in np.linspace(sig.min(), sig.max(), 400):
        f = f1_score(gt, sig >= th)  # boundary when δ_prev ≥ δ*
        if f > best_f1:
            best_f1, best_th = f, float(th)
    return best_th, best_f1, len(sig), int(gt.sum())


def main():
    dialogs = load_train()
    print(f"TIAGE train: n_conv={len(dialogs)}\n")
    print(f"{'encoder':40s}  δ* (best-F1)   train F1   n_trans  GT_shifts")
    for label, name in ENCODERS.items():
        ds, f1, n, g = best_delta_star(dialogs, name)
        flag = "  <-- 0.5557 과 일치" if abs(ds - 0.5557) < 0.01 else ""
        print(f"{label:40s}  {ds:.4f}         {f1:.3f}      {n:6d}   {g:5d}{flag}")
    print("\n기록: δ*=0.5557, train F1=0.437, n_conv=300 (decision-log 2026-05-18)")


if __name__ == "__main__":
    main()
