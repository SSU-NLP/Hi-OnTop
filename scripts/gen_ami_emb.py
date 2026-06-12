#!/usr/bin/env python3
"""AMI topic json → turn 임베딩 캐시 생성 (minilm-int8, outputs/runs/_misc/ami_emb/<mid>.pkl).

ami_topic_prep.py 산출(data/ami/topic/<mid>.json)의 turns 를 인코딩. 인코더는
run_encoder_comparison._encoder('minilm-int8')(torch.int4 shim 포함) 재사용.
재실행 시 이미 있는 pkl 은 skip. 임베딩만 만들면 이후 eval 들은 pkl 만 읽음(torch 불필요).
"""
from __future__ import annotations
import json, glob, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import _encoder

TOPIC = Path("data/ami/topic")
CACHE = Path("outputs/runs/_misc/ami_emb"); CACHE.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(str(TOPIC / "*.json"))
                  if not p.endswith("manifest.json"))
    enc = _encoder("minilm-int8")
    for i, mid in enumerate(mids, 1):
        cp = CACHE / f"{mid}.pkl"
        if cp.exists():
            print(f"[{i}/{len(mids)}] {mid} skip", flush=True); continue
        turns = [t["text"] for t in json.load(open(TOPIC / f"{mid}.json"))["turns"]]
        emb = np.asarray(enc.encode(turns, normalize_embeddings=True,
                                    show_progress_bar=False), dtype=np.float64)
        with open(cp, "wb") as f:
            pickle.dump(emb, f)
        print(f"[{i}/{len(mids)}] {mid} {emb.shape} done", flush=True)
    print(f"DONE {len(mids)} meetings -> {CACHE}")
