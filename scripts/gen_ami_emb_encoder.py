#!/usr/bin/env python3
"""AMI turn 임베딩을 임의 SentenceTransformer 인코더로 생성 (인코더 강도 비교용).

gen_ami_emb.py(minilm-int8 전용)의 일반화 버전. reset 부트스트랩 벽이 인코더-의존인지
(약한 int8 인코더 아티팩트인지) 검증하기 위해 더 강한 인코더로 재인코딩한다.

사용: python scripts/gen_ami_emb_encoder.py <hf_model_id> <out_subdir>
  예: python scripts/gen_ami_emb_encoder.py sentence-transformers/all-mpnet-base-v2 ami_emb_mpnet
출력: outputs/runs/_misc/<out_subdir>/<mid>.pkl  (단위정규화 float64)
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # 인코딩은 CPU 병렬 허용 (latency job 아님)

TOPIC = Path("data/ami/topic")


def main() -> None:
    model_id = sys.argv[1]
    out_sub = sys.argv[2]
    cache = Path("outputs/runs/_misc") / out_sub
    cache.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_id, device="cpu")

    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(str(TOPIC / "*.json"))
                  if not p.endswith("manifest.json"))
    for i, mid in enumerate(mids, 1):
        cp = cache / f"{mid}.pkl"
        if cp.exists():
            print(f"[{i}/{len(mids)}] {mid} skip", flush=True)
            continue
        turns = [t["text"] for t in json.load(open(TOPIC / f"{mid}.json"))["turns"]]
        emb = np.asarray(enc.encode(turns, normalize_embeddings=True,
                                    show_progress_bar=False, batch_size=64), dtype=np.float64)
        with open(cp, "wb") as f:
            pickle.dump(emb, f)
        print(f"[{i}/{len(mids)}] {mid} {emb.shape} done", flush=True)
    print(f"DONE {len(mids)} meetings -> {cache} (model={model_id})")


if __name__ == "__main__":
    main()
