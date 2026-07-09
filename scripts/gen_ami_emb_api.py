#!/usr/bin/env python3
"""AMI turn 임베딩을 OpenAI-호환 API(Crts 프록시) 인코더로 생성.

gen_ami_emb_encoder.py(로컬 SentenceTransformer)의 API 버전. 로컬 GPU(Blackwell sm_120)가
설치된 torch 와 비호환이라, 현대 강한 임베딩(text-embedding-3-large 등)은 API 로 받는다.

사용: python scripts/gen_ami_emb_api.py <api_model> <out_subdir>
  예: python scripts/gen_ami_emb_api.py openrouter/openai/text-embedding-3-large ami_emb_te3large
출력: outputs/runs/_misc/<out_subdir>/<mid>.pkl  (단위정규화 float64, per-meeting 캐시)
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

TOPIC = Path("data/ami/topic")
BATCH = 256


def _load_env() -> None:
    for line in open(".env"):
        m = re.match(r"([A-Z_]+)=(.*)", line.strip())
        if m:
            os.environ.setdefault(m.group(1), m.group(2))


def _client():
    from openai import OpenAI
    hdr = {}
    if os.environ.get("CF_ACCESS_CLIENT_ID"):
        hdr["CF-Access-Client-Id"] = os.environ["CF_ACCESS_CLIENT_ID"]
        hdr["CF-Access-Client-Secret"] = os.environ["CF_ACCESS_CLIENT_SECRET"]
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                  base_url=os.environ["OPENAI_BASE_URL"],
                  default_headers=hdr or None)


def _embed(cli, model: str, texts: list[str]) -> np.ndarray:
    """배치 임베딩 (빈 문자열은 공백으로 치환, 3회 재시도)."""
    texts = [t if t.strip() else " " for t in texts]
    for attempt in range(3):
        try:
            r = cli.embeddings.create(model=model, input=texts)
            return np.asarray([d.embedding for d in r.data], dtype=np.float64)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"    retry {attempt+1} ({str(e)[:60]})", flush=True); time.sleep(2 + 2 * attempt)


def main() -> None:
    model = sys.argv[1]
    out_sub = sys.argv[2]
    _load_env()
    cli = _client()
    cache = Path("outputs/runs/_misc") / out_sub
    cache.mkdir(parents=True, exist_ok=True)

    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(str(TOPIC / "*.json"))
                  if not p.endswith("manifest.json"))
    for i, mid in enumerate(mids, 1):
        cp = cache / f"{mid}.pkl"
        if cp.exists():
            print(f"[{i}/{len(mids)}] {mid} skip", flush=True)
            continue
        turns = [t["text"] for t in json.load(open(TOPIC / f"{mid}.json"))["turns"]]
        vecs = []
        for s in range(0, len(turns), BATCH):
            vecs.append(_embed(cli, model, turns[s:s + BATCH]))
        emb = np.concatenate(vecs, axis=0)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)  # 단위정규화
        with open(cp, "wb") as f:
            pickle.dump(emb, f)
        print(f"[{i}/{len(mids)}] {mid} {emb.shape} done", flush=True)
    print(f"DONE {len(mids)} meetings -> {cache} (model={model})")


if __name__ == "__main__":
    main()
