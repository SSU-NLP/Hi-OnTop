"""LLMLingua-2 compression for SeCom memory units (segments only).

Mirrors SeCom's ``experiment/compress.py`` but compresses ONLY the
``segments`` field (we don't need turn/session granularity for our paper
comparison) and writes to a separate result file per method.

LLMLingua-2 (microsoft/llmlingua-2-xlm-roberta-large-meetingbank) is a
local HF model (~2 GB). Runs on CPU; slow but tractable for 11 conv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/SeCom"))


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--compress_rate", type=float, default=0.75)
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    from secom import SeCom
    from omegaconf import OmegaConf
    cfg_full = OmegaConf.load(REPO_ROOT / "benchmarks/SeCom/secom/configs/mpnet.yaml")
    comp_only_cfg = REPO_ROOT / "benchmarks/SeCom/secom/configs/_comp_only_tmp.yaml"
    OmegaConf.save({"compressor": cfg_full.compressor}, comp_only_cfg)
    secom = SeCom(config_path=str(comp_only_cfg))

    data = load_jsonl(Path(args.load_path))
    print(f"n_conv: {len(data)}, compress_rate={args.compress_rate}", flush=True)

    results = []
    for idx, sample in enumerate(tqdm(data, desc="compress")):
        segments = sample["segments"]
        assert (
            isinstance(segments, list)
            and len(segments) > 0
            and isinstance(segments[0], list)
            and isinstance(segments[0][0], str)
        )
        sample["comp_segments"] = secom.compress(segments, compress_rate=args.compress_rate)
        results.append(sample)
        with open(args.save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_seg = len(segments)
        print(f"  conv {idx}: {n_seg} segments compressed", flush=True)


if __name__ == "__main__":
    main()
