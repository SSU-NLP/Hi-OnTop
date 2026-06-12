"""Retrieve top-k segments per question (mpnet + FAISS).

Mirrors SeCom's ``retrieve.py`` with two simplifications:
- segments-granularity only (not turn/session)
- no resume, single overwrite write (small dataset)

Writes ``sample["retrieved_texts"]: List[str]`` aligned with
``sample["questions"]: List[str]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/SeCom"))

# Force HF embeddings to CPU (no GPU available in this env).
os.environ.setdefault("HF_HUB_OFFLINE", "0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--config_path",
        default=str(REPO_ROOT / "benchmarks/SeCom/secom/configs/mpnet.yaml"),
    )
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    from secom import SeCom
    from omegaconf import OmegaConf
    cfg_full = OmegaConf.load(args.config_path)
    ret_only_cfg = REPO_ROOT / "benchmarks/SeCom/secom/configs/_ret_only_tmp.yaml"
    OmegaConf.save({"retriever": cfg_full.retriever}, ret_only_cfg)
    secom = SeCom(config_path=str(ret_only_cfg))

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"n_conv: {len(data)}", flush=True)

    # Patch retriever device (mpnet config defaults to cuda; force cpu)
    orig_init = secom.config.retriever.get("device_map", "cuda")
    secom.config.retriever["device_map"] = args.device

    results = []
    n_ex_list, n_token_list = [], []
    for idx, sample in enumerate(tqdm(data, desc="retrieve")):
        requests = sample["questions"]
        units = sample["segments"]
        comp_units = sample.get("comp_segments")
        if comp_units is not None:
            ret_texts, n_ex, n_tok = secom.retrieve_external_memory(
                requests, units, comp_units, retrieve_topk=args.topk
            )
        else:
            ret_texts, n_ex, n_tok = secom.retrieve_external_memory(
                requests, units, retrieve_topk=args.topk
            )
        sample["retrieved_texts"] = ret_texts
        results.append(sample)
        n_ex_list.append(n_ex)
        n_token_list.append(n_tok)
        print(
            f"  conv {idx}: {len(units)} segs, {len(requests)} q, "
            f"avg_n_ex={n_ex:.1f}, avg_n_tok={n_tok:.1f}",
            flush=True,
        )

    with open(args.save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {
        "n_ex_avg": sum(n_ex_list) / max(1, len(n_ex_list)),
        "n_token_avg": sum(n_token_list) / max(1, len(n_token_list)),
        "topk": args.topk,
        "n_conv": len(data),
    }
    metrics_path = Path(args.save_path).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nretrieval metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
