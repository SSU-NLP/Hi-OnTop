"""RoBERTa-supervised segmentation runner — MTB+ adapter.

Wraps ``methods/RoBERTa/online/segment.py`` 's strict-causal online inference
into the secom_swap segment.jsonl schema (per-conv ``sample["segments"]:
List[List[str]]``) consumed by 05_compress → 06_retrieve → 07_chat → 08_eval.

Same eval_online logic as upstream (window W, causal): for each session, for
each turn ``t >= 1`` build window ``u_{max(0,t-W+1)..t}``, encode, classify
the first ``</s>`` of utt ``t-1`` to decide boundary ``(t-1, t)``. Per-session
fresh state (matches baseline streaming adapter semantics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaForTokenClassification

REPO_ROOT = Path(__file__).resolve().parents[2]

# Reuse offline training helpers (encode_window) and the canonical Run-1 ckpt.
sys.path.insert(0, str(REPO_ROOT / "methods" / "RoBERTa" / "offline"))
from train import encode_window  # noqa: E402

DEFAULT_MODEL_DIR = (
    REPO_ROOT / "methods" / "RoBERTa" / "_roberta_unzip"
    / "roberta_seg_out" / "roberta_supervised" / "model"
)

sys.path.insert(0, str(REPO_ROOT / "scripts" / "secom_swap"))
from _streaming_adapter import session_segments_from_boundaries  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def predict_session(model, tok, utts: list[str], cfg: dict, device,
                    timing: dict | None = None) -> list[int]:
    """Strict-causal online preds yp for one session — yp[t]=1 ⇒ boundary (t-1,t).

    If ``timing`` is given, accumulates ``preprocess_sec`` (encode + RoBERTa
    forward) and ``seg_sec`` (logit argmax / threshold).
    """
    n = len(utts)
    yp = [0] * n
    if n < 2:
        return yp
    W, pad = cfg["sliding_window"], tok.pad_token_id
    win_ids, meta = [], []
    _pp = time.perf_counter()
    for t in range(1, n):                      # boundary (t-1, t) decided at turn t
        ws = max(0, t - W + 1)                 # causal window start
        wu = utts[ws:t + 1]                    # u_{ws..t} (no future)
        ids, _, cls_pos = encode_window(
            tok, wu, [0] * len(wu), "test",
            cfg["max_utt_len"], cfg["max_seq_len"])
        local = (t - 1) - ws                   # index of utt t-1 inside window
        win_ids.append(torch.tensor(ids))
        meta.append((t, cls_pos[local]))
    batch = pad_sequence(win_ids, batch_first=True, padding_value=pad).to(device)
    logits = model(input_ids=batch,
                   attention_mask=batch.ne(pad).long()).logits.cpu().numpy()
    if timing is not None:
        timing["preprocess_sec"] = timing.get("preprocess_sec", 0.0) + (
            time.perf_counter() - _pp)
    _ds = time.perf_counter()
    for w, (t, j) in enumerate(meta):
        yp[t] = int(logits[w, j].argmax())     # decision = argmax (binary threshold)
    yp[0] = 0                                  # by construction (no boundary before first)
    if timing is not None:
        timing["seg_sec"] = timing.get("seg_sec", 0.0) + (
            time.perf_counter() - _ds)
    return yp


def yp_to_boundaries_1based(yp: list[int]) -> list[int]:
    """yp[t]=1 ⇒ utt t-1 (0-based) is LAST of its segment ⇒ 1-based pos = t."""
    return [t for t in range(1, len(yp)) if yp[t] == 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--load_path",
        default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"),
    )
    ap.add_argument(
        "--save_path",
        default=str(
            REPO_ROOT
            / "benchmarks/SeCom/experiment/result/mtbp/roberta/segments.jsonl"
        ),
    )
    ap.add_argument(
        "--latency_path",
        default=str(
            REPO_ROOT
            / "outputs/experiments/2026-05-21_v413_secom_swap/latency_roberta.json"
        ),
    )
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--sliding_window", type=int, default=20)
    ap.add_argument("--max_utt_len", type=int, default=25)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        sys.exit(f"[error] ckpt not found: {model_dir}")

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latency_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[device] {device} | ckpt {model_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = RobertaForTokenClassification.from_pretrained(str(model_dir)).to(device).eval()
    cfg = dict(sliding_window=args.sliding_window,
               max_utt_len=args.max_utt_len, max_seq_len=args.max_seq_len)

    data = load_jsonl(Path(args.load_path))
    print(f"n_conv: {len(data)}", flush=True)

    results = []
    per_conv_lat = []
    tot_pp = 0.0
    tot_ds = 0.0
    for idx, sample in enumerate(tqdm(data, desc="roberta-seg")):
        sessions = sample["sessions"]
        n_ex = sum(len(s) for s in sessions)
        t0 = time.perf_counter()
        seg_all: list[list[str]] = []
        timing: dict = {}
        for sess in sessions:
            if not sess:
                continue
            yp = predict_session(model, tok, sess, cfg, device, timing=timing)
            bnd = yp_to_boundaries_1based(yp)
            seg_all.extend(session_segments_from_boundaries(sess, bnd))
        t = time.perf_counter() - t0
        sample["segments"] = seg_all
        results.append(sample)
        pp = timing.get("preprocess_sec", 0.0)
        ds = timing.get("seg_sec", 0.0)
        tot_pp += pp
        tot_ds += ds
        lat = {
            "conversation_id": sample["conversation_id"],
            "n_sessions": len(sessions),
            "n_exchanges": n_ex,
            "n_segments": len(seg_all),
            "total_sec": t,
            "preprocess_sec": pp,
            "seg_sec": ds,
            "sec_per_exchange": t / max(1, n_ex),
        }
        per_conv_lat.append(lat)
        print(f"  conv {idx} ({sample['conversation_id']}): "
              f"{n_ex} ex → {len(seg_all)} segs, {t:.2f}s, "
              f"{lat['sec_per_exchange']*1000:.1f}ms/ex", flush=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ex_total = sum(l["n_exchanges"] for l in per_conv_lat)
    n_seg_total = sum(l["n_segments"] for l in per_conv_lat)
    total_sec = sum(l["total_sec"] for l in per_conv_lat)
    summary = {
        "method": "roberta_supervised_online",
        "model_dir": str(model_dir),
        "n_conv": len(per_conv_lat),
        "n_exchanges": n_ex_total,
        "n_segments": n_seg_total,
        "avg_exchanges_per_segment": n_ex_total / max(1, n_seg_total),
        "total_sec": total_sec,
        "preprocess_sec": tot_pp,
        "seg_sec": tot_ds,
        "ms_per_exchange": total_sec * 1000 / max(1, n_ex_total),
        "ms_per_exchange_preprocess": tot_pp * 1000 / max(1, n_ex_total),
        "ms_per_exchange_seg": tot_ds * 1000 / max(1, n_ex_total),
        "device": str(device),
        "cfg": cfg,
        "per_conv": per_conv_lat,
    }
    with open(args.latency_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nroberta latency -> {args.latency_path}", flush=True)
    print(f"roberta: {n_seg_total} segments / {n_ex_total} exchanges, "
          f"{summary['ms_per_exchange']:.1f} ms/exchange", flush=True)


if __name__ == "__main__":
    main()
