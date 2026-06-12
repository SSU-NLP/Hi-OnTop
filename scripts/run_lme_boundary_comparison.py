#!/usr/bin/env python3
"""Boundary comparison on LongMemEval dataset.

Same 6 methods as SHARE (run_share_boundary_comparison.py):
  LLM segmenters : GPT-5, Qwen3.5-27B, Qwen3.5-122B
  Hi-OnTop       : MiniLM-int8 at p60, p70, p80

Gold = session boundaries from haystack_sessions structure.
Each instance has ~48 sessions of user-assistant pairs.

Outputs
-------
  outputs/experiments/2026-06-02_lme_boundary_comparison/
      segments_<method>.jsonl
      results.json
  outputs/figures/figure_T_lme_boundary_agreement.{pdf,png}

Usage
-----
  # Quick test (10 instances, first 8 sessions each, skip LLMs)
  uv run python scripts/run_lme_boundary_comparison.py --limit 10 --max-sessions 8 --skip-llm

  # Full run (all 500 instances, first 10 sessions each)
  uv run python scripts/run_lme_boundary_comparison.py --max-sessions 10
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(4)
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/SeCom"))
from hi_ontop.hi_ontop import HiOnTop

LME_PATH = REPO / "benchmarks/longmemeval/data/longmemeval_s.json"
OUT_EXP   = REPO / "outputs/experiments/2026-06-02_lme_boundary_comparison"
OUT_FIG   = REPO / "outputs/figures"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

M, RHO, A = 2, 0.7, 0.5
ENCODER_CFG = {
    "model":     "sentence-transformers/all-MiniLM-L6-v2",
    "backend":   "onnx",
    "file_name": "onnx/model_quint8_avx2.onnx",
}
HIONTOP_P_VALUES = [60, 70, 80]

METHOD_REGISTRY = [
    ("gpt5",     "GPT-5",                      "#8E44AD"),
    ("int8_p60", r"Hi-OnTop (int8, $p_{60}$)", "#2980B9"),
    ("int8_p70", r"Hi-OnTop (int8, $p_{70}$)", "#1A5276"),
    ("int8_p80", r"Hi-OnTop (int8, $p_{80}$)", "#0A2744"),
]
ALL_KEYS  = [k for k, _, _ in METHOD_REGISTRY]
KEY_LABEL = {k: l for k, l, _ in METHOD_REGISTRY}
KEY_COLOR = {k: c for k, _, c in METHOD_REGISTRY}


# ── data loading ──────────────────────────────────────────────────────────────

def load_lme(limit: int | None, max_sessions: int) -> list[dict]:
    print(f"  loading LongMemEval _s (limit={limit}, max_sessions={max_sessions})", flush=True)
    with open(LME_PATH) as f:
        raw = json.load(f)

    conversations = []
    for inst in raw:
        sessions_raw = inst["haystack_sessions"][:max_sessions]
        sessions: list[list[str]] = []
        for sess in sessions_raw:
            turns = [f"[{t['role'].upper()}] {t['content']}" for t in sess
                     if t.get("content", "").strip()]
            if turns:
                sessions.append(turns)
        if len(sessions) < 2:
            continue

        gold_bnd: list[int] = []
        cursor = -1
        for i, sess in enumerate(sessions):
            cursor += len(sess)
            if i < len(sessions) - 1:
                gold_bnd.append(cursor)

        conversations.append({
            "conversation_id": inst["question_id"],
            "sessions":        sessions,
            "gold_boundaries": gold_bnd,
        })
        if limit and len(conversations) >= limit:
            break

    print(f"  loaded {len(conversations)} instances", flush=True)
    return conversations


# ── Hi-OnTop ──────────────────────────────────────────────────────────────────

def make_encoder() -> SentenceTransformer:
    return SentenceTransformer(
        ENCODER_CFG["model"], backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": ENCODER_CFG["file_name"]})


def encode_conversations(enc, conversations: list[dict], cache: Path) -> dict:
    if cache.exists():
        print(f"  using cached embeddings {cache.name}", flush=True)
        return pickle.loads(cache.read_bytes())
    emb_map = {}
    for conv in conversations:
        turns = [t for sess in conv["sessions"] for t in sess]
        vecs  = enc.encode(turns, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        emb_map[conv["conversation_id"]] = vecs.astype(np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(emb_map))
    print(f"  cached → {cache.name}", flush=True)
    return emb_map


def run_hiontop_all_p(conversations, emb_map) -> dict[str, dict]:
    pool_vals, deff_map = [], {}
    for conv in conversations:
        cid  = conv["conversation_id"]
        vecs = emb_map[cid]
        seg  = HiOnTop(dim=vecs.shape[1], delta_star=1.0,
                      ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
        for v in vecs:
            seg.assign(v)
        deffs = np.array([float(h["delta_eff"]) for h in seg.history()[1:]])
        pool_vals.extend(deffs.tolist())
        deff_map[cid] = deffs

    pool = np.array(pool_vals)
    print(f"  δ_eff pool: n={pool.size}  mean={pool.mean():.4f}  std={pool.std():.4f}", flush=True)

    results = {}
    for p in HIONTOP_P_VALUES:
        key   = f"int8_p{p}"
        dstar = float(np.percentile(pool, p))
        print(f"  Hi-OnTop int8 p{p}: δ*={dstar:.4f}", flush=True)
        per_conv = {conv["conversation_id"]: [i for i, d in enumerate(deff_map[conv["conversation_id"]]) if d >= dstar]
                    for conv in conversations}
        results[key] = per_conv
    return results


def save_hiontop_segments(conversations, emb_map, hiontop_bnd, key):
    out = OUT_EXP / f"segments_{key}.jsonl"
    lines = []
    for conv in conversations:
        cid       = conv["conversation_id"]
        all_turns = [t for sess in conv["sessions"] for t in sess]
        bnd_set   = set(hiontop_bnd[key][cid])
        segs, seg = [], []
        for i, turn in enumerate(all_turns):
            seg.append(turn)
            if i in bnd_set:
                segs.append(seg); seg = []
        if seg:
            segs.append(seg)
        lines.append(json.dumps({"conversation_id": cid,
                                 "sessions": conv["sessions"],
                                 "segments": segs}, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n")
    print(f"  saved {out.name}", flush=True)


# ── LLM segmenters ────────────────────────────────────────────────────────────

def _make_client(model_slug: str):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL",
                                            os.environ.get("OPENAI_API_BASE")))
    is_gpt5 = "gpt-5" in model_slug.lower()
    _orig   = client.chat.completions.create

    def _patched(*a, **kw):
        if is_gpt5:
            if "max_tokens" in kw:
                kw["max_completion_tokens"] = max(kw.pop("max_tokens"), 2000)
            kw.pop("temperature", None); kw.pop("top_p", None); kw.pop("seed", None)
            kw.setdefault("reasoning_effort", "minimal")
        else:
            kw.setdefault("reasoning_effort", "none")
        return _orig(*a, **kw)

    client.chat.completions.create = _patched
    return client, model_slug


def _build_prompt(turns: list[str]) -> str:
    exchanges = "".join(f"[Exchange {i}]: {t}\n\n" for i, t in enumerate(turns))
    return f"""# Instruction
Segment the following multi-turn conversation into topically coherent units.
Output jsonl inside <segmentation></segmentation> tags. Each line:
{{"segment_id": int, "start_exchange_number": int, "end_exchange_number": int, "num_exchanges": int}}
Cover ALL exchanges (start=0, end={len(turns)-1}, no gaps/overlaps).

# Data
{exchanges.strip()}

# Output
<segmentation>
"""


def _parse_segments(response: str, turns: list[str]) -> list[list[str]]:
    import re, traceback
    m = re.search(r"<segmentation>([\s\S]*?)(?:</segmentation>|$)", response)
    if not m:
        return [turns[i:i+3] for i in range(0, len(turns), 3)]
    segs, cursor, ok = [], 0, True
    for line in m.group(1).strip().splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            d = json.loads(line)
            n = int(d["num_exchanges"])
            segs.append(turns[cursor:cursor+n]); cursor += n
        except Exception:
            ok = False; break
    if not ok or cursor != len(turns):
        return [turns[i:i+3] for i in range(0, len(turns), 3)]
    return segs


def _segs_to_bnd(segs: list[list[str]]) -> list[int]:
    bnd, cursor = [], -1
    for i, s in enumerate(segs):
        cursor += len(s)
        if i < len(segs) - 1 and s:
            bnd.append(cursor)
    return bnd


def run_llm_segmenter(conversations, key, model_slug) -> dict[str, list[int]]:
    out_path = OUT_EXP / f"segments_{key}.jsonl"
    done = {}
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if not ln.strip(): continue
            r = json.loads(ln)
            done[r["conversation_id"]] = _segs_to_bnd(r["segments"])
        print(f"  [{key}] resuming: {len(done)} done", flush=True)

    client, slug = _make_client(model_slug)
    per_conv = dict(done)

    with out_path.open("a", encoding="utf-8") as fh:
        for conv in conversations:
            cid = conv["conversation_id"]
            if cid in done:
                continue
            turns  = [t for sess in conv["sessions"] for t in sess]
            prompt = _build_prompt(turns)
            print(f"  [{key}] {cid}: {len(turns)} turns … ", flush=True, end="")
            from time import sleep
            resp = ""
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=slug, messages=[{"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=2048)
                    resp = r.choices[0].message.content
                    break
                except Exception as e:
                    print(f"err({attempt+1}:{e}) ", flush=True, end="")
                    sleep(5 * (attempt+1))
            segs = _parse_segments(resp, turns)
            bnd  = _segs_to_bnd(segs)
            per_conv[cid] = bnd
            fh.write(json.dumps({"conversation_id": cid, "sessions": conv["sessions"],
                                  "segments": segs}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"→ {len(segs)} segs", flush=True)

    return per_conv


# ── metrics + figure ──────────────────────────────────────────────────────────

def pairwise_f1(pred, gold):
    if not pred and not gold: return 1.0
    if not pred or not gold:  return 0.0
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs); p = tp/max(1,len(ps)); r = tp/max(1,len(gs))
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0


def make_figure(boundaries, cids):
    n   = len(ALL_KEYS)
    mat = np.zeros((n, n))
    for i, a in enumerate(ALL_KEYS):
        for j, b in enumerate(ALL_KEYS):
            mat[i, j] = float(np.mean(
                [pairwise_f1(boundaries[a][c], boundaries[b][c]) for c in cids]))

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="equal")
    labels = [KEY_LABEL[k] for k in ALL_KEYS]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if v > 0.5 else "#222222")
    ax.set_title(f"Pairwise boundary F1 on LongMemEval-S ({len(cids)} instances, {n} methods)",
                 fontsize=11, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025).set_label("Pairwise boundary F1", fontsize=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT_FIG / f"figure_T_lme_boundary_agreement.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"saved {p}", flush=True)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-sessions", type=int, default=10,
                    help="Use only first N sessions per instance (default 10)")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--gpt5-model",    default="openrouter/openai/gpt-5")
    ap.add_argument("--qwen27b-model", default="openrouter/qwen/qwen3.5-27b")
    ap.add_argument("--qwen122b-model", default="openrouter/qwen/qwen3.5-122b-a10b")
    return ap.parse_args()


def main():
    args = parse_args()
    load_dotenv(REPO / ".env")
    os.environ.setdefault("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", ""))

    print(f"\n[1/5] loading LME (limit={args.limit}, max_sessions={args.max_sessions})", flush=True)
    conversations = load_lme(args.limit, args.max_sessions)
    gold_map = {c["conversation_id"]: c["gold_boundaries"] for c in conversations}
    cids     = sorted(gold_map.keys())

    if not args.plot_only:
        print(f"\n[2/5] encoding MiniLM-int8 ({len(cids)} instances)", flush=True)
        tag   = f"lim{args.limit or 'all'}_sess{args.max_sessions}"
        cache = OUT_EXP / f"emb_minilm_int8_{tag}.pkl"
        enc   = make_encoder()
        emb_map = encode_conversations(enc, conversations, cache)
        del enc

        print("\n[3/5] Hi-OnTop segmentation", flush=True)
        hiontop_bnd = run_hiontop_all_p(conversations, emb_map)
        for p in HIONTOP_P_VALUES:
            save_hiontop_segments(conversations, emb_map, hiontop_bnd, f"int8_p{p}")

        if not args.skip_llm:
            print("\n[4/5] LLM segmenters", flush=True)
            print(f"\n  running gpt5 ({args.gpt5_model})", flush=True)
            run_llm_segmenter(conversations, "gpt5", args.gpt5_model)
        else:
            print("\n[4/5] skipping LLM (--skip-llm)", flush=True)

    print("\n[5/5] evaluating + plotting", flush=True)
    boundaries: dict[str, dict] = {}
    for p in HIONTOP_P_VALUES:
        key = f"int8_p{p}"
        jl  = OUT_EXP / f"segments_{key}.jsonl"
        if not jl.exists():
            print(f"  WARNING: {jl.name} missing", flush=True); continue
        pc = {}
        for ln in jl.read_text().splitlines():
            if not ln.strip(): continue
            r = json.loads(ln); pc[r["conversation_id"]] = _segs_to_bnd(r["segments"])
        boundaries[key] = pc

    for key in ("gpt5",):
        jl = OUT_EXP / f"segments_{key}.jsonl"
        if not jl.exists():
            boundaries[key] = {c: [] for c in cids}; continue
        pc = {}
        for ln in jl.read_text().splitlines():
            if not ln.strip(): continue
            r = json.loads(ln); pc[r["conversation_id"]] = _segs_to_bnd(r["segments"])
        boundaries[key] = pc

    complete = [c for c in cids if all(c in boundaries.get(k, {}) for k in ALL_KEYS)]
    print(f"  complete: {len(complete)}/{len(cids)}", flush=True)
    if not complete:
        print("  no complete instances yet — run LLM segmenters first.", flush=True); return

    print("\n" + "="*60, flush=True)
    print("PAIRWISE BOUNDARY F1 — LongMemEval-S", flush=True)
    print("="*60, flush=True)
    results = {"n_conv": len(complete), "pairwise": {}}
    for a in ALL_KEYS:
        for b in ALL_KEYS:
            results["pairwise"][f"{a}_vs_{b}"] = float(np.mean(
                [pairwise_f1(boundaries[a][c], boundaries[b][c]) for c in complete]))

    hiontop_keys = ["int8_p60", "int8_p70", "int8_p80"]
    print(f"\n  {'':30s}  {'GPT-5':>8}", flush=True)
    for hk in hiontop_keys:
        v = results["pairwise"][f"{hk}_vs_gpt5"]
        print(f"  {KEY_LABEL[hk]:30s}  {v:8.4f}", flush=True)

    (OUT_EXP / "results.json").write_text(json.dumps(results, indent=2))
    make_figure(boundaries, complete)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
