#!/usr/bin/env python3
"""Boundary comparison on SHARE dataset.

6 methods compared vs SHARE gold session boundaries (pairwise F1):
  LLM segmenters : GPT-5, Qwen3.5-27B, Qwen3.5-122B
  Hi-OnTop       : MiniLM-int8 at p60, p70, p80

SHARE gold = session boundaries from the multi-session movie dialogue structure.
Each character-pair episode has ~5.5 sessions; session transitions are the
ground-truth topic boundaries.

Outputs
-------
  outputs/experiments/2026-06-02_share_boundary_comparison/
      segments_<method>.jsonl   # resumable; one line per conversation
      results.json
  outputs/figures/figure_R_share_boundary_agreement.{pdf,png}

Usage
-----
  # Full run (test split, ~320 conversations):
  uv run python scripts/run_share_boundary_comparison.py

  # Quick smoke test (first 10 conversations, skip LLMs):
  uv run python scripts/run_share_boundary_comparison.py --limit 10 --skip-llm

  # Custom model slugs:
  uv run python scripts/run_share_boundary_comparison.py \
      --gpt5-model openai/gpt-5 \
      --qwen27b-model qwen/qwen3.5-27b \
      --qwen122b-model qwen/qwen3.5-122b
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

from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

# ── directories ─────────────────────────────────────────────────────────────
SHARE_DIR = REPO / "benchmarks/SHARE"
OUT_EXP   = REPO / "outputs/experiments/2026-06-02_share_boundary_comparison"
OUT_FIG   = REPO / "outputs/figures"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ── Hi-OnTop defaults ────────────────────────────────────────────────────────
M, RHO, A  = 2, 0.7, 0.5
ENCODER_CFG = {
    "model":      "sentence-transformers/all-MiniLM-L6-v2",
    "backend":    "onnx",
    "file_name":  "onnx/model_quint8_avx2.onnx",
}
HIONTOP_P_VALUES = [60, 70, 80]

# ── method registry ──────────────────────────────────────────────────────────
# (key, display_label, color)
METHOD_REGISTRY = [
    ("gpt5",          "GPT-5",                "#8E44AD"),
    ("qwen27b",       "Qwen3.5-27B",          "#1F8E3F"),
    ("qwen122b",      "Qwen3.5-122B",         "#0D5E1F"),
    ("int8_p60",      r"Hi-OnTop (int8, $p_{60}$)", "#2980B9"),
    ("int8_p70",      r"Hi-OnTop (int8, $p_{70}$)", "#1A5276"),
    ("int8_p80",      r"Hi-OnTop (int8, $p_{80}$)", "#0A2744"),
]
ALL_KEYS   = [k for k, _, _ in METHOD_REGISTRY]
KEY_LABEL  = {k: l for k, l, _ in METHOD_REGISTRY}
KEY_COLOR  = {k: c for k, _, c in METHOD_REGISTRY}


# ── SHARE loading ────────────────────────────────────────────────────────────

def download_share(split: str = "test") -> Path:
    """Download SHARE split JSON from HuggingFace if not cached."""
    path = SHARE_DIR / "data" / f"{split}.json"
    if path.exists():
        print(f"  SHARE {split}.json already cached at {path}", flush=True)
        return path
    print(f"  downloading SHARE {split}.json from HuggingFace…", flush=True)
    from huggingface_hub import hf_hub_download
    tmp = hf_hub_download(
        repo_id="eunwoneunwon/SHARE",
        filename=f"data/{split}.json",
        repo_type="dataset",
        local_dir=str(SHARE_DIR),
    )
    return Path(tmp)


def load_share(path: Path, limit: int | None = None) -> list[dict]:
    """Parse SHARE JSON into conversation dicts.

    Each dict:
        conversation_id : str        (character-pair key)
        sessions        : List[List[str]]  (each session = list of "SPEAKER: text")
        gold_boundaries : List[int]  (global turn indices of session-end turns,
                                      excluding the final session)
    """
    raw = json.loads(path.read_text())
    conversations = []
    for pair_key, info in raw.items():
        sessions: list[list[str]] = []
        for sess_info in info.get("dialogue", []):
            turns = [
                f"{d['speaker']}: {d['text']}"
                for d in sess_info.get("dialogues", [])
                if d.get("text", "").strip()
            ]
            if turns:
                sessions.append(turns)
        if len(sessions) < 2:
            continue  # no boundary to detect

        # gold: last turn index of each session except the final
        gold_bnd: list[int] = []
        cursor = -1
        for i, sess in enumerate(sessions):
            cursor += len(sess)
            if i < len(sessions) - 1:
                gold_bnd.append(cursor)

        conversations.append({
            "conversation_id": pair_key,
            "sessions":        sessions,
            "gold_boundaries": gold_bnd,
        })
        if limit and len(conversations) >= limit:
            break

    print(f"  loaded {len(conversations)} episodes from SHARE", flush=True)
    return conversations


# ── Hi-OnTop segmentation ────────────────────────────────────────────────────

def make_encoder() -> SentenceTransformer:
    return SentenceTransformer(
        ENCODER_CFG["model"],
        backend="onnx",
        model_kwargs={
            "provider":  "CPUExecutionProvider",
            "file_name": ENCODER_CFG["file_name"],
        },
    )


def encode_conversations(enc, conversations: list[dict], cache: Path) -> dict:
    """Returns {conv_id: np.ndarray (n_turns, dim)}."""
    if cache.exists():
        print(f"  using cached embeddings {cache.name}", flush=True)
        return pickle.loads(cache.read_bytes())
    emb_map: dict[str, np.ndarray] = {}
    for conv in conversations:
        cid   = conv["conversation_id"]
        turns = [t for sess in conv["sessions"] for t in sess]
        vecs  = enc.encode(turns, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        emb_map[cid] = vecs.astype(np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(emb_map))
    print(f"  cached embeddings → {cache.name}", flush=True)
    return emb_map


def hiontop_boundaries(vecs: np.ndarray, dstar: float) -> list[int]:
    """Global turn indices where a boundary occurs AFTER that turn."""
    n = len(vecs)
    seg = HiOnTop(dim=vecs.shape[1], delta_star=dstar,
                 ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for v in vecs:
        seg.assign(v)
    history = seg.history()
    bnd = []
    for i in range(1, n):
        if float(history[i]["delta_eff"]) >= dstar:
            bnd.append(i - 1)
    return bnd


def run_hiontop_all_p(conversations: list[dict], emb_map: dict) -> dict[str, dict]:
    """Run Hi-OnTop at each percentile in HIONTOP_P_VALUES.

    Returns {key: {conv_id: List[int] boundaries}}
    """
    # pool δ_eff across all conversations
    pool_vals: list[float] = []
    deff_map: dict[str, np.ndarray] = {}
    for conv in conversations:
        cid  = conv["conversation_id"]
        vecs = emb_map[cid]
        seg  = HiOnTop(dim=vecs.shape[1], delta_star=1.0,
                      ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
        for v in vecs:
            seg.assign(v)
        hist   = seg.history()
        deffs  = np.array([float(h["delta_eff"]) for h in hist[1:]])
        pool_vals.extend(deffs.tolist())
        deff_map[cid] = deffs

    pool = np.array(pool_vals)
    print(f"  δ_eff pool: n={pool.size}  mean={pool.mean():.4f} "
          f"std={pool.std():.4f}", flush=True)

    results: dict[str, dict] = {}
    for p in HIONTOP_P_VALUES:
        key    = f"int8_p{p}"
        dstar  = float(np.percentile(pool, p))
        print(f"  Hi-OnTop int8 p{p}: δ*={dstar:.4f}", flush=True)
        per_conv: dict[str, list[int]] = {}
        for conv in conversations:
            cid   = conv["conversation_id"]
            deffs = deff_map[cid]
            bnd   = [i for i, d in enumerate(deffs) if d >= dstar]
            per_conv[cid] = bnd
        results[key] = per_conv
    return results


def save_hiontop_segments(conversations: list[dict], emb_map: dict,
                          hiontop_bnd: dict[str, dict], key: str) -> None:
    """Save Hi-OnTop segments as segments.jsonl (SeCom-compatible format)."""
    out = OUT_EXP / f"segments_{key}.jsonl"
    lines = []
    for conv in conversations:
        cid      = conv["conversation_id"]
        all_turns = [t for sess in conv["sessions"] for t in sess]
        bnd_set  = set(hiontop_bnd[key][cid])
        # convert boundary-after-turn indices → segments (List[List[str]])
        segs: list[list[str]] = []
        seg: list[str] = []
        for i, turn in enumerate(all_turns):
            seg.append(turn)
            if i in bnd_set:
                segs.append(seg)
                seg = []
        if seg:
            segs.append(seg)
        lines.append(json.dumps({
            "conversation_id": cid,
            "sessions":        conv["sessions"],
            "segments":        segs,
        }, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n")
    print(f"  saved {out.name}", flush=True)


# ── LLM segmenters ────────────────────────────────────────────────────────────

def _make_llm_client(model_slug: str):
    """Construct OpenAI client pointed at Crts + patch for thinking/GPT-5."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL",
                                os.environ.get("OPENAI_API_BASE")),
    )
    is_gpt5 = "gpt-5" in model_slug.lower()
    _orig   = client.chat.completions.create

    def _patched(*a, **kw):
        if is_gpt5:
            if "max_tokens" in kw:
                kw["max_completion_tokens"] = max(kw.pop("max_tokens"), 2000)
            kw.pop("temperature", None)
            kw.pop("top_p", None)
            kw.pop("seed", None)
            kw.setdefault("reasoning_effort", "minimal")
        else:
            # Disable Qwen hybrid-thinking
            kw.setdefault("reasoning_effort", "none")
        return _orig(*a, **kw)

    client.chat.completions.create = _patched
    return client, model_slug


def _call_llm(client, model_slug: str, prompt: str, max_tokens: int = 2048) -> str:
    from time import sleep
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_slug,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    LLM error (attempt {attempt+1}): {e}", flush=True)
            sleep(5 * (attempt + 1))
    return ""


def _build_segment_prompt(turns: list[str]) -> str:
    """SeCom-style prompt adapted for two-speaker dialogue."""
    exchanges_str = ""
    for i, turn in enumerate(turns):
        exchanges_str += f"[Exchange {i}]: {turn}\n\n"

    return f"""# Instruction

## Context

- **Goal**: Your task is to segment a multi-turn dialogue between two speakers into \
topically coherent units. Exchanges about the same topic should be grouped together; \
create a new unit when the topic shifts.
- **Data**: A series of exchanges separated by "\\n\\n". Each exchange starts with \
"[Exchange (N)]: ".

## Output Format

Output segmentation results as **jsonl lines** inside <segmentation></segmentation> tags. \
Each line is a JSON dict with keys:
  - segment_id          : int (starting from 0)
  - start_exchange_number : int (first exchange in segment)
  - end_exchange_number   : int (last exchange in segment)
  - num_exchanges         : int (= end - start + 1)

Example:
<segmentation>
{{"segment_id": 0, "start_exchange_number": 0, "end_exchange_number": 4, "num_exchanges": 5}}
{{"segment_id": 1, "start_exchange_number": 5, "end_exchange_number": 7, "num_exchanges": 3}}
</segmentation>

## Constraints

- Cover ALL exchanges (no gaps, no overlaps).
- start_exchange_number of the first segment = 0.
- end_exchange_number of the last segment = {len(turns) - 1}.

# Data

{exchanges_str.strip()}

# Output

<segmentation>
"""


def _parse_llm_segments(response: str, turns: list[str]) -> list[list[str]]:
    """Extract segments from LLM response; fallback to 3-turn chunks on failure."""
    import re
    import traceback
    m = re.search(r"<segmentation>([\s\S]*?)(?:</segmentation>|$)", response)
    if not m:
        print("    parse fail: no <segmentation> tag", flush=True)
        return [turns[i:i+3] for i in range(0, len(turns), 3)]

    block = m.group(1).strip()
    segs: list[list[str]] = []
    cursor = 0
    ok = True
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            d    = json.loads(line)
            n_ex = int(d["num_exchanges"])
            segs.append(turns[cursor:cursor + n_ex])
            cursor += n_ex
        except Exception:
            print(f"    parse fail on line: {line!r}", flush=True)
            traceback.print_exc()
            ok = False
            break
    if not ok or cursor != len(turns):
        print(f"    fallback: cursor={cursor} expected={len(turns)}", flush=True)
        return [turns[i:i+3] for i in range(0, len(turns), 3)]
    return segs


def run_llm_segmenter(conversations: list[dict], key: str,
                       model_slug: str) -> dict[str, list[int]]:
    """Segment all conversations with the given LLM; save/resume incrementally."""
    out_path = OUT_EXP / f"segments_{key}.jsonl"

    # Load already-processed ids
    done: dict[str, list[int]] = {}
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if not ln.strip():
                continue
            r   = json.loads(ln)
            cid = r["conversation_id"]
            done[cid] = _segs_to_boundaries(r["segments"])
        print(f"  [{key}] resuming: {len(done)} already done", flush=True)

    client, slug = _make_llm_client(model_slug)
    per_conv: dict[str, list[int]] = dict(done)

    with out_path.open("a", encoding="utf-8") as fh:
        for conv in conversations:
            cid = conv["conversation_id"]
            if cid in done:
                continue
            turns  = [t for sess in conv["sessions"] for t in sess]
            prompt = _build_segment_prompt(turns)
            print(f"  [{key}] conv {cid!r}: {len(turns)} turns …", flush=True,
                  end=" ")
            resp   = _call_llm(client, slug, prompt)
            segs   = _parse_llm_segments(resp, turns)
            bnd    = _segs_to_boundaries(segs)
            per_conv[cid] = bnd
            record = {
                "conversation_id": cid,
                "sessions":        conv["sessions"],
                "segments":        segs,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"→ {len(segs)} segs, {len(bnd)} boundaries", flush=True)

    return per_conv


def _segs_to_boundaries(segments: list[list[str]]) -> list[int]:
    """Convert segments (List[List[str]]) to global turn boundary indices."""
    bnd: list[int] = []
    cursor = -1
    for i, seg in enumerate(segments):
        cursor += len(seg)
        if i < len(segments) - 1 and seg:
            bnd.append(cursor)
    return bnd


# ── metrics ──────────────────────────────────────────────────────────────────

def pairwise_f1(pred: list[int], gold: list[int]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    ps, gs = set(pred), set(gold)
    tp = len(ps & gs)
    p  = tp / max(1, len(ps))
    r  = tp / max(1, len(gs))
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(boundaries: dict[str, dict], cids: list[str]) -> None:
    """6×6 pairwise boundary F1 heatmap between methods (no gold comparison).

    Same methodology as figure_F (boundary_agreement) and
    calibrate_p_via_llm_distillation.py: methods compared against each other.
    """
    n   = len(ALL_KEYS)
    mat = np.zeros((n, n))
    for i, a in enumerate(ALL_KEYS):
        for j, b in enumerate(ALL_KEYS):
            mat[i, j] = float(np.mean(
                [pairwise_f1(boundaries[a][c], boundaries[b][c]) for c in cids]
            ))

    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="equal")
    labels = [KEY_LABEL[k] for k in ALL_KEYS]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v     = mat[i, j]
            color = "white" if v > 0.5 else "#222222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color)
    ax.set_title(
        f"Pairwise boundary F1 on SHARE ({len(cids)} conv, {n} methods)",
        fontsize=11, pad=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cbar.set_label("Pairwise boundary F1", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = OUT_FIG / f"figure_R_share_boundary_agreement.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"saved {p}", flush=True)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    help="SHARE split to use: train / valid / test (default: test)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on number of conversations (None = all)")
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip LLM segmenters (useful for smoke-testing Hi-OnTop)")
    ap.add_argument("--gpt5-model",    default="openrouter/openai/gpt-5")
    ap.add_argument("--qwen27b-model", default="openrouter/qwen/qwen3.5-27b")
    ap.add_argument("--qwen122b-model", default="openrouter/qwen/qwen3.5-122b-a10b")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip segmentation; only regenerate figure from saved jsonl")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO / ".env")
    os.environ.setdefault("OPENAI_API_BASE",
                          os.environ.get("OPENAI_BASE_URL", ""))

    # ── 1. Load SHARE ─────────────────────────────────────────────────────────
    print(f"\n[1/5] loading SHARE {args.split} split", flush=True)
    share_json = download_share(args.split)
    conversations = load_share(share_json, limit=args.limit)
    gold_map      = {c["conversation_id"]: c["gold_boundaries"]
                     for c in conversations}
    cids          = sorted(gold_map.keys())

    if not args.plot_only:
        # ── 2. Hi-OnTop embeddings ─────────────────────────────────────────────
        print(f"\n[2/5] encoding with MiniLM-int8 ({len(cids)} conversations)",
              flush=True)
        cache = OUT_EXP / "emb_minilm_int8.pkl"
        enc   = make_encoder()
        emb_map = encode_conversations(enc, conversations, cache)
        del enc

        # ── 3. Hi-OnTop segmentation (p60/p70/p80) ────────────────────────────
        print("\n[3/5] Hi-OnTop segmentation (global δ* per percentile)", flush=True)
        hiontop_bnd = run_hiontop_all_p(conversations, emb_map)
        for p in HIONTOP_P_VALUES:
            save_hiontop_segments(conversations, emb_map, hiontop_bnd, f"int8_p{p}")

        # ── 4. LLM segmenters ─────────────────────────────────────────────────
        if not args.skip_llm:
            print("\n[4/5] LLM segmenters", flush=True)
            llm_specs = [
                ("gpt5",    args.gpt5_model),
                ("qwen27b", args.qwen27b_model),
                ("qwen122b", args.qwen122b_model),
            ]
            for key, slug in llm_specs:
                print(f"\n  running {key} ({slug})", flush=True)
                run_llm_segmenter(conversations, key, slug)
        else:
            print("\n[4/5] skipping LLM segmenters (--skip-llm)", flush=True)

    # ── 5. Collect boundaries & evaluate ──────────────────────────────────────
    print("\n[5/5] evaluating + plotting", flush=True)

    boundaries: dict[str, dict] = {}

    # load Hi-OnTop boundaries
    for p in HIONTOP_P_VALUES:
        key = f"int8_p{p}"
        jl  = OUT_EXP / f"segments_{key}.jsonl"
        if not jl.exists():
            print(f"  WARNING: {jl.name} not found; skipping {key}", flush=True)
            continue
        per_conv: dict[str, list[int]] = {}
        for ln in jl.read_text().splitlines():
            if not ln.strip():
                continue
            r   = json.loads(ln)
            cid = r["conversation_id"]
            per_conv[cid] = _segs_to_boundaries(r["segments"])
        boundaries[key] = per_conv

    # load LLM boundaries
    for key in ("gpt5", "qwen27b", "qwen122b"):
        jl = OUT_EXP / f"segments_{key}.jsonl"
        if not jl.exists():
            print(f"  WARNING: {jl.name} not found; using empty for {key}",
                  flush=True)
            boundaries[key] = {c: [] for c in cids}
            continue
        per_conv = {}
        for ln in jl.read_text().splitlines():
            if not ln.strip():
                continue
            r   = json.loads(ln)
            cid = r["conversation_id"]
            per_conv[cid] = _segs_to_boundaries(r["segments"])
        boundaries[key] = per_conv

    # restrict to conversations present in ALL methods
    complete_cids = [c for c in cids
                     if all(c in boundaries.get(k, {}) for k in ALL_KEYS)]
    print(f"  conversations with all methods complete: {len(complete_cids)}",
          flush=True)
    if not complete_cids:
        print("  no complete conversations yet — run LLM segmenters first.",
              flush=True)
        # still save partial results
    else:
        # summary table
        print("\n" + "=" * 60, flush=True)
        print("PAIRWISE BOUNDARY F1 — SHARE (vs LLM references)", flush=True)
        print("=" * 60, flush=True)
        llm_keys    = ["gpt5", "qwen27b", "qwen122b"]
        hiontop_keys = ["int8_p60", "int8_p70", "int8_p80"]
        results: dict = {"n_conv": len(complete_cids), "pairwise": {}}
        for a in ALL_KEYS:
            for b in ALL_KEYS:
                f1s = [pairwise_f1(boundaries[a][c], boundaries[b][c])
                       for c in complete_cids]
                results["pairwise"][f"{a}_vs_{b}"] = float(np.mean(f1s))

        # print Hi-OnTop vs each LLM reference (the key comparison)
        print(f"\n  {'':30s}  {'GPT-5':>8}  {'Qwen27B':>8}  {'Qwen122B':>8}",
              flush=True)
        for hk in hiontop_keys:
            row = "  " + KEY_LABEL[hk].ljust(30)
            for lk in llm_keys:
                v = results["pairwise"][f"{hk}_vs_{lk}"]
                row += f"  {v:8.4f}"
            print(row, flush=True)

        # also print LLM vs LLM
        print(f"\n  LLM agreement:", flush=True)
        for i, a in enumerate(llm_keys):
            for b in llm_keys[i+1:]:
                v = results["pairwise"][f"{a}_vs_{b}"]
                print(f"    {KEY_LABEL[a]} vs {KEY_LABEL[b]}: {v:.4f}", flush=True)

        (OUT_EXP / "results.json").write_text(json.dumps(results, indent=2))
        make_figure(boundaries, complete_cids)

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
