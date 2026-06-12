#!/usr/bin/env python3
"""TIAGE full compare — v1 / v3.1.1 / v3.3.1 / v3.3.2 / v3.3.3 / v3.3.4 /
v3.3.3-2 / v3.3.4-2 (8 method × N seed).

Default HP: each method's recommended config (matches LoCoMo decision-log).
Random source: ``torch.manual_seed`` + ``numpy.random.seed`` per (method, seed).
Output: REPORT.md. **Target = WD↓ / F1↑ / Pk↓** (literature-comparable;
user 2026-05-18, supersedes 2026-05-17 ARI-primary). ARI / n_topics /
collapse retained as GUARD columns (F1 is gamed by every-turn splitting;
degenerate collapse ≥ 50% flagged † and excluded from every 'best').
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from nltk.metrics.segmentation import pk as _pk
from nltk.metrics.segmentation import windowdiff as _windowdiff
from sklearn.metrics import adjusted_rand_score, f1_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_ontop.embedding import QueryEncoder  # noqa: E402
from hi_ontop.sem_core import HiOnTopSegmenter  # noqa: E402
from hi_ontop.sem_core_optimize import HiOnTopSegmenterV3  # noqa: E402
from hi_ontop.sem_core_v331_rnn import HiOnTopSegmenterV331  # noqa: E402
from hi_ontop.sem_core_v332_rnn_pe import HiOnTopSegmenterV332  # noqa: E402
from hi_ontop.sem_core_v333_rnn_f0 import HiOnTopSegmenterV333  # noqa: E402
from hi_ontop.sem_core_v334_rnn_var import HiOnTopSegmenterV334  # noqa: E402
from hi_ontop.sem_core_v333_2 import HiOnTopSegmenterV333_2  # noqa: E402
from hi_ontop.sem_core_v334_2 import HiOnTopSegmenterV334_2  # noqa: E402
from hi_ontop.sem_core_v335 import HiOnTopSegmenterV335  # noqa: E402
from hi_ontop.sem_core_v336 import HiOnTopSegmenterV336  # noqa: E402
from hi_ontop.sem_core_v337 import HiOnTopSegmenterV337  # noqa: E402
from hi_ontop.sem_core_v338 import HiOnTopSegmenterV338  # noqa: E402
from hi_ontop.sem_core_v339 import HiOnTopSegmenterV339  # noqa: E402
from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411  # noqa: E402


DATA_DIR = REPO_ROOT / "benchmarks" / "tiage" / "data" / "personachat" / "anno"


def load_split(split: str):
    path = DATA_DIR / split / f"anno_{split}.json"
    raw = json.loads(path.read_text())
    return {cid: [(t[0], t[1]) for t in dialog] for cid, dialog in raw.items()}


def gt_shifts(dialog):
    labels = [t[1] for t in dialog]
    return [labels[i] == "1" for i in range(1, len(labels))]


def _bnd_str(shifts) -> str:
    """Transition-bool list → nltk segmentation string ('1' = boundary)."""
    return "".join("1" if b else "0" for b in shifts)


def _pk_wd_one(ref_bool, hyp_bool):
    """Pk / WindowDiff for a single conversation. Returns (pk, wd) or None.

    k = half the mean reference segment length (nltk convention), clamped to
    [2, len-1]. Conversations with <2 transition points are skipped (k ill-
    defined). No-boundary refs use k = len//2 fallback so empty-ref convs still
    contribute rather than silently inflating the score.
    """
    ref, hyp = _bnd_str(ref_bool), _bnd_str(hyp_bool)
    n = len(ref)
    if n < 2:
        return None
    n_bnd = ref.count("1")
    k = round(n / (2 * n_bnd)) if n_bnd else n // 2
    k = max(2, min(k, n - 1))
    return float(_pk(ref, hyp, k=k)), float(_windowdiff(ref, hyp, k))


def _gt_seg_labels(shifts) -> list[int]:
    """Transition-bool list → per-turn GT segment id (cumsum of shifts)."""
    labels = [0]
    for b in shifts:
        labels.append(labels[-1] + (1 if b else 0))
    return labels


def _ari_one(shifts, pred_ids) -> float | None:
    """Adjusted Rand Index for one conversation: GT segment partition vs
    predicted topic-id partition (both length n_turns). Penalises BOTH
    over-segmentation and over-merging (collapse), unlike boundary-F1.
    Returns None for conversations with <2 turns (ARI ill-defined)."""
    gt = _gt_seg_labels(shifts)
    if len(gt) < 2 or len(gt) != len(pred_ids):
        return None
    return float(adjusted_rand_score(gt, pred_ids))


# Method configs: (label, factory, uses_rng).
# factory(dim) → segmenter; HP = LoCoMo 2026-05-08~ recommended defaults.
def _factory_v1(dim):
    return HiOnTopSegmenter(dim=dim, alpha=1.0, lmda=10.0, sigma0_sq=0.01)


def _factory_v311(dim):
    return HiOnTopSegmenterV3(dim=dim, alpha=1.0, lmda=10.0, tau=50.0, cos_threshold=0.7)


def _factory_v331(dim):
    return HiOnTopSegmenterV331(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v332(dim):
    return HiOnTopSegmenterV332(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25, pe_threshold=0.5,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v333(dim):
    return HiOnTopSegmenterV333(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25, pe_threshold=0.5,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v334(dim):
    return HiOnTopSegmenterV334(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v333_2(dim):
    return HiOnTopSegmenterV333_2(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25, pe_threshold=0.5,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v334_2(dim):
    return HiOnTopSegmenterV334_2(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


# v3.3.5~8: methodology doc (context/methodology/v3.3.{5,6,7,8}.md, 2026-05-17)
# 권장/평가 regime = α=1, λ=10 + 모듈 default (cos_threshold=0.9,
# pe_var_sigma0_sq=0.04, pe_prior=1.0 …). doc 의 idx374 결과가 모두 이 HP
# 기준. α=100 은 fresh-prior 폭주로 degenerate (older 와 동일). 나머지는
# 각 클래스 default 에 위임 (inspect_longmemeval_segmentation 구성과 동일).
def _factory_v335(dim):
    return HiOnTopSegmenterV335(dim=dim, alpha=1.0, lmda=10.0)


def _factory_v336(dim):
    return HiOnTopSegmenterV336(dim=dim, alpha=1.0, lmda=10.0)


def _factory_v337(dim):
    return HiOnTopSegmenterV337(dim=dim, alpha=1.0, lmda=10.0)


def _factory_v338(dim):
    return HiOnTopSegmenterV338(dim=dim, alpha=1.0, lmda=10.0)

def _factory_v339(dim):
    return HiOnTopSegmenterV339(dim=dim, alpha=1.0, lmda=10.0)


def _factory_v411(dim):
    return HiOnTopSegmenterV411(dim=dim, alpha=1.0, lmda=10.0)


METHODS = [
    ("v1", _factory_v1, False),
    ("v3.1.1", _factory_v311, False),
    ("v3.3.1", _factory_v331, True),
    ("v3.3.2", _factory_v332, True),
    ("v3.3.3", _factory_v333, True),
    ("v3.3.4", _factory_v334, True),
    ("v3.3.3-2", _factory_v333_2, True),
    ("v3.3.4-2", _factory_v334_2, True),
    ("v3.3.5", _factory_v335, True),
    ("v3.3.6", _factory_v336, True),
    ("v3.3.7", _factory_v337, True),
    ("v3.3.8", _factory_v338, True),
    ("v3.3.9", _factory_v339, True),
    ("v4.1.1", _factory_v411, True),
]


# ─── HP sweep (Stage A coarse default) ───────────────────────────────
# Swept methods + their classes. NOTE: ``pe_prior`` does NOT exist on
# v3.3.5/6/7 (v3.3.8-only fresh-baseline knob). The decision-log/codex
# "alpha·lmda·pe_prior" maps here to ``pe_var_sigma0_sq`` — the PE
# variance prior mode, i.e. the likelihood-calibration knob codex named
# as the real bottleneck for v3.3.5~7. (decision-log 2026-05-17 정정.)
SWEEP_CLASSES = {
    "v3.3.5": HiOnTopSegmenterV335,
    "v3.3.6": HiOnTopSegmenterV336,
    "v3.3.7": HiOnTopSegmenterV337,
    "v3.3.9": HiOnTopSegmenterV339,
    "v4.1.1": HiOnTopSegmenterV411,
}

# Stage A coarse grid (27 combos/method). Override via --grid.
DEFAULT_GRID = {
    "alpha": [0.5, 1.0, 2.0],
    "lmda": [5.0, 10.0, 20.0],
    "pe_var_sigma0_sq": [0.01, 0.04, 0.1],
}


def _parse_grid(spec: str | None) -> dict[str, list[float]]:
    """``"alpha=0.5,1,2;lmda=5,10,20;pe_var_sigma0_sq=0.01,0.04"`` →
    dict of axis→values. None → DEFAULT_GRID (Stage A coarse)."""
    if not spec:
        return DEFAULT_GRID
    grid: dict[str, list[float]] = {}
    for part in spec.split(";"):
        key, _, vals = part.partition("=")
        grid[key.strip()] = [float(v) for v in vals.split(",") if v.strip()]
    return grid


def _build_sweep_segmenter(method: str, dim: int, hp: dict):
    """Parameterised factory: swept HP + each class's other defaults."""
    return SWEEP_CLASSES[method](dim=dim, **hp)


# multiprocessing worker globals (loaded once per worker via initializer).
_W_DIALOGS = None
_W_EMB = None


def _sweep_init(cache_path: str) -> None:
    global _W_DIALOGS, _W_EMB
    import torch as _t

    _t.set_num_threads(1)  # 8 workers must not oversubscribe BLAS/torch
    with open(cache_path, "rb") as fh:
        blob = pickle.load(fh)
    _W_DIALOGS, _W_EMB = blob["dialogs"], blob["embeddings"]


def _sweep_task(task: tuple):
    """One (method, hp, seed) cell over the full split. Returns metrics +
    config. Reuses run_one (ARI/Pk/WD/F1/collapse) with a closure factory."""
    method, hp, seed = task
    factory = lambda dim: _build_sweep_segmenter(method, dim, hp)  # noqa: E731
    r = run_one(factory, _W_DIALOGS, _W_EMB, seed=seed)
    return {"method": method, "hp": hp, "seed": seed, **r}


# Priority order — sweep the contenders that conflict on ARI vs WD/n_topics
# first (v3.3.6 best ARI=primary, v3.3.5 best WD + n_topics closest to GT).
# v3.3.7 is consistently 3rd on every axis → deferred (opt-in via --sweep-methods).
SWEEP_PRIORITY = ["v3.3.6", "v3.3.5", "v3.3.7"]


def run_sweep(args, dialogs, embeddings, n_convs, n_turns, n_shifts) -> None:
    grid = _parse_grid(args.grid)
    methods = [m.strip() for m in args.sweep_methods.split(",") if m.strip()]
    methods.sort(key=lambda m: SWEEP_PRIORITY.index(m)
                 if m in SWEEP_PRIORITY else 99)
    axes = list(grid)
    combos = [dict(zip(axes, vals)) for vals in itertools.product(*grid.values())]

    exp_dir = REPO_ROOT / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    cache_path = exp_dir / "_emb_cache.pkl"
    with open(cache_path, "wb") as fh:  # encode-once → workers reuse
        pickle.dump({"dialogs": dialogs, "embeddings": embeddings}, fh)

    # Tasks emitted in method-priority order so partial results are useful.
    tasks = [
        (m, hp, s)
        for m in methods
        for hp in combos
        for s in args.seeds
    ]
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(
        f"[sweep] {len(methods)} method × {len(combos)} combo × "
        f"{len(args.seeds)} seed = {len(tasks)} run · {workers} workers · "
        f"grid={grid}"
    )

    ctx = mp.get_context("spawn")
    raw = []
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_sweep_init,
                  initargs=(str(cache_path),)) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_sweep_task, tasks), 1
        ):
            raw.append(res)
            print(
                f"  [{i}/{len(tasks)}] {res['method']} "
                f"{res['hp']} seed={res['seed']} "
                f"ARI={res['ari']:.3f} WD={res['wd']:.3f} "
                f"nt={res['n_topics_mean']:.1f} "
                f"col={res['collapse_rate']:.0%} "
                f"({time.perf_counter() - t0:.0f}s elapsed)"
            )
            _write_sweep_report(  # incremental: partial results survive
                args, raw, grid, methods, combos, n_convs, n_turns, n_shifts
            )
    cache_path.unlink(missing_ok=True)
    print(f"\n[sweep] done in {time.perf_counter() - t0:.0f}s")


def _write_sweep_report(args, raw, grid, methods, combos,
                        n_convs, n_turns, n_shifts) -> None:
    """Aggregate (method, hp) over seeds → ARI/WD/n_topics REPORT,
    sorted by ARI desc. Rewritten every cell so a killed run keeps
    whatever finished."""
    by_cfg: dict = {}
    for r in raw:
        key = (r["method"], tuple(sorted(r["hp"].items())))
        by_cfg.setdefault(key, []).append(r)
    rows = []
    for (method, hp_items), rs in by_cfg.items():
        hp = dict(hp_items)
        a = [x["ari"] for x in rs]
        rows.append({
            "method": method, "hp": hp, "n": len(rs),
            "ari_m": float(np.mean(a)), "ari_s": float(np.std(a)),
            "wd_m": float(np.mean([x["wd"] for x in rs])),
            "pk_m": float(np.mean([x["pk"] for x in rs])),
            "f1_m": float(np.mean([x["f1"] for x in rs])),
            "f1_s": float(np.std([x["f1"] for x in rs])),
            "nt_m": float(np.mean([x["n_topics_mean"] for x in rs])),
            "col_m": float(np.mean([x["collapse_rate"] for x in rs])),
        })
    rows.sort(key=lambda r: r["f1_m"], reverse=True)
    axes = list(grid)
    out = [
        f"# TIAGE {args.split} HP sweep — {'/'.join(methods)} "
        f"(target: WD/F1/Pk)\n",
        f"n_convs={n_convs} · n_turns={n_turns} · n_shifts={n_shifts}\n",
        f"grid: {grid}\n",
        f"swept HP = {axes} · seeds={args.seeds} · "
        f"{len(rows)}/{len(methods) * len(combos)} configs aggregated "
        f"(incremental — partial-safe).\n",
        "GT n_topics/conv ≈ 4.15. **Target = WD↓/F1↑/Pk↓** "
        "(literature-comparable; user 2026-05-18, supersedes ARI-primary). "
        "Rows sorted by F1↓. **ARI/n_topics/collapse = guard** (F1 gamed "
        "by every-turn split; † = degenerate collapse≥50% EXCLUDED from "
        "best).\n",
        "| method | " + " | ".join(axes)
        + " | F1 ↑ (m±s) | WD ↓ | Pk ↓ | ARI (guard) | n_topics | collapse |",
        "|---|" + "---:|" * (len(axes) + 6),
    ]
    for r in rows:
        deg = r["col_m"] >= 0.5
        out.append(
            f"| {r['method']}{'†' if deg else ''} | "
            + " | ".join(f"{r['hp'][a]:g}" for a in axes)
            + f" | {r['f1_m']:.3f} ± {r['f1_s']:.3f} | "
            f"{r['wd_m']:.3f} | {r['pk_m']:.3f} | {r['ari_m']:.3f} | "
            f"{r['nt_m']:.1f} | {r['col_m']:.0%} |"
        )
    out.append("")
    nondeg = [r for r in rows if r["col_m"] < 0.5] or rows
    if rows:
        bf = max(nondeg, key=lambda r: r["f1_m"])
        out.append(
            f"**best F1 (target)**: `{bf['method']}` "
            f"{ {a: bf['hp'][a] for a in axes} } — {bf['f1_m']:.3f} "
            f"(WD {bf['wd_m']:.3f}, Pk {bf['pk_m']:.3f}, "
            f"n_topics {bf['nt_m']:.1f}, ARI {bf['ari_m']:.3f})"
        )
        bw = min(nondeg, key=lambda r: r["wd_m"])
        out.append(
            f"**best WD (target)**: `{bw['method']}` "
            f"{ {a: bw['hp'][a] for a in axes} } — {bw['wd_m']:.3f} "
            f"(F1 {bw['f1_m']:.3f}, n_topics {bw['nt_m']:.1f})"
        )
        bp = min(nondeg, key=lambda r: r["pk_m"])
        out.append(
            f"**best Pk (target)**: `{bp['method']}` "
            f"{ {a: bp['hp'][a] for a in axes} } — {bp['pk_m']:.3f}"
        )
    (REPO_ROOT / "outputs" / "experiments" / args.name / "REPORT.md").write_text(
        "\n".join(out) + "\n"
    )


def run_one(factory, dialogs, embeddings, *, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gt_all, pred_all = [], []
    n_topics = []
    pks, wds, aris = [], [], []
    n_collapsed = 0  # conversations the method merged into a single topic
    t0 = time.perf_counter()
    n_turns = 0
    for (cid, dialog), emb in zip(dialogs.items(), embeddings):
        torch.manual_seed(seed)
        seg = factory(emb.shape[1])
        ids = [seg.assign(s)[0] for s in emb]
        gt = gt_shifts(dialog)
        pred = [ids[i] != ids[i - 1] for i in range(1, len(ids))]
        gt_all.extend(gt)
        pred_all.extend(pred)
        n_topics.append(len(set(ids)))
        if len(set(ids)) == 1:
            n_collapsed += 1
        n_turns += len(dialog)
        pw = _pk_wd_one(gt, pred)
        if pw is not None:
            pks.append(pw[0])
            wds.append(pw[1])
        a = _ari_one(gt, ids)
        if a is not None:
            aris.append(a)
    f1 = f1_score(gt_all, pred_all)
    p, r, _, _ = precision_recall_fscore_support(
        gt_all, pred_all, average="binary", zero_division=0
    )
    return {
        "f1": f1,
        "precision": p,
        "recall": r,
        "pk": float(np.mean(pks)),  # macro avg over conversations, ↓ better
        "wd": float(np.mean(wds)),  # macro avg over conversations, ↓ better
        "ari": float(np.mean(aris)),  # macro avg, ↑ better, collapse-immune
        "n_topics_mean": float(np.mean(n_topics)),
        "collapse_rate": n_collapsed / len(dialogs),  # 1-topic degenerate frac
        "wall_s": time.perf_counter() - t0,
        "n_turns": n_turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--name", type=str, default="2026-05-10_tiage_iter1",
        help="Experiment name → outputs/experiments/<name>/REPORT.md",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output path. Default = outputs/experiments/<name>/REPORT.md",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="HP sweep mode (v3.3.5/6/7 × grid, parallel, embedding-cached).",
    )
    parser.add_argument(
        "--sweep-methods", type=str, default="v3.3.6,v3.3.5",
        help="Comma list, priority-ordered (contenders first; v3.3.7 deferred).",
    )
    parser.add_argument(
        "--grid", type=str, default=None,
        help='"alpha=0.5,1,2;lmda=5,10,20;pe_var_sigma0_sq=0.01,0.04,0.1". '
             "Default = Stage A coarse (27 combos/method).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default = cpu_count-1).",
    )
    args = parser.parse_args()

    print(f"[1/3] loading TIAGE {args.split}...")
    dialogs = load_split(args.split)
    n_convs = len(dialogs)
    n_turns = sum(len(d) for d in dialogs.values())
    n_shifts = sum(sum(gt_shifts(d)) for d in dialogs.values())
    print(f"  {n_convs} conv / {n_turns} turns / {n_shifts} GT shifts")

    print("[2/3] encoding...")
    enc = QueryEncoder(device=args.device)
    embeddings = []
    for cid, dialog in dialogs.items():
        utts = [t[0] for t in dialog]
        embeddings.append(np.asarray(enc.encode(utts)))
    print(f"  encoded — dim {embeddings[0].shape[1]}")

    if args.sweep:
        run_sweep(args, dialogs, embeddings, n_convs, n_turns, n_shifts)
        return

    print("[3/3] running methods...")
    results = {}  # name → list of dicts (one per seed)
    for name, factory, uses_rng in METHODS:
        per_seed = []
        seeds = args.seeds if uses_rng else [args.seeds[0]]
        for s in seeds:
            r = run_one(factory, dialogs, embeddings, seed=s)
            per_seed.append(r)
            print(
                f"  {name:10s} seed={s} F1={r['f1']:.3f} "
                f"P={r['precision']:.3f} R={r['recall']:.3f} "
                f"Pk={r['pk']:.3f} WD={r['wd']:.3f} ARI={r['ari']:.3f} "
                f"n_topics={r['n_topics_mean']:.1f} "
                f"collapse={r['collapse_rate']:.0%} wall={r['wall_s']:.1f}s"
            )
        results[name] = per_seed

    # Aggregate.
    rows = []
    for name, per_seed in results.items():
        f1s = [x["f1"] for x in per_seed]
        ps = [x["precision"] for x in per_seed]
        rs = [x["recall"] for x in per_seed]
        pksd = [x["pk"] for x in per_seed]
        wdsd = [x["wd"] for x in per_seed]
        arisd = [x["ari"] for x in per_seed]
        cols = [x["collapse_rate"] for x in per_seed]
        nts = [x["n_topics_mean"] for x in per_seed]
        wall = [x["wall_s"] for x in per_seed]
        rows.append({
            "method": name,
            "n_seeds": len(per_seed),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "p_mean": float(np.mean(ps)),
            "r_mean": float(np.mean(rs)),
            "pk_mean": float(np.mean(pksd)),
            "pk_std": float(np.std(pksd)),
            "wd_mean": float(np.mean(wdsd)),
            "wd_std": float(np.std(wdsd)),
            "ari_mean": float(np.mean(arisd)),
            "ari_std": float(np.std(arisd)),
            "collapse_mean": float(np.mean(cols)),
            "n_topics_mean": float(np.mean(nts)),
            "wall_per_turn_ms": float(np.mean(wall)) / per_seed[0]["n_turns"] * 1000,
        })

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = REPO_ROOT / "outputs" / "experiments" / args.name / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# TIAGE {args.split} — full compare ({len(METHODS)} methods × {len(args.seeds)} seeds)\n",
        f"n_convs={n_convs} · n_turns={n_turns} · n_shifts={n_shifts}\n",
        "**Target = WD↓ / F1↑ / Pk↓** (literature-comparable; user "
        "decision 2026-05-18, supersedes 2026-05-17 ARI-primary). Rows "
        "sorted by F1↓. Pk/WD macro-avg, k = half mean ref seg length. "
        "**Guard columns retained**: ARI (over-seg detector — F1 is gamed "
        "by every-turn splitting, e.g. v3.3.1~4 F1 0.354 / ARI≈0), "
        "n_topics (GT≈4.15), collapse. † = degenerate (collapse ≥ 50%) "
        "→ EXCLUDED from every best (else v3.3.8 1-topic wins 'best WD').\n",
        "| method | n_seeds | F1 ↑ (m ± s) | WD ↓ | Pk ↓ | P | R | "
        "ARI (guard) | n_topics | collapse | ms/turn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows.sort(key=lambda r: r["f1_mean"], reverse=True)
    for r in rows:
        deg = r["collapse_mean"] >= 0.5
        mark = "†" if deg else ""
        lines.append(
            f"| {r['method']}{mark} | {r['n_seeds']} | "
            f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f} | "
            f"{r['wd_mean']:.3f} | {r['pk_mean']:.3f} | "
            f"{r['p_mean']:.3f} | {r['r_mean']:.3f} | "
            f"{r['ari_mean']:.3f} | "
            f"{r['n_topics_mean']:.1f} | {r['collapse_mean']:.0%} | "
            f"{r['wall_per_turn_ms']:.2f} |"
        )
    lines.append("")
    nondeg = [r for r in rows if r["collapse_mean"] < 0.5] or rows
    best_f1 = max(nondeg, key=lambda r: r["f1_mean"])
    best_wd = min(nondeg, key=lambda r: r["wd_mean"])
    best_pk = min(nondeg, key=lambda r: r["pk_mean"])
    best_ari = max(rows, key=lambda r: r["ari_mean"])  # guard only
    lines.append(
        f"**best F1 (target)**: `{best_f1['method']}` — "
        f"{best_f1['f1_mean']:.3f} ± {best_f1['f1_std']:.3f} "
        f"(WD {best_f1['wd_mean']:.3f}, n_topics "
        f"{best_f1['n_topics_mean']:.1f}, ARI {best_f1['ari_mean']:.3f})"
    )
    lines.append(
        f"**best WD (target)**: `{best_wd['method']}` — "
        f"{best_wd['wd_mean']:.3f} ± {best_wd['wd_std']:.3f} "
        f"(F1 {best_wd['f1_mean']:.3f}, n_topics "
        f"{best_wd['n_topics_mean']:.1f})"
    )
    lines.append(
        f"**best Pk (target)**: `{best_pk['method']}` — "
        f"{best_pk['pk_mean']:.3f} ± {best_pk['pk_std']:.3f}"
    )
    lines.append(
        f"_guard_ **best ARI**: `{best_ari['method']}` — "
        f"{best_ari['ari_mean']:.3f} (over-seg sanity, not a target)"
    )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nreport → {out_path}")


if __name__ == "__main__":
    main()
