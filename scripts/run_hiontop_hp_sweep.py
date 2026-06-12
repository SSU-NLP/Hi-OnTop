#!/usr/bin/env python3
"""Hi-OnTop full HP sweep — all 4 hyper-parameters, single grid.

Hi-OnTop (``src/hi_ontop/hi_ontop.py``) has exactly 4 hyper-parameters, all live
(verified by the 2026-05-22 audit — no dead HPs remain). They jointly shape
the causal-window context vector and the boundary threshold, so the whole
set is swept as one grid (no OAT phase needed):

    delta_star, ctx_window, ctx_decay, ctx_blend_a

Tuning set (no test leakage): ``tiage/train`` + ``superseg/validation`` +
``dialseg711`` 30%-tune split (711-dialog test set, seeded 30/70 shuffle —
30% joins tuning, 70% held out for the report).
Final report: ``tiage/test``, ``dialseg711`` 70% held-out, ``superseg/test``.

Metric: official SuperDialseg Pk/WD (k=auto) + binary F1,
``Score = 0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD)``. Embeddings = cached mpnet
pkl (``outputs/runs/_misc/sds_emb_*``). Tuning objective = mean Score over
the 3 tuning sets.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
MPNET_SUFFIX = "_sentence-transformers_multi-qa-mpnet-base-dot-v1"


# --------------------------------------------------------------------------
# data + metric
# --------------------------------------------------------------------------

def load_dialogs(ds: str, split: str):
    raw = json.loads((SDS / ds / f"segmentation_file_{split}.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if yt:
            yt[-1] = 0
        if len(utts) >= 2:
            out.append((utts, yt))
    return out


def load_embs(ds: str, split: str):
    for cand in (CACHE / f"sds_emb_{ds}_{split}.pkl",
                 CACHE / f"sds_emb_{ds}_{split}{MPNET_SUFFIX}.pkl"):
        if cand.exists():
            with open(cand, "rb") as fh:
                return pickle.load(fh)
    raise FileNotFoundError(f"no cached embedding for {ds}/{split}")


def split_frac(dialogs, embs, tune_frac: float = 0.30, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(len(dialogs))
    cut = int(round(len(idx) * tune_frac))
    tune_i, test_i = idx[:cut], idx[cut:]
    tune = ([dialogs[i] for i in tune_i], [embs[i] for i in tune_i])
    test = ([dialogs[i] for i in test_i], [embs[i] for i in test_i])
    return tune, test


def official_pk_wd(yt, yp):
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


# --------------------------------------------------------------------------
# Hi-OnTop run
# --------------------------------------------------------------------------

def seg_pred(emb: np.ndarray, hp: dict) -> list[int]:
    seg = HiOnTop(dim=emb.shape[1], **hp)
    ids = [seg.assign(s.astype(np.float64))[0] for s in emb]
    return [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]


def eval_on(dialogs, embs, hp: dict) -> dict:
    pks, wds, g, p = [], [], [], []
    for (u, yt), e in zip(dialogs, embs):
        yp = seg_pred(e, hp)
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    return dict(pk=pk_m, wd=wd_m, f1=f1, score=score)


def tuning_score(tune_sets, hp: dict) -> tuple[float, dict]:
    per = {n: eval_on(d, e, hp) for n, (d, e) in tune_sets.items()}
    mean = float(np.mean([per[n]["score"] for n in per]))
    return mean, per


# --------------------------------------------------------------------------
# sweep grid — all 4 Hi-OnTop HPs
# --------------------------------------------------------------------------

GRID = dict(
    delta_star=[0.40, 0.45, 0.50, 0.5594, 0.62, 0.70, 0.78],
    ctx_window=[2, 3, 4, 5, 6, 8],
    ctx_decay=[0.5, 0.7, 0.9],
    ctx_blend_a=[0.0, 0.3, 0.5, 0.7, 1.0],
)
# v4.1.x canonical config (= v4.1.3 reported-Score config)
DEFAULT_HP = dict(delta_star=0.5594, ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-22_hiontop_hp_sweep")
    args = ap.parse_args()

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_json = exp_dir / "sweep_results.json"

    print("[load] tuning + test sets", flush=True)
    ds711_dia = load_dialogs("dialseg711", "test")
    ds711_emb = load_embs("dialseg711", "test")
    ds711_tune, ds711_test = split_frac(ds711_dia, ds711_emb, 0.30, 0)
    tune_sets = {
        "tiage_train": (load_dialogs("tiage", "train"), load_embs("tiage", "train")),
        "superseg_val": (load_dialogs("superseg", "validation"),
                         load_embs("superseg", "validation")),
        "dialseg711_tune": ds711_tune,
    }
    test_sets = {
        "tiage": (load_dialogs("tiage", "test"), load_embs("tiage", "test")),
        "dialseg711": ds711_test,
        "superseg": (load_dialogs("superseg", "test"), load_embs("superseg", "test")),
    }
    for n, (d, _) in tune_sets.items():
        print(f"  tune/{n}: {len(d)} dialogs", flush=True)
    for n, (d, _) in test_sets.items():
        print(f"  test/{n}: {len(d)} dialogs", flush=True)

    grid = [(ds, m, rho, a)
            for ds in GRID["delta_star"]
            for m in GRID["ctx_window"]
            for rho in GRID["ctx_decay"]
            for a in GRID["ctx_blend_a"]]
    results = []
    if out_json.exists():
        results = json.loads(out_json.read_text()).get("grid", [])
        print(f"[resume] {len(results)} grid rows loaded", flush=True)
    done = {(r["delta_star"], r["ctx_window"], r["ctx_decay"], r["ctx_blend_a"])
            for r in results}

    print(f"\n[sweep] {len(grid)} grid points ({len(done)} done)", flush=True)
    for i, (ds, m, rho, a) in enumerate(grid):
        if (ds, m, rho, a) in done:
            continue
        hp = dict(delta_star=ds, ctx_window=m, ctx_decay=rho, ctx_blend_a=a)
        t0 = time.perf_counter()
        mean, per = tuning_score(tune_sets, hp)
        results.append(dict(
            delta_star=ds, ctx_window=m, ctx_decay=rho, ctx_blend_a=a,
            tuning_score=mean, per_set={n: per[n]["score"] for n in per}))
        if i % 10 == 0 or i == len(grid) - 1:
            out_json.write_text(json.dumps({"grid": results}, indent=2))
        print(f"  [{i+1}/{len(grid)}] δ*={ds} m={m} ρ={rho} a={a}: "
              f"tune={mean:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    out_json.write_text(json.dumps({"grid": results}, indent=2))

    best = max(results, key=lambda r: r["tuning_score"])
    print(f"\n[best] {best}", flush=True)

    # final: default vs best on test sets
    default_test = {ds: eval_on(d, e, DEFAULT_HP) for ds, (d, e) in test_sets.items()}
    best_hp = {k: best[k] for k in ("delta_star", "ctx_window", "ctx_decay", "ctx_blend_a")}
    best_test = {ds: eval_on(d, e, best_hp) for ds, (d, e) in test_sets.items()}
    print("\n[final] default vs swept-best on test:", flush=True)
    for ds in test_sets:
        print(f"  {ds}: default {default_test[ds]['score']:.4f} "
              f"→ best {best_test[ds]['score']:.4f}", flush=True)

    # REPORT
    L = ["# Hi-OnTop full HP sweep — all 4 HPs", "",
         "Hi-OnTop has exactly 4 hyper-parameters, all live. Single grid sweep.",
         "Tuned on tiage/train + superseg/validation + dialseg711 30%-tune; "
         "reported on the 3 held-out test sets. Metric = SuperDialseg Score.", "",
         f"Grid: {len(grid)} points. delta_star={GRID['delta_star']}, "
         f"ctx_window={GRID['ctx_window']}, ctx_decay={GRID['ctx_decay']}, "
         f"ctx_blend_a={GRID['ctx_blend_a']}", "",
         "## Top-15 configs (by tuning Score)", "",
         "| δ* | m | ρ | a | tuning | tiage_tr | superseg_val | dialseg711_tune |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda r: -r["tuning_score"])[:15]:
        ps = r["per_set"]
        L.append(f"| {r['delta_star']} | {r['ctx_window']} | {r['ctx_decay']} | "
                 f"{r['ctx_blend_a']} | **{r['tuning_score']:.4f}** | "
                 f"{ps['tiage_train']:.4f} | {ps['superseg_val']:.4f} | "
                 f"{ps['dialseg711_tune']:.4f} |")
    L += ["", "## default vs swept-best (TEST sets)", "",
          "default = canonical v4.1.x config (δ*=0.5594, m=2, ρ=0.7, a=0.5)", "",
          "| dataset | default Score | swept-best Score | Δ |",
          "|---|---:|---:|---:|"]
    for ds in test_sets:
        d, s = default_test[ds]["score"], best_test[ds]["score"]
        L.append(f"| {ds} | {d:.4f} | {s:.4f} | {s-d:+.4f} |")
    dm = np.mean([default_test[ds]["score"] for ds in test_sets])
    sm = np.mean([best_test[ds]["score"] for ds in test_sets])
    L.append(f"| **mean-3** | {dm:.4f} | {sm:.4f} | {sm-dm:+.4f} |")
    L += ["", "## swept-best HP", "", "```python"]
    for k, v in best_hp.items():
        L.append(f"{k}={v}")
    L += ["```", "",
          "## 한계 / 검증 미해결",
          "- swept-best 는 tuning set 에 fit — test Δ 가 음수면 default 가 robust "
          "(dataset-specific tuning 불필요) 라는 negative result.",
          "- dialseg711 은 train split 없음 → 30/70 seeded split. tune 30% 가 "
          "test 70% 와 disjoint (no leakage).",
          "- canonical default m=2 는 v4.1.3 보고 Score 가 나온 config (decision-log "
          "2026-05-22).", ""]
    (exp_dir / "REPORT.md").write_text("\n".join(L))
    print(f"\nDONE → {exp_dir}/REPORT.md", flush=True)


if __name__ == "__main__":
    main()
