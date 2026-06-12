#!/usr/bin/env python3
"""v4.1.3 full hyper-parameter sweep (segmentation Score).

2-phase sweep with explicit tune/held-out split for datasets lacking train:

- **Phase 1 — interacting grid**: ``(ctx_window, ctx_decay, ctx_blend_a,
  delta_star)``. These four jointly shape the causal-window context vector
  and the boundary threshold, so they interact and are swept as a grid.
- **Phase 2 — OAT**: every remaining HP swept one-at-a-time, holding the
  Phase-1 winner fixed.

Tuning set: ``tiage/train`` + ``superseg/validation`` + ``dialseg711`` tune
split. dialseg711 has no official train split, so its 711-dialog test set is
split 30/70 by a seeded shuffle — the 30% tune split joins the tuning
objective, and the 70% held-out split is the dialseg711 report row.
Final report: ``tiage/test``, ``dialseg711`` held-out split, ``superseg/test``.

Metric: official SuperDialseg Pk/WD (k=auto) + binary F1,
``Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)``. Embeddings = cached
mpnet pkl (``outputs/runs/_misc/sds_emb_*``).

Tuning objective = mean Score over the tuning sets.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
# v4.1.3 full SEM2 form archived (2026-05-22). This sweep probes the full
# HP set (incl. now-dead params), so it imports the archived class.
sys.path.insert(0, str(REPO / "archive" / "legacy_sem_ablation"))
from sem_core_v413 import HiOnTopSegmenterV413  # noqa: E402

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
    """Load cached mpnet embeddings. Tries no-suffix then mpnet-suffix name."""
    for cand in (CACHE / f"sds_emb_{ds}_{split}.pkl",
                 CACHE / f"sds_emb_{ds}_{split}{MPNET_SUFFIX}.pkl"):
        if cand.exists():
            with open(cand, "rb") as fh:
                return pickle.load(fh)
    raise FileNotFoundError(
        f"no cached embedding for {ds}/{split} "
        f"(looked for sds_emb_{ds}_{split}[{MPNET_SUFFIX}].pkl)")


def split_frac(dialogs, embs, tune_frac: float = 0.30, seed: int = 0):
    """Seeded split → (tune, test), each (dialogs, embs).

    tune_frac=0.30: 30% tune / 70% test. The tuning objective averages
    per-set Scores (each set weighted 1/N regardless of size), so a small
    tune split still contributes its full 1/N weight — a larger test split
    therefore gives a more reliable reported number at no tuning cost.
    """
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
# segmenter run
# --------------------------------------------------------------------------

# v4.1.3 default HPs (v4.1.1-inherited). Sweep overrides merge on top.
BASE_HP = dict(
    alpha=1.0, lmda=10.0, beta=0.25, pe_threshold=1.0, cos_threshold=0.9,
    delta_star=0.5594, sigma_delta_c=0.0625,
    ctx_window=3, ctx_decay=0.7, ctx_blend_a=0.5,
    pe_var_sigma0_sq=0.04, pe_var_df0=1.0, pe_prior=1.0,
    f0_min_starts=2, restart_pe_threshold=0.5,
)


def seg_pred(emb: np.ndarray, hp: dict):
    seg = HiOnTopSegmenterV413(dim=emb.shape[1], **hp)
    torch.manual_seed(0)
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
    """Mean Score over the tuning sets. Returns (mean_score, per_set)."""
    per = {}
    for name, (dia, emb) in tune_sets.items():
        per[name] = eval_on(dia, emb, hp)
    mean = float(np.mean([per[n]["score"] for n in per]))
    return mean, per


# --------------------------------------------------------------------------
# sweep spaces
# --------------------------------------------------------------------------

PHASE1_GRID = dict(
    ctx_window=[2, 3, 4, 5, 6, 7, 8],
    ctx_decay=[0.5, 0.7, 0.9],
    ctx_blend_a=[0.3, 0.5, 0.7],
    delta_star=[0.45, 0.52, 0.5594, 0.62, 0.70],
)

PHASE2_OAT = dict(
    alpha=[0.5, 1.0, 2.0, 5.0],
    lmda=[1.0, 5.0, 10.0, 20.0, 50.0],
    beta=[0.1, 0.25, 0.5, 1.0],
    pe_threshold=[0.5, 0.8, 1.0],
    cos_threshold=[0.8, 0.9, 0.95],
    sigma_delta_c=[0.03, 0.0625, 0.125, 0.25],
    pe_var_sigma0_sq=[0.01, 0.04, 0.09],
    pe_var_df0=[0.5, 1.0, 2.0],
    pe_prior=[0.8, 1.0, 1.2],
    # NOTE: f0_min_starts and restart_pe_threshold are EXCLUDED — empirically
    # verified (2026-05-22) to produce byte-identical output for all values
    # ≥2 / any value. The f0/restart/re-entry branch is dead code at v4.1.3
    # defaults (re-entry circular-deadlocks at f0_min_starts≥2; same-topic
    # restart never changes topic_id → never a boundary). f0_min_starts=1
    # would activate re-entry but is known-bad (v4.1.3.md: Score collapse).
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-22_v413_hp_sweep")
    ap.add_argument("--out_json", default=None,
                    help="resumable partial-results JSON")
    args = ap.parse_args()

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else exp_dir / "sweep_results.json"

    # ---- load tuning + test sets ----
    print("[load] tuning + test sets", flush=True)
    # dialseg711: no official train → 30/70 seeded split.
    ds711_dia = load_dialogs("dialseg711", "test")
    ds711_emb = load_embs("dialseg711", "test")
    (ds711_tune, ds711_test) = split_frac(ds711_dia, ds711_emb, tune_frac=0.30, seed=0)

    tune_sets = {
        "tiage_train": (load_dialogs("tiage", "train"), load_embs("tiage", "train")),
        "superseg_val": (load_dialogs("superseg", "validation"),
                         load_embs("superseg", "validation")),
        "dialseg711_tune": ds711_tune,
    }
    for n, (d, e) in tune_sets.items():
        print(f"  tune/{n}: {len(d)} dialogs", flush=True)
    test_sets = {
        "tiage": (load_dialogs("tiage", "test"), load_embs("tiage", "test")),
        "dialseg711": ds711_test,  # held-out split (disjoint from dialseg711_tune)
        "superseg": (load_dialogs("superseg", "test"), load_embs("superseg", "test")),
    }
    for n, (d, e) in test_sets.items():
        print(f"  test/{n}: {len(d)} dialogs", flush=True)

    results = {"phase1": [], "phase2": {}, "base_hp": dict(BASE_HP)}
    if out_json.exists():
        results = json.loads(out_json.read_text())
        print(f"[resume] loaded {len(results['phase1'])} phase-1 rows", flush=True)

    def save():
        out_json.write_text(json.dumps(results, indent=2))

    # ---- Phase 1: interacting grid ----
    grid = [(m, r, a, d)
            for m in PHASE1_GRID["ctx_window"]
            for r in PHASE1_GRID["ctx_decay"]
            for a in PHASE1_GRID["ctx_blend_a"]
            for d in PHASE1_GRID["delta_star"]]
    done_p1 = {(r["ctx_window"], r["ctx_decay"], r["ctx_blend_a"], r["delta_star"])
               for r in results["phase1"]}
    print(f"\n[phase1] {len(grid)} grid points "
          f"({len(done_p1)} already done)", flush=True)
    for i, (m, rho, a, d) in enumerate(grid):
        if (m, rho, a, d) in done_p1:
            continue
        hp = dict(BASE_HP, ctx_window=m, ctx_decay=rho, ctx_blend_a=a, delta_star=d)
        t0 = time.perf_counter()
        mean, per = tuning_score(tune_sets, hp)
        results["phase1"].append(dict(
            ctx_window=m, ctx_decay=rho, ctx_blend_a=a, delta_star=d,
            tuning_score=mean,
            per_set={n: per[n]["score"] for n in per},
        ))
        if i % 5 == 0 or i == len(grid) - 1:
            save()
        print(f"  [{i+1}/{len(grid)}] m={m} ρ={rho} a={a} δ*={d}: "
              f"tune={mean:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    save()

    best_p1 = max(results["phase1"], key=lambda r: r["tuning_score"])
    print(f"\n[phase1 best] {best_p1}", flush=True)
    p1_hp = dict(BASE_HP, ctx_window=best_p1["ctx_window"],
                 ctx_decay=best_p1["ctx_decay"], ctx_blend_a=best_p1["ctx_blend_a"],
                 delta_star=best_p1["delta_star"])
    results["phase1_best"] = best_p1

    # ---- Phase 2: OAT ----
    print(f"\n[phase2] OAT over {len(PHASE2_OAT)} HPs", flush=True)
    for hp_name, values in PHASE2_OAT.items():
        if hp_name in results["phase2"]:
            continue
        rows = []
        for v in values:
            hp = dict(p1_hp, **{hp_name: v})
            t0 = time.perf_counter()
            mean, per = tuning_score(tune_sets, hp)
            rows.append(dict(value=v, tuning_score=mean,
                             per_set={n: per[n]["score"] for n in per}))
            print(f"  {hp_name}={v}: tune={mean:.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        results["phase2"][hp_name] = rows
        save()

    # ---- assemble final best config ----
    final_hp = dict(p1_hp)
    for hp_name, rows in results["phase2"].items():
        best_v = max(rows, key=lambda r: r["tuning_score"])
        # only adopt if it beats the phase-1-best baseline value
        base_v = BASE_HP[hp_name]
        base_row = next((r for r in rows if r["value"] == base_v), None)
        if base_row is None or best_v["tuning_score"] > base_row["tuning_score"]:
            final_hp[hp_name] = best_v["value"]
    results["final_hp"] = final_hp
    save()

    # ---- final eval on test sets: default vs swept ----
    print(f"\n[final] default vs swept on test sets", flush=True)
    default_hp = dict(BASE_HP)
    final_test, default_test = {}, {}
    for ds, (dia, emb) in test_sets.items():
        default_test[ds] = eval_on(dia, emb, default_hp)
        final_test[ds] = eval_on(dia, emb, final_hp)
        print(f"  {ds}: default Score={default_test[ds]['score']:.4f} "
              f"→ swept Score={final_test[ds]['score']:.4f}", flush=True)
    results["test_default"] = default_test
    results["test_swept"] = final_test
    save()

    # ---- write REPORT ----
    _write_report(exp_dir, results)
    print(f"\nDONE → {exp_dir}/REPORT.md", flush=True)


def _write_report(exp_dir: Path, R: dict) -> None:
    L = ["# v4.1.3 full HP sweep — segmentation Score", "",
         "2-phase sweep (interacting grid + OAT), tuned on "
         "`tiage/train` + `superseg/validation` + a seeded `dialseg711` tune split.",
         "Metric: Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD), official "
         "SuperDialseg Pk/WD. Encoder = mpnet (cached).", ""]
    bp = R.get("phase1_best", {})
    L += ["## Phase 1 — interacting grid (ctx_window × ctx_decay × ctx_blend_a × δ*)", "",
          f"Grid points: {len(R['phase1'])}. **Best**: "
          f"ctx_window={bp.get('ctx_window')}, ctx_decay={bp.get('ctx_decay')}, "
          f"ctx_blend_a={bp.get('ctx_blend_a')}, δ*={bp.get('delta_star')} "
          f"(tuning Score {bp.get('tuning_score', 0):.4f})", ""]
    top = sorted(R["phase1"], key=lambda r: -r["tuning_score"])[:10]
    set_names = list(top[0]["per_set"].keys()) if top else []
    L += ["### Top-10 grid configs", "",
          "| m | ρ | a | δ* | tuning Score | " + " | ".join(set_names) + " |",
          "|---:|---:|---:|---:|---:|" + "---:|" * len(set_names)]
    for r in top:
        cells = " | ".join(f"{r['per_set'][n]:.4f}" for n in set_names)
        L.append(f"| {r['ctx_window']} | {r['ctx_decay']} | {r['ctx_blend_a']} | "
                 f"{r['delta_star']} | **{r['tuning_score']:.4f}** | {cells} |")
    L.append("")
    L += ["## Phase 2 — OAT (holding Phase-1 best)", ""]
    for hp_name, rows in R.get("phase2", {}).items():
        best = max(rows, key=lambda r: r["tuning_score"])
        sn = list(rows[0]["per_set"].keys()) if rows else []
        L.append(f"### {hp_name} → best = {best['value']} "
                 f"(tuning {best['tuning_score']:.4f})")
        L.append("")
        L.append("| value | tuning Score | " + " | ".join(sn) + " |")
        L.append("|---:|---:|" + "---:|" * len(sn))
        for r in rows:
            mark = " ⭐" if r["value"] == best["value"] else ""
            cells = " | ".join(f"{r['per_set'][n]:.4f}" for n in sn)
            L.append(f"| {r['value']}{mark} | {r['tuning_score']:.4f} | {cells} |")
        L.append("")
    # final
    L += ["## Final config — default vs swept (TEST sets)", "",
          "| dataset | default Score | swept Score | Δ |",
          "|---|---:|---:|---:|"]
    dt, st = R.get("test_default", {}), R.get("test_swept", {})
    for ds in ("tiage", "dialseg711", "superseg"):
        if ds in dt and ds in st:
            d, s = dt[ds]["score"], st[ds]["score"]
            L.append(f"| {ds} | {d:.4f} | {s:.4f} | {s-d:+.4f} |")
    if dt and st:
        dm = np.mean([dt[ds]["score"] for ds in dt])
        sm = np.mean([st[ds]["score"] for ds in st])
        L.append(f"| **mean-3** | {dm:.4f} | {sm:.4f} | {sm-dm:+.4f} |")
    L += ["", "## Final HP config", "", "```python"]
    for k, v in R.get("final_hp", {}).items():
        L.append(f"{k}={v}")
    L += ["```", "",
          "## 한계 / 검증 미해결",
          "- Phase-2 OAT 는 interaction 무시 (Phase-1 best 고정 후 1축씩). "
          "Phase-1 의 4 HP 외 상호작용은 미탐색.",
          "- dialseg711 은 official train split 이 없어 test dialogs 를 seeded "
          "30% tune / 70% held-out 으로 나눔. 따라서 `dialseg711` test row 는 "
          "full official test 가 아니라 held-out split 이며, literature-comparable "
          "full-test 숫자로 직접 인용하면 안 됨.",
          "- superseg 는 validation (1322) 으로 tune, train(6948) 미사용 "
          "(인코딩 비용). validation = test 와 동일 크기라 대표성 OK.",
          ""]
    (exp_dir / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
