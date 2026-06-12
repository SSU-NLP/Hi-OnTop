#!/usr/bin/env python3
"""v4.3.1-exp η sweep — DialoGPT-small surprisal 이 δ_model 자리.

설정:
- mpnet channel = v4.1.1 default HP (m=2, ρ=0.7, a=0.5, δ*=0.5594)
- LM channel    = DialoGPT-small mean-token NLL, causal window m=5
                  (precompute_v431_nll.py 산물 재사용)
- η = mpnet weight, (1-η) = LM surprisal weight
- δ* = 0.5594 (mpnet 그대로, raw NLL scale mismatch caveat 존재)

Sanity:
- η=1.0 → v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상)
- η<1.0 → LM surprisal 활성. raw NLL ≈ 2~8 (vs δ_adj ≈ 0.4~0.6)
  이므로 η 매우 작을 때 δ_eff 가 NLL 의존성으로 급팽창 → 과분절 위험.

Datasets: tiage, dialseg711, superseg test.
- mpnet embedding cache: outputs/runs/_misc/sds_emb_{ds}_test_{mpnet}.pkl
- LM NLL cache:          outputs/runs/_misc/sds_nll_{ds}_test_{dialogpt}.pkl
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
from hi_ontop.sem_core_v431_exp import HiOnTopSegmenterV431Exp  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
ENC_TOPIC = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
LM_NAME = "microsoft/DialoGPT-small"

# v4.1.1 mpnet TIAGE-train best
MPNET_M, MPNET_RHO, MPNET_A, MPNET_DSTAR = 2, 0.7, 0.5, 0.5594
LM_CTX_WINDOW = 5

V411_BASELINE = {"tiage": 0.4675, "dialseg711": 0.5897, "superseg": 0.4631}


def load_dialogs(dataset, split):
    raw = json.loads((SDS / dataset / f"segmentation_file_{split}.json").read_text())
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


def official_pk_wd(yt, yp):
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


def load_cache_emb(ds, split, encoder_name):
    safe = encoder_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{ds}_{split}_{safe}.pkl"
    if not cp.exists():
        raise SystemExit(f"emb cache missing: {cp}")
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def load_cache_nll(ds, split, lm_name):
    safe = lm_name.replace("/", "_")
    cp = CACHE / f"sds_nll_{ds}_{split}_{safe}.pkl"
    if not cp.exists():
        raise SystemExit(
            f"nll cache missing: {cp}\n  run: uv run python scripts/precompute_v431_nll.py"
        )
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def run_v431(embs_topic, nlls_lm, eta):
    out = []
    for et, nll in zip(embs_topic, nlls_lm):
        seg = HiOnTopSegmenterV431Exp(
            dim=et.shape[1], alpha=1.0, lmda=10.0,
            delta_star=MPNET_DSTAR,
            ctx_window=MPNET_M, ctx_decay=MPNET_RHO, ctx_blend_a=MPNET_A,
            eta_prev=eta,
            lm_name=LM_NAME,
            context_window_lm=LM_CTX_WINDOW,
        )
        torch.manual_seed(0)
        ids = []
        for i, st in enumerate(et):
            nll_t = float(nll[i]) if i < len(nll) else 0.0
            k, _ = seg.assign_pair(st.astype(np.float64), nll_t)
            ids.append(k)
        yp = [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]
        out.append(yp)
    return out


def eval_metrics(dialogs, preds):
    pks, wds, g, p = [], [], [], []
    for (_, yt), yp in zip(dialogs, preds):
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    return dict(pk=pk_m, wd=wd_m, f1=f1, score=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-21_v431_eta_sweep")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--etas", nargs="+", type=float,
                    default=[1.0, 0.99, 0.95, 0.9, 0.75, 0.5])
    args = ap.parse_args()

    print("[load] datasets + caches")
    data = {}
    for ds in args.datasets:
        dia = load_dialogs(ds, "test")
        et = load_cache_emb(ds, "test", ENC_TOPIC)
        nll = load_cache_nll(ds, "test", LM_NAME)
        # quick NLL stat
        all_vals = np.concatenate(nll)
        data[ds] = (dia, et, nll)
        print(f"  {ds}: n_dial={len(dia)}, nll mean={all_vals.mean():.3f}, "
              f"std={all_vals.std():.3f}")

    print(f"\n[sweep] η ∈ {args.etas}")
    grid = {}
    for eta in args.etas:
        for ds in args.datasets:
            dia, et, nll = data[ds]
            t0 = time.perf_counter()
            preds = run_v431(et, nll, eta)
            m = eval_metrics(dia, preds)
            wall = time.perf_counter() - t0
            grid[(eta, ds)] = (m, wall)
            print(f"  η={eta:.3f}  {ds:12s}  Score={m['score']:.4f}  "
                  f"F1={m['f1']:.4f}  Pk={m['pk']:.4f}  WD={m['wd']:.4f}  ({wall:.0f}s)")

    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        f"# v4.3.1-exp η sweep — DialoGPT-small surprisal 이 δ_model 자리",
        "",
        "**Setup**:",
        f"- mpnet channel (parent): `{ENC_TOPIC}`, "
        f"m={MPNET_M}, ρ={MPNET_RHO}, a={MPNET_A}, δ*={MPNET_DSTAR} (v4.1.1 TIAGE-train).",
        f"- LM channel (δ_model slot): `{LM_NAME}` mean-token NLL, "
        f"causal context window m={LM_CTX_WINDOW}, raw EOS-concat format.",
        "- δ_eff² = η·δ_adj² + (1−η)·δ_model² (v4.1.1 식 그대로, RNN PE 자리에 LM surprisal).",
        "- δ* = 0.5594 (mpnet 그대로 — **raw NLL scale (≈2~8) mismatch caveat**).",
        "- Sanity: η=1.0 == v4.1.1 mpnet-only (0.4675 / 0.5897 / 0.4631 매치 예상).",
        "",
        "## Score matrix (행=η, 열=dataset, mean = row mean)",
        "",
        "| η | " + " | ".join(args.datasets) + " | mean |",
        "|---:|" + "---:|" * (len(args.datasets) + 1),
    ]
    for eta in args.etas:
        scores = [grid[(eta, ds)][0]["score"] for ds in args.datasets]
        mean = sum(scores) / len(scores)
        row = [f"{eta:.3f}"] + [f"{s:.4f}" for s in scores] + [f"{mean:.4f}"]
        L.append("| " + " | ".join(row) + " |")

    L += [
        "",
        "## Detailed metrics",
        "",
        "| η | dataset | Score ↑ | F1 ↑ | Pk ↓ | WD ↓ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for eta in args.etas:
        for ds in args.datasets:
            m, _ = grid[(eta, ds)]
            L.append(
                f"| {eta:.3f} | {ds} | {m['score']:.4f} | {m['f1']:.4f} | "
                f"{m['pk']:.4f} | {m['wd']:.4f} |"
            )

    L += [
        "",
        "## Per-dataset best",
        "",
        "| dataset | best η | Score | vs v4.1.1 (η=1.0) |",
        "|---|---:|---:|---:|",
    ]
    for ds in args.datasets:
        best_eta, best_score = max(
            ((eta, grid[(eta, ds)][0]["score"]) for eta in args.etas),
            key=lambda x: x[1],
        )
        delta = best_score - V411_BASELINE.get(ds, 0.0)
        L.append(f"| {ds} | {best_eta:.3f} | {best_score:.4f} | {delta:+.4f} |")

    L += [
        "",
        "## 해석 / 판정",
        "",
        "(채울 것 — η sweep 결과 본 후 작성)",
        "",
        "## 한계 / 검증 미해결",
        "- **Raw scale mismatch**: NLL ≈ 2~8 (nats/token) vs δ_adj ≈ 0.4~0.6.",
        "  η<1 에서 δ_eff 가 NLL 의존성으로 급팽창 → 같은 δ\\* 에서 과분절 위험.",
        "- **δ\\* re-calibration 미수행** — η 별 TIAGE-train δ\\* 후속.",
        "- **DialoGPT-small** = 117M, Reddit 기반. domain-shifted (TIAGE 자연 대화,",
        "  Dialseg711 정형 도메인, SuperDialseg gov FAQ). surprisal 의 절대 scale 이",
        "  dataset 마다 다를 가능성.",
        "- **Single seed, single LM** — multi-seed variance / model-size scan 미수행.",
        "- **Causal context window m=5** — m sweep 미수행.",
        "- **EOS-concat 포맷** vs ChatML 비교 미수행 (학습 분포 적합성 명목 우선).",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
