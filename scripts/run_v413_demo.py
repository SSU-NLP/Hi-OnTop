#!/usr/bin/env python3
"""v4.1.3 demo — graded boundary score visualization.

각 test 데이터셋에서 첫 N개 대화를 v4.1.3 으로 segment 한 뒤,
- per-turn graded_score (δ_eff / δ*) sequence
- boundary strength band histogram (Ben-Yakov & Henson 2018)
- per-band precision (graded score 가 진짜 calibrated 인지)

paper 의 graded boundary contribution 정량 표현.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
# v4.1.3 full SEM2 form archived to archive/legacy_sem_ablation/ (2026-05-22).
# This historical demo exercises the full machinery, so it imports from there.
sys.path.insert(0, str(REPO / "archive" / "legacy_sem_ablation"))
from sem_core_v413 import HiOnTopSegmenterV413  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

ENC = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
MPNET_M, MPNET_RHO, MPNET_A, MPNET_DSTAR = 2, 0.7, 0.5, 0.5594


def load_dialogs(dataset, split):
    raw = json.loads((SDS / dataset / f"segmentation_file_{split}.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if yt: yt[-1] = 0
        if len(utts) >= 2:
            out.append((utts, yt))
    return out


def load_cache(ds, split, enc_name=ENC):
    safe = enc_name.replace("/", "_")
    cp = CACHE / f"sds_emb_{ds}_{split}_{safe}.pkl"
    if not cp.exists():
        cp = CACHE / f"sds_emb_{ds}_{split}.pkl"
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def band_of(score):
    if score < 0.7: return "very_weak"
    if score < 1.0: return "weak"
    if score < 1.3: return "normal"
    return "strong"


def render_dialog(idx, utts, yt, history):
    L = [
        f"### Dialog {idx} ({len(utts)} turns, GT boundaries: {sum(yt)})",
        "",
        "| turn | topic | GT | pred | graded | band |",
        "|---:|---:|:---:|:---:|---:|---|",
    ]
    for h, gt in zip(history, yt):
        t = int(h["turn"])
        k = int(h["topic_id"])
        bnd = "**B**" if h["is_boundary"] else " "
        gt_b = "**B**" if gt else " "
        gs = float(h["graded_score"])
        b = band_of(gs) if t > 0 else "—"
        mark = ""
        if gs >= 1.3:
            mark = " 💪"
        elif gs >= 1.0:
            mark = ""
        L.append(f"| {t} | {k} | {gt_b} | {bnd} | {gs:.3f} | {b}{mark} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711"])
    ap.add_argument("--n-dialogs", type=int, default=2)
    ap.add_argument("--name", default="2026-05-21_v413_demo")
    args = ap.parse_args()

    sections = []
    summary = []

    for ds in args.datasets:
        print(f"\n[load] {ds}/test")
        dialogs = load_dialogs(ds, "test")
        embs = load_cache(ds, "test")
        print(f"  n_dial={len(dialogs)}")

        # Run V413 on all dialogs
        all_records = []
        per_band_conf = {b: {"tp": 0, "fp": 0, "fn": 0} for b in
                         ("very_weak", "weak", "normal", "strong")}
        total_strength = {"very_weak": 0, "weak": 0, "normal": 0, "strong": 0}
        all_yt, all_yp = [], []
        for i, ((u, yt), e) in enumerate(zip(dialogs, embs)):
            seg = HiOnTopSegmenterV413(
                dim=e.shape[1], alpha=1.0, lmda=10.0,
                delta_star=MPNET_DSTAR,
                ctx_window=MPNET_M, ctx_decay=MPNET_RHO, ctx_blend_a=MPNET_A,
            )
            torch.manual_seed(0)
            for s in e:
                seg.assign(s.astype(np.float64))
            hist = seg.history()
            all_records.append({"i": i, "history": hist, "utts": u, "yt": yt})
            for k, v in seg.boundary_strength().items():
                total_strength[k] += v
            # build yp from topic_id changes
            ids = [h["topic_id"] for h in hist]
            yp = [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]
            # per-band confusion (using boundary AFTER turn i)
            for j in range(len(hist) - 1):
                gs = float(hist[j + 1]["graded_score"])
                b = band_of(gs)
                gt_j = yt[j]
                pred_j = yp[j]
                if pred_j and gt_j:
                    per_band_conf[b]["tp"] += 1
                elif pred_j and not gt_j:
                    per_band_conf[b]["fp"] += 1
                elif gt_j:
                    per_band_conf[b]["fn"] += 1
            all_yt += yt
            all_yp += yp

        # Aggregate metrics
        f1 = float(f1_score(all_yt, all_yp, zero_division=0))
        pks, wds = [], []
        for r in all_records:
            yt = r["yt"]
            ids = [h["topic_id"] for h in r["history"]]
            yp = [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]
            n_seg = sum(yt) + 1
            k = max(2, int(round(len(yt) / n_seg / 2)))
            ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
            pks.append(float(_nltk_pk(ts, ps, k=k)))
            wds.append(float(_nltk_wd(ts, ps, k=k)))
        pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
        score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)

        summary.append({
            "dataset": ds,
            "n_dialogs": len(all_records),
            "n_turns": sum(len(r["history"]) for r in all_records),
            "n_boundaries_pred": sum(all_yp),
            "n_boundaries_gt": sum(all_yt),
            "score": score, "f1": f1, "pk": pk_m, "wd": wd_m,
            "strength": total_strength,
            "per_band_conf": per_band_conf,
        })
        print(f"  Score={score:.4f}  F1={f1:.4f}  Pk={pk_m:.4f}  WD={wd_m:.4f}")
        print(f"  strength: {total_strength}")

        # Pick example dialogs: longest with diverse banded scores
        picked = sorted(
            all_records,
            key=lambda r: (
                -sum(1 for h in r["history"]
                     if band_of(float(h["graded_score"])) in ("normal", "strong")),
                -len(r["history"]),
            ),
        )[:args.n_dialogs]

        sections.append(f"\n## {ds}\n")
        sections.append(f"- **Score={score:.4f}** (= v4.1.1 baseline, algorithm 무변경)")
        sections.append(f"- F1={f1:.4f}, Pk={pk_m:.4f}, WD={wd_m:.4f}")
        sections.append(f"- 총 turns: {summary[-1]['n_turns']}")
        sections.append(f"- boundary 예측: {summary[-1]['n_boundaries_pred']}, GT: {summary[-1]['n_boundaries_gt']}")
        sections.append("")
        sections.append("### Per-band precision (graded score discriminator 검증)")
        sections.append("")
        sections.append("| band | n_pred_bnd | TP | FP | precision |")
        sections.append("|---|---:|---:|---:|---:|")
        for b in ("very_weak", "weak", "normal", "strong"):
            c = per_band_conf[b]
            n_pred = c["tp"] + c["fp"]
            prec = c["tp"] / max(1, n_pred)
            sections.append(f"| {b} | {n_pred} | {c['tp']} | {c['fp']} | {prec:.3f} |")
        sections.append("")
        sections.append("### Example dialogs")
        sections.append("")
        for r in picked:
            sections.append(render_dialog(r["i"], r["utts"], r["yt"], r["history"]))
            sections.append("")

    # REPORT
    out = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# v4.1.3 demo — graded boundary score",
        "",
        "**Algorithm = v4.1.1 identical**. v4.1.3 adds `graded_score = δ_eff / δ*`",
        "per-turn output (Ben-Yakov & Henson 2018 hippocampal graded boundary profile mapping).",
        "",
        "Boundary strength bands:",
        "- `< 0.7` very weak (downstream consumer 보류 권장)",
        "- `0.7-1.0` weak (repeat 우세)",
        "- `1.0-1.3` normal (정상 경계)",
        "- `≥ 1.3` strong (즉시 commit 권장)",
        "",
        f"Encoder: `{ENC}` (TIAGE-train HP: m={MPNET_M}, ρ={MPNET_RHO}, a={MPNET_A}, δ*={MPNET_DSTAR}).",
        "",
        "## 데이터셋별 통계",
        "",
        "| dataset | n_dial | n_turns | n_bnd_pred | n_bnd_gt | Score | F1 | very_weak | weak | normal | strong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        L.append(
            f"| {s['dataset']} | {s['n_dialogs']} | {s['n_turns']} | "
            f"{s['n_boundaries_pred']} | {s['n_boundaries_gt']} | "
            f"{s['score']:.4f} | {s['f1']:.4f} | "
            f"{s['strength']['very_weak']} | {s['strength']['weak']} | "
            f"{s['strength']['normal']} | {s['strength']['strong']} |"
        )
    L += sections
    out.write_text("\n".join(L) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
