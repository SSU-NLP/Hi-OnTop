#!/usr/bin/env python3
"""v4.1.3 graded_score / re-entry 분석 — false-positive bucket + case study.

질문:
1. 86-94% re-entry 중 얼마나 GT 와 일치하는가? (TP / FP 분리)
2. graded_score band 별 precision (boundary 신호의 calibration)
3. paper-quality case study (TP / FP 각 예시 2~3)

Boundary 정의 (eval 규약):
- y_p[i] = 1 if topic_id[i] != topic_id[i+1]  (i 와 i+1 사이 경계)
- y_t[i] = segmentation_label at turn i; last turn forced 0
- → y_t[i] 가 "i 이후 경계" 의미 (SuperDialseg 규약)
- 따라서 history[i+1].is_boundary 와 y_t[i] 가 같은 위치를 가리킴
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
# v4.1.3 full SEM2 form archived (2026-05-22). Re-entry analysis needs the
# full f0/restart machinery, so it imports the archived class.
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
        if yt:
            yt[-1] = 0
        if len(utts) >= 2:
            out.append((utts, yt))
    return out


def load_cache(ds, split):
    safe = ENC.replace("/", "_")
    cp = CACHE / f"sds_emb_{ds}_{split}_{safe}.pkl"
    if not cp.exists():
        cp = CACHE / f"sds_emb_{ds}_{split}.pkl"
    with open(cp, "rb") as fh:
        return pickle.load(fh)


def band_of(score: float) -> str:
    if score < 0.7:
        return "very_weak"
    if score < 1.0:
        return "weak"
    if score < 1.3:
        return "normal"
    return "strong"


def analyze_dialog(utts, yt, emb):
    """Run V413, classify each pred boundary as TP / FP and bucket by band."""
    seg = HiOnTopSegmenterV413(
        dim=emb.shape[1], alpha=1.0, lmda=10.0,
        delta_star=MPNET_DSTAR,
        ctx_window=MPNET_M, ctx_decay=MPNET_RHO, ctx_blend_a=MPNET_A,
        # f0_min_starts=1 default
    )
    torch.manual_seed(0)
    for s in emb:
        seg.assign(s.astype(np.float64))
    hist = seg.history()

    # boundary 위치 매핑: history[i+1].is_boundary 와 yt[i] 가 같은 위치 비교.
    # 즉 transition_t (= boundary between turn t and t+1) → 마지막 turn 제외.
    records = []
    for i in range(len(hist) - 1):
        gt = int(yt[i])
        h_next = hist[i + 1]
        pb = bool(h_next["is_boundary"])
        is_re = bool(h_next["is_reentry"])
        is_slr = bool(h_next.get("is_same_label_restart", False))
        is_ctr = bool(h_next.get("is_cross_topic_reentry", False))
        gs = float(h_next["graded_score"])
        records.append({
            "transition": i,
            "topic_id_before": int(hist[i]["topic_id"]),
            "topic_id_after": int(h_next["topic_id"]),
            "gt": gt,
            "pred": int(pb),
            "is_reentry": int(is_re),
            "is_same_label_restart": int(is_slr),
            "is_cross_topic_reentry": int(is_ctr),
            "graded_score": gs,
            "band": band_of(gs),
            "utt_before": utts[i],
            "utt_after": utts[i + 1],
        })
    return records


def aggregate(ds_records):
    """Per-band confusion + re-entry TP/FP (전체 / same_label / cross_topic 각각)."""
    by_band = defaultdict(lambda: {"n": 0, "pred_bnd": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
    re_stats = {
        "total": {"tp": 0, "fp": 0},
        "same_label": {"tp": 0, "fp": 0},
        "cross_topic": {"tp": 0, "fp": 0},
    }
    for d_idx, recs in ds_records:
        for r in recs:
            b = r["band"]
            by_band[b]["n"] += 1
            if r["pred"]:
                by_band[b]["pred_bnd"] += 1
                if r["gt"]:
                    by_band[b]["tp"] += 1
                else:
                    by_band[b]["fp"] += 1
                if r["is_reentry"]:
                    key = "tp" if r["gt"] else "fp"
                    re_stats["total"][key] += 1
                    if r["is_same_label_restart"]:
                        re_stats["same_label"][key] += 1
                    elif r["is_cross_topic_reentry"]:
                        re_stats["cross_topic"][key] += 1
            else:
                if r["gt"]:
                    by_band[b]["fn"] += 1
                else:
                    by_band[b]["tn"] += 1
    return by_band, re_stats


def find_examples(ds_records, kind: str, max_examples=2):
    """Pick example dialogs for case study.

    kind:
      'reentry_tp_rich'  — many re-entries that match GT
      'reentry_fp_rich'  — many re-entries with NO GT boundary nearby (false alarms)
      'strong_boundary_match' — strong band predictions matching GT
    """
    scored = []
    for d_idx, recs in ds_records:
        tp_re = sum(1 for r in recs if r["is_reentry"] and r["gt"])
        fp_re = sum(1 for r in recs if r["is_reentry"] and not r["gt"])
        strong_match = sum(1 for r in recs if r["band"] == "strong" and r["pred"] and r["gt"])
        if kind == "reentry_tp_rich":
            score = tp_re - 0.3 * fp_re
        elif kind == "reentry_fp_rich":
            score = fp_re - 0.3 * tp_re
        elif kind == "strong_boundary_match":
            score = strong_match
        else:
            score = 0
        # prefer mid-length dialogs (not too short)
        if 8 <= len(recs) + 1 <= 25:
            score += 0.5
        scored.append((score, d_idx, recs))
    scored.sort(key=lambda x: -x[0])
    return scored[:max_examples]


def render_dialog(d_idx, recs):
    L = [f"### Dialog {d_idx} ({len(recs) + 1} turns, GT boundaries: {sum(r['gt'] for r in recs)})", ""]
    L.append("| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |")
    L.append("|---:|:---:|:---:|:---:|---:|---|---|---|")
    for r in recs:
        pred_mark = "**B**" if r["pred"] else " "
        gt_mark = "**B**" if r["gt"] else " "
        re_mark = "🔁" if r["is_reentry"] else ""
        tp_fp = ""
        if r["pred"] and r["gt"]:
            tp_fp = " ✓"
        elif r["pred"] and not r["gt"]:
            tp_fp = " ✗"
        elif r["gt"]:
            tp_fp = " (miss)"
        ub = (r["utt_before"][:35] + "…") if len(r["utt_before"]) > 35 else r["utt_before"]
        ua = (r["utt_after"][:35] + "…") if len(r["utt_after"]) > 35 else r["utt_after"]
        L.append(
            f"| {r['transition']} | {pred_mark}{tp_fp} | {gt_mark} | {re_mark} | "
            f"{r['graded_score']:.3f} | {r['band']} | "
            f"{r['topic_id_before']}→{r['topic_id_after']} | "
            f"`{ub}` → `{ua}` |"
        )
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["tiage", "dialseg711"])
    ap.add_argument("--name", default="2026-05-21_v413_reentry_analysis")
    args = ap.parse_args()

    out_md = REPO / "outputs" / "experiments" / args.name / "REPORT.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    overall = []

    for ds in args.datasets:
        print(f"\n[load] {ds}/test")
        dialogs = load_dialogs(ds, "test")
        embs = load_cache(ds, "test")
        ds_records = []
        for i, ((u, yt), e) in enumerate(zip(dialogs, embs)):
            recs = analyze_dialog(u, yt, e)
            ds_records.append((i, recs))

        by_band, re_stats = aggregate(ds_records)
        n_total_trans = sum(b["n"] for b in by_band.values())
        n_pred_bnd = sum(b["pred_bnd"] for b in by_band.values())
        n_gt_bnd = sum(b["tp"] + b["fn"] for b in by_band.values())

        def _prec(d):
            tot = d["tp"] + d["fp"]
            return (d["tp"] / max(1, tot), tot)

        re_prec_total, re_total = _prec(re_stats["total"])
        re_prec_slr, re_total_slr = _prec(re_stats["same_label"])
        re_prec_ctr, re_total_ctr = _prec(re_stats["cross_topic"])

        overall.append({
            "ds": ds, "n_dial": len(dialogs), "n_trans": n_total_trans,
            "n_gt": n_gt_bnd, "n_pred": n_pred_bnd,
            "re_total": re_total, "re_tp": re_stats["total"]["tp"], "re_fp": re_stats["total"]["fp"],
            "re_precision": re_prec_total,
            "slr_total": re_total_slr, "slr_tp": re_stats["same_label"]["tp"],
            "slr_fp": re_stats["same_label"]["fp"], "slr_prec": re_prec_slr,
            "ctr_total": re_total_ctr, "ctr_tp": re_stats["cross_topic"]["tp"],
            "ctr_fp": re_stats["cross_topic"]["fp"], "ctr_prec": re_prec_ctr,
        })
        print(f"  trans={n_total_trans}  GT_bnd={n_gt_bnd}  pred_bnd={n_pred_bnd}")
        print(f"  total re-entries={re_total}  TP={re_stats['total']['tp']}  FP={re_stats['total']['fp']}  prec={re_prec_total:.3f}")
        print(f"  same_label_restart={re_total_slr}  TP={re_stats['same_label']['tp']}  FP={re_stats['same_label']['fp']}  prec={re_prec_slr:.3f}")
        print(f"  cross_topic_reentry={re_total_ctr}  TP={re_stats['cross_topic']['tp']}  FP={re_stats['cross_topic']['fp']}  prec={re_prec_ctr:.3f}")

        # per-band table
        sections.append(f"\n## {ds}\n")
        sections.append("### Per-band confusion")
        sections.append("")
        sections.append("| band | n_trans | n_pred_bnd | TP | FP | precision | recall (of band) |")
        sections.append("|---|---:|---:|---:|---:|---:|---:|")
        for b in ("very_weak", "weak", "normal", "strong"):
            d = by_band.get(b, {"n": 0, "pred_bnd": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
            prec = d["tp"] / max(1, d["pred_bnd"])
            # recall within this band: TP / (TP + FN where FN is GT positives missed in this band)
            rec_band = d["tp"] / max(1, d["tp"] + d["fn"])
            sections.append(
                f"| {b} | {d['n']} | {d['pred_bnd']} | {d['tp']} | {d['fp']} | "
                f"{prec:.3f} | {rec_band:.3f} |"
            )
        sections.append("")
        sections.append("### Re-entry precision (sub-type 분리)")
        sections.append("")
        sections.append(f"- **Total re-entries**: {re_total}")
        sections.append(f"  - TP: **{re_stats['total']['tp']}** ({re_stats['total']['tp']/max(1,re_total):.1%})")
        sections.append(f"  - FP: **{re_stats['total']['fp']}** ({re_stats['total']['fp']/max(1,re_total):.1%})")
        sections.append("")
        sections.append(f"- **same_label_restart** (V411 _is_restart 경로, topic_id 유지): {re_total_slr}")
        sections.append(f"  - TP: {re_stats['same_label']['tp']} / FP: {re_stats['same_label']['fp']} / **prec {re_prec_slr:.3f}**")
        sections.append("")
        sections.append(f"- **cross_topic_reentry** (옛 topic 으로 복귀, 진짜 non-linear): {re_total_ctr}")
        sections.append(f"  - TP: {re_stats['cross_topic']['tp']} / FP: {re_stats['cross_topic']['fp']} / **prec {re_prec_ctr:.3f}**")
        sections.append("")

        # Case studies — TP-rich dialog
        sections.append("### Case study — TP-rich (schema reinstatement 성공 예시)")
        sections.append("")
        for score, d_idx, recs in find_examples(ds_records, "reentry_tp_rich", max_examples=2):
            sections.append(render_dialog(d_idx, recs))
            sections.append("")

        # Case studies — FP-rich dialog (paper false-alarm 예시)
        sections.append("### Case study — FP-rich (false re-entry, generic opener 등)")
        sections.append("")
        for score, d_idx, recs in find_examples(ds_records, "reentry_fp_rich", max_examples=2):
            sections.append(render_dialog(d_idx, recs))
            sections.append("")

    # Top-level summary
    L = [
        "# v4.1.3 graded_score / re-entry 분석 — false-positive bucket + case study",
        "",
        "**Setup**:",
        f"- v4.1.3 default ({{f0_min_starts=1, m=2, ρ=0.7, a=0.5, δ*={MPNET_DSTAR}}})",
        f"- Encoder: `{ENC}`",
        "- Boundary 매핑: history[i+1].is_boundary ↔ y_t[i] (i,i+1 사이 경계)",
        "- Re-entry sub-types:",
        "  - `same_label_restart`: boundary AND topic_id == prev_k (V411 _is_restart 경로)",
        "  - `cross_topic_reentry`: boundary AND topic_id != prev_k AND counts[new]>1 (옛 topic 복귀)",
        "  - `is_reentry` = OR of the two (backward compat)",
        "",
        "## Overall — re-entry precision per dataset (sub-type breakdown)",
        "",
        "| dataset | n_trans | GT_bnd | pred_bnd | total re | same_label_restart (TP/FP/prec) | cross_topic_reentry (TP/FP/prec) | total re-prec |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for o in overall:
        L.append(
            f"| {o['ds']} | {o['n_trans']} | {o['n_gt']} | {o['n_pred']} | "
            f"{o['re_total']} | "
            f"{o['slr_tp']}/{o['slr_fp']}/{o['slr_prec']:.3f} (n={o['slr_total']}) | "
            f"{o['ctr_tp']}/{o['ctr_fp']}/{o['ctr_prec']:.3f} (n={o['ctr_total']}) | "
            f"**{o['re_precision']:.3f}** |"
        )
    L += sections
    L += [
        "",
        "## 해석",
        "",
        "(자동 채울 것)",
    ]
    out_md.write_text("\n".join(L) + "\n")
    print(f"\n→ {out_md}")


if __name__ == "__main__":
    main()
