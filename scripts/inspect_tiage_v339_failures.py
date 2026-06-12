#!/usr/bin/env python3
"""v3.3.9 TIAGE failure-case dump (inspection only — NOT tuning).

Per-turn: GT shift, predicted topic id, pred boundary, δ_prev (the actual
adjacent-cosine signal the model thresholds), and FP/FN flag. Plus the
separability view: δ_prev distribution at GT-shift vs non-shift turns —
if these overlap heavily, no single global δ* can do better (L4 is
structural). Output → outputs/runs/_misc/tiage_v339_failures.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.embedding import QueryEncoder  # noqa: E402
from hi_ontop.sem_core_v339 import HiOnTopSegmenterV339  # noqa: E402

SPLIT = "test"
DATA = REPO / "benchmarks" / "tiage" / "data" / "personachat" / "anno"
OUT = REPO / "outputs" / "runs" / "_misc" / "tiage_v339_failures.txt"


def load_split(split):
    raw = json.loads((DATA / split / f"anno_{split}.json").read_text())
    return {cid: [(t[0], t[1]) for t in d] for cid, d in raw.items()}


def gt_shifts(dialog):
    lab = [t[1] for t in dialog]
    return [lab[i] == "1" for i in range(1, len(lab))]


def main() -> None:
    dialogs = load_split(SPLIT)
    enc = QueryEncoder()
    embs = {cid: np.asarray(enc.encode([t[0] for t in d]))
            for cid, d in dialogs.items()}

    seg0 = HiOnTopSegmenterV339(dim=next(iter(embs.values())).shape[1],
                             alpha=1.0, lmda=10.0)
    dstar = seg0.delta_star
    sig = seg0._sigma_delta_sq

    gt_all, pred_all = [], []
    d_shift, d_noshift = [], []  # δ_prev split by GT
    conv_err = []  # (n_err, cid, n_turns, n_gt, n_pred)
    fn_rows = []  # missed GT shifts:  (dprev, cid, i, text, prev_text)
    fp_rows = []  # spurious boundary: (dprev, cid, i, text, prev_text)

    for cid, dialog in dialogs.items():
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
        seg = HiOnTopSegmenterV339(dim=embs[cid].shape[1], alpha=1.0, lmda=10.0)
        e = embs[cid]
        ids, dprev = [], [None]
        for i, s in enumerate(e):
            if i > 0:
                a, b = e[i - 1], s
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                dprev.append(
                    1.0 - float(np.dot(a, b) / (na * nb))
                    if na > 1e-12 and nb > 1e-12 else None
                )
            ids.append(seg.assign(s.astype(np.float64))[0])
        gt = gt_shifts(dialog)
        pred = [ids[i] != ids[i - 1] for i in range(1, len(ids))]
        gt_all += gt
        pred_all += pred
        for j, g in enumerate(gt):
            dp = dprev[j + 1]
            if dp is not None:
                (d_shift if g else d_noshift).append(dp)

        n_err = sum(1 for g, p in zip(gt, pred) if g != p)
        conv_err.append((n_err, cid, len(dialog), sum(gt), sum(pred)))
        for i in range(1, len(dialog)):
            g, p, dp = gt[i - 1], pred[i - 1], dprev[i]
            if g == p or dp is None:
                continue
            row = (dp, cid, i, dialog[i][0][:78], dialog[i - 1][0][:78])
            (fn_rows if g and not p else fp_rows).append(row)

    f1 = f1_score(gt_all, pred_all)
    p, r, _, _ = precision_recall_fscore_support(
        gt_all, pred_all, average="binary", zero_division=0)
    tp = sum(1 for g, q in zip(gt_all, pred_all) if g and q)
    fp = sum(1 for g, q in zip(gt_all, pred_all) if q and not g)
    fn = sum(1 for g, q in zip(gt_all, pred_all) if g and not q)

    ds, dn = np.array(d_shift), np.array(d_noshift)

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    hdr = [
        "v3.3.9 TIAGE test FAILURE DUMP (inspection only, seed=0, "
        "α=1 λ=10, default HP)",
        f"δ* = {dstar:.4f} (train-estimated, fixed)   σδ² = {sig:.5f}",
        "",
        f"turns={len(gt_all)}  GT_shifts={int(np.sum(gt_all))}  "
        f"pred_bnd={int(np.sum(pred_all))}",
        f"F1={f1:.3f}  P={p:.3f}  R={r:.3f}   TP={tp} FP={fp} FN={fn}",
        "",
        "── δ_prev SEPARABILITY (the core L4 question) ──",
        "  GT-SHIFT turns   (should be HIGH δ_prev → boundary):",
        f"    n={len(ds)}  mean={ds.mean():.3f}  "
        f"p10={pct(ds,10):.3f} p25={pct(ds,25):.3f} "
        f"p50={pct(ds,50):.3f} p75={pct(ds,75):.3f}",
        "  NON-shift turns  (should be LOW δ_prev → no boundary):",
        f"    n={len(dn)}  mean={dn.mean():.3f}  "
        f"p25={pct(dn,25):.3f} p50={pct(dn,50):.3f} "
        f"p75={pct(dn,75):.3f} p90={pct(dn,90):.3f}",
        f"  → δ*={dstar:.3f}.  GT-shift below δ* (missed, FN-prone): "
        f"{float((ds < dstar).mean()) * 100:.0f}%   "
        f"non-shift above δ* (spurious, FP-prone): "
        f"{float((dn > dstar).mean()) * 100:.0f}%",
        "  (Heavy overlap ⇒ no single global δ* separates ⇒ L4 structural;"
        " windowed/adaptive needed.)",
        "",
    ]

    body: list[str] = []
    # ── Section 1: FN (missed GT shifts), δ_prev ASC — smooth/low-δ
    #    shifts first (the structurally hard ones).
    fn_rows.sort(key=lambda r: r[0])
    body += [
        "",
        "═══════════════════════════════════════════════════════════════",
        f" FN — MISSED GT SHIFTS  ({len(fn_rows)})  · δ_prev ASCENDING",
        " (low δ_prev = topic changed but wording stayed similar →",
        "  no single threshold can catch these)",
        "═══════════════════════════════════════════════════════════════",
        " δ_prev | conv | t  | utterance   ⤺ prev utterance",
    ]
    for dp, cid, i, txt, ptxt in fn_rows:
        body.append(f"  {dp:.3f} | {cid:>4} | {i:2d} | {txt}")
        body.append(f"        |      |    |   ⤺ {ptxt}")

    # ── Section 2: FP (spurious boundaries), δ_prev DESC — strongest
    #    spurious first (high δ_prev within same topic).
    fp_rows.sort(key=lambda r: -r[0])
    body += [
        "",
        "═══════════════════════════════════════════════════════════════",
        f" FP — SPURIOUS BOUNDARIES  ({len(fp_rows)})  · δ_prev DESCENDING",
        " (high δ_prev but SAME topic = within-topic chit-chat diversity",
        "  → over-segmentation, the dominant error mode)",
        "═══════════════════════════════════════════════════════════════",
        " δ_prev | conv | t  | utterance   ⤺ prev utterance",
    ]
    for dp, cid, i, txt, ptxt in fp_rows:
        body.append(f"  {dp:.3f} | {cid:>4} | {i:2d} | {txt}")
        body.append(f"        |      |    |   ⤺ {ptxt}")

    # ── Section 3: per-conversation error density (worst first).
    conv_err.sort(key=lambda x: -x[0])
    body += [
        "",
        "═══════════════════════════════════════════════════════════════",
        " PER-CONV ERROR DENSITY (worst first)",
        "═══════════════════════════════════════════════════════════════",
        " errors | conv | turns | GT_shifts | pred_bnd",
    ]
    for n_err, cid, nt, ng, npd in conv_err:
        if n_err == 0:
            continue
        body.append(
            f"   {n_err:3d}  | {cid:>4} |  {nt:3d}  |    {ng:2d}     "
            f"|   {npd:2d}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(hdr + body) + "\n")
    print("\n".join(hdr))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
