#!/usr/bin/env python3
"""Reconstruct Pk/WD/F1/Score per dataset from CSM-online jsonl sidecar.

CSM-online run completed tiage/dialseg711 but exited before SuperDialseg
(see turns_*.jsonl in 2026-05-23_csm_online_delay2/). This script reads
the per-turn predictions and gold boundaries (from Def-DTS bundle) and
computes the aggregate metrics, matching methods/CSM/online/delay2.py.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hi_ontop.baselines._seg_utils import (  # noqa: E402
    boundary_set_f1, latency_stats, load_defdts, pk_wd,
)

EXP_DIR = REPO / "outputs" / "experiments" / "2026-05-23_csm_online_delay2"
DEFDTS_DIR = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"


def main() -> None:
    rows = []
    for ds in ("tiage", "dialseg711"):
        sidecar = EXP_DIR / f"turns_{ds}.jsonl"
        if not sidecar.exists():
            print(f"  [{ds}] sidecar missing → skip")
            continue

        # group predictions by dialogue id, dedupe by (id, t)
        # (sidecar may contain interleaved writes from a re-run; keep last)
        recs_by_idt: dict[tuple[str, int], dict] = {}
        with sidecar.open() as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs_by_idt[(rec["id"], int(rec["t"]))] = rec

        preds_by_did: dict[str, list[int]] = defaultdict(list)
        lat_ms = []
        for (did, t), rec in sorted(recs_by_idt.items()):
            preds_by_did[did].extend(rec["new_bs"])
            if t >= 2:
                lat_ms.append(float(rec["ms"]))

        # load Def-DTS gold boundaries for this dataset
        dialogues = load_defdts(ds, DEFDTS_DIR)
        gold_by_did = {did: gold for did, _utts, gold in dialogues}
        utts_by_did = {did: utts for did, utts, _gold in dialogues}

        pk_v, wd_v, f1_v = [], [], []
        n_pred_total, n_gold_total, n_turn = 0, 0, 0
        for did, gold in gold_by_did.items():
            if did not in preds_by_did:
                continue
            pred = sorted(set(int(b) for b in preds_by_did[did]))
            utts = utts_by_did[did]
            n = len(utts)
            pk, wd = pk_wd(pred, gold, n)
            f1 = boundary_set_f1(pred, gold)
            pk_v.append(pk); wd_v.append(wd); f1_v.append(f1)
            n_pred_total += len(pred); n_gold_total += len(gold)
            n_turn += n

        pk_m = float(np.mean(pk_v)) if pk_v else float("nan")
        wd_m = float(np.mean(wd_v)) if wd_v else float("nan")
        f1_m = float(np.mean(f1_v)) if f1_v else float("nan")
        score = 0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
        st = latency_stats(lat_ms) if lat_ms else {"mean": float("nan"), "p95": float("nan"), "n": 0}

        print(
            f"  [{ds}] n_dial={len(pk_v)} n_turn={n_turn}  "
            f"Pk={pk_m:.4f} WD={wd_m:.4f} F1={f1_m:.4f} Score={score:.4f}  "
            f"lat={st['mean']:.2f}ms p95={st['p95']:.2f}ms (n={st['n']})  "
            f"pred={n_pred_total} gold={n_gold_total}"
        )

        rows.append(
            {"ds": ds, "n_dial": len(pk_v), "n_turn": n_turn,
             "pk": pk_m, "wd": wd_m, "f1": f1_m, "score": score,
             "lat_mean_ms": st["mean"], "lat_p95_ms": st["p95"],
             "n_lat": st["n"], "n_pred": n_pred_total, "n_gold": n_gold_total}
        )

    out = EXP_DIR / "metrics_reconstructed.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
