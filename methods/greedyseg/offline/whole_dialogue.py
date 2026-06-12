#!/usr/bin/env python3
"""GreedySeg **offline** (원본 SuperDialseg `GreedySegmenter`, whole-dialogue).

원본 코드는 `benchmarks/superdialseg/src/super_dialseg/models/greedyseg/`
(Coldog2333/SuperDialseg, read-only) 의 `GreedySegmenter` — BERT 임베딩
+ greedy argmin similarity + 비가역 segment 선택, **대화 전체** 에 적용
(미래 포함, prefix 아님). 코드 복사 없이 sys.path import (CLAUDE.md:
benchmarks read-only).

online 판(`methods/greedyseg/online/delay2.py`) 의 `delay=2` bounded-
lookahead 와 달리 여기는 horizon 무제한. 같은 score 공식·HP — emission
만 offline.

데이터 = Def-DTS 번들 test, metric = segeval Pk/WD + boundary-set F1,
Score = 0.5F1 + 0.25(1−Pk) + 0.25(1−WD).
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
SDS_SRC = REPO / "benchmarks" / "superdialseg" / "src"
DEFDTS_DIR = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"
DATASETS = ("tiage", "dialseg711", "superseg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-23_greedyseg_offline")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument("--limit", type=int, default=0,
                    help="0=full test set; N=앞 N 대화")
    ap.add_argument("--backbone", default="bert-base-uncased")
    ap.add_argument("--max-utterance-len", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(SDS_SRC))
    from super_dialseg.models.greedyseg.modeling_greedyseg import GreedySegmenter  # noqa: E402
    from hi_ontop.baselines._seg_utils import (
        boundary_set_f1, latency_stats, load_defdts, pk_wd)

    print(f"[setup] backbone={args.backbone} device={args.device}")
    seg = GreedySegmenter(backbone=args.backbone,
                          max_utterance_len=args.max_utterance_len)
    # GreedySegmenter 의 __init__ 가 device 자동 결정 (cuda if available else
    # cpu). CPU 강제하려면 to(cpu) 재바인딩.
    import torch
    seg.device = torch.device(args.device)
    seg.model.to(seg.device)

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        full = load_defdts(ds, DEFDTS_DIR)
        if args.limit:
            full = full[: args.limit]
        print(f"\n=== {ds}: {len(full)} dialogues ===")

        pk_v, wd_v, f1_v, lat = [], [], [], []
        n_pred_total = n_gold_total = 0
        miss = 0
        for did, utts, gold in full:
            n = len(utts)
            t0 = time.perf_counter()
            try:
                preds = seg.forward({"utterances": utts})
            except Exception as e:
                miss += 1
                print(f"  miss {did}: {type(e).__name__}: {e}")
                continue
            lat.append((time.perf_counter() - t0) * 1000.0)
            # preds[i]=1 ⇒ 경계 (SuperDialseg convention) ⇒ utt i+2 1-based
            # 새 segment 시작. preds[-1]=0 강제.
            pred_bs = [i + 2 for i, p in enumerate(preds[:-1]) if p == 1]
            pk, wd = pk_wd(pred_bs, gold, n)
            f1 = boundary_set_f1(pred_bs, gold)
            pk_v.append(pk); wd_v.append(wd); f1_v.append(f1)
            n_pred_total += len(pred_bs); n_gold_total += len(gold)

        pk_m = float(np.mean(pk_v)) if pk_v else float("nan")
        wd_m = float(np.mean(wd_v)) if wd_v else float("nan")
        f1_m = float(np.mean(f1_v)) if f1_v else float("nan")
        score = 0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
        st = latency_stats(lat)
        rows.append(dict(ds=ds, n_dial=len(full), pk=pk_m, wd=wd_m,
                         f1=f1_m, score=score, n_pred=n_pred_total,
                         n_gold=n_gold_total, miss=miss, **st))
        r = rows[-1]
        print(f"  Pk={pk_m:.4f} WD={wd_m:.4f} F1={f1_m:.4f} Score={score:.4f}"
              f" | lat/dial mean={st['mean']:.1f}ms | miss={miss}")

    L = [
        "# GreedySeg **offline** (SuperDialseg `GreedySegmenter`, whole-dialogue)",
        "",
        f"backbone={args.backbone} max_utt_len={args.max_utterance_len} "
        f"device={args.device}",
        "데이터=Def-DTS 번들 test, metric=segeval Pk/WD + boundary-set F1, "
        "Score=0.5F1+0.25(1-Pk)+0.25(1-WD).",
        "online 판 (methods/greedyseg/online/delay2.py) 과 동일 harness.",
        "",
        "| dataset | n_dial | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/dial(ms) | pred bs | gold bs | miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n_dial']} | {r['pk']:.4f} | "
                 f"{r['wd']:.4f} | {r['f1']:.4f} | {r['score']:.4f} | "
                 f"{r['mean']:.1f} | {r['n_pred']} | {r['n_gold']} | "
                 f"{r['miss']} |")
    L += ["",
          "## 한계",
          "- offline = 대화 전체(미래 포함). online 판은 "
          "`methods/greedyseg/online/delay2.py` (delay-2 bounded-lookahead).",
          "- BERT-base-uncased pretrained 가중치 (별도 학습 ckpt 불필요).",
          "- BERT forward 비용으로 lat/dial 큼. 진짜 per-turn latency 는 "
          "online 판 표 참조."]
    (exp_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nreport → {exp_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
