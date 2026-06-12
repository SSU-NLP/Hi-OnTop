#!/usr/bin/env python3
"""Hi-OnTop vs v4.1.x (HiOnTopSegmenterV411) output-parity verification.

Runs both segmenters on identical cached embeddings (TIAGE / Dialseg711 /
SuperDialseg test) under identical HP and checks:
  1. byte-identical boundary sequences (turn-level diff count)
  2. identical official SuperDialseg Score / F1 / Pk / WD

HiOnTopSegmenterV413 == HiOnTopSegmenterV411 for the segmentation decision
(v4.1.3 only adds output attributes), so V411 stands in for v4.1.3.
"""

from __future__ import annotations

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
from hi_ontop.sem_core_v411 import HiOnTopSegmenterV411  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"

# canonical v4.1.x "TIAGE-cfg" that produced the headline numbers
M, RHO, A, DSTAR = 2, 0.7, 0.5, 0.5594


def load_dialogs(ds: str, split: str = "test"):
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


def official_pk_wd(yt, yp):
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


def boundaries_from_ids(ids):
    return [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]


def seg_v411(emb):
    seg = HiOnTopSegmenterV411(
        dim=emb.shape[1], alpha=1.0, lmda=10.0, delta_star=DSTAR,
        ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    return boundaries_from_ids([seg.assign(s.astype(np.float64))[0] for s in emb])


def seg_hiontop(emb):
    seg = HiOnTop(
        dim=emb.shape[1], delta_star=DSTAR,
        ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    return boundaries_from_ids([seg.assign(s.astype(np.float64))[0] for s in emb])


def score_of(dialogs, preds):
    pks, wds, g, p = [], [], [], []
    for (u, yt), yp in zip(dialogs, preds):
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    return dict(f1=f1, pk=pk_m, wd=wd_m,
                score=0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m))


def main():
    print(f"config: m={M} rho={RHO} a={A} delta*={DSTAR}\n")
    lines = [
        "# Hi-OnTop ↔ v4.1.3 output-parity 검증",
        "",
        f"config: m={M} · ρ={RHO} · a={A} · δ*={DSTAR} (canonical TIAGE-cfg)",
        "v4.1.3 의 segmentation 결정 = HiOnTopSegmenterV411 (v4.1.3 는 출력 "
        "attribute 만 추가) → V411 로 대조.",
        "metric = 공식 SuperDialseg (Pk/WD k=auto, F1 binary, "
        "Score=0.5F1+0.25(1−Pk)+0.25(1−WD)).",
        "",
        "| dataset | n_dial | n_turn | diff turns | "
        "Score v411 | Score Hi-OnTop | F1 | Pk | WD | parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    all_ok = True
    for ds in ["tiage", "dialseg711", "superseg"]:
        dia = load_dialogs(ds)
        with open(CACHE / f"sds_emb_{ds}_test.pkl", "rb") as fh:
            embs = pickle.load(fh)
        t0 = time.perf_counter()
        p_v411 = [seg_v411(e) for e in embs]
        p_dots = [seg_hiontop(e) for e in embs]
        wall = time.perf_counter() - t0

        n_turn = sum(len(yp) for yp in p_v411)
        diff = sum(int(a != b)
                   for yv, yd in zip(p_v411, p_dots)
                   for a, b in zip(yv, yd))
        s_v411 = score_of(dia, p_v411)
        s_dots = score_of(dia, p_dots)
        ok = (diff == 0
              and abs(s_v411["score"] - s_dots["score"]) < 1e-12)
        all_ok &= ok
        print(f"[{ds:11s}] n_dial={len(dia):5d} n_turn={n_turn:6d}  "
              f"diff_turns={diff:4d}  "
              f"Score v411={s_v411['score']:.4f} Hi-OnTop={s_dots['score']:.4f}  "
              f"{'IDENTICAL' if ok else 'MISMATCH'}  ({wall:.0f}s)")
        lines.append(
            f"| {ds} | {len(dia)} | {n_turn} | {diff} | "
            f"{s_v411['score']:.4f} | {s_dots['score']:.4f} | "
            f"{s_dots['f1']:.4f} | {s_dots['pk']:.4f} | {s_dots['wd']:.4f} | "
            f"{'✅' if ok else '❌'} |")

    verdict = ("**모든 데이터셋에서 byte-identical** — Hi-OnTop 출력이 "
               "v4.1.3 와 완전히 동일."
               if all_ok else "**불일치 발견** — 아래 표 확인.")
    lines += ["", f"## 판정", "", verdict, "",
              "diff turns = v411 과 Hi-OnTop 의 boundary 예측이 다른 turn 수. "
              "0 이면 두 segmenter 의 turn-level 출력이 완전히 같다는 뜻."]
    out = REPO / "outputs" / "experiments" / "2026-05-22_hiontop_parity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\n{'ALL IDENTICAL' if all_ok else 'MISMATCH'}  →  {out/'REPORT.md'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
