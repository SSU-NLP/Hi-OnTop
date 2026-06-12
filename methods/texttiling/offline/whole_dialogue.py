#!/usr/bin/env python3
"""TextTiling **offline** (원본 SuperDialseg 설정, whole-dialogue).

원본 코드는 benchmarks/superdialseg (Coldog2333/SuperDialseg, read-only)
의 TexttilingSegmenter 와 동일 동작 — nltk TextTilingTokenizer(w=10,k=6)
를 대화 **전체**에 적용(미래 포함, prefix 아님). 코드 복사 없이 동일
알고리즘을 호출만 함(CLAUDE.md: benchmarks read-only).

online 판(prefix-causal)과 **같은 데이터(Def-DTS 번들)·같은 metric
(autoseg segeval Pk/WD + F1, Score=0.5F1+0.25(1-Pk)+0.25(1-WD))** 로
산출 → offline vs online 을 Hi-OnTop 내에서 apples-to-apples 비교.
※ 논문(SuperDialseg) 보고치(tiage .363/superseg .471/dialseg711 .382)
는 *그쪽 데이터·공식 metric* 산출이라 정확히 같지 않을 수 있음 —
방향/타당성 검증용 (REPORT 명시).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
DEFDTS = REPO / "benchmarks" / "Def-DTS"
DATASETS = ["tiage", "dialseg711", "superseg"]


def _stub_anthropic() -> None:
    m = types.ModuleType("anthropic")
    m.Anthropic = lambda **kw: None
    sys.modules["anthropic"] = m


def _utts(dialogue: str) -> list[str]:
    return [s for s in dialogue.split("[NEWLINE]")
            if s.strip() not in ("[BOUNDARY]", "")]


def _tile_end_flags(utts, tt):
    """SuperDialseg TexttilingSegmenter.forward 동일: per-utterance
    1=tile 의 마지막(경계 AFTER), 마지막 강제 0. None=실패/길이불일치."""
    doc = "\n\n".join((u.strip() or ".") for u in utts)
    try:
        tiles = tt.tokenize(doc)
    except Exception:
        return None
    pred = []
    for tile in tiles:
        lines = [x for x in tile.strip().split("\n\n") if x != ""]
        if not lines:
            continue
        pred += [0] * len(lines)
        pred[-1] = 1
    if not pred or len(pred) != len(utts):
        return None
    pred[-1] = 0
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_texttiling_offline")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--limit", type=int, default=0,
                    help="0=full test set (paper 비교용); N=앞 N 대화")
    ap.add_argument("-w", type=int, default=10)
    ap.add_argument("-k", type=int, default=6)
    args = ap.parse_args()

    import nltk
    for p in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.download(p, quiet=True)
        except Exception:
            pass
    from nltk.tokenize import TextTilingTokenizer

    _stub_anthropic()
    sys.path.insert(0, str(DEFDTS))
    os.chdir(DEFDTS)
    import src.autoseg as A  # noqa: E402

    exp = REPO / "outputs" / "experiments" / args.name
    exp.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        data = list(A.alternative_load_dataset(ds, "test"))
        if args.limit:
            data = data[: args.limit]
        tt = TextTilingTokenizer(w=args.w, k=args.k)
        preds, labels, miss, lat = [], [], 0, []
        for d in data:
            utts = _utts(d["dialogue"])
            if len(utts) < 2:
                continue
            t0 = time.perf_counter()
            flags = _tile_end_flags(utts, tt)
            lat.append((time.perf_counter() - t0) * 1000.0)
            if flags is None:
                miss += 1
                uttr = [False] * len(utts)
            else:
                # flags[i]=1 → 경계 after i  ⇒  uttr[i+1]=새 세그 시작
                uttr = [False] * len(utts)
                for i in range(len(utts) - 1):
                    if flags[i] == 1:
                        uttr[i + 1] = True
            uttr[0] = False
            pred = A.extract_pred(uttr)
            lbl, _ = A.extract_label(d["dialogue"].split("[NEWLINE]"), True)
            pred, lbl = A.align_pred_label(pred, lbl)
            preds.append(pred)
            labels.append(lbl)
        m = A.compute_metrics(preds, labels)
        score = 0.5 * m["f1"] + 0.25 * (1 - m["pk"]) + 0.25 * (1 - m["wd"])
        rows.append(dict(ds=ds, n=len(data), pk=m["pk"], wd=m["wd"],
                         f1=m["f1"], score=score, miss=miss,
                         lat_ms=float(np.mean(lat)) if lat else float("nan")))
        r = rows[-1]
        print(f"{ds:11s} n={r['n']:4d} Pk={r['pk']:.4f} WD={r['wd']:.4f} "
              f"F1={r['f1']:.4f} Score={r['score']:.4f} "
              f"lat/dial={r['lat_ms']:.1f}ms miss={miss}")

    L = ["# TextTiling **offline** (원본 SuperDialseg 설정, whole-dialogue)",
         "",
         "원본=benchmarks/superdialseg TexttilingSegmenter 동일 알고리즘"
         "(nltk w/k, 코드 복사 없이 호출). data=Def-DTS 번들, "
         "metric=autoseg Pk/WD+F1, Score=0.5F1+0.25(1-Pk)+0.25(1-WD).",
         "online(prefix-causal)판과 동일 harness → offline↔online 직접 비교.",
         f"limit={args.limit or 'full'} · w={args.w} k={args.k}",
         "",
         "| dataset | n_dial | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
         "lat/dial(ms) | miss |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n']} | {r['pk']:.4f} | {r['wd']:.4f} "
                 f"| {r['f1']:.4f} | {r['score']:.4f} | {r['lat_ms']:.1f} "
                 f"| {r['miss']} |")
    L += ["",
          "## 한계",
          "- 원 SuperDialseg 논문 보고치(tiage .363/superseg .471/"
          "dialseg711 .382)는 *그쪽 데이터·공식 metric* — 본 표는 Hi-OnTop "
          "harness(Def-DTS 데이터+autoseg+Score)라 정확 일치 아님, "
          "방향·정상동작 검증용.",
          "- offline = 대화 전체(미래 포함). online 판은 methods/"
          "texttiling/online.py (prefix-causal, AUXILIARY).",
          "- non-LLM CPU, calls/turn=0 tok/turn=0."]
    (exp / "REPORT.md").write_text("\n".join(L) + "\n")
    print("report →", exp / "REPORT.md")


if __name__ == "__main__":
    main()
