"""Shared utilities for online segmentation baseline runners.

Def-DTS 번들 (`benchmarks/Def-DTS/data/DTS_session_datasets/*_test.jsonl`)
형식 파싱 + segeval Pk/WD masses 변환 + boundary-set F1 + latency stats.

데이터 형식: 각 row 의 ``dialogue`` 가 ``U1[NEWLINE]U2[NEWLINE][BOUNDARY]
[NEWLINE]U3...`` 형태로 utterance 와 boundary marker 가 한 문자열에.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATASETS = ("tiage", "dialseg711", "superseg")


def parse_defdts_dialogue(dialogue: str) -> tuple[list[str], list[int]]:
    """Def-DTS dialogue 문자열 → (utterances, boundary_idxs_1based).

    ``[BOUNDARY]`` 가 두 발화 사이에 나오면 *다음 발화부터* 새 topic
    (= boundary BEFORE 그 발화). 첫 발화는 정의상 경계 아님.
    """
    parts = dialogue.split("[NEWLINE]")
    utts: list[str] = []
    bnds: list[int] = []
    pending = False
    for seg in parts:
        s = seg.strip()
        if s == "":
            continue
        if s == "[BOUNDARY]":
            pending = True
            continue
        utts.append(seg)
        if pending and len(utts) >= 2:
            bnds.append(len(utts))
        pending = False
    return utts, bnds


def load_defdts(dataset: str, defdts_dir: Path) -> list[tuple[str, list[str], list[int]]]:
    """반환: [(dialogue_id, utterances, gold_boundary_idxs_1based)]."""
    path = defdts_dir / f"{dataset}_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{dataset} test set 없음: {path}")
    out: list[tuple[str, list[str], list[int]]] = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        utts, bnds = parse_defdts_dialogue(row["dialogue"])
        if len(utts) >= 2:
            out.append((row["id"], utts, bnds))
    return out


def bnds_to_masses(boundaries: list[int], n: int) -> tuple[int, ...]:
    """boundary utterance indices (1-based, before-utt) → segment-length tuple.

    boundary i 는 "utts[i] 부터 새 segment" 의미. 결과 길이 합 = n.
    """
    bs = sorted({b for b in boundaries if 2 <= b <= n})
    masses: list[int] = []
    prev = 1
    for b in bs:
        masses.append(b - prev)
        prev = b
    masses.append(n - prev + 1)
    return tuple(masses)


def boundary_set_f1(pred_bs: list[int], gold_bs: list[int]) -> float:
    p = set(pred_bs)
    g = set(gold_bs)
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def latency_stats(xs: list[float]) -> dict:
    if not xs:
        return {k: float("nan") for k in
                ("n", "mean", "std", "min", "p50", "p90", "p95", "p99", "max")}
    a = np.asarray(xs, float)
    return dict(
        n=len(a), mean=float(a.mean()), std=float(a.std()),
        min=float(a.min()), p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)), max=float(a.max()))


def pk_wd(pred_bs: list[int], gold_bs: list[int], n: int) -> tuple[float, float]:
    """segeval Pk / WindowDiff. degenerate 시 (1.0, 1.0)."""
    from segeval.window.pk import pk as _pk
    from segeval.window.windowdiff import window_diff as _wd
    pm = bnds_to_masses(pred_bs, n)
    gm = bnds_to_masses(gold_bs, n)
    try:
        return float(_pk(pm, gm)), float(_wd(pm, gm))
    except Exception:
        return 1.0, 1.0
