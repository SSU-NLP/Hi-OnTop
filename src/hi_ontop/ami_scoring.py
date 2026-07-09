"""AMI 화제분절 단일 공식 채점기 (확정 2026-06-15).

HANDOFF_04 가 DTS 채점기를 확정했듯, AMI 도 단일 채점기로 확정한다. AMI 의 모든 수치
(de-neut threshold AMI Score≈0.372 등)를 만든 `scripts/ami_adaptive_deneut_deploy.py::{ev,tol_f1}`
와 **동일 스펙**이며, 이를 정본 모듈로 승격한 것.

DTS(HANDOFF_04 `dts_scoring`) 와의 차이는 **딱 두 가지**, 나머지는 동일:
  1. **F1 = ±2 tolerant** boundary F1 (DTS 는 exact binary). AMI gold 는 구어/turn 분절로
     경계 시점이 부정확·sparse 해서 tolerance 필요.
  2. **정렬 = shift 0** (DTS 는 -1). **AMI gold `bnd_top` = start-turn 규약**(새 top-level topic 의
     첫 turn; = depth-1 `topic_levels.start_turn` 에서 최초 시작 제외) — 데이터 검증(2026-06-15).
     임베딩 신호도 새 segment 첫 turn 에서 솟으므로 같은 규약 → shift 0.

동일한 부분 (DTS 와 같음):
  - Pk/WD = nltk pk/windowdiff, window=auto = max(2, round(len/(sum+1)/2)) (Fournier 2013).
  - Score = 0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD).
  - per-meeting 계산 후 평균 (macro-over-meetings). 마지막 turn label·pred = 0.
  - 예측 interior filter: 0 < p < n-1.

권위 출처(동치): `scripts/ami_adaptive_deneut_deploy.py::{ev, tol_f1, load_ami}`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd

TOL = 2  # ±2 tolerant boundary F1


def tol_f1(gold: Sequence[int], pred: Sequence[int], tol: int = TOL) -> float:
    """±tol tolerant boundary F1 (gold/pred = 경계 turn index 리스트)."""
    gold = list(gold); pred = list(pred)
    if not pred or not gold:
        return 0.0
    p = sum(1 for i in pred if any(abs(i - j) <= tol for j in gold)) / len(pred)
    r = sum(1 for j in gold if any(abs(i - j) <= tol for i in pred)) / len(gold)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _official_pk_wd(yt: list[int], yp: list[int]) -> tuple[float, float]:
    """nltk Pk/WD, window=auto (DTS 와 동일)."""
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


def score_meetings(golds: Sequence[Sequence[int]], preds: Sequence[Sequence[int]],
                   *, tol: int = TOL) -> dict:
    """AMI 공식 채점. golds/preds = meeting 별 per-turn 0/1 (start-turn 규약, shift 0).

    Returns: ``{'f1','pk','wd','score','n'}`` — 전부 meeting 별 평균.
    """
    if len(golds) != len(preds):
        raise ValueError(f"meeting 수 불일치: {len(golds)} vs {len(preds)}")
    F1, PK, WD = [], [], []
    for yt, yp in zip(golds, preds):
        if len(yt) != len(yp):
            raise ValueError("turn 수 불일치")
        n = len(yt)
        yt = list(yt); yp = list(yp)
        yt[-1] = 0; yp[-1] = 0
        gold = [i for i in range(n) if yt[i] == 1]
        pred = [i for i in range(n) if yp[i] == 1 and 0 < i < n - 1]
        F1.append(tol_f1(gold, pred, tol))
        pk, wd = _official_pk_wd(yt, yp)
        PK.append(pk); WD.append(wd)
    f1, pk, wd = float(np.mean(F1)), float(np.mean(PK)), float(np.mean(WD))
    return {"f1": f1, "pk": pk, "wd": wd,
            "score": 0.5 * f1 + 0.25 * (1 - pk) + 0.25 * (1 - wd), "n": len(golds)}


def boundaries_to_pred(n: int, boundary_idx: Sequence[int]) -> list[int]:
    """경계 turn index 리스트 → per-turn 0/1 (shift 0, start-turn 규약)."""
    bset = set(boundary_idx)
    return [1 if t in bset else 0 for t in range(n)]
