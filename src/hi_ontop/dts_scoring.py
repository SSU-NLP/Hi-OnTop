"""DTS 단일 공식 채점기 — SuperDialseg ``SegmentationEvaluation`` 래퍼.

확정 채점기 (HANDOFF_04, 2026-06-13)
------------------------------------
DTS(tiage/dialseg711/superseg) 의 **유일 공식 채점 기준**은 SuperDialseg
(`Coldog2333/SuperDialseg`) 의 ``SegmentationEvaluation`` 이다. Def-DTS(segeval/풀링
F1/개수제외) 와 TIAGE(풀링 분류 F1) 는 쓰지 않는다.

스펙 (전부 레포 코드에서 강제):
- 집계 = **대화별 계산 후 대화 수로 평균 (per-dialogue)** — Pk·WD·F1 전부.
- F1 = ``f1_score(average='binary')`` (경계 positive 클래스만).
- Pk/WD = ``nltk.metrics.{pk,windowdiff}``, window = ``max(2, round(len/(sum+1)/2))``.
- Score = ``0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD)``.
- 마지막 turn label·pred = 0.
- **경계 정렬 = 끝-turn 규약**: gold label=1 = segment 의 *마지막* turn.

본 모듈은 레포 클래스를 **import 만** 한다 (복사 금지, 읽기 전용 레포). 무거운
``super_dialseg/__init__`` (gensim/torch eager import) 를 우회해 metrics 서브모듈만
namespace 로 로드한다.

권위 출처: ``benchmarks/superdialseg/src/super_dialseg/metrics/{segmentation,basic,
classification}.py``. 검증: ``scripts/validate_official_scorer.py`` (paper TextTiling 재현).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[2]
_SDS_PKG = _REPO / "benchmarks" / "superdialseg" / "src" / "super_dialseg"

_SegEvalCls = None


def _load_segmentation_evaluation():
    """레포 ``SegmentationEvaluation`` 을 ``__init__`` 우회하여 로드 (1회 캐시)."""
    global _SegEvalCls
    if _SegEvalCls is not None:
        return _SegEvalCls
    if "prettytable" not in sys.modules:  # basic.py 가 show_performance 에서만 사용
        shim = types.ModuleType("prettytable")
        shim.PrettyTable = object
        sys.modules["prettytable"] = shim
    base = str(_SDS_PKG)
    for name, path in [("super_dialseg", base), ("super_dialseg.metrics", base + "/metrics")]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod
    _SegEvalCls = importlib.import_module("super_dialseg.metrics.segmentation").SegmentationEvaluation
    return _SegEvalCls


def new_evaluation():
    """fresh 공식 ``SegmentationEvaluation`` (window_size='auto')."""
    return _load_segmentation_evaluation()(window_size="auto")


def score_dialogues(
    golds: Sequence[Sequence[int]],
    preds: Sequence[Sequence[int]],
    *,
    force_last_zero: bool = True,
) -> dict:
    """대화별 per-turn 0/1 라벨/예측 리스트를 공식 채점기로 채점.

    Args:
        golds: 대화별 gold per-turn 0/1 (끝-turn 규약). 각 길이 = 그 대화 turn 수.
        preds: 대화별 예측 per-turn 0/1. ``golds`` 와 대화 수·각 길이 동일해야 함.
        force_last_zero: 각 대화 마지막 turn 의 label·pred 를 0 으로 강제 (공식 규약).

    Returns:
        ``{'pk','wd','f1','score','n'}`` — 전부 대화별 평균 (n=대화 수).
    """
    if len(golds) != len(preds):
        raise ValueError(f"대화 수 불일치: golds={len(golds)} preds={len(preds)}")
    ev = new_evaluation()
    for yt, yp in zip(golds, preds):
        if len(yt) != len(yp):
            raise ValueError(f"turn 수 불일치: gold={len(yt)} pred={len(yp)}")
        yt, yp = list(yt), list(yp)
        if force_last_zero and yt:
            yt[-1] = 0
            yp[-1] = 0
        ev.add(yt, yp)
    r = ev.compute()
    return {
        "pk": float(r["pk"]),
        "wd": float(r["windowdiff"]),
        "f1": float(r["f1(binary)"]),
        "score": float(r["total_score"]),
        "n": int(r["#sample"]),
    }


def signal_to_pred(delta_seq: Sequence[float], threshold: float) -> list[int]:
    """거리 신호(δ_eff/V 등) → 끝-turn 규약 per-turn 예측.

    임베딩 거리 신호는 새 segment 의 *첫* turn(t) 에서 솟지만 gold 경계는 이전
    segment 의 *끝* turn(t-1) 에 찍힌다 ⇒ **-1 매핑**. 결과는 길이 n 의 0/1 리스트
    (마지막 turn 0). ``run_encoder_comparison.boundaries`` 와 동일 정렬.

    boundary(t-1) = 1  iff  delta_seq[t] >= threshold   (t=1..n-1)
    """
    n = len(delta_seq)
    return [1 if delta_seq[i] >= threshold else 0 for i in range(1, n)] + [0]
