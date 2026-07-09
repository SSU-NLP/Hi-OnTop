"""공식 SuperDialseg 채점기 검증 — 독립 reference(paper offline TextTiling) 재현.

목적
----
DTS 채점이 그동안 세 harness(SuperDialseg / Def-DTS / TIAGE)로 섞여 있어
점수 신뢰가 깨졌다 (micro vs per-dialogue F1, nltk vs segeval, off-by-one 정렬).
본 스크립트는 **확정 채점기 = SuperDialseg 공식 ``SegmentationEvaluation``** 가
독립 reference(Jiang et al. 2023 의 offline TextTiling 공개 점수)를 재현하는지
확인한다. 재현되면 "이 채점기가 진짜 공식"임이 증명된다.

확정 채점기 스펙 (이 스크립트가 그대로 호출)
---------------------------------------------
- 권위 출처: ``benchmarks/superdialseg/src/super_dialseg/metrics/segmentation.py``
  의 ``SegmentationEvaluation`` (읽기 전용 레포, **복사 금지·import 만**).
- 집계: 모든 지표를 **대화별 계산 후 대화 수로 평균 (per-dialogue / macro-over-dialogues)**.
  (basic.py ``_compute``: 대화 1개씩 add → 루프 합산 → ``/#sample``.)
- F1: ``f1_score(average='binary')`` = 경계(positive) 클래스만. (paper: "we use F1
  (binary) instead of F1 (macro) ... we only care about the segmentation points".)
- Pk/WD: ``nltk.metrics.{pk,windowdiff}``, window = ``max(2, round(len/(sum+1)/2))``
  (= 그 대화 평균 segment 길이의 절반, Fournier 2013).
- Score = ``0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD)``.
- 마지막 turn label·pred = 0 강제.
- **경계 정렬 = 끝-turn 규약**: gold 의 label=1 은 segment 의 *마지막* turn.
  TextTiling 은 tile 끝줄=1 로 내므로 정렬 일치(shift 없음). (δ_eff/임베딩 신호는
  새 segment *첫* turn 에서 솟으므로 채점 시 -1 매핑 필요 — 본 검증 대상 아님.)

실행
----
    python scripts/validate_official_scorer.py

데이터 = ``benchmarks/superdialseg_data/{tiage,dialseg711,superseg}/segmentation_file_test.json``
(공식 superdialseg_data, ``run_encoder_comparison.load_dialogs`` 로 로드).
모델 = 공식 ``TexttilingSegmenter`` 와 동일 알고리즘 (nltk ``TextTilingTokenizer``
w=10,k=6, forward 로직 동일).
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SDS_SRC = REPO / "benchmarks" / "superdialseg" / "src"

# Jiang et al. 2023 (SuperDialseg) Table 3 offline TextTiling 공개 Score.
# (dts_result.md 의 dialseg711 0.482 는 전사 오류 — 실제 0.382, 본 스크립트로 확인.)
PAPER_TEXTTILING = {"tiage": 0.363, "dialseg711": 0.382, "superseg": 0.471}


def load_official_scorer():
    """레포 ``SegmentationEvaluation`` 을 무거운 패키지 ``__init__`` 우회하여 import.

    ``super_dialseg/__init__`` 는 모든 모델(gensim/torch 등)을 eager import 하므로,
    metrics 서브모듈만 namespace 패키지로 직접 로드한다.
    """
    if "prettytable" not in sys.modules:  # basic.py 가 show_performance 에서만 사용
        shim = types.ModuleType("prettytable")
        shim.PrettyTable = object
        sys.modules["prettytable"] = shim
    base = str(SDS_SRC / "super_dialseg")
    for name, path in [("super_dialseg", base), ("super_dialseg.metrics", base + "/metrics")]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod
    return importlib.import_module("super_dialseg.metrics.segmentation").SegmentationEvaluation


def texttiling_predict(tt, utts):
    """공식 ``TexttilingSegmenter.forward`` 와 동일 로직 (끝-turn=1 규약)."""
    utterances = [u.strip("\n") for u in utts]
    document = "\n\n".join(utterances)
    tiles = tt.tokenize(document)
    predictions: list[int] = []
    for tile in tiles:
        lines = tile.strip().split("\n\n")
        if lines == [""]:
            continue
        predictions.extend([0] * len(lines))
        predictions[-1] = 1
    predictions[-1] = 0
    return predictions


def main() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import nltk

    nltk.download("stopwords", quiet=True)
    from nltk.tokenize import TextTilingTokenizer

    from run_encoder_comparison import load_dialogs

    SegEval = load_official_scorer()
    tt = TextTilingTokenizer(w=10, k=6)

    print("=== 공식 SegmentationEvaluation 검증 (offline TextTiling, superdialseg_data) ===")
    ok = True
    for ds in ["tiage", "dialseg711", "superseg"]:
        ev = SegEval(window_size="auto")
        miss = 0
        for utts, yt in load_dialogs(ds, "test"):
            try:
                pred = texttiling_predict(tt, utts)
            except Exception:
                miss += 1
                continue
            if len(pred) != len(yt):
                miss += 1
                continue
            yt2, pred = list(yt), list(pred)
            yt2[-1] = 0
            pred[-1] = 0
            ev.add(yt2, pred)
        r = ev.compute()
        paper = PAPER_TEXTTILING[ds]
        match = abs(r["total_score"] - paper) <= 0.01
        ok = ok and match
        print(
            f"{ds:11s}: F1={r['f1(binary)']:.4f} Pk={r['pk']:.4f} "
            f"WD={r['windowdiff']:.4f} Score={r['total_score']:.4f} "
            f"(#dlg={r['#sample']}, miss={miss}) | paper={paper} "
            f"{'OK' if match else 'MISMATCH'}"
        )
    print("\n검증", "통과 — 공식 채점기 확정" if ok else "실패 — 차이 점검 필요")


if __name__ == "__main__":
    main()
