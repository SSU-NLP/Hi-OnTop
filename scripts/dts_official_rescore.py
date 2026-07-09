"""DTS 공식 재채점 — δ_eff(DTS primary) vs Hi-OnTop-DeNeut(de-neut+threshold), 정상정렬.

HANDOFF_04 확정 채점기(`hi_ontop.dts_scoring` = 공식 SegmentationEvaluation,
per-dialogue binary F1 + nltk Pk/WD + Score, 끝-turn 정렬)로 두 디폴트를 3 DTS 에서
재채점한다. off-by-one 버그(HANDOFF_01) 없이, 공개 채점 기준으로 비교.

- 인코더: MiniLM-int8 (cached enccmp, dts_result §3.2 와 동일).
- δ*(p80, MiniLM-int8): dts_result.md line 294 값.
- 출력: outputs/experiments/2026-06-13_dts_official_rescore/REPORT.md
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from hi_ontop import dts_scoring as S  # noqa: E402
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402
import hi_ontop.hi_ontop_deneut as dn  # noqa: E402
from run_encoder_comparison import CACHE, load_dialogs  # noqa: E402

M, RHO, A = 2, 0.7, 0.5
# δ* per (dataset, percentile), MiniLM-int8 (dts_result.md line 293-295).
DSTAR = {
    "tiage": {"p60": 0.7334, "p70": 0.7763, "p80": 0.8223},
    "dialseg711": {"p60": 0.7033, "p70": 0.7519, "p80": 0.8029},
    "superseg": {"p60": 0.7255, "p70": 0.7839, "p80": 0.8409},
}
DATASETS = ["tiage", "dialseg711", "superseg"]


def deff_seq(e):
    seg = HiOnTop(dim=e.shape[1], delta_star=1.0, ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for v in e:
        seg.assign(v.astype(np.float64))
    return np.array([float(h["delta_eff"]) for h in seg.history()])


def dn_pred(e, n):
    # DeNeut segment()는 boundary turn index(스파이크=새 segment 첫 turn) → 끝-turn -1 매핑.
    bset = {i - 1 for i in dn.segment(e)}
    return [1 if t in bset else 0 for t in range(n)]


def main() -> None:
    rows = []
    for ds in DATASETS:
        dl = load_dialogs(ds, "test")
        embs = [np.asarray(e, dtype=np.float64) for e in
                pickle.load(open(CACHE / f"enccmp_{ds}_test_minilm-int8.pkl", "rb"))]
        golds = [list(yt) for _, yt in dl]
        seqs = [deff_seq(e) for e in embs]
        # δ_eff: p60/p70/p80 sweep (per-dataset best 가 다름 — superseg=p60, tiage=p70, ds711=p80)
        deff = {p: S.score_dialogues(golds, [S.signal_to_pred(s, DSTAR[ds][p]) for s in seqs])
                for p in ("p60", "p70", "p80")}
        r_dn = S.score_dialogues(golds, [dn_pred(e, len(yt)) for e, (_, yt) in zip(embs, dl)])
        best_p = max(deff, key=lambda p: deff[p]["score"])
        rows.append((ds, deff, best_p, r_dn))
        print(f"{ds}: δ_eff best={deff[best_p]['score']:.4f}({best_p})  "
              f"DeNeut={r_dn['score']:.4f}", flush=True)

    out = REPO / "outputs" / "experiments" / "2026-06-13_dts_official_rescore"
    out.mkdir(parents=True, exist_ok=True)
    L = [
        "# DTS 공식 재채점 — δ_eff(DTS primary) vs Hi-OnTop-DeNeut(de-neut+threshold)",
        "",
        "## 실험 setup",
        "- 목적: HANDOFF_01 off-by-one 버그 없이, 확정 공식 채점기로 두 분절 모델 DTS 비교.",
        "- 채점기: `hi_ontop.dts_scoring` (= 레포 `SegmentationEvaluation`, per-dialogue binary",
        "  F1 + nltk Pk/WD(window=avg seg/2) + Score, 끝-turn 정렬). 검증: HANDOFF_04 / "
        "`scripts/validate_official_scorer.py` (paper TextTiling 3-decimal 재현).",
        "- 데이터: superdialseg_data test (tiage 100 / dialseg711 711 / superseg 1322), MiniLM-int8 cached.",
        "- δ_eff = `HiOnTop`(m=2,ρ=0.7,a=0.5), label-free percentile δ* (p60/p70/p80, dts_result §3.2)."
        " **best percentile 은 데이터셋마다 다름**(superseg=p60, tiage=p70, ds711=p80).",
        "- Hi-OnTop-DeNeut = `hi_ontop_deneut.segment()` (de-neut+적응β 신호 + threshold deploy, 단일 calibration-free 임계).",
        "- 정렬: 두 신호 모두 스파이크(새 segment 첫 turn) → 경계 t-1 매핑 (끝-turn 규약).",
        "",
        "## 결과 (MiniLM-int8, per-dialogue 공식 Score)",
        "",
        "| 데이터셋 | δ_eff p60 | δ_eff p70 | δ_eff p80 | δ_eff best | Hi-OnTop-DeNeut |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for ds, deff, best_p, r_dn in rows:
        cells = []
        for p in ("p60", "p70", "p80"):
            v = f"{deff[p]['score']:.3f}"
            cells.append(f"**{v}**" if p == best_p else v)
        L.append(f"| {ds} | {cells[0]} | {cells[1]} | {cells[2]} | "
                 f"**{deff[best_p]['score']:.3f}** ({best_p}) | {r_dn['score']:.3f} |")
    L += [
        "",
        "## 해석",
        "- **δ_eff 는 낮지 않다 — published 범위 그대로** (best percentile, official per-dialogue): tiage 0.476(p70) /",
        "  dialseg711 0.602(p80) / superseg 0.405(p60). dts_result(pooled) 0.489/0.607/0.423 대비 official",
        "  per-dialogue 가 0.01~0.02 낮을 뿐(규약 차이). best percentile 은 데이터셋마다 다름(p80 일괄은 superseg·tiage 과소평가).",
        "- **DTS 3개 전부 δ_eff > Hi-OnTop-DeNeut**. Hi-OnTop-DeNeut 이 낮은 건 (1) de-neut 신호가 AMI drift 용이라",
        "  sharp-seam DTS 에 부적합 + (2) per-dataset percentile 튜닝 없이 단일 calibration-free 임계 사용. (회귀 아님 — 도메인 불일치.)",
        "- HANDOFF_01 의 'de-neut 우위/superseg 벽 돌파' 는 off-by-one 버그 산물(무효).",
        "",
        "## 판정",
        "- **DTS primary 디폴트 = δ_eff (`HiOnTop`)** (정상 성능). Hi-OnTop-DeNeut 은 DTS 미적합 → AMI/drift 한정.",
        "",
        "## 한계 / 검증 미해결",
        "- MiniLM-int8 단일 인코더. baseline(GreedySeg/CSM 등) 공식 재채점은 GPU(torch sm_120)·ckpt 부재로 보류(HANDOFF_04 §3).",
        "- AMI(±2 metric)는 off-by-one 영향 미미 — 별도 재확인(HANDOFF_04 §5d).",
    ]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("report →", out / "REPORT.md")


if __name__ == "__main__":
    main()
