#!/usr/bin/env python3
"""Hi-OnTop 실제 분절 시연 — 3개 벤치마크의 진짜 대화로.

각 벤치(TIAGE / Dialseg711 / SuperDialseg) test set 에서 예시 대화를 골라
Hi-OnTop(default config)를 turn 단위로 돌리고, 발화 텍스트 + δ_eff +
graded_score + 신뢰도(밴드별 경계 precision) + 예측/정답 경계를 한 표로
보여준다.

신뢰도(비율) = graded 밴드별로, 그 밴드에 속한 turn 중 실제 GT 경계인
비율 (= 밴드 precision). 해당 벤치 test set 전체에서 계산.

출력: outputs/experiments/2026-05-22_hiontop_segmentation_demo/REPORT.md
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
HP = dict(delta_star=0.5594, ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
BANDS = ["매우약", "약", "정상", "강"]
BAND_RANGE = {"매우약": "<0.7", "약": "0.7~1.0", "정상": "1.0~1.3", "강": "≥1.3"}


def load_raw(ds: str):
    """필터(len>=2) 후 dialog 리스트 — 임베딩 pkl 과 정렬 동일."""
    raw = json.loads((SDS / ds / "segmentation_file_test.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        turns = d["turns"]
        if len(turns) < 2:
            continue
        utts = [t["utterance"] for t in turns]
        roles = [str(t.get("role", t.get("speaker", ""))) for t in turns]
        yt = [int(t.get("segmentation_label", 0)) for t in turns]
        yt[-1] = 0
        out.append((utts, roles, yt))
    return out


def pick(dialogs, n_want=2, lo=14, hi=26, seg_lo=3, seg_hi=7):
    cand = [i for i, (u, r, yt) in enumerate(dialogs)
            if lo <= len(u) <= hi and seg_lo <= sum(yt) <= seg_hi]
    return cand[:n_want] if cand else list(range(min(n_want, len(dialogs))))


def trunc(s: str, n: int = 66) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def band(g: float) -> str:
    """graded_score → boundary strength 밴드 (Ben-Yakov & Henson 2018)."""
    if g < 0.7:
        return "매우약"
    if g < 1.0:
        return "약"
    if g < 1.3:
        return "정상"
    return "강"


def segment(embs):
    seg = HiOnTop(dim=embs.shape[1], **HP)
    rows = []
    for s in embs:
        tid, isb = seg.assign(s.astype(np.float64))
        rows.append((tid, isb, seg.last_delta_eff, seg.last_graded_score))
    return rows


def calibrate(dialogs, embs_all):
    """벤치 test set 전체에서 밴드별 [turn 수, GT 경계 수] → precision(신뢰도)."""
    cnt = {b: [0, 0] for b in BANDS}
    for di, (utts, roles, yt) in enumerate(dialogs):
        rows = segment(embs_all[di])
        n = len(utts)
        gt_new = [False] + [yt[i - 1] == 1 for i in range(1, n)]
        for i in range(1, n):  # turn 0 은 graded 미정의 → 제외
            b = band(rows[i][3])
            cnt[b][0] += 1
            if gt_new[i]:
                cnt[b][1] += 1
    prec = {b: (cnt[b][1] / cnt[b][0] if cnt[b][0] else 0.0) for b in BANDS}
    return cnt, prec


def main() -> None:
    out_dir = REPO / "outputs" / "experiments" / "2026-05-22_hiontop_segmentation_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    L = [
        "# Hi-OnTop 실제 분절 시연 — 3개 벤치마크 진짜 대화 + 신뢰도",
        "",
        "Hi-OnTop default config (`δ*=0.5594, m=2, ρ=0.7, a=0.5`) 를 각 벤치",
        "test set 의 실제 대화에 turn 단위로 돌린 결과. 발화는 데이터셋 원문",
        "(긴 것은 `…` 절단).",
        "",
        "## 신뢰도(비율)란",
        "",
        "Hi-OnTop 는 매 turn `graded = δ_eff / δ*` 를 낸다. 이걸 4 밴드로 나눈다:",
        "`매우약`(<0.7) · `약`(0.7~1.0) · `정상`(1.0~1.3) · `강`(≥1.3).",
        "**신뢰도(비율) = 그 밴드에 속한 turn 중 실제 GT 경계인 비율**(= 밴드",
        "precision). 각 벤치 test set *전체*에서 계산한다 — 즉 \"graded 가 이",
        "밴드면, 진짜 경계일 확률이 경험적으로 몇 %\"인가.",
        "",
        "→ 예측 경계(▶, graded≥1.0)는 `정상` 또는 `강` 밴드. 같은 ▶ 라도",
        "밴드 신뢰도가 다르므로, downstream 은 신뢰도로 약/강 경계를 구분",
        "처리할 수 있다.",
        "",
        "**표 읽는 법** — 행 = 한 turn.",
        "",
        "- `topic` : Hi-OnTop segment id (바뀌면 새 주제로 분절).",
        "- `δ_eff` : 그 turn 의 놀람 `a·δ_prev + (1−a)·δ_ctx`.",
        "- `graded` : `δ_eff / δ*`. **≥ 1.0 이면 경계 판정.**",
        "- `신뢰도` : 그 turn 의 graded 밴드 precision (벤치 전체 기준 비율).",
        "- `예측` : Hi-OnTop 가 새 주제 시작이라 판정한 turn → `▶`.",
        "- `정답` : 데이터셋 GT 경계 → `●`.  `평가` : ✓hit / ✗FP / ✗FN.",
        "",
        "turn 0 은 첫 segment 시작이라 경계로 세지 않음(δ_eff 미정의).",
        "",
    ]

    DS = [("tiage", "TIAGE — 인간 주석 잡담(PersonaChat 계열)"),
          ("dialseg711", "Dialseg711 — 합성(서로 다른 task 대화 concat), 경계 선명"),
          ("superseg", "SuperDialseg — 문서 기반 task 대화")]

    for ds, title in DS:
        dialogs = load_raw(ds)
        with open(CACHE / f"sds_emb_{ds}_test.pkl", "rb") as fh:
            embs_all = pickle.load(fh)
        cnt, prec = calibrate(dialogs, embs_all)
        L += [f"## {title}", "",
              f"### 신뢰도 보정표 ({ds} test 전체 {len(dialogs)} dialog)", "",
              "| 밴드 | graded 범위 | turn 수 | GT 경계 수 | **신뢰도(경계 비율)** |",
              "|:--:|:--:|--:|--:|--:|"]
        for b in BANDS:
            L.append(f"| {b} | {BAND_RANGE[b]} | {cnt[b][0]} | {cnt[b][1]} | "
                     f"**{prec[b]*100:.1f}%** |")
        L.append("")

        idxs = pick(dialogs)
        for ex_n, di in enumerate(idxs, 1):
            utts, roles, yt = dialogs[di]
            rows = segment(embs_all[di])
            n = len(utts)
            gt_new = [False] + [yt[i - 1] == 1 for i in range(1, n)]
            pred_new = [False] + [rows[i][0] != rows[i - 1][0] for i in range(1, n)]
            n_gt, n_pred = sum(gt_new), sum(pred_new)
            hit = sum(1 for i in range(n) if gt_new[i] and pred_new[i])
            L += [
                f"### 예시 {ex_n} — dialog #{di} ({n} turns)",
                "",
                f"GT 경계 {n_gt}개 · Hi-OnTop 예측 {n_pred}개 · "
                f"hit {hit} / FP {n_pred-hit} / FN {n_gt-hit}",
                "",
                "| # | topic | 발화 | δ_eff | graded | 신뢰도 | 예측 | 정답 | 평가 |",
                "|--:|:--:|:--|--:|--:|--:|:--:|:--:|:--:|",
            ]
            for i in range(n):
                tid, isb, deff, grd = rows[i]
                role = (roles[i] + ": ") if roles[i] else ""
                pm = "▶" if pred_new[i] else ""
                gm = "●" if gt_new[i] else ""
                ev = ("✓" if pred_new[i] and gt_new[i]
                      else "✗FP" if pred_new[i]
                      else "✗FN" if gt_new[i] else "")
                if i == 0:
                    deff_s = grd_s = conf_s = "—"
                else:
                    deff_s, grd_s = f"{deff:.3f}", f"{grd:.2f}"
                    conf_s = f"{prec[band(grd)]*100:.0f}%"
                L.append(
                    f"| {i} | {tid} | {trunc(role + utts[i])} | "
                    f"{deff_s} | {grd_s} | {conf_s} | {pm} | {gm} | {ev} |"
                )
            L.append("")

    L += [
        "## 해석",
        "",
        "- 신뢰도(밴드 precision)가 `강` 밴드로 갈수록 올라가면 graded 가",
        "  calibrated 된 신호라는 뜻 — 보정표에서 `매우약`→`강` 순으로 비율이",
        "  단조 증가하는지 확인.",
        "- 같은 예측 경계(▶)라도 신뢰도가 다르다: `강` 밴드 ▶ 는 믿고 commit,",
        "  `정상` 밴드 ▶ 는 약한 경계 — downstream 이 비율로 구분 처리 가능.",
        "- TIAGE 잡담형은 신뢰도 자체가 낮고(밴드 precision 낮음) FP 가 많다.",
        "  Dialseg711 은 `강` 밴드 신뢰도가 높아 hit 비율이 높다 — 벤치 Score",
        "  차이가 신뢰도 보정표에서 그대로 드러난다.",
        "",
        "## 한계",
        "",
        "- 신뢰도(밴드 precision)는 해당 벤치 test set 전체에서 계산 — 같은",
        "  test 로 보정·표시라 held-out 신뢰도 주장은 아님(시연용 calibration).",
        "- 예시 대화는 '길이 14~26 + GT 경계 3~7개' 앞쪽을 기계 선별",
        "  (cherry-pick 아님, 잘된 것·못된 것 그대로).",
        "- 밴드 경계값(0.7/1.0/1.3)은 휴리스틱.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"DONE → {out_dir/'REPORT.md'}")


if __name__ == "__main__":
    main()
